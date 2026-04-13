"""Unified non-hardware LOCO benchmark for Stanford and Warwick.

This script evaluates both datasets through the same code path:
1. Load per-cell raw EIS features
2. Apply train-only StandardScaler + PCA(6) in each outer LOCO fold
3. Use group-aware inner CV for classical hyperparameter search
4. Use group-aware nested depth search for QRC

It is intended to remove cross-dataset configuration drift from the
paper-facing evaluation code.

Examples
--------
python -m src.manuscript_support.unified_loco_benchmark
python -m src.manuscript_support.unified_loco_benchmark --datasets stanford warwick --models qrc xgboost ridge
python -m src.manuscript_support.unified_loco_benchmark --datasets warwick --align-warwick-to-stanford-freqs
python -m src.manuscript_support.unified_loco_benchmark --max-folds 2
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    _SRC_DIR = Path(__file__).resolve().parents[1]
    if str(_SRC_DIR) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR))

    from config import N_PCA, NOMINAL_CAPACITIES, PROJECT_ROOT, RANDOM_STATE
    from data_loader import load_stanford_data
    from data_loader_warwick import load_warwick_data
    from phase_3.config import MODEL_NAMES, MLP_MODEL_NAMES
    from phase_3.models import get_model_pipeline, get_param_grid
    from phase_4.qrc_model import QuantumReservoir
else:
    from ..config import N_PCA, NOMINAL_CAPACITIES, PROJECT_ROOT, RANDOM_STATE
    from ..data_loader import load_stanford_data
    from ..data_loader_warwick import load_warwick_data
    from ..phase_3.config import MODEL_NAMES, MLP_MODEL_NAMES
    from ..phase_3.models import get_model_pipeline, get_param_grid
    from ..phase_4.qrc_model import QuantumReservoir


DEFAULT_MODELS = ["qrc", "xgboost", "ridge", "svr", "rff", "esn"]
QRC_DEPTHS = [1, 2, 3, 4]


@dataclass
class DatasetBundle:
    name: str
    cell_data: dict[str, dict[str, np.ndarray]]
    cell_ids: list[str]
    nominal_ah: float
    feature_variant: str
    raw_feature_dim: int


def _json_dumps(value: Any) -> str:
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    if isinstance(value, dict):
        value = {str(k): _convert(v) for k, v in value.items()}
    return json.dumps(value, sort_keys=True)


def _fit_projection_in_fold(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    n_components: int = N_PCA,
) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(n_components, X_train_raw.shape[0], X_train_raw.shape[1])
    scaler = StandardScaler()
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    return X_train_pca, X_test_pca


def _make_train_groups(
    cell_data: dict[str, dict[str, np.ndarray]],
    train_cells: list[str],
) -> np.ndarray:
    return np.concatenate([
        np.full(len(cell_data[cell_id]["y"]), cell_id, dtype=object)
        for cell_id in train_cells
    ])


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nominal_ah: float,
) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mae_pct": float(mae * 100.0),
        "rmse_pct": float(rmse * 100.0),
        "mae_ah": float(mae * nominal_ah),
        "rmse_ah": float(rmse * nominal_ah),
    }


def _align_warwick_to_reference_freqs(
    warwick_data: dict[str, dict[str, np.ndarray]],
    reference_freq: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    sample_cell = next(iter(sorted(warwick_data.keys())))
    warwick_freq = np.asarray(warwick_data[sample_cell]["freq"], dtype=float)
    ref_freq = np.asarray(reference_freq, dtype=float)

    log_w = np.log10(warwick_freq)
    log_ref = np.log10(ref_freq)
    indices = np.array([int(np.argmin(np.abs(log_w - f))) for f in log_ref], dtype=int)

    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Frequency alignment produced duplicate Warwick indices: {indices.tolist()}")

    aligned: dict[str, dict[str, np.ndarray]] = {}
    n_w = len(warwick_freq)
    for cell_id, record in warwick_data.items():
        X_raw = np.asarray(record["X_raw"], dtype=float)
        re_z = X_raw[:, :n_w]
        im_z = X_raw[:, n_w:]
        X_aligned = np.concatenate([re_z[:, indices], im_z[:, indices]], axis=1)
        aligned[cell_id] = {
            **record,
            "X_raw": X_aligned,
            "freq": warwick_freq[indices],
        }

    return aligned, indices


def load_dataset_bundle(
    dataset: str,
    align_warwick_to_stanford_freqs: bool = False,
) -> DatasetBundle:
    if dataset == "stanford":
        cell_data = load_stanford_data()
        cell_ids = list(cell_data.keys())
        raw_feature_dim = int(next(iter(cell_data.values()))["X_raw"].shape[1])
        return DatasetBundle(
            name="stanford",
            cell_data=cell_data,
            cell_ids=cell_ids,
            nominal_ah=NOMINAL_CAPACITIES["stanford"],
            feature_variant="native_pca6",
            raw_feature_dim=raw_feature_dim,
        )

    if dataset == "warwick":
        cell_data = load_warwick_data()
        feature_variant = "native_pca6"
        if align_warwick_to_stanford_freqs:
            stanford_freq = next(iter(load_stanford_data().values()))["freq"]
            cell_data, indices = _align_warwick_to_reference_freqs(cell_data, stanford_freq)
            feature_variant = f"aligned19_pca6_idx_{'-'.join(map(str, indices.tolist()))}"

        raw_feature_dim = int(next(iter(cell_data.values()))["X_raw"].shape[1])
        return DatasetBundle(
            name="warwick",
            cell_data=cell_data,
            cell_ids=sorted(cell_data.keys()),
            nominal_ah=NOMINAL_CAPACITIES["warwick"],
            feature_variant=feature_variant,
            raw_feature_dim=raw_feature_dim,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def _fit_classical_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    model = get_model_pipeline(model_name)
    if model_name == "xgboost":
        model.set_params(xgb__n_jobs=1)

    param_grid = get_param_grid(model_name)
    meta: dict[str, Any] = {
        "inner_mae": float("nan"),
        "selected_depth": float("nan"),
        "selected_ridge_alpha": float("nan"),
        "selected_params": "",
        "search_trace": "",
    }

    if param_grid:
        n_groups = len(np.unique(train_groups))
        if n_groups >= 2:
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=LeaveOneGroupOut(),
                scoring="neg_mean_absolute_error",
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train, groups=train_groups)
        else:
            cv_folds = min(3, len(y_train))
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring="neg_mean_absolute_error",
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)

        model = search.best_estimator_
        meta["inner_mae"] = float(-search.best_score_)
        meta["selected_params"] = _json_dumps(search.best_params_)
    else:
        model.fit(X_train, y_train)

    return model, meta


def _run_qrc_nested_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    logo = LeaveOneGroupOut()
    depth_rows: list[dict[str, Any]] = []

    for depth in QRC_DEPTHS:
        fold_maes: list[float] = []
        fold_best_alphas: list[float] = []

        for inner_train_idx, inner_val_idx in logo.split(X_train, y_train, groups=train_groups):
            X_inner_train = X_train[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]
            X_inner_val = X_train[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]
            inner_groups = train_groups[inner_train_idx]

            qrc = QuantumReservoir(
                depth=depth,
                use_zz=True,
                ridge_alpha=None,
                use_classical_fallback=False,
                add_random_rotations=True,
                observable_set="Z",
            )
            qrc.fit(X_inner_train, y_inner_train, groups=inner_groups)
            y_inner_pred = qrc.predict(X_inner_val)
            fold_maes.append(float(mean_absolute_error(y_inner_val, y_inner_pred)))
            fold_best_alphas.append(float(getattr(qrc, "best_alpha_", np.nan)))

        depth_rows.append({
            "depth": depth,
            "mean_inner_mae": float(np.mean(fold_maes)),
            "std_inner_mae": float(np.std(fold_maes, ddof=0)),
            "mean_inner_alpha": float(np.nanmean(fold_best_alphas)),
        })

    depth_df = pd.DataFrame(depth_rows).sort_values(
        ["mean_inner_mae", "depth"], ascending=[True, True]
    ).reset_index(drop=True)
    best_depth = int(depth_df.iloc[0]["depth"])
    best_inner_mae = float(depth_df.iloc[0]["mean_inner_mae"])

    final_qrc = QuantumReservoir(
        depth=best_depth,
        use_zz=True,
        ridge_alpha=None,
        use_classical_fallback=False,
        add_random_rotations=True,
        observable_set="Z",
    )
    final_qrc.fit(X_train, y_train, groups=train_groups)
    y_pred = final_qrc.predict(X_test)

    meta = {
        "inner_mae": best_inner_mae,
        "selected_depth": best_depth,
        "selected_ridge_alpha": float(getattr(final_qrc, "best_alpha_", np.nan)),
        "selected_params": _json_dumps({
            "depth": best_depth,
            "use_zz": True,
            "observable_set": "Z",
            "ridge_alpha": float(getattr(final_qrc, "best_alpha_", np.nan)),
        }),
        "search_trace": _json_dumps({
            f"depth_{int(row.depth)}": {
                "mean_inner_mae": float(row.mean_inner_mae),
                "std_inner_mae": float(row.std_inner_mae),
                "mean_inner_alpha": float(row.mean_inner_alpha),
            }
            for row in depth_df.itertuples(index=False)
        }),
    }
    return y_pred, meta


def run_unified_loco(
    bundle: DatasetBundle,
    models: list[str],
    max_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    test_cell_ids = bundle.cell_ids[:max_folds] if max_folds is not None else bundle.cell_ids

    print(f"\nDataset: {bundle.name}")
    print(f"  Cells: {len(bundle.cell_ids)} total, evaluating {len(test_cell_ids)} outer folds")
    print(f"  Raw feature dim: {bundle.raw_feature_dim}")
    print(f"  Variant: {bundle.feature_variant}")
    print(f"  Models: {', '.join(models)}")

    for outer_fold, test_cell in enumerate(test_cell_ids, start=1):
        train_cells = [cell_id for cell_id in bundle.cell_ids if cell_id != test_cell]
        X_train_raw = np.vstack([bundle.cell_data[cell_id]["X_raw"] for cell_id in train_cells])
        y_train = np.concatenate([bundle.cell_data[cell_id]["y"] for cell_id in train_cells])
        train_groups = _make_train_groups(bundle.cell_data, train_cells)
        X_test_raw = bundle.cell_data[test_cell]["X_raw"]
        y_test = bundle.cell_data[test_cell]["y"]

        X_train_pca, X_test_pca = _fit_projection_in_fold(X_train_raw, X_test_raw)
        naive_mae = float(mean_absolute_error(y_test, np.full_like(y_test, y_train.mean())))

        print(f"  [Fold {outer_fold:02d}/{len(test_cell_ids)}] hold out {test_cell}")
        for model_name in models:
            if model_name == "qrc":
                y_pred, meta = _run_qrc_nested_search(
                    X_train=X_train_pca,
                    y_train=y_train,
                    train_groups=train_groups,
                    X_test=X_test_pca,
                )
            else:
                estimator, meta = _fit_classical_model(
                    model_name=model_name,
                    X_train=X_train_pca,
                    y_train=y_train,
                    train_groups=train_groups,
                )
                y_pred = estimator.predict(X_test_pca)

            metrics = _compute_metrics(y_test, y_pred, nominal_ah=bundle.nominal_ah)
            row = {
                "dataset": bundle.name,
                "feature_variant": bundle.feature_variant,
                "model": model_name,
                "outer_fold": outer_fold,
                "test_cell": test_cell,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "raw_feature_dim": int(bundle.raw_feature_dim),
                "pca_components": int(X_train_pca.shape[1]),
                "naive_mean_mae": naive_mae,
                "beats_naive": metrics["mae"] < naive_mae,
                **metrics,
                **meta,
            }
            fold_rows.append(row)
            print(f"    {model_name:<8} MAE={metrics['mae_pct']:.3f}%")

    folds_df = pd.DataFrame(fold_rows)
    summary_df = (
        folds_df.groupby(["dataset", "feature_variant", "model"], as_index=False)["mae"]
        .agg(mae_mean="mean", mae_std="std", n_folds="count")
        .sort_values(["dataset", "mae_mean", "model"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    summary_df["mae_pct_mean"] = summary_df["mae_mean"] * 100.0
    summary_df["mae_pct_std"] = summary_df["mae_std"] * 100.0
    return folds_df, summary_df


def _get_output_dir() -> Path:
    out_dir = PROJECT_ROOT / "result" / "manuscript_support" / "unified_loco" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _make_run_tag(args: argparse.Namespace) -> str:
    datasets_tag = "-".join(args.datasets)
    models_tag = "-".join(args.models)
    variant_tag = "aligned19" if args.align_warwick_to_stanford_freqs else "native"
    folds_tag = f"max{args.max_folds}" if args.max_folds is not None else "full"
    return f"{datasets_tag}__{models_tag}__{variant_tag}__{folds_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified non-hardware LOCO benchmark for Stanford and Warwick."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["stanford", "warwick"],
        default=["stanford", "warwick"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models to evaluate. Use 'qrc' plus any classical model from phase_3 "
            f"({', '.join(MODEL_NAMES + MLP_MODEL_NAMES)})."
        ),
    )
    parser.add_argument(
        "--align-warwick-to-stanford-freqs",
        action="store_true",
        default=False,
        help="Project Warwick 61-frequency raw EIS onto the Stanford 19-frequency grid before PCA.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Evaluate only the first N outer folds for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_models = set(MODEL_NAMES + MLP_MODEL_NAMES + ["qrc"])
    invalid_models = [model for model in args.models if model not in valid_models]
    if invalid_models:
        raise ValueError(f"Unknown model(s): {invalid_models}")

    out_dir = _get_output_dir()
    run_tag = _make_run_tag(args)
    all_folds: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []

    print("=" * 72)
    print("Unified Non-Hardware LOCO Benchmark")
    print("=" * 72)
    print(f"Output dir: {out_dir}")

    for dataset in args.datasets:
        bundle = load_dataset_bundle(
            dataset=dataset,
            align_warwick_to_stanford_freqs=args.align_warwick_to_stanford_freqs,
        )
        folds_df, summary_df = run_unified_loco(
            bundle=bundle,
            models=args.models,
            max_folds=args.max_folds,
        )
        all_folds.append(folds_df)
        all_summary.append(summary_df)

    combined_folds = pd.concat(all_folds, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)

    folds_path = out_dir / f"unified_loco__{run_tag}__folds.csv"
    summary_path = out_dir / f"unified_loco__{run_tag}__summary.csv"
    meta_path = out_dir / f"unified_loco__{run_tag}__run_metadata.json"

    combined_folds.to_csv(folds_path, index=False)
    combined_summary.to_csv(summary_path, index=False)
    meta = {
        "datasets": args.datasets,
        "models": args.models,
        "align_warwick_to_stanford_freqs": bool(args.align_warwick_to_stanford_freqs),
        "max_folds": args.max_folds,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(combined_summary.to_string(index=False))
    print(f"\nSaved folds:   {folds_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved meta:    {meta_path}")


if __name__ == "__main__":
    main()
