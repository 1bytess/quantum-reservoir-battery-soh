"""Computational cost benchmarking for all models.

Times train + predict cycles and measures peak memory usage.
Output: benchmark_timing.csv with wall-clock seconds and memory (MB).
"""

import time
import tracemalloc
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.decomposition import PCA

from .config import CELL_IDS, RANDOM_STATE
from src.phase_3.models import get_model_pipeline
from src.phase_3.config import MODEL_NAMES
from ..phase_4.qrc_model import QuantumReservoir
from ..phase_4.config import USE_ZZ_CORRELATORS

# ── PCA defaults ──────────────────────────────────────────────────────────
N_PCA_COMPONENTS = 6
PCA_RANDOM_STATE = 42


def _fit_pca_in_fold(
    X_train_72d: np.ndarray,
    X_test_72d: np.ndarray,
    n_components: int = N_PCA_COMPONENTS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit PCA on training data only, transform both train and test."""
    n_components = min(n_components, X_train_72d.shape[0], X_train_72d.shape[1])
    pca = PCA(n_components=n_components, random_state=PCA_RANDOM_STATE)
    X_train_6d = pca.fit_transform(X_train_72d)
    X_test_6d = pca.transform(X_test_72d)
    return X_train_6d, X_test_6d


def _get_one_fold(cell_data: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get train/test split from the first LOCO fold for benchmarking."""
    test_cell = CELL_IDS[0]
    train_cells = [c for c in CELL_IDS if c != test_cell]

        X_train_72d = np.vstack([cell_data[c]["X_72d"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test_72d = cell_data[test_cell]["X_72d"]
        y_test = cell_data[test_cell]["y"]

        X_train, X_test = _fit_pca_in_fold(X_train_72d, X_test_72d)
        return X_train, y_train, X_test, y_test


def benchmark_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_runs: int = 3,
) -> Dict:
    """Benchmark a single model's train + predict cycle.

    Returns:
        Dict with timing and memory stats.
    """
    times = []
    peak_mems = []

    for _ in range(n_runs):
        model = get_model_pipeline(model_name)

        tracemalloc.start()
        t0 = time.perf_counter()

        model.fit(X_train, y_train)
        _ = model.predict(X_test)

        t1 = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(t1 - t0)
        peak_mems.append(peak_mem / 1024 / 1024)  # bytes → MB

    return {
        "model": model_name,
        "time_mean_s": np.mean(times),
        "time_std_s": np.std(times),
        "time_min_s": np.min(times),
        "peak_mem_mb": np.mean(peak_mems),
    }


def benchmark_qrc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    depth: int = 3,
    n_runs: int = 3,
) -> Dict:
    """Benchmark QRC's train + predict cycle."""
    times = []
    peak_mems = []

    for _ in range(n_runs):
        qrc = QuantumReservoir(
            depth=depth,
            use_zz=USE_ZZ_CORRELATORS,
            use_classical_fallback=True,
            add_random_rotations=True,
        )

        tracemalloc.start()
        t0 = time.perf_counter()

        qrc.fit(X_train, y_train)
        _ = qrc.predict(X_test)

        t1 = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(t1 - t0)
        peak_mems.append(peak_mem / 1024 / 1024)

    return {
        "model": f"qrc_d{depth}",
        "time_mean_s": np.mean(times),
        "time_std_s": np.std(times),
        "time_min_s": np.min(times),
        "peak_mem_mb": np.mean(peak_mems),
    }


def run_benchmark(
    cell_data: Dict[str, Dict],
    models: List[str] = None,
    qrc_depths: List[int] = None,
    output_dir: Path = None,
    n_runs: int = 3,
) -> pd.DataFrame:
    """Run computational cost benchmarks for all models.

    Args:
        cell_data: Per-cell EIS data
        models: Classical model names to benchmark (default: MODEL_NAMES)
        qrc_depths: QRC depths to benchmark (default: [2, 3, 4])
        output_dir: Directory to save results
        n_runs: Number of timing runs per model

    Returns:
        DataFrame with timing and memory benchmarks
    """
    if models is None:
        models = MODEL_NAMES
    if qrc_depths is None:
        qrc_depths = [2, 3, 4]

    # Get a single fold for benchmarking (consistency)
    X_train, y_train, X_test, y_test = _get_one_fold(cell_data)

    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║  Computational Cost Benchmark            ║")
    print("  ╚══════════════════════════════════════════╝")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  n_runs per model: {n_runs}\n")

    results = []

    # Classical models
    for model_name in models:
        try:
            r = benchmark_model(model_name, X_train, y_train, X_test, n_runs)
            results.append(r)
            print(f"  [OK] {model_name:15s}: {r['time_mean_s']:.4f}s ± {r['time_std_s']:.4f}s, "
                  f"mem={r['peak_mem_mb']:.2f}MB")
        except Exception as e:
            print(f"  [X] {model_name:15s}: FAILED ({e})")

    # QRC
    for depth in qrc_depths:
        try:
            r = benchmark_qrc(X_train, y_train, X_test, depth, n_runs)
            results.append(r)
            print(f"  [OK] {r['model']:15s}: {r['time_mean_s']:.4f}s ± {r['time_std_s']:.4f}s, "
                  f"mem={r['peak_mem_mb']:.2f}MB")
        except Exception as e:
            print(f"  [X] qrc_d{depth}:         FAILED ({e})")

    results_df = pd.DataFrame(results)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "benchmark_timing.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\n  Saved {out_path}")

    return results_df
