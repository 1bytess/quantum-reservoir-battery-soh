"""Shared plotting constants -- Tier-1 publication defaults.

Palette strategy
----------------
- One base color per model family.
- Sub-variants (38D vs PCA-6D) share the same color and differ by hatch.
- Quantum models use deeper hues so they stand out from classical baselines.
"""

import matplotlib.pyplot as plt

FAMILY_COLORS = {
    "ridge": "#009E73",
    "svr": "#E69F00",
    "rff": "#CC79A7",
    "esn": "#0072B2",
    "xgboost": "#56B4E9",
    "mlp": "#F0E442",
    "gp": "#44AA99",
    "cnn1d": "#DDCC77",
    "qrc_noiseless": "#D55E00",
    "qrc_noisy": "#882255",
    "qrc_hardware": "#332288",
    "qrc_nested": "#AA4499",
    "naive": "#999999",
}

_MODEL_TO_FAMILY = {
    "ridge": "ridge",
    "svr": "svr",
    "rff": "rff",
    "esn": "esn",
    "xgboost": "xgboost",
    "mlp": "mlp",
    "gp": "gp",
    "cnn1d": "cnn1d",
    "ridge_72d": "ridge",
    "svr_72d": "svr",
    "rff_72d": "rff",
    "esn_72d": "esn",
    "xgboost_72d": "xgboost",
    "ridge_38d": "ridge",
    "svr_38d": "svr",
    "rff_38d": "rff",
    "esn_38d": "esn",
    "xgboost_38d": "xgboost",
    "qrc_d1": "qrc_noiseless",
    "qrc_d2": "qrc_noiseless",
    "qrc_d3": "qrc_noiseless",
    "qrc_d4": "qrc_noiseless",
    "qrc_noiseless": "qrc_noiseless",
    "qrc_noiseless_d1": "qrc_noiseless",
    "qrc_noiseless_d2": "qrc_noiseless",
    "qrc_noiseless_d3": "qrc_noiseless",
    "qrc_noiseless_d4": "qrc_noiseless",
    "qrc_d1_noisy": "qrc_noisy",
    "qrc_d2_noisy": "qrc_noisy",
    "qrc_d3_noisy": "qrc_noisy",
    "qrc_d4_noisy": "qrc_noisy",
    "qrc_noisy": "qrc_noisy",
    "qrc_hardware": "qrc_hardware",
    "qrc_nested": "qrc_nested",
    "QRC": "qrc_nested",
    "XGBoost": "xgboost",
    "ESN": "esn",
}

HATCH_38D = "///"
HATCH_PCA6D = ""

LINESTYLE_38D = "--"
LINESTYLE_PCA6D = "-"


def is_38d(model_name: str) -> bool:
    """Return True if the model name refers to a raw-38D variant."""
    return model_name.endswith("_38d") or "38D" in model_name or "38d" in model_name


MODEL_LABELS = {
    "ridge": "Ridge (6D)",
    "svr": "SVR (6D)",
    "rff": "RFF (6D)",
    "esn": "ESN (6D)",
    "xgboost": "XGBoost (6D)",
    "mlp": "MLP (6D)",
    "gp": "GP (6D)",
    "cnn1d": "1D-CNN",
    "ridge_72d": "Ridge (72D)",
    "svr_72d": "SVR (72D)",
    "rff_72d": "RFF (72D)",
    "esn_72d": "ESN (72D)",
    "xgboost_72d": "XGBoost (72D)",
    "ridge_38d": "Ridge (38D)",
    "svr_38d": "SVR (38D)",
    "rff_38d": "RFF (38D)",
    "esn_38d": "ESN (38D)",
    "xgboost_38d": "XGBoost (38D)",
    "qrc_d1": "QRC d=1",
    "qrc_d2": "QRC d=2",
    "qrc_d3": "QRC d=3",
    "qrc_d4": "QRC d=4",
    "qrc_noiseless": "QRC (noiseless)",
    "qrc_noiseless_d1": "QRC d=1",
    "qrc_noiseless_d2": "QRC d=2",
    "qrc_d1_noisy": "QRC d=1 (noisy)",
    "qrc_d2_noisy": "QRC d=2 (noisy)",
    "qrc_d3_noisy": "QRC d=3 (noisy)",
    "qrc_d4_noisy": "QRC d=4 (noisy)",
    "qrc_noisy": "QRC (noisy)",
    "qrc_hardware": "QRC (hardware)",
    "qrc_nested": "QRC (nested)",
    "QRC": "QRC",
    "XGBoost": "XGBoost",
    "ESN": "ESN",
}

CELL_IDS_SECL = ["W3", "W8", "W9", "W10", "V4", "V5"]
_CELL_CMAP = plt.get_cmap("tab10")
CELL_COLORS = {
    cid: _CELL_CMAP(i / max(1, len(CELL_IDS_SECL) - 1))
    for i, cid in enumerate(CELL_IDS_SECL)
}

DPI = 300

MODEL_COLORS = {
    "ridge": FAMILY_COLORS["ridge"],
    "svr": FAMILY_COLORS["svr"],
    "rff": FAMILY_COLORS["rff"],
    "esn": FAMILY_COLORS["esn"],
    "xgboost": FAMILY_COLORS["xgboost"],
    "mlp": FAMILY_COLORS["mlp"],
    "gp": FAMILY_COLORS["gp"],
    "cnn1d": FAMILY_COLORS["cnn1d"],
    "qrc_noiseless": FAMILY_COLORS["qrc_noiseless"],
    "qrc_noisy": FAMILY_COLORS["qrc_noisy"],
    "qrc_hardware": FAMILY_COLORS["qrc_hardware"],
    "qrc_nested": FAMILY_COLORS["qrc_nested"],
    "temporal_qrc": FAMILY_COLORS["qrc_hardware"],
    "qrc_expanded": FAMILY_COLORS["qrc_nested"],
}

PAPER_MODELS = [
    "ridge",
    "svr",
    "xgboost",
    "linear_pc1",
    "rff",
    "esn",
    "gp",
    "cnn1d",
    "ridge_72d",
    "svr_72d",
    "xgboost_72d",
    "rff_72d",
    "esn_72d",
]


def tier1_rc(double_column: bool = False):
    """Return rcParams dict for publication figures."""
    return {
        "font.family": "serif",
        "font.size": 7 if not double_column else 8,
        "axes.titlesize": 8 if not double_column else 9,
        "axes.labelsize": 7 if not double_column else 8,
        "xtick.labelsize": 6.5 if not double_column else 7,
        "ytick.labelsize": 6.5 if not double_column else 7,
        "legend.fontsize": 6 if not double_column else 6.5,
        "legend.framealpha": 0.85,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
    }


def model_color(name: str) -> str:
    """Get color for a model name, resolving through the family map."""
    family = _MODEL_TO_FAMILY.get(name, None)
    if family:
        return FAMILY_COLORS[family]
    nl = name.lower()
    if "hardware" in nl:
        return FAMILY_COLORS["qrc_hardware"]
    if "noisy" in nl or "noise" in nl:
        return FAMILY_COLORS["qrc_noisy"]
    if "qrc" in nl or "quantum" in nl:
        return FAMILY_COLORS["qrc_noiseless"]
    if "xgboost" in nl or "xgb" in nl:
        return FAMILY_COLORS["xgboost"]
    if "mlp" in nl:
        return FAMILY_COLORS["mlp"]
    if "esn" in nl:
        return FAMILY_COLORS["esn"]
    if "ridge" in nl:
        return FAMILY_COLORS["ridge"]
    if "svr" in nl:
        return FAMILY_COLORS["svr"]
    if "rff" in nl:
        return FAMILY_COLORS["rff"]
    return "#999999"


def model_hatch(name: str) -> str:
    """Get hatch pattern for a model."""
    return HATCH_38D if is_38d(name) else HATCH_PCA6D


def model_label(name: str) -> str:
    """Get display label for a model name."""
    return MODEL_LABELS.get(name, name)


def save_fig(fig, directory, name):
    """Save as PNG and PDF and close the figure."""
    from pathlib import Path
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png/pdf -> {d}")
