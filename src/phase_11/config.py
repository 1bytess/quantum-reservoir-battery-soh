"""Configuration for Phase 11: Reviewer Response — Statistical Robustness & Baseline Verification.

Addresses the following reviewer concerns:
  1. CNN1D baseline: verify real PyTorch was used (not SVR fallback)
  2. Pre-registered QRC vs XGBoost: single primary comparison, raw p-value
  3. Bootstrap CI as primary statistical evidence
  4. Holm correction analysis with proper reviewer-response framing
  5. Few-shot reframe: "best accuracy-efficiency trade-off in 9-18 cell regime"
  6. Limitations audit: PCA leakage (W3), temporal QRC, transfer learning

Primary dataset: Warwick DIB (24 NMC811 cells, LOCO-CV)
  - 24-fold LOCO gives substantially more statistical power than Stanford (6-fold)
  - QRC MAE = 0.93% vs XGBoost MAE = 1.51% — 38% improvement (strong result)
  - QRC beats ALL baselines on Warwick (unlike Stanford where Ridge wins)
  - Reviewer flagged p=0.063 Holm-corrected specifically for Warwick data

Secondary dataset: Stanford (6 LCO cells)
  - Presented as "competitive performance on small dataset"
  - Ridge beats QRC on Stanford — acknowledged honestly in Limitations
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_11_ROOT = PROJECT_ROOT / "result" / "phase_11"

# Inherited constants
RANDOM_STATE = 42
N_PCA = 6

# Nominal capacities
WARWICK_NOMINAL_AH = 5.0
STANFORD_NOMINAL_AH = 4.85

# Bootstrap resamples for CI estimation
N_BOOTSTRAP = 10_000

# Few-shot reframe: regime of interest from reviewer
# "best accuracy-efficiency trade-off in 9-18 cell regime"
FEWSHOT_REGIME_LOW = 9
FEWSHOT_REGIME_HIGH = 18

# ── Warwick LOCO model name mapping (phase_8 stage_2 naming convention) ──────
# Models as stored in result/phase_8/stage_2/data/nested_warwick_loco_*.csv
WARWICK_QRC_MODEL = "qrc"
WARWICK_BASELINES = {
    "xgboost": "xgboost",
    "ridge":   "ridge",
    "svr":     "svr",
    "esn":     "esn",
    "rff":     "rff",
}
WARWICK_DISPLAY_NAMES = {
    "qrc":     "QRC",
    "xgboost": "XGBoost",
    "ridge":   "Ridge",
    "svr":     "SVR",
    "esn":     "ESN",
    "rff":     "RFF",
}

# Warwick condition
WARWICK_TEMP = "25degC"
WARWICK_SOC  = "50SOC"

# Phase 8 result paths (primary source — load before re-running)
PHASE_8_LOCO_CSV     = PROJECT_ROOT / "result" / "phase_8" / "stage_2" / "data" / "nested_warwick_loco_predictions.csv"
PHASE_8_SUMMARY_CSV  = PROJECT_ROOT / "result" / "phase_8" / "stage_2" / "data" / "nested_warwick_loco_summary.csv"

# Primary pre-registered comparison (single test, no correction needed)
PRIMARY_COMPARISON = (WARWICK_QRC_MODEL, WARWICK_BASELINES["xgboost"])

# All Warwick baseline model keys (phase_8 naming)
ALL_WARWICK_BASELINES = list(WARWICK_BASELINES.values())


def get_stage_paths(stage_name: str) -> tuple[Path, Path]:
    """Create and return (data_dir, plot_dir) for a Phase 11 stage."""
    data_dir = PHASE_11_ROOT / stage_name / "data"
    plot_dir = PHASE_11_ROOT / stage_name / "plot"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, plot_dir
