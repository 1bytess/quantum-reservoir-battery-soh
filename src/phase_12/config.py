"""Configuration for Phase 12: ECM baseline development on Warwick."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_12_ROOT = PROJECT_ROOT / "result" / "phase_12"

RANDOM_STATE = 42
WARWICK_TEMP = "25degC"
WARWICK_SOC = "50SOC"
WARWICK_NOMINAL_AH = 5.0

# Reuse the same ridge grid used elsewhere in the manuscript for a fair readout.
RIDGE_ALPHAS = [
    1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1,
    1.0, 3.0, 10.0, 30.0, 100.0,
]

# Legacy proxy feature set retained as a simple fallback summary of raw impedance.
ECM_PROXY_FEATURES = [
    "r_ohm_ohm",
    "r_lowfreq_ohm",
    "delta_r_ohm",
    "im_abs_peak_ohm",
    "peak_freq_hz",
    "tau_peak_s",
    "c_pseudo_f",
    "area_abs_im_ohm_loghz",
    "lowfreq_re_slope",
]

ECM_CANDIDATE_MODELS = ["L_R_Rc", "L_R_Rcpe", "L_R_Rcpe_w"]
ECM_PARAMETER_FEATURES = [
    "L_h",
    "r0_ohm",
    "r1_ohm",
    "c1_f",
    "q1",
    "alpha1",
    "warburg_sigma",
]


def get_stage_paths(stage_name: str) -> tuple[Path, Path]:
    """Create and return (data_dir, plot_dir) for a Phase 12 stage."""
    data_dir = PHASE_12_ROOT / stage_name / "data"
    plot_dir = PHASE_12_ROOT / stage_name / "plot"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, plot_dir
