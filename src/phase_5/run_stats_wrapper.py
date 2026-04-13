
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from src.phase_5.statistical_tests import run_statistical_tests
from src.phase_3.config import Phase3LabPaths
from src.phase_5.config import Phase5LabPaths

if __name__ == "__main__":
    p3 = Phase3LabPaths()
    p5 = Phase5LabPaths("stage_stats_wrapper")
    p3.ensure_dirs()
    p5.ensure_dirs()
    
    # Load data
    try:
        loco_df = pd.read_csv(p3.data_dir / "loco_results.csv")
        qrc_df = pd.read_csv(str(p3.data_dir).replace("phase_3", "phase_4") + "/qrc_noisy.csv")
        
        # Prepare comparison
        print("Comparing SVR vs QRC (Noisy)...")
        # Filter for best depth QRC (e.g. depth=1 or 2 with min MAE)
        best_depth = qrc_df.groupby("depth")["mae"].mean().idxmin()
        qrc_best = qrc_df[qrc_df["depth"] == best_depth].copy()
        qrc_best["model"] = "qrc_noisy"
        
        svr_df = loco_df[loco_df["model"] == "svr"].copy()
        print(f"SVR count: {len(svr_df)}, QRC count: {len(qrc_best)}")
        
        combined = pd.concat([svr_df, qrc_best])
        combined_path = p5.data_dir / "combined_svr_qrc.csv"
        combined.to_csv(combined_path, index=False)
        
        # Run tests
        stats = run_statistical_tests(combined_path, qrc_model="qrc_noisy", output_dir=p5.data_dir)
        print("\nStatistical Test Results:")
        print(stats)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
