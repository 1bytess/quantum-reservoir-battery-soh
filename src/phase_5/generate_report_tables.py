
import pandas as pd
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent.parent
    res_dir = root / "result"
    phase_5_dir = res_dir / "phase_5"
    
    with open(root / "src" / "phase_5" / "report_tables.md", "w", encoding="utf-8") as f:
        # 1. Phase 3 (Baselines)
        loco_path = res_dir / "phase_3" / "data" / "loco_results.csv"
        if loco_path.exists():
            df = pd.read_csv(loco_path)
            f.write("\n## Phase 3 Baselines (LOCO)\n")
            f.write("| Model | Mean MAE | Std MAE | Count |\n")
            f.write("|-------|----------|---------|-------|\n")
            stats = df.groupby("model")["mae"].agg(["mean", "std", "count"]).sort_values("mean")
            for idx, row in stats.iterrows():
                f.write(f"| {idx} | {row['mean']:.4e} | {row['std']:.4e} | {int(row['count'])} |\n")

        # 2. Phase 4 (QRC Noiseless)
        nl_path = res_dir / "phase_4" / "data" / "qrc_noiseless.csv"
        if nl_path.exists():
            df = pd.read_csv(nl_path)
            f.write("\n## Phase 4 QRC Noiseless (LOCO)\n")
            # Best noiseless config
            best_nl_depth = df.groupby("depth")["mae"].mean().idxmin()
            best_nl_mae = df.groupby("depth")["mae"].mean().min()
            f.write(f"\n**Best Noiseless QRC:** Depth={best_nl_depth} -> MAE={best_nl_mae:.4f}\n")
            
            f.write("\n| Depth | Mean MAE | Std MAE |\n")
            f.write("|-------|----------|---------|\n")
            stats = df.groupby("depth")["mae"].agg(["mean", "std"]).sort_values("mean")
            for idx, row in stats.iterrows():
                f.write(f"| {idx} | {row['mean']:.4f} | {row['std']:.4f} |\n")

        # 3. Phase 4 (QRC Noisy)
        n_path = res_dir / "phase_4" / "data" / "qrc_noisy.csv"
        if n_path.exists():
            df = pd.read_csv(n_path)
            f.write("\n## Phase 4 QRC Noisy (LOCO)\n")
            # Best overall configuration
            best_cfg = df.groupby(["depth", "shots"])["mae"].mean().idxmin()
            best_mae = df.groupby(["depth", "shots"])["mae"].mean().min()
            f.write(f"\n**Best Noisy QRC:** Depth={best_cfg[0]}, Shots={best_cfg[1]} -> MAE={best_mae:.4f}\n")
            
            f.write("\n| Depth | Shots | Mean MAE | Std MAE |\n")
            f.write("|-------|-------|----------|---------|\n")
            stats = df.groupby(["depth", "shots"])["mae"].agg(["mean", "std"]).sort_values("mean")
            for idx, row in stats.iterrows():
                f.write(f"| {idx[0]} | {idx[1]} | {row['mean']:.4f} | {row['std']:.4f} |\n")

        # 4. Phase 5 (PCA Ablation)
        abl_path = phase_5_dir / "stage_pca_ablation" / "data" / "ablation_summary.csv"
        if abl_path.exists():
            df = pd.read_csv(abl_path)
            f.write("\n## Phase 5 PCA Ablation\n")
            f.write("| Model | In-Fold PCA | Global PCA | Raw 72D | Leakage Delta |\n")
            f.write("|-------|-------------|------------|---------|---------------|\n")
            for idx, row in df.iterrows():
                # Handling large numbers or small numbers
                m_if = f"{row['mae_in_fold_pca']:.4e}" if row['mae_in_fold_pca'] > 10 else f"{row['mae_in_fold_pca']:.4f}"
                m_g = f"{row['mae_global_pca']:.4e}" if row['mae_global_pca'] > 10 else f"{row['mae_global_pca']:.4f}"
                m_r = f"{row['mae_raw_72d']:.4e}" if row['mae_raw_72d'] > 10 else f"{row['mae_raw_72d']:.4f}"
                ld = f"{row['leakage_delta']:.4e}"
                f.write(f"| {row['model']} | {m_if} | {m_g} | {m_r} | {ld} |\n")
                
    print(f"Report saved to {root / 'src' / 'phase_5' / 'report_tables.md'}")

if __name__ == "__main__":
    main()
