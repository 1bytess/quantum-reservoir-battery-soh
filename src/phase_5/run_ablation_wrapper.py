
import sys
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root))

from src.phase_3.data_loader import get_cell_data
from src.phase_3.config import Phase3LabPaths
from src.phase_5.ablation_pca import run_pca_ablation
from src.phase_5.config import Phase5LabPaths

if __name__ == "__main__":
    paths = Phase3LabPaths()
    print("Loading data...")
    cell_data = get_cell_data(paths)
    
    out_dir = Phase5LabPaths("stage_pca_ablation").data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running PCA ablation...")
    run_pca_ablation(cell_data, models=["ridge", "svr"], output_dir=out_dir)
