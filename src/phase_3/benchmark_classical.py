"""Benchmark classical execution time (XGBoost) for comparison with QRC."""
import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_loader import load_stanford_data
from phase_3.models import get_model_pipeline
from config import CELL_IDS

def main():
    print("Loading data...")
    cell_data = load_stanford_data()
    
    print("Running XGBoost timing benchmark...")
    start_time = time.time()
    
    # We do a single LOCO pass to mimic what QRC does
    model = get_model_pipeline("xgboost")
    
    for test_cell in CELL_IDS:
        train_cells = [c for c in CELL_IDS if c != test_cell]
        X_train_raw = np.vstack([cell_data[c]["X_raw"] for c in train_cells])
        y_train = np.concatenate([cell_data[c]["y"] for c in train_cells])
        X_test_raw = cell_data[test_cell]["X_raw"]
        
        # Fit and predict
        model.fit(X_train_raw, y_train)
        model.predict(X_test_raw)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print("-" * 50)
    print(f"Classical (XGBoost) Execution Time (6 LOCO folds): {total_time:.4f} seconds")
    print("-" * 50)

if __name__ == "__main__":
    main()
