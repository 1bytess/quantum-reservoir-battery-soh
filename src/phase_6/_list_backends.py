"""List available backends for a given IBM account."""
import os, sys
from dotenv import load_dotenv
import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
acct = int(sys.argv[1]) if len(sys.argv) > 1 else 3
from qiskit_ibm_runtime import QiskitRuntimeService
svc = QiskitRuntimeService(channel="ibm_cloud", token=os.environ[f"IBM_ACC{acct}_API"], instance=os.environ[f"IBM_ACC{acct}_CRN"])
backends = svc.backends()
for b in backends:
    st = b.status()
    print(f"  {b.name}: {b.num_qubits}q, operational={st.operational}, pending={st.pending_jobs}")
