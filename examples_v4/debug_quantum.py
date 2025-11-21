import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sigma_c.adapters.quantum import QuantumAdapter
import inspect

print(f"QuantumAdapter file: {inspect.getfile(QuantumAdapter)}")
qpu = QuantumAdapter()
print(f"Has run_optimization: {hasattr(qpu, 'run_optimization')}")
print(f"Dir: {dir(qpu)}")
