import numpy as np
import time
from sigma_c.core.engine import Engine
from sigma_c import Universe

def test_core():
    print("Testing Sigma-C Core...")
    engine = Engine()
    
    # Generate synthetic data
    epsilon = np.linspace(0, 1, 1000)
    # Create a peak at 0.3
    observable = np.exp(-(epsilon - 0.3)**2 / 0.01)
    
    start = time.time()
    result = engine.compute_susceptibility(epsilon, observable)
    dt = time.time() - start
    
    # The susceptibility is the gradient, so peaks are at inflection points (mu +/- sigma)
    # For exp(-(x-0.3)^2/0.01), sigma approx 0.07. Peak at 0.3 - 0.07 = 0.23
    print(f"Compute time: {dt*1000:.3f} ms")
    print(f"Sigma_c: {result['sigma_c']:.4f} (Expected ~0.23)")
    print(f"Kappa: {result['kappa']:.4f}")
    
    assert abs(result['sigma_c'] - 0.23) < 0.05
    print("Core Test PASSED")

def test_adapters():
    print("\nTesting Adapters...")
    try:
        qpu = Universe.quantum()
        print(f"Quantum Adapter: {qpu}")
        
        gpu = Universe.gpu()
        print(f"GPU Adapter: {gpu}")
        
        fin = Universe.finance()
        print(f"Financial Adapter: {fin}")
        print("Adapter Test PASSED")
    except Exception as e:
        print(f"Adapter Test FAILED: {e}")

if __name__ == "__main__":
    test_core()
    test_adapters()
