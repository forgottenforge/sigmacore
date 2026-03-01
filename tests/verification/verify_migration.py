import sys
import traceback

print("Starting verification script...", flush=True)

try:
    import numpy as np
    import pandas as pd
    from unittest.mock import MagicMock
    
    # Mock yfinance
    sys.modules['yfinance'] = MagicMock()
    sys.modules['yfinance'].download.return_value = pd.DataFrame({
        'Close': np.exp(np.cumsum(np.random.normal(0, 0.01, 1000))),
        'Volume': np.random.randint(100, 1000, 1000)
    })
    print("Mocks set up.", flush=True)

    from sigma_c import Universe
    print("Imports successful.", flush=True)

    def test_quantum():
        print("Testing Quantum...", flush=True)
        qpu = Universe.quantum(device='simulator')
        print("Quantum Adapter created.", flush=True)
        res = qpu.run_optimization(epsilon_values=[0.1], shots=10)
        print(f"Quantum Result: {res['sigma_c']}", flush=True)

    def test_gpu():
        print("Testing GPU...", flush=True)
        gpu = Universe.gpu()
        print("GPU Adapter created.", flush=True)
        res = gpu.auto_tune(alpha_levels=[0.1], epsilon_points=5)
        print(f"GPU Result: {res['statistics']['jonckheere_terpstra']['p_value']}", flush=True)

    def test_finance():
        print("Testing Finance...", flush=True)
        fin = Universe.finance()
        print("Financial Adapter created.", flush=True)
        res = fin.detect_regime(symbol='TEST')
        print(f"Finance Result: {res['sigma_c']}", flush=True)

    def test_climate():
        print("Testing Climate...", flush=True)
        clim = Universe.climate()
        print("Climate Adapter created.", flush=True)
        # Mock data
        data = pd.DataFrame({
            'lat': np.random.uniform(35, 70, 1000),
            'lon': np.random.uniform(-10, 40, 1000),
            'value': np.random.normal(15, 5, 1000)
        })
        res = clim.analyze_spatial_scaling(data)
        print(f"Climate Result: {res['sigma_c']}", flush=True)

    def test_seismic():
        print("Testing Seismic...", flush=True)
        seis = Universe.seismic()
        print("Seismic Adapter created.", flush=True)
        # Mock catalog
        catalog = pd.DataFrame({
            'latitude': np.random.uniform(32, 42, 100),
            'longitude': np.random.uniform(-125, -114, 100),
            'mag': np.random.uniform(2.5, 7.0, 100),
            'time': pd.date_range('2020-01-01', periods=100)
        })
        res = seis.analyze_criticality(catalog)
        print(f"Seismic Result: {res['sigma_c']}", flush=True)

    if __name__ == "__main__":
        test_quantum()
        test_gpu()
        test_finance()
        test_climate()
        test_seismic()
        print("ALL TESTS PASSED", flush=True)

except Exception:
    print("CRITICAL ERROR:", flush=True)
    traceback.print_exc()
    sys.exit(1)
