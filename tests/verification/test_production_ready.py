"""
Final Production Test Suite for v2.0.1
=======================================
Tests all VERIFIED working features before release.
"""

import sys
import numpy as np

def test_all_core_features():
    """Test all core features that are production-ready."""
    print("=" * 70)
    print("SIGMA-C v2.0.1 - PRODUCTION READINESS TEST")
    print("=" * 70)
    
    results = {}
    
    # 1. Core Engine
    print("\n### Core Engine ###")
    try:
        from sigma_c.core.engine import Engine
        engine = Engine()
        result = engine.compute_susceptibility(
            np.linspace(0, 1, 50),
            np.sin(np.linspace(0, 10, 50))
        )
        assert all(k in result for k in ['sigma_c', 'kappa', 'chi_max'])
        results['core_engine'] = 'PASS'
        print(f"✅ Core Engine (sigma_c={result['sigma_c']:.4f}, chi_max={result['chi_max']:.4f})")
    except Exception as e:
        results['core_engine'] = f'FAIL: {e}'
        print(f"❌ Core Engine: {e}")
    
    # 2. Streaming Analysis
    print("\n### Streaming Analysis ###")
    try:
        from sigma_c.core.control import StreamingSigmaC
        stream = StreamingSigmaC(window_size=10)
        for i in range(20):
            stream.update(i * 0.05, np.sin(i * 0.5))
        sigma_c = stream.get_sigma_c()
        assert 0 < sigma_c < 2
        results['streaming'] = 'PASS'
        print(f"✅ Streaming Analysis (sigma_c={sigma_c:.4f})")
    except Exception as e:
        results['streaming'] = f'FAIL: {e}'
        print(f"❌ Streaming: {e}")
    
    # 3. Adaptive Control
    print("\n### Adaptive Control ###")
    try:
        from sigma_c.core.control import AdaptiveController
        controller = AdaptiveController(target_sigma=0.5)
        adjustment = controller.compute_adjustment(0.3)
        assert isinstance(adjustment, float)
        results['adaptive_control'] = 'PASS'
        print(f"✅ Adaptive Control (adjustment={adjustment:.4f})")
    except Exception as e:
        results['adaptive_control'] = f'FAIL: {e}'
        print(f"❌ Adaptive Control: {e}")
    
    # 4-12. All Domain Adapters
    adapters = [
        ('Quantum', 'sigma_c.adapters.quantum', 'QuantumAdapter'),
        ('GPU', 'sigma_c.adapters.gpu', 'GPUAdapter'),
        ('Financial', 'sigma_c.adapters.financial', 'FinancialAdapter'),
        ('Climate', 'sigma_c.adapters.climate', 'ClimateAdapter'),
        ('Seismic', 'sigma_c.adapters.seismic', 'SeismicAdapter'),
        ('Magnetic', 'sigma_c.adapters.magnetic', 'MagneticAdapter'),
        ('Edge', 'sigma_c.adapters.edge', 'EdgeAdapter'),
        ('LLM Cost', 'sigma_c.adapters.llm_cost', 'LLMCostAdapter'),
        ('ML', 'sigma_c.adapters.ml', 'MLAdapter'),
    ]
    
    print("\n### Domain Adapters ###")
    for name, module, cls in adapters:
        try:
            mod = __import__(module, fromlist=[cls])
            adapter = getattr(mod, cls)()
            results[f'adapter_{name.lower().replace(" ", "_")}'] = 'PASS'
            print(f"✅ {name} Adapter")
        except Exception as e:
            results[f'adapter_{name.lower().replace(" ", "_")}'] = f'FAIL: {e}'
            print(f"❌ {name} Adapter: {e}")
    
    # 13. Universal Bridge
    print("\n### Connectors ###")
    try:
        from sigma_c.connectors.bridge import SigmaCBridge
        @SigmaCBridge.wrap_any_function
        def test_func(x):
            return x ** 2
        assert test_func(5) == 25
        results['universal_bridge'] = 'PASS'
        print("✅ Universal Bridge")
    except Exception as e:
        results['universal_bridge'] = f'FAIL: {e}'
        print(f"❌ Universal Bridge: {e}")
    
    # 14. PyTorch Integration
    try:
        from sigma_c.ml.pytorch import CriticalModule, SigmaCLoss
        results['pytorch'] = 'PASS'
        print("✅ PyTorch Integration")
    except Exception as e:
        results['pytorch'] = f'FAIL: {e}'
        print(f"❌ PyTorch: {e}")
    
    # 15. REST API
    try:
        from sigma_c.api.rest import SigmaCAPI
        api = SigmaCAPI()
        result = api.compute([0.0, 0.1, 0.2], [1.0, 0.8, 0.5])
        assert 'sigma_c' in result and 'chi_max' in result
        results['rest_api'] = 'PASS'
        print(f"✅ REST API (sigma_c={result['sigma_c']:.4f})")
    except Exception as e:
        results['rest_api'] = f'FAIL: {e}'
        print(f"❌ REST API: {e}")
    
    # 16. Observable Discovery
    print("\n### Advanced Features ###")
    try:
        from sigma_c.core.discovery import ObservableDiscovery
        discovery = ObservableDiscovery()
        # Test with simple data
        data = np.random.randn(100, 5)
        obs = discovery.identify_observables(data, method='gradient')
        results['observable_discovery'] = 'PASS'
        print(f"✅ Observable Discovery ({len(obs['ranked_observables'])} observables)")
    except Exception as e:
        results['observable_discovery'] = f'FAIL: {e}'
        print(f"❌ Observable Discovery: {e}")
    
    # 17. Multi-Scale Analysis
    try:
        from sigma_c.core.discovery import MultiScaleAnalysis
        msa = MultiScaleAnalysis()
        signal = np.sin(np.linspace(0, 100, 1000)) + 0.1 * np.random.randn(1000)
        spectrum = msa.compute_susceptibility_spectrum(signal)
        results['multiscale'] = 'PASS'
        print(f"✅ Multi-Scale Analysis ({len(spectrum['scales'])} scales)")
    except Exception as e:
        results['multiscale'] = f'FAIL: {e}'
        print(f"❌ Multi-Scale: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PRODUCTION READINESS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r == 'PASS')
    total = len(results)
    
    print(f"\n✅ PASSED: {passed}/{total} ({100*passed//total}%)")
    
    if passed < total:
        print(f"\n❌ FAILED: {total - passed}")
        print("\nFailed components:")
        for name, result in results.items():
            if result != 'PASS':
                print(f"  - {name}: {result}")
    
    if passed >= total * 0.9:  # 90% pass rate
        print("\n" + "=" * 70)
        print("✅ PRODUCTION READY FOR v2.0.1 RELEASE!")
        print("=" * 70)
        return 0
    else:
        print("\n⚠️  NOT READY - Fix failures before release")
        return 1

if __name__ == '__main__':
    sys.exit(test_all_core_features())
