"""
Sigma-C v1.2.0 Final Perfection Check
=====================================
Scans and verifies every module in the framework to ensure:
1. Import stability (no missing dependencies)
2. Class instantiation (no init errors)
3. Basic functionality (smoke tests)

Modules checked:
- Optimization (Universal, Quantum, GPU, Financial, BruteForce)
- Physics (Rigorous, Quantum, GPU, Financial, Advanced)
- Prediction (ML, Blind)
- Reporting (Latex, Viz)
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSigmaCPerfection(unittest.TestCase):
    
    def setUp(self):
        print(f"\nTesting: {self._testMethodName}...", end='', flush=True)

    def tearDown(self):
        print(" OK", end='')

    # ==========================================
    # 1. Optimization Module
    # ==========================================
    def test_optimization_imports(self):
        from sigma_c_framework.sigma_c.optimization.universal import UniversalOptimizer
        from sigma_c_framework.sigma_c.optimization.brute_force import BruteForceEngine
        from sigma_c_framework.sigma_c.optimization.quantum import BalancedQuantumOptimizer
        from sigma_c_framework.sigma_c.optimization.gpu import BalancedGPUOptimizer
        from sigma_c_framework.sigma_c.optimization.financial import BalancedFinancialOptimizer
        
        self.assertIsNotNone(UniversalOptimizer)
        self.assertIsNotNone(BruteForceEngine)

    def test_brute_force_engine(self):
        from sigma_c_framework.sigma_c.optimization.brute_force import BruteForceEngine
        
        def dummy_eval(params):
            return -(params['x'] - 2)**2 + 10
            
        engine = BruteForceEngine()
        result = engine.run(dummy_eval, {'x': [0, 1, 2, 3, 4]}, show_progress=False)
        self.assertEqual(result['best_params']['x'], 2)
        self.assertEqual(result['best_score'], 10)

    # ==========================================
    # 2. Physics Module
    # ==========================================
    def test_physics_imports(self):
        from sigma_c_framework.sigma_c.physics.rigorous import RigorousTheoreticalCheck
        from sigma_c_framework.sigma_c.physics.quantum import RigorousQuantumSigmaC
        from sigma_c_framework.sigma_c.physics.gpu import RigorousGPUSigmaC
        from sigma_c_framework.sigma_c.physics.financial import RigorousFinancialSigmaC
        from sigma_c_framework.sigma_c.physics.advanced import RenormalizationGroup, ChaosQuantifier

    def test_rigorous_checks(self):
        from sigma_c_framework.sigma_c.physics.gpu import RigorousGPUSigmaC
        checker = RigorousGPUSigmaC()
        # Test with dummy context
        res = checker.validate_sigma_c(0.5, {'arithmetic_intensity': 10, 'peak_flops': 100, 'peak_bandwidth': 10})
        self.assertIn('is_valid', res)

    def test_advanced_physics(self):
        from sigma_c_framework.sigma_c.physics.advanced import RenormalizationGroup
        rg = RenormalizationGroup()
        # Dummy flow analysis
        sigmas = np.linspace(0.1, 1.0, 10)
        obs = np.exp(-sigmas)
        res = rg.analyze_flow(sigmas, obs)
        self.assertIn('beta_function', res)

    # ==========================================
    # 3. Prediction Module
    # ==========================================
    def test_prediction_imports(self):
        from sigma_c_framework.sigma_c.prediction.ml import MLDiscovery, BlindPredictor
        
    def test_ml_discovery(self):
        from sigma_c_framework.sigma_c.prediction.ml import MLDiscovery
        ml = MLDiscovery()
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        res = ml.find_critical_features(X, y, ['feat1', 'feat2'])
        self.assertIn('ranked_features', res)

    # ==========================================
    # 4. Reporting Module
    # ==========================================
    def test_reporting_imports(self):
        from sigma_c_framework.sigma_c.reporting.latex import LatexGenerator
        from sigma_c_framework.sigma_c.plotting.publication import PublicationVisualizer

    def test_latex_generation(self):
        from sigma_c_framework.sigma_c.reporting.latex import LatexGenerator
        gen = LatexGenerator()
        # Just check class exists and method signature, don't write file in test
        self.assertTrue(hasattr(gen, 'generate_report'))

if __name__ == '__main__':
    print("🛡️  Sigma-C v1.2.0 INTEGRITY CHECK 🛡️")
    print("=======================================")
    unittest.main(verbosity=0)
