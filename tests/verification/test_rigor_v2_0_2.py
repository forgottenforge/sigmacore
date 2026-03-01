import unittest
import numpy as np
from sigma_c.core.control import StreamingSigmaC
from sigma_c.adapters.llm_cost import LLMCostAdapter
from sigma_c.adapters.seismic import SeismicAdapter

class TestRigorV202(unittest.TestCase):
    
    def test_welford_stability(self):
        """Test Welford's algorithm for numerical stability with large offsets."""
        stream = StreamingSigmaC(window_size=100)
        
        # Large offset, small variance
        # Naive sum_sq: sum((x - 0)^2) would be huge, loss of precision for variance
        offset = 1e9
        data = np.random.normal(0, 1, 100) + offset
        
        for x in data:
            stream.update(x)
            
        # Check variance (sigma_c is 1/(1+var))
        # We access internal state to check variance directly if possible, 
        # or infer from sigma_c
        
        # Reconstruct variance from sigma_c: sigma = 1/(1+chi) -> chi = (1/sigma) - 1
        sigma_c = stream.get_sigma_c()
        chi = (1.0 / sigma_c) - 1.0
        
        expected_var = np.var(data) # Population variance by default in numpy? No, sample?
        # Numpy var is population by default (ddof=0). My implementation used count (population).
        
        print(f"Welford Variance: {chi:.6f}, Expected: {expected_var:.6f}")
        self.assertAlmostEqual(chi, expected_var, places=5)
        
    def test_welford_sliding_window(self):
        """Test that sliding window correctly removes old values."""
        window_size = 5
        stream = StreamingSigmaC(window_size=window_size)
        
        # Add 5 values: 1, 2, 3, 4, 5
        for i in range(1, 6):
            stream.update(float(i))
            
        # Variance of [1,2,3,4,5] is 2.0
        sigma_c = stream.get_sigma_c()
        chi = (1.0 / sigma_c) - 1.0
        self.assertAlmostEqual(chi, 2.0, places=5)
        
        # Add 6th value: 6. Window becomes [2,3,4,5,6]. Variance still 2.0.
        stream.update(6.0)
        sigma_c = stream.get_sigma_c()
        chi = (1.0 / sigma_c) - 1.0
        self.assertAlmostEqual(chi, 2.0, places=5)
        
        # Verify count stays at window size
        self.assertEqual(stream.count, 5)

    def test_llm_safety_constraint(self):
        """Test that unsafe models are rejected."""
        adapter = LLMCostAdapter()
        
        models = [
            {'name': 'Unsafe_Cheap', 'cost': 0.01, 'hallucination_rate': 0.90}, # Should be rejected
            {'name': 'Safe_Expensive', 'cost': 1.00, 'hallucination_rate': 0.01}, # Should be picked
            {'name': 'Mid_Mid', 'cost': 0.50, 'hallucination_rate': 0.10} # Safe enough
        ]
        
        result = adapter.analyze_cost_safety(models)
        print(f"Selected Model: {result['best_model']}")
        
        # Unsafe_Cheap has score ~111 (if no constraint), Safe_Expensive ~100.
        # Without constraint, Unsafe_Cheap wins.
        # With constraint (0.15), Unsafe_Cheap is disqualified.
        # Between Safe_Expensive (score 100) and Mid_Mid (score 1/(0.05) = 20), Safe_Expensive wins?
        # Mid_Mid score: 1 / (0.5 * 0.1) = 1 / 0.05 = 20.
        # Safe_Expensive score: 1 / (1.0 * 0.01) = 100.
        # So Safe_Expensive should win.
        
        self.assertNotEqual(result['best_model'], 'Unsafe_Cheap')
        self.assertEqual(result['best_model'], 'Safe_Expensive')

    def test_seismic_significance(self):
        """Test significance calculation."""
        adapter = SeismicAdapter()
        
        # Create random data (should have low significance / high p-value for being "ordered")
        # Actually, compute_significance checks if observed_stat is extreme compared to surrogates.
        # Let's just check it runs.
        data = np.random.random(100)
        observed_stat = 0.5
        
        p_value = adapter.compute_significance(observed_stat, data, n_surrogates=50)
        print(f"Seismic p-value: {p_value}")
        
        self.assertTrue(0.0 <= p_value <= 1.0)

if __name__ == '__main__':
    unittest.main()
