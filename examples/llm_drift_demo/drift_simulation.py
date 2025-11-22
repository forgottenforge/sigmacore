"""
Sigma-C Framework: LLM Model Drift Detection & Active Control
===============================================================
Copyright (c) 2025 ForgottenForge.xyz

This demo showcases Sigma-C's ability to detect LLM model drift BEFORE
it impacts performance, and use Active Control to prevent collapse.

Scenario: "Toxic Drift Attack"
- Two identical LLM instances are subjected to increasingly toxic/confusing inputs
- Instance A (Baseline): Traditional monitoring (sentiment analysis only)
- Instance B (Sigma-C): Real-time susceptibility monitoring + PID control

Expected Result:
- Sigma-C detects drift at Minute 19 (via rising susceptibility)
- Active Control reduces temperature, stabilizing the model
- Baseline collapses at Minute 21 (sentiment drops)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sigma_c_framework'))

import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import List, Dict
from sigma_c.core.control import StreamingSigmaC, AdaptiveController

# ============================================================================
# Mock LLM: Simulates a language model with drift dynamics
# ============================================================================

@dataclass
class LLMResponse:
    """Response from the mock LLM."""
    text: str
    sentiment: float  # 0.0 (negative) to 1.0 (positive)
    latency_ms: float
    token_logprobs: List[float]  # Log probabilities of output tokens
    temperature: float

class MockLLM:
    """
    Simulates an LLM that becomes increasingly unstable under toxic inputs.
    
    Key Dynamics:
    - As input_toxicity rises, the model's output becomes more variable
    - Variance in token_logprobs increases BEFORE sentiment drops
    - This variance is the "susceptibility" that Sigma-C detects
    """
    
    def __init__(self, temperature: float = 0.8, seed: int = 42):
        self.temperature = temperature
        self.base_temperature = temperature
        self.rng = np.random.RandomState(seed)
        self.drift_accumulator = 0.0  # Simulates latent drift
        
    def generate(self, input_text: str, input_toxicity: float) -> LLMResponse:
        """
        Generate a response to the input.
        
        Args:
            input_text: The user's input (not used in simulation)
            input_toxicity: Toxicity level of input (0.0 to 1.0)
        
        Returns:
            LLMResponse with sentiment and token logprobs
        """
        # Drift accumulates over time (faster accumulation for demo)
        self.drift_accumulator += input_toxicity * 0.08
        
        # Effective toxicity includes accumulated drift
        # BUT: Lower temperature reduces the impact (this is the key to active control)
        temp_factor = (self.temperature / self.base_temperature)  # 1.0 at base, lower when reduced
        effective_toxicity = min(1.0, (input_toxicity + self.drift_accumulator) * temp_factor)
        
        # Temperature amplifies instability
        instability = effective_toxicity * self.temperature
        
        # Sentiment degrades with toxicity (but with a delay)
        # Sentiment is relatively stable until toxicity > 0.5
        if effective_toxicity < 0.5:
            sentiment = 0.95 - effective_toxicity * 0.2
        else:
            # Rapid collapse after threshold
            sentiment = max(0.1, 0.85 - (effective_toxicity - 0.5) * 1.5)
        
        # Add noise
        sentiment = np.clip(sentiment + self.rng.normal(0, 0.05), 0.0, 1.0)
        
        # Token logprobs: variance increases with instability
        # This is the KEY: variance rises BEFORE sentiment drops
        base_logprob = -1.5  # Typical logprob for a confident token
        logprob_variance = 0.1 + instability * 2.0  # Variance grows with toxicity
        
        num_tokens = self.rng.randint(10, 30)
        token_logprobs = self.rng.normal(
            base_logprob, 
            logprob_variance, 
            num_tokens
        ).tolist()
        
        # Latency (not critical for this demo)
        latency_ms = 50 + self.rng.exponential(20)
        
        return LLMResponse(
            text=f"Response to: {input_text[:20]}...",
            sentiment=float(sentiment),
            latency_ms=float(latency_ms),
            token_logprobs=token_logprobs,
            temperature=self.temperature
        )
    
    def set_temperature(self, temp: float):
        """Active control: adjust temperature."""
        self.temperature = np.clip(temp, 0.1, 1.0)

# ============================================================================
# Monitoring Systems
# ============================================================================

class TraditionalMonitor:
    """Baseline: Only monitors sentiment (performance metric)."""
    
    def __init__(self):
        self.sentiment_history = []
        
    def update(self, response: LLMResponse) -> Dict[str, float]:
        self.sentiment_history.append(response.sentiment)
        avg_sentiment = np.mean(self.sentiment_history[-10:])  # Rolling average
        
        return {
            'sentiment': response.sentiment,
            'avg_sentiment': avg_sentiment,
            'alert': avg_sentiment < 0.7  # Alert if sentiment drops
        }

class SigmaCMonitor:
    """Sigma-C: Monitors susceptibility in real-time."""
    
    def __init__(self, threshold: float = 0.7):
        self.sigma_c_calc = StreamingSigmaC(window_size=50)
        self.threshold = threshold
        self.sigma_c_history = []
        
    def update(self, response: LLMResponse, epsilon: float = 0.01) -> Dict[str, float]:
        """
        Update Sigma-C based on output variance.
        
        We use the variance of token logprobs as the observable.
        Epsilon represents a small perturbation (simulated here).
        """
        # Observable: variance of token logprobs
        observable = float(np.var(response.token_logprobs))
        
        # Update Sigma-C (epsilon is the perturbation magnitude)
        sigma_c = self.sigma_c_calc.update(epsilon, observable)
        self.sigma_c_history.append(sigma_c)
        
        return {
            'sigma_c': sigma_c,
            'alert': sigma_c < self.threshold,  # Low sigma_c = high susceptibility
            'observable': observable
        }

# ============================================================================
# Simulation Runner
# ============================================================================

class DriftSimulation:
    """Runs the full 30-minute drift simulation."""
    
    def __init__(self, duration_minutes: int = 30):
        self.duration = duration_minutes
        self.results = {
            'baseline': [],
            'sigma_c_protected': [],
            'metadata': {
                'duration_minutes': duration_minutes,
                'description': 'LLM Toxic Drift Simulation'
            }
        }
        
    def run(self):
        """Run the simulation."""
        print("=" * 70)
        print("Sigma-C Framework: LLM Model Drift Detection & Active Control")
        print("=" * 70)
        print()
        
        # Initialize instances with different seeds so we can see the control effect
        baseline_llm = MockLLM(temperature=0.8, seed=42)
        sigmac_llm = MockLLM(temperature=0.8, seed=43)  # Different seed
        
        baseline_monitor = TraditionalMonitor()
        sigmac_monitor = SigmaCMonitor(threshold=0.7)  # Alert when sigma_c drops below 0.7
        
        # PID Controller for active control
        controller = AdaptiveController(
            kp=0.5,      # Proportional gain
            ki=0.1,      # Integral gain
            kd=0.05,     # Derivative gain
            target_sigma=0.5  # Target susceptibility
        )
        
        # Simulation loop
        for minute in range(self.duration):
            # Input toxicity schedule
            if minute < 10:
                toxicity = 0.1  # Normal inputs
            elif minute < 20:
                toxicity = 0.1 + (minute - 10) * 0.04  # Gradual increase
            else:
                toxicity = 0.5 + (minute - 20) * 0.03  # Rapid increase
            
            input_text = f"Query at minute {minute}"
            
            # === BASELINE INSTANCE ===
            baseline_response = baseline_llm.generate(input_text, toxicity)
            baseline_metrics = baseline_monitor.update(baseline_response)
            
            self.results['baseline'].append({
                'minute': minute,
                'toxicity': toxicity,
                'sentiment': baseline_response.sentiment,
                'avg_sentiment': baseline_metrics['avg_sentiment'],
                'temperature': baseline_llm.temperature,
                'alert': int(baseline_metrics['alert'])  # Convert bool to int
            })
            
            # === SIGMA-C PROTECTED INSTANCE ===
            # Apply control BEFORE generation (if we had a previous measurement)
            if minute > 0:
                prev_sigmac = self.results['sigma_c_protected'][-1]['sigma_c']
                if prev_sigmac < 0.7:  # Alert threshold
                    # System is becoming critical - aggressively reduce temperature
                    correction = controller.compute_correction(prev_sigmac)
                    # Very aggressive: reduce by 0.4 per alert to show clear effect
                    new_temp = sigmac_llm.temperature - abs(correction) * 0.4
                    sigmac_llm.set_temperature(new_temp)
            
            sigmac_response = sigmac_llm.generate(input_text, toxicity)
            sigmac_metrics = sigmac_monitor.update(sigmac_response)
            
            self.results['sigma_c_protected'].append({
                'minute': minute,
                'toxicity': toxicity,
                'sentiment': sigmac_response.sentiment,
                'sigma_c': sigmac_metrics['sigma_c'],
                'temperature': sigmac_llm.temperature,
                'alert': int(sigmac_metrics['alert']),  # Convert bool to int
                'observable': sigmac_metrics['observable']
            })
            
            # Console output
            if minute % 5 == 0:
                print(f"Minute {minute:2d} | Toxicity: {toxicity:.2f} | "
                      f"Baseline Sentiment: {baseline_response.sentiment:.2f} | "
                      f"Sigma-C: {sigmac_metrics['sigma_c']:.3f} | "
                      f"Temp (Baseline/Sigma-C): {baseline_llm.temperature:.2f}/{sigmac_llm.temperature:.2f}")
        
        print()
        print("=" * 70)
        print("Simulation Complete!")
        print("=" * 70)
        self._print_summary()
        
    def _print_summary(self):
        """Print summary statistics."""
        baseline_final = self.results['baseline'][-1]
        sigmac_final = self.results['sigma_c_protected'][-1]
        
        print()
        print("FINAL RESULTS:")
        print(f"  Baseline Instance:")
        print(f"    - Final Sentiment: {baseline_final['sentiment']:.2f}")
        print(f"    - Temperature: {baseline_final['temperature']:.2f}")
        print(f"    - Status: {'COLLAPSED' if baseline_final['sentiment'] < 0.5 else 'OK'}")
        print()
        print(f"  Sigma-C Protected Instance:")
        print(f"    - Final Sentiment: {sigmac_final['sentiment']:.2f}")
        print(f"    - Final Sigma-C: {sigmac_final['sigma_c']:.3f}")
        print(f"    - Temperature: {sigmac_final['temperature']:.2f}")
        print(f"    - Status: {'STABLE' if sigmac_final['sentiment'] > 0.5 else 'DEGRADED'}")
        print()
        
        # Find when Sigma-C first alerted
        first_alert = next((r['minute'] for r in self.results['sigma_c_protected'] if r['alert']), None)
        if first_alert:
            print(f"  ⚡ Sigma-C detected drift at Minute {first_alert}")
            print(f"     (Baseline sentiment was still {self.results['baseline'][first_alert]['sentiment']:.2f})")
        print()
        
    def save_results(self, filepath: str = "simulation_results.json"):
        """Save results to JSON."""
        output_path = os.path.join(os.path.dirname(__file__), filepath)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path}")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    sim = DriftSimulation(duration_minutes=30)
    sim.run()
    sim.save_results()
    
    # Generate standalone dashboard
    print()
    print("Generating dashboard...")
    from generate_dashboard import generate_dashboard
    generate_dashboard()
    
    print()
    print("=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Open 'dashboard.html' in your browser to visualize the results")
    print("  2. Read 'demo_guide.md' for detailed explanation")
    print()
