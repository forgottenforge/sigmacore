#!/usr/bin/env python3
"""
Sigma-C Framework v1.1.0 - Universal Diagnostics Demo
=======================================================
Copyright (c) 2025 ForgottenForge.xyz

Demonstrates the new Universal Diagnostics System across all 6 domains.

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

from sigma_c import Universe
import numpy as np
import pandas as pd

def demo_quantum_diagnostics():
    """Demonstrate Quantum adapter diagnostics."""
    print("\n" + "="*60)
    print("QUANTUM DOMAIN DIAGNOSTICS")
    print("="*60)
    
    universe = Universe()
    
    try:
        # Create a quantum circuit
        circuit = universe.quantum.create_grover_with_noise(n_qubits=2, epsilon=0.01)
        
        # Run diagnostics
        print("\n📊 Running diagnostics on quantum circuit...")
        diag = universe.quantum.diagnose(circuit)
        
        print(f"Status: {diag['status']}")
        print(f"Issues: {diag['issues']}")
        print(f"Recommendations: {diag['recommendations']}")
        print(f"Details: {diag['details']}")
        
        # Auto-search for optimal parameters
        print("\n🔍 Auto-searching optimal parameters...")
        search_result = universe.quantum.auto_search(circuit_type='grover', n_qubits=2, n_shots=30)
        print(f"Best params: {search_result['best_params']}")
        print(f"Recommendation: {search_result['recommendation']}")
        
        # Validate techniques
        print("\n✓ Validating quantum techniques...")
        validation = universe.quantum.validate_techniques(circuit)
        print(f"Validation: {validation}")
        
        # Run actual analysis
        print("\n🚀 Running quantum optimization...")
        result = universe.quantum.run_optimization(circuit_type='grover', n_qubits=2, shots=50)
        
        # Explain results
        print("\n📖 Explanation:")
        explanation = universe.quantum.explain(result)
        print(explanation)
        
    except Exception as e:
        print(f"⚠️ Quantum demo skipped: {e}")

def demo_gpu_diagnostics():
    """Demonstrate GPU adapter diagnostics."""
    print("\n" + "="*60)
    print("GPU DOMAIN DIAGNOSTICS")
    print("="*60)
    
    universe = Universe()
    
    try:
        # Run diagnostics
        print("\n📊 Running GPU diagnostics...")
        diag = universe.gpu.diagnose()
        
        print(f"Status: {diag['status']}")
        print(f"Issues: {diag['issues']}")
        print(f"Recommendations: {diag['recommendations']}")
        
        # Auto-search for optimal alpha
        print("\n🔍 Auto-searching optimal cache parameters...")
        search_result = universe.gpu.auto_search(epsilon_points=15)
        print(f"Best params: {search_result['best_params']}")
        print(f"Recommendation: {search_result['recommendation']}")
        
        # Validate
        print("\n✓ Validating GPU techniques...")
        validation = universe.gpu.validate_techniques()
        print(f"Validation: {validation}")
        
    except Exception as e:
        print(f"⚠️ GPU demo skipped: {e}")

def demo_financial_diagnostics():
    """Demonstrate Financial adapter diagnostics."""
    print("\n" + "="*60)
    print("FINANCIAL DOMAIN DIAGNOSTICS")
    print("="*60)
    
    universe = Universe()
    
    try:
        # Generate synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=300)
        prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.02))
        price_data = pd.Series(prices, index=dates)
        
        # Run diagnostics
        print("\n📊 Running financial data diagnostics...")
        diag = universe.financial.diagnose(price_data=price_data)
        
        print(f"Status: {diag['status']}")
        print(f"Issues: {diag['issues']}")
        print(f"Recommendations: {diag['recommendations']}")
        print(f"Details: {diag['details']}")
        
        # Validate
        print("\n✓ Validating financial techniques...")
        validation = universe.financial.validate_techniques(price_data=price_data)
        print(f"Validation: {validation}")
        
        print("\n💡 For live market data, use: universe.financial.diagnose(symbol='^GSPC')")
        
    except Exception as e:
        print(f"⚠️ Financial demo skipped: {e}")

def demo_climate_diagnostics():
    """Demonstrate Climate adapter diagnostics."""
    print("\n" + "="*60)
    print("CLIMATE DOMAIN DIAGNOSTICS")
    print("="*60)
    
    universe = Universe()
    
    try:
        # Generate synthetic climate grid
        np.random.seed(42)
        grid = np.random.randn(64, 64)
        
        # Run diagnostics
        print("\n📊 Running climate grid diagnostics...")
        diag = universe.climate.diagnose(data=grid)
        
        print(f"Status: {diag['status']}")
        print(f"Issues: {diag['issues']}")
        print(f"Recommendations: {diag['recommendations']}")
        print(f"Details: {diag['details']}")
        
        # Validate
        print("\n✓ Validating climate techniques...")
        validation = universe.climate.validate_techniques(data=grid)
        print(f"Validation: {validation}")
        
    except Exception as e:
        print(f"⚠️ Climate demo skipped: {e}")

def demo_seismic_diagnostics():
    """Demonstrate Seismic adapter diagnostics."""
    print("\n" + "="*60)
    print("SEISMIC DOMAIN DIAGNOSTICS")
    print("="*60)
    
    universe = Universe()
    
    try:
        # Generate synthetic earthquake catalog
        np.random.seed(42)
        n_events = 200
        catalog = pd.DataFrame({
            'latitude': np.random.uniform(30, 40, n_events),
            'longitude': np.random.uniform(-120, -110, n_events),
            'magnitude': np.random.exponential(2.0, n_events) + 2.0,
            'depth': np.random.uniform(0, 50, n_events)
        })
        
        # Run diagnostics
        print("\n📊 Running seismic catalog diagnostics...")
        diag = universe.seismic.diagnose(catalog=catalog)
        
        print(f"Status: {diag['status']}")
        print(f"Issues: {diag['issues']}")
        print(f"Recommendations: {diag['recommendations']}")
        print(f"Details: {diag['details']}")
        
        # Validate
        print("\n✓ Validating seismic techniques...")
        validation = universe.seismic.validate_techniques(catalog=catalog)
        print(f"Validation: {validation}")
        
    except Exception as e:
        print(f"⚠️ Seismic demo skipped: {e}")

def demo_magnetic_diagnostics():
    """Demonstrate Magnetic adapter diagnostics."""
    print("\n" + "="*60)
    print("MAGNETIC DOMAIN DIAGNOSTICS")
    print("="*60)
    
    universe = Universe()
    
    try:
        # Run diagnostics
        print("\n📊 Running magnetic system diagnostics...")
        diag = universe.magnetic.diagnose(lattice_size=32)
        
        print(f"Status: {diag['status']}")
        print(f"Issues: {diag['issues']}")
        print(f"Recommendations: {diag['recommendations']}")
        print(f"Details: {diag['details']}")
        
        # Validate
        print("\n✓ Validating magnetic techniques...")
        validation = universe.magnetic.validate_techniques(lattice_size=32)
        print(f"Validation: {validation}")
        
    except Exception as e:
        print(f"⚠️ Magnetic demo skipped: {e}")

def main():
    """Run all diagnostics demos."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         Sigma-C Framework v1.1.0                            ║
║         Universal Diagnostics System Demo                    ║
║                                                              ║
║  Demonstrates intelligent diagnostics, auto-search, and     ║
║  recommendations across all 6 domains.                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run all domain diagnostics
    demo_quantum_diagnostics()
    demo_gpu_diagnostics()
    demo_financial_diagnostics()
    demo_climate_diagnostics()
    demo_seismic_diagnostics()
    demo_magnetic_diagnostics()
    
    print("\n" + "="*60)
    print("✅ Universal Diagnostics Demo Complete!")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  • diagnose() - Domain-specific health checks")
    print("  • auto_search() - Automated parameter optimization")
    print("  • validate_techniques() - Technique validation")
    print("  • explain() - Human-readable result interpretation")
    print("\nFor more details, see DOCUMENTATION.md")

if __name__ == '__main__':
    main()
