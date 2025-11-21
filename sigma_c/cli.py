"""
Sigma-C Command Line Interface
==============================
Run optimizations and visualize results from the terminal.

Usage:
    sigma-c run <config.yaml>
    sigma-c visualize <result.json>

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

"""
Sigma-C Command Line Interface
==============================
Copyright (c) 2025 ForgottenForge.xyz
"""
"""
Sigma-C Command Line Interface
==============================
Copyright (c) 2025 ForgottenForge.xyz
"""
import argparse


import yaml
import json
import sys
from pathlib import Path
from typing import Dict, Any

from sigma_c.core.callbacks import LoggingCallback, CheckpointCallback
from sigma_c.visualization import plot_convergence, plot_pareto_frontier

def load_class(path: str):
    """Load a class from a string path (e.g., 'module.submodule.ClassName')."""
    components = path.split('.')
    module_path = '.'.join(components[:-1])
    class_name = components[-1]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def run_optimization(config_path: str):
    """Run optimization based on a YAML config file."""
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 1. Initialize Adapter
    adapter_cls = load_class(config['adapter']['class'])
    adapter_config = config['adapter'].get('config', {})
    adapter = adapter_cls(config=adapter_config) if adapter_config else adapter_cls()
    
    # 2. Initialize Optimizer
    optimizer_cls = load_class(config['optimizer']['class'])
    opt_config = config['optimizer'].get('config', {})
    optimizer = optimizer_cls(adapter, **opt_config)
    
    # 3. Setup Callbacks
    callbacks = [LoggingCallback()]
    if 'checkpoint' in config:
        callbacks.append(CheckpointCallback(config['checkpoint']['path']))
        
    # 4. Run Optimization
    print("Starting optimization...")
    param_space = config['param_space']
    strategy = config.get('strategy', 'brute_force')
    
    # Note: We assume the optimizer's optimize method signature matches
    # Ideally we would have a unified 'optimize_system' method, but for now
    # we might need to adapt based on the optimizer type.
    # However, UniversalOptimizer defines optimize(system, param_space, ...)
    # But specific optimizers like BalancedQuantumOptimizer use optimize_circuit...
    # This is a slight API inconsistency we need to handle or fix.
    # For v1.2.3, let's assume the user provides the target factory in the config?
    # Actually, for CLI, it's hard to pass a function factory.
    # We might need a 'target' field in config that points to a python function.
    
    target_factory = None
    if 'target' in config:
        target_factory = load_class(config['target'])
        
    # Dispatch based on optimizer type (temporary fix for API inconsistency)
    if 'Quantum' in optimizer_cls.__name__:
        result = optimizer.optimize_circuit(target_factory, param_space, strategy=strategy)
    elif 'ML' in optimizer_cls.__name__:
        result = optimizer.optimize_model(target_factory, param_space)
    elif 'Financial' in optimizer_cls.__name__:
        result = optimizer.optimize_strategy(param_space, strategy=strategy)
    elif 'GPU' in optimizer_cls.__name__:
        # GPU optimizer needs a kernel function
        result = optimizer.optimize_kernel(target_factory, param_space)
    else:
        # Fallback to generic optimize
        result = optimizer.optimize(adapter, param_space, strategy=strategy, callbacks=callbacks)
        
    # 5. Save Result
    output_path = config.get('output', 'result.json')
    print(f"Optimization complete. Saving to {output_path}...")
    
    # Manually save if the specific optimizer doesn't return a Universal Result with save method
    # But our UniversalOptimizer has save/load now.
    # The result object is OptimizationResult dataclass.
    
    # We need to serialize the dataclass
    data = {
        'optimal_params': result.optimal_params,
        'score': result.score,
        'history': result.history,
        'sigma_c_after': result.sigma_c_after,
        'performance_after': result.performance_after
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print("Done.")

def visualize_result(result_path: str):
    """Visualize a result JSON file."""
    print(f"Loading result from {result_path}...")
    with open(result_path, 'r') as f:
        data = json.load(f)
        
    # Reconstruct a mock result object for plotting
    from sigma_c.optimization.universal import OptimizationResult
    result = OptimizationResult(
        optimal_params=data['optimal_params'],
        score=data['score'],
        history=data['history'],
        sigma_c_before=0.0, # Unknown from simple JSON
        sigma_c_after=data['sigma_c_after'],
        performance_metric_name="Unknown",
        performance_before=0.0,
        performance_after=data['performance_after'],
        strategy_used="unknown"
    )
    
    print("Generating plots...")
    base_name = Path(result_path).stem
    plot_convergence(result, save_path=f"{base_name}_convergence.png")
    plot_pareto_frontier(result, save_path=f"{base_name}_pareto.png")
    print(f"Saved plots to {base_name}_convergence.png and {base_name}_pareto.png")

def main():
    parser = argparse.ArgumentParser(description="Sigma-C Framework CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Run Command
    run_parser = subparsers.add_parser('run', help='Run optimization from config')
    run_parser.add_argument('config', help='Path to YAML config file')
    
    # Visualize Command
    viz_parser = subparsers.add_parser('visualize', help='Visualize optimization results')
    viz_parser.add_argument('result', help='Path to result JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        run_optimization(args.config)
    elif args.command == 'visualize':
        visualize_result(args.result)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
