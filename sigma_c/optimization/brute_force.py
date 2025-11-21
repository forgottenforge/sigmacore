"""
Brute Force Engine
==================
Parallelized parameter sweeper for the Universal Optimizer.
Inspired by `auto_opti2.py` from the QPU development scripts.
"""

import itertools
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm

class BruteForceEngine:
    """
    Executes exhaustive search over parameter spaces with parallelization support.
    """
    
    def __init__(self, max_workers: int = 4, use_processes: bool = False):
        self.max_workers = max_workers
        self.use_processes = use_processes
        
    def run(self, 
            eval_function: Callable[[Dict[str, Any]], float],
            param_space: Dict[str, List[Any]],
            show_progress: bool = True) -> Dict[str, Any]:
        """
        Run the brute force sweep.
        
        Args:
            eval_function: Function taking params dict and returning a score (float).
            param_space: Dictionary of parameter names and list of possible values.
            show_progress: Whether to show a tqdm progress bar.
            
        Returns:
            Dict containing 'best_params', 'best_score', and 'all_results'.
        """
        keys = list(param_space.keys())
        values = list(param_space.values())
        combinations = list(itertools.product(*values))
        total_combos = len(combinations)
        
        results = []
        
        # Helper for mapping
        def _worker(combo):
            params = dict(zip(keys, combo))
            try:
                score = eval_function(params)
                return {'params': params, 'score': score, 'error': None}
            except Exception as e:
                return {'params': params, 'score': -float('inf'), 'error': str(e)}

        Executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with Executor(max_workers=self.max_workers) as executor:
            if show_progress:
                futures = list(tqdm(executor.map(_worker, combinations), 
                                  total=total_combos, 
                                  desc="Brute Force Sweep"))
            else:
                futures = list(executor.map(_worker, combinations))
                
            results = futures

        # Find best
        valid_results = [r for r in results if r['error'] is None]
        if not valid_results:
            raise RuntimeError("All parameter combinations failed evaluation.")
            
        best_result = max(valid_results, key=lambda x: x['score'])
        
        return {
            'best_params': best_result['params'],
            'best_score': best_result['score'],
            'all_results': results
        }

    def grid_refinement(self, 
                        eval_function: Callable[[Dict[str, Any]], float],
                        initial_center: Dict[str, float],
                        param_ranges: Dict[str, float],
                        steps: int = 5,
                        depth: int = 3) -> Dict[str, Any]:
        """
        Adaptive grid refinement (zoom-in) strategy.
        Useful for continuous parameters.
        """
        current_center = initial_center.copy()
        current_ranges = param_ranges.copy()
        
        best_global = None
        
        for d in range(depth):
            # Create grid around center
            local_space = {}
            for key, val in current_center.items():
                radius = current_ranges[key]
                local_space[key] = np.linspace(val - radius, val + radius, steps).tolist()
                
            # Run sweep
            result = self.run(eval_function, local_space, show_progress=False)
            
            # Update best
            if best_global is None or result['best_score'] > best_global['best_score']:
                best_global = result
                
            # Zoom in
            current_center = result['best_params']
            for key in current_ranges:
                current_ranges[key] /= 2.0  # Halve the search radius
                
        return best_global
