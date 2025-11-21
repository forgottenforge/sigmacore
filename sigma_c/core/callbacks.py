"""
Sigma-C Callbacks System
========================
Provides hooks for monitoring and controlling the optimization process.

Copyright (c) 2025 ForgottenForge.xyz
Licensed under the AGPL-3.0-or-later OR Commercial License.
"""

from typing import Dict, Any, Optional
import json
import time
from pathlib import Path

class OptimizationCallback:
    """Base class for all callbacks."""
    
    def on_optimization_start(self, optimizer: Any, params: Dict[str, Any]):
        """Called before optimization begins."""
        pass
        
    def on_step_end(self, optimizer: Any, step: int, logs: Dict[str, Any]):
        """Called after each optimization step."""
        pass
        
    def on_optimization_end(self, optimizer: Any, result: Any):
        """Called after optimization completes."""
        pass

class EarlyStopping(OptimizationCallback):
    """Stop optimization if metric stops improving."""
    
    def __init__(self, monitor: str = 'score', patience: int = 5, min_delta: float = 0.0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = -float('inf')
        self.wait = 0
        self.stopped_step = 0
        
    def on_step_end(self, optimizer: Any, step: int, logs: Dict[str, Any]):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if current > self.best_value + self.min_delta:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_step = step
                optimizer.stop_optimization = True
                print(f"\nEarly stopping triggered at step {step}. Best {self.monitor}: {self.best_value:.4f}")

class LoggingCallback(OptimizationCallback):
    """Log optimization progress to console or file."""
    
    def __init__(self, interval: int = 1, log_file: Optional[str] = None):
        self.interval = interval
        self.log_file = log_file
        self.start_time = 0
        
    def on_optimization_start(self, optimizer: Any, params: Dict[str, Any]):
        self.start_time = time.time()
        msg = f"Starting optimization with {len(params)} parameters..."
        print(msg)
        if self.log_file:
            with open(self.log_file, 'w') as f:
                f.write(msg + "\n")
                
    def on_step_end(self, optimizer: Any, step: int, logs: Dict[str, Any]):
        if step % self.interval == 0:
            elapsed = time.time() - self.start_time
            msg = f"Step {step}: Score={logs.get('score', 0):.4f} | Perf={logs.get('performance', 0):.4f} | Stab={logs.get('stability', 0):.4f} ({elapsed:.1f}s)"
            print(msg)
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(msg + "\n")

class CheckpointCallback(OptimizationCallback):
    """Save optimizer state periodically."""
    
    def __init__(self, filepath: str, interval: int = 10):
        self.filepath = filepath
        self.interval = interval
        
    def on_step_end(self, optimizer: Any, step: int, logs: Dict[str, Any]):
        if step % self.interval == 0:
            # Assuming optimizer has a save method (we will add this)
            if hasattr(optimizer, 'save'):
                optimizer.save(f"{self.filepath}_step_{step}.json")
