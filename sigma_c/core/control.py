"""
Sigma-C Control Module
======================
Copyright (c) 2025 ForgottenForge.xyz

Implements active control systems and real-time streaming analysis 
for maintaining systems at criticality.
"""

import numpy as np
from collections import deque
from typing import Optional, Callable, Deque

class StreamingSigmaC:
    """
    Real-time, incremental calculation of Critical Susceptibility (sigma_c).
    Enables O(1) complexity updates for streaming data.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.epsilon_window: Deque[float] = deque(maxlen=window_size)
        self.obs_window: Deque[float] = deque(maxlen=window_size)
        self.sum_x = 0.0
        self.sum_sq_x = 0.0
        self.count = 0

    def update(self, epsilon: float, observable: Optional[float] = None) -> float:
        """
        Add a new data point and return the current sigma_c.
        
        Args:
            epsilon: Control parameter value (or single observable if observable=None)
            observable: Observable value (optional)
            
        Returns:
            Current sigma_c value
        """
        # If only one argument, treat as single value
        if observable is None:
            value = epsilon
        else:
            # Store epsilon-observable pair
            self.epsilon_window.append(epsilon)
            self.obs_window.append(observable)
            value = observable
        
        # Remove old value if window is full
        if len(self.obs_window if observable is not None else [value]) == self.window_size:
            if observable is not None:
                old_val = self.obs_window[0]
            else:
                old_val = value
            self.sum_x -= old_val
            self.sum_sq_x -= old_val * old_val
        else:
            self.count += 1
            
        # Add new value
        self.sum_x += value
        self.sum_sq_x += value * value
        
        return self.compute_sigma_c()

    def compute_sigma_c(self) -> float:
        """
        Computes sigma_c based on variance of the current window.
        chi ~ Variance (Fluctuation-Dissipation Theorem approximation)
        sigma_c = 1 / (1 + chi)
        """
        if self.count < 2:
            return 1.0
            
        mean = self.sum_x / self.count
        variance = (self.sum_sq_x / self.count) - (mean * mean)
        
        # In many physical systems, susceptibility is proportional to variance
        chi = max(0, variance)  # Ensure non-negative
        
        return 1.0 / (1.0 + chi)
    
    def get_sigma_c(self) -> float:
        """Get current sigma_c value without updating."""
        return self.compute_sigma_c()

class AdaptiveController:
    """
    PID-based controller to maintain a system at a target critical point.
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05, target_sigma: float = 0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target_sigma
        
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_correction(self, current_sigma: float, dt: float = 1.0) -> float:
        """
        Computes the control signal (parameter adjustment) to return to target.
        
        Args:
            current_sigma: Measured sigma_c
            dt: Time step
            
        Returns:
            Control signal (delta to apply to control parameter)
        """
        error = self.target - current_sigma
        
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        self.prev_error = error
        
        return p_term + i_term + d_term
    
    def compute_adjustment(self, current_sigma: float) -> float:
        """Alias for compute_correction with default dt."""
        return self.compute_correction(current_sigma)

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
