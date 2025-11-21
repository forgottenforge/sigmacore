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
        self.window: Deque[float] = deque(maxlen=window_size)
        self.sum_x = 0.0
        self.sum_sq_x = 0.0
        self.count = 0

    def update(self, value: float) -> float:
        """
        Add a new data point and return the current sigma_c.
        
        Args:
            value: New observable measurement
            
        Returns:
            Current sigma_c value
        """
        # Remove old value if window is full
        if len(self.window) == self.window_size:
            old_val = self.window[0]
            self.sum_x -= old_val
            self.sum_sq_x -= old_val * old_val
        else:
            self.count += 1
            
        # Add new value
        self.window.append(value)
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
        chi = variance 
        
        return 1.0 / (1.0 + chi)

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

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
