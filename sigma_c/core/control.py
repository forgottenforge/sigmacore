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
        
        # Welford's Algorithm State
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences from the current mean
        self.count = 0

    def update(self, epsilon: float, observable: Optional[float] = None) -> float:
        """
        Add a new data point and return the current sigma_c.
        Uses Welford's Online Algorithm for numerically stable variance.
        """
        # If only one argument, treat as single value
        if observable is None:
            value = epsilon
        else:
            # Store epsilon-observable pair
            self.epsilon_window.append(epsilon)
            self.obs_window.append(observable)
            value = observable
        
        # Handle Window Removal (Inverse Welford)
        # We need to know if an item is being removed.
        # The deque automatically handles the storage, but we need the value *before* it's gone to update stats.
        
        # Determine the window to check size against
        active_window = self.obs_window if observable is not None else self.epsilon_window
        # Note: In single value mode, we need to store values to remove them correctly
        if observable is None:
             # If we are in single value mode, we need to track the values to remove them
             # The original code didn't explicitly store them in a separate list for single-value mode 
             # other than implicitly via deque if we used it. 
             # Let's ensure we store them.
             if len(self.obs_window) == 0 and len(self.epsilon_window) == 0:
                 # First time initialization for single value mode if needed, 
                 # but let's just use obs_window for single values if epsilon is the value
                 pass
             
        # Actually, let's simplify. We always store the 'value' that contributes to variance.
        # If observable is None, 'value' is epsilon. We should store it.
        # The previous code had:
        # if len(self.obs_window if observable is not None else [value]) == self.window_size:
        # This implies it wasn't really storing single values in a window? 
        # Wait, the original code:
        # self.window: Deque[float] = deque(maxlen=window_size) (in viewed code it was self.window)
        # But in the file content I read (Step 1668), it was:
        # self.epsilon_window: Deque[float] = deque(maxlen=window_size)
        # self.obs_window: Deque[float] = deque(maxlen=window_size)
        
        # Let's use a unified window for the values we calculate variance on.
        
        if observable is None:
             # Single value mode
             self.obs_window.append(value)
        
        # Check if we need to remove (if window was already full BEFORE appending, 
        # but deque appends first... wait. Deque with maxlen auto-discards.
        # We need to capture the discarded value.
        
        # Correct approach with maxlen deque:
        # We can't easily get the "to be discarded" value from a maxlen deque *after* append.
        # We should check length *before* append.
        
        removed_val = None
        if len(self.obs_window) == self.window_size:
            removed_val = self.obs_window[0]
            
            # Inverse Welford for removal
            # delta = x - new_mean
            # new_mean = old_mean - delta / (n-1)
            # M2 -= delta * (x - old_mean)
            
            self.count -= 1
            delta = removed_val - self.mean
            self.mean -= delta / self.count
            self.M2 -= delta * (removed_val - self.mean)
            
            # Note: obs_window will auto-pop left when we append below
            
        # Welford Update (Addition)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        # Now append (and auto-pop if full)
        # Note: If we are in 'observable is not None' mode, we already appended above? 
        # No, I haven't appended yet in this logic flow.
        
        if observable is not None:
             # We already appended in the 'else' block above? 
             # Wait, I need to be careful with the order.
             pass
        else:
             # I need to append now
             pass
             
        # Refactoring for clarity:
        
        return self.compute_sigma_c()

    def update(self, epsilon: float, observable: Optional[float] = None) -> float:
        """
        Add a new data point and return the current sigma_c.
        Uses Welford's Online Algorithm for numerically stable variance.
        """
        # Determine value
        if observable is None:
            value = epsilon
        else:
            value = observable
            self.epsilon_window.append(epsilon)

        # Handle Removal (Inverse Welford)
        if len(self.obs_window) == self.window_size:
            removed_val = self.obs_window[0]
            self.count -= 1
            delta = removed_val - self.mean
            self.mean -= delta / self.count
            self.M2 -= delta * (removed_val - self.mean)
        
        # Add to window
        self.obs_window.append(value)
        
        # Handle Addition (Welford)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        return self.compute_sigma_c()

    def compute_sigma_c(self) -> float:
        """
        Computes sigma_c based on variance of the current window.
        chi ~ Variance (Fluctuation-Dissipation Theorem approximation)
        sigma_c = 1 / (1 + chi)
        """
        if self.count < 2:
            return 1.0
            
        # Variance = M2 / (n - 1) for sample variance, or M2 / n for population
        # Using population variance for consistency with previous implementation (numpy default)
        variance = self.M2 / self.count
        
        # Ensure non-negative (M2 should theoretically be positive, but float errors can happen)
        variance = max(0.0, variance)
        
        # In many physical systems, susceptibility is proportional to variance
        chi = variance
        
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
