"""
Sigma-C Control Module
======================
Copyright (c) 2025 ForgottenForge.xyz

Implements active control systems and real-time streaming analysis
for maintaining systems at criticality.
"""

import numpy as np
from collections import deque
from typing import Optional, Deque

class StreamingSigmaC:
    """
    Real-time, incremental calculation of Critical Susceptibility (sigma_c).
    Uses Welford's Online Algorithm for amortized O(1) updates.
    """

    def __init__(self, window_size: int = 1000):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.window_size = window_size
        self.epsilon_window: Deque[float] = deque(maxlen=window_size)
        self.obs_window: Deque[float] = deque(maxlen=window_size)

        # Welford's Algorithm State
        self.mean = 0.0
        self.M2 = 0.0
        self.count = 0

    def update(self, epsilon: float, observable: Optional[float] = None) -> float:
        """
        Add a new data point and return the current sigma_c.
        Uses Welford's Online Algorithm for numerically stable variance.
        """
        if observable is None:
            value = epsilon
        else:
            value = observable
            self.epsilon_window.append(epsilon)

        # Handle Removal (Inverse Welford) - must happen before append
        if len(self.obs_window) == self.window_size:
            removed_val = self.obs_window[0]
            self.count -= 1
            if self.count > 0:
                delta = removed_val - self.mean
                self.mean -= delta / self.count
                self.M2 -= delta * (removed_val - self.mean)
            else:
                self.mean = 0.0
                self.M2 = 0.0

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

        variance = self.M2 / self.count
        variance = max(0.0, variance)
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
