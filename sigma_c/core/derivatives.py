#!/usr/bin/env python3
"""
Extended Derivative Estimation Module
==========================================================
Copyright (c) 2025-2026 ForgottenForge.xyz

Provides multiple methods for estimating derivatives of noisy data:
- Savitzky-Golay filtering
- Regularized cubic spline
- Gaussian Process regression (with uncertainty)
- Automatic method selection

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional


def savitzky_golay_derivative(x: np.ndarray, y: np.ndarray,
                              window_length: int = 11,
                              polyorder: int = 3) -> np.ndarray:
    """
    Compute derivative using Savitzky-Golay filter.

    Args:
        x: Independent variable (must be uniformly spaced or nearly so)
        y: Dependent variable
        window_length: Filter window length (odd integer)
        polyorder: Polynomial order for local fit

    Returns:
        Estimated derivative dy/dx
    """
    from scipy.signal import savgol_filter

    n = len(y)
    window_length = min(window_length, n)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, polyorder + 2)
    if window_length % 2 == 0:
        window_length += 1

    # Compute derivative using savgol_filter with deriv=1
    dx = np.mean(np.diff(x))
    if abs(dx) < 1e-15:
        return np.zeros_like(y)

    dy = savgol_filter(y, window_length, polyorder, deriv=1, delta=dx)
    return dy


def spline_derivative(x: np.ndarray, y: np.ndarray,
                      smoothing_factor: Optional[float] = None) -> np.ndarray:
    """
    Compute derivative using regularized cubic spline.

    Args:
        x: Independent variable
        y: Dependent variable
        smoothing_factor: Spline smoothing parameter (None = automatic)

    Returns:
        Estimated derivative dy/dx
    """
    from scipy.interpolate import UnivariateSpline

    # Sort by x
    sort_idx = np.argsort(x)
    xs, ys = x[sort_idx], y[sort_idx]

    if smoothing_factor is None:
        # Automatic: minimal smoothing to preserve derivative structure
        smoothing_factor = max(0.01, len(y) * np.var(y) * 0.001)

    try:
        spline = UnivariateSpline(xs, ys, s=smoothing_factor)
        dy = spline.derivative()(xs)
        # Unsort
        result = np.empty_like(dy)
        result[sort_idx] = dy
        return result
    except Exception:
        # Fallback to numpy gradient
        return np.gradient(y, x)


def gp_regression_derivative(x: np.ndarray, y: np.ndarray,
                             length_scale: Optional[float] = None,
                             noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute derivative using Gaussian Process regression.
    Returns both mean derivative and uncertainty.

    Args:
        x: Independent variable
        y: Dependent variable
        length_scale: GP kernel length scale (None = auto from data range)
        noise_level: Observation noise level

    Returns:
        Tuple of (mean_derivative, std_derivative)
    """
    n = len(x)
    if length_scale is None:
        length_scale = (np.max(x) - np.min(x)) / 5.0

    # RBF kernel: k(x, x') = exp(-0.5 * (x-x')^2 / l^2)
    X = x.reshape(-1, 1)
    diffs = X - X.T
    K = np.exp(-0.5 * diffs**2 / length_scale**2)
    K_noise = K + noise_level**2 * np.eye(n)

    # Derivative of kernel w.r.t. x_*: dk/dx_* = -(x_* - x') / l^2 * k(x_*, x')
    # For prediction at training points:
    try:
        L = np.linalg.cholesky(K_noise)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        # Fallback if Cholesky fails
        alpha = np.linalg.solve(K_noise + 1e-6 * np.eye(n), y)

    # Derivative at each training point
    dy_mean = np.zeros(n)
    dy_std = np.zeros(n)

    for i in range(n):
        dk_dx = -(x[i] - x) / length_scale**2 * K[i, :]
        dy_mean[i] = dk_dx @ alpha

        # Uncertainty from GP posterior derivative variance
        # d^2k/dx_*^2 at x_* = x[i]: (1/l^2)(1 - 0) = 1/l^2 (self-point)
        # Posterior variance of derivative: d2k_self - v^T v
        try:
            v = np.linalg.solve(L, dk_dx)
            d2k_self = 1.0 / length_scale**2
            dy_var = d2k_self - v @ v
            dy_std[i] = np.sqrt(max(0, dy_var))
        except Exception:
            dy_std[i] = noise_level

    return dy_mean, dy_std


def select_best_method(x: np.ndarray, y: np.ndarray) -> str:
    """
    Automatically select the best derivative estimation method.

    Selection criteria:
    - n < 20: spline (more stable for small samples)
    - n < 50: savitzky_golay (good balance)
    - n >= 50 with low noise: savitzky_golay
    - n >= 50 with high noise: gp (uncertainty quantification useful)

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        Method name: 'savitzky_golay', 'spline', or 'gp'
    """
    n = len(x)

    if n < 20:
        return 'spline'
    elif n < 50:
        return 'savitzky_golay'
    else:
        # Estimate noise level via residuals from local polynomial
        from scipy.signal import savgol_filter
        window = min(11, n if n % 2 == 1 else n - 1)
        try:
            smoothed = savgol_filter(y, window, 3)
            residuals = y - smoothed
            snr = np.std(smoothed) / (np.std(residuals) + 1e-12)
            if snr < 5:
                return 'gp'
        except Exception:
            pass
        return 'savitzky_golay'


def compute_derivative(x: np.ndarray, y: np.ndarray,
                       method: str = 'auto',
                       **kwargs) -> Dict[str, Any]:
    """
    Unified derivative computation interface.

    Args:
        x: Independent variable
        y: Dependent variable
        method: 'gaussian' (default/legacy), 'savitzky_golay', 'spline',
                'gp', or 'auto'
        **kwargs: Method-specific parameters

    Returns:
        Dict with 'derivative', 'method_used', and optionally 'uncertainty'
    """
    if method == 'auto':
        method = select_best_method(x, y)

    result = {'method_used': method}

    if method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = kwargs.get('kernel_sigma', 0.6)
        smoothed = gaussian_filter1d(y, sigma)
        result['derivative'] = np.abs(np.gradient(smoothed, x))

    elif method == 'savitzky_golay':
        window = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 3)
        result['derivative'] = np.abs(savitzky_golay_derivative(x, y, window, polyorder))

    elif method == 'spline':
        sf = kwargs.get('smoothing_factor', None)
        result['derivative'] = np.abs(spline_derivative(x, y, sf))

    elif method == 'gp':
        ls = kwargs.get('length_scale', None)
        noise = kwargs.get('noise_level', 0.1)
        dy_mean, dy_std = gp_regression_derivative(x, y, ls, noise)
        result['derivative'] = np.abs(dy_mean)
        result['uncertainty'] = dy_std

    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Use 'gaussian', 'savitzky_golay', 'spline', 'gp', or 'auto'.")

    return result
