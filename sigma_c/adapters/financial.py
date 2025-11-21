#!/usr/bin/env python3
"""
Sigma-C Financial Adapter
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Adapter for Financial Markets and Regime Detection.

For commercial licensing without AGPL-3.0 obligations, contact:
[nfo@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from ..core.base import SigmaCAdapter
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List
from pathlib import Path
import hashlib
import warnings

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False

class FinancialAdapter(SigmaCAdapter):
    """
    Adapter for Financial Markets.
    Ported from finone.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.cache_dir = Path(self.config.get('cache_dir', 'cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_market_data(self, symbol: str = '^GSPC', start_date='2000-01-01') -> pd.DataFrame:
        """
        Fetch and cache market data.
        """
        if not _HAS_YFINANCE:
            raise ImportError("yfinance is required for FinancialAdapter")
            
        cache_key = hashlib.sha256(f"{symbol}{start_date}".encode()).hexdigest()[:8]
        cache_file = self.cache_dir / f"market_{symbol}_{cache_key}.pkl"
        
        if cache_file.exists():
            return pd.read_pickle(cache_file)
            
        df = yf.download(symbol, start=start_date, progress=False, auto_adjust=True)
        if not df.empty:
            df['Return'] = np.log(df['Close']).diff()
            df.dropna(inplace=True)
            df.to_pickle(cache_file)
            
        return df
        
    def get_observable(self, data: np.ndarray, **kwargs) -> float:
        """
        Compute volatility clustering observable (autocorrelation).
        """
        # data is assumed to be absolute returns series
        if len(data) < 4:
            return 0.0
            
        # Autocorrelation at lag 1
        return float(pd.Series(data).autocorr(lag=1))

    def detect_regime(self, symbol: str = '^GSPC', window_days: int = 252):
        """
        Detect current market regime using sigma_c.
        """
        df = self.fetch_market_data(symbol)
        returns = df['Return'].values
        
        # Use last window
        subset = returns[-window_days:]
        
        # Compute sigma_c over time scales
        sigma_range = np.logspace(0, np.log10(min(window_days/4, 100)), 20)
        observables = []
        
        for sigma in sigma_range:
            w = max(2, int(sigma))
            # Rolling volatility
            vol = pd.Series(np.abs(subset)).rolling(w, min_periods=2).std()
            # Mean volatility clustering (autocorr of vol)
            if len(vol.dropna()) > w:
                obs = vol.autocorr(lag=1)
                observables.append(obs if not np.isnan(obs) else 0)
            else:
                observables.append(0)
                
        analysis = self.compute_susceptibility(sigma_range, np.array(observables))
        
        return {
            'symbol': symbol,
            'sigma_c': analysis['sigma_c'],
            'kappa': analysis['kappa'],
            'regime': 'Critical' if analysis['kappa'] > 10 else 'Stable',
            'sigma_values': sigma_range,
            'observable': observables
        }

    # ========== v1.1.0: Financial Diagnostics ==========
    
    def _domain_specific_diagnose(self, price_data: Optional[pd.Series] = None, symbol: str = None, **kwargs) -> Dict[str, Any]:
        """Financial-specific diagnostics."""
        issues, recommendations, details = [], [], {}
        
        if symbol and price_data is None:
            if not _HAS_YFINANCE:
                return {'status': 'error', 'issues': ['yfinance not installed'], 'recommendations': ['pip install yfinance'], 'auto_fix': None, 'details': {}}
            try:
                df = self.fetch_market_data(symbol)
                price_data = df['Close']
            except Exception as e:
                return {'status': 'error', 'issues': [f'Failed to fetch {symbol}'], 'recommendations': ['Check symbol'], 'auto_fix': None, 'details': {}}
        
        if price_data is None:
            return {'status': 'error', 'issues': ['No data'], 'recommendations': ['Provide price_data or symbol'], 'auto_fix': None, 'details': {}}
        
        details['data_length'] = len(price_data)
        if len(price_data) < 100:
            issues.append(f"Insufficient data: {len(price_data)}")
            recommendations.append("Fetch more data")
        
        status = 'ok' if not issues else ('error' if any('not installed' in i for i in issues) else 'warning')
        return {'status': status, 'issues': issues, 'recommendations': recommendations, 'auto_fix': None, 'details': details}
    
    def _domain_specific_auto_search(self, symbol: str = '^GSPC', param_ranges: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """Auto-search optimal financial parameters."""
        if not _HAS_YFINANCE:
            return {'best_params': {}, 'all_results': [], 'convergence_data': {}, 'recommendation': 'yfinance not installed'}
        
        lookback_windows = np.linspace(20, 200, 8, dtype=int) if param_ranges is None else np.linspace(*param_ranges['lookback'], 8, dtype=int)
        results = []
        
        for lookback in lookback_windows:
            try:
                result = self.detect_regime(symbol, window_days=int(lookback))
                results.append({'lookback': int(lookback), 'sigma_c': result['sigma_c'], 'kappa': result['kappa'], 'success': True})
            except Exception as e:
                results.append({'lookback': int(lookback), 'sigma_c': 0, 'kappa': 0, 'success': False, 'error': str(e)})
        
        successful = [r for r in results if r.get('success', False)]
        if not successful:
            return {'best_params': {}, 'all_results': results, 'convergence_data': {}, 'recommendation': 'No successful runs'}
        
        best = max(successful, key=lambda x: x['kappa'])
        return {'best_params': {'lookback': best['lookback']}, 'all_results': results, 'convergence_data': {'n_successful': len(successful), 'n_failed': len(results) - len(successful)}, 'recommendation': f"Use lookback={best['lookback']} (κ={best['kappa']:.2f})"}
    
    def _domain_specific_validate(self, price_data: Optional[pd.Series] = None, **kwargs) -> Dict[str, bool]:
        """Validate financial techniques."""
        checks = {'yfinance_installed': _HAS_YFINANCE}
        if price_data is not None:
            checks.update({'sufficient_data': len(price_data) >= 100, 'no_missing': price_data.isna().sum() == 0, 'positive_prices': (price_data > 0).all()})
        return checks
    
    def _domain_specific_explain(self, result: Dict[str, Any], **kwargs) -> str:
        """Financial-specific explanation."""
        return f"# Financial Regime Analysis\n\nσ_c: {result.get('sigma_c', 'N/A')}\nκ: {result.get('kappa', 'N/A')}\n\nLower σ_c = more frequent regime changes\nHigher κ = sharper transitions"

    # ========================================================================
    # v1.2.0: Universal Rigor Integration
    # ========================================================================

    def optimize_strategy(self, 
                         param_space: Dict[str, List[Any]],
                         strategy: str = 'brute_force') -> Dict[str, Any]:
        """
        Optimize a financial strategy using the BalancedFinancialOptimizer.
        
        Args:
            param_space: Dictionary of parameter names and values to sweep.
            strategy: Optimization strategy ('brute_force').
            
        Returns:
            OptimizationResult as a dictionary.
        """
        from ..optimization.financial import BalancedFinancialOptimizer
        
        optimizer = BalancedFinancialOptimizer(self)
        result = optimizer.optimize_strategy(param_space, strategy)
        
        return {
            'optimal_params': result.optimal_params,
            'score': result.score,
            'sigma_c_before': result.sigma_c_before,
            'sigma_c_after': result.sigma_c_after,
            'performance_improvement': result.performance_after - result.performance_before
        }

    def validate_rigorously(self, sigma_c_value: float, n_assets: int = 500) -> Dict[str, Any]:
        """
        Validate a sigma_c result against rigorous Financial bounds (RMT).
        """
        from ..physics.financial import RigorousFinancialSigmaC
        
        checker = RigorousFinancialSigmaC()
        context = {
            'T': 252,
            'N': n_assets
        }
        
        return checker.validate_sigma_c(sigma_c_value, context)

    # ========== v2.0: Rigorous Physics Implementation ==========

    def compute_hurst_exponent(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Computes the Hurst Exponent (H) to classify market regime.
        H < 0.5: Mean-reverting (Anti-persistent)
        H = 0.5: Random Walk (Brownian Motion)
        H > 0.5: Trending (Persistent)
        
        Uses R/S (Rescaled Range) Analysis.
        """
        ts = np.array(time_series)
        if len(ts) < 100:
            return {'hurst': 0.5, 'regime': 'insufficient_data'}
            
        # Create sub-series of different lengths
        min_chunk = 10
        max_chunk = len(ts) // 2
        scales = np.unique(np.logspace(np.log10(min_chunk), np.log10(max_chunk), 10).astype(int))
        
        rs_values = []
        for n in scales:
            # Split into chunks of size n
            n_chunks = len(ts) // n
            chunk_rs = []
            for i in range(n_chunks):
                chunk = ts[i*n : (i+1)*n]
                mean = np.mean(chunk)
                # Cumulative deviation
                y = np.cumsum(chunk - mean)
                # Range
                r = np.max(y) - np.min(y)
                # Standard deviation
                s = np.std(chunk)
                if s == 0: s = 1e-9
                chunk_rs.append(r / s)
            rs_values.append(np.mean(chunk_rs))
            
        # Log-log fit: log(R/S) ~ H * log(n)
        log_n = np.log(scales)
        log_rs = np.log(rs_values)
        hurst, intercept = np.polyfit(log_n, log_rs, 1)
        
        regime = 'random_walk'
        if hurst < 0.45: regime = 'mean_reverting'
        elif hurst > 0.55: regime = 'trending'
        
        return {
            'hurst': hurst,
            'regime': regime,
            'confidence': float(1.0 - abs(hurst - 0.5)) # Heuristic confidence
        }

    def analyze_volatility_clustering(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Analyzes volatility clustering using a GARCH(1,1) model.
        sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        
        High alpha+beta -> High persistence (clustering).
        """
        from scipy.optimize import minimize
        
        # Simple GARCH(1,1) Likelihood
        def garch_likelihood(params, rets):
            omega, alpha, beta = params
            n = len(rets)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(rets)
            
            log_lik = 0
            for t in range(1, n):
                sigma2[t] = omega + alpha * rets[t-1]**2 + beta * sigma2[t-1]
                if sigma2[t] <= 0: return 1e10 # Penalty
                log_lik += -0.5 * (np.log(sigma2[t]) + rets[t]**2 / sigma2[t])
                
            return -log_lik

        # Fit parameters
        initial_guess = [np.var(returns)*0.01, 0.1, 0.8]
        bounds = ((1e-6, None), (0, 1), (0, 1))
        
        try:
            res = minimize(garch_likelihood, initial_guess, args=(returns,), 
                          bounds=bounds, method='L-BFGS-B')
            omega, alpha, beta = res.x
            persistence = alpha + beta
        except:
            omega, alpha, beta, persistence = 0, 0, 0, 0
            
        # Sigma_c in this context relates to the persistence of volatility shocks
        # As persistence -> 1, shocks last forever (criticality)
        sigma_c_garch = 1.0 / (1.0 + (1.0 - persistence)) # Maps 1->1, 0->0.5
        
        return {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': persistence,
            'sigma_c': sigma_c_garch
        }

    def analyze_order_flow(self, imbalance_series: np.ndarray) -> Dict[str, float]:
        """
        Analyzes Order Flow Imbalance (OFI) for liquidity crashes.
        OFI = Buy_Volume - Sell_Volume at best bid/ask.
        
        Criticality occurs when OFI cumulative sum diverges (liquidity collapse).
        """
        # Cumulative OFI
        cofi = np.cumsum(imbalance_series)
        
        # Detect structural breaks in OFI trend
        # We look for super-diffusive behavior: <x^2> ~ t^gamma with gamma > 1
        
        lags = np.unique(np.logspace(0.5, np.log10(len(cofi)//4), 10).astype(int))
        msd = [] # Mean Squared Displacement
        
        for tau in lags:
            diffs = cofi[tau:] - cofi[:-tau]
            msd.append(np.mean(diffs**2))
            
        log_tau = np.log(lags)
        log_msd = np.log(msd)
        gamma, _ = np.polyfit(log_tau, log_msd, 1)
        
        # gamma = 1: Normal diffusion
        # gamma > 1: Super-diffusion (Trend/Crash)
        # gamma = 2: Ballistic
        
        return {
            'diffusion_exponent': gamma,
            'crash_risk': 'high' if gamma > 1.5 else 'low',
            'sigma_c_flow': min(1.0, max(0.0, (gamma - 1.0))) # 1 -> 0, 2 -> 1
        }
