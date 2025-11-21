"""
Sigma-C QuantLib Extension
===========================
Copyright (c) 2025 ForgottenForge.xyz

QuantLib integration for criticality-adjusted pricing.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    import QuantLib as ql
    _HAS_QUANTLIB = True
except ImportError:
    _HAS_QUANTLIB = False
    ql = None


class CriticalPricing:
    """
    QuantLib pricing with criticality adjustments.
    
    Usage:
        from sigma_c.finance.quantlib import CriticalPricing
        
        price = CriticalPricing.black_scholes(
            S=100, K=110, r=0.05, sigma=0.2, T=1.0,
            criticality_adjustment=True
        )
    """
    
    @staticmethod
    def black_scholes(S: float, K: float, r: float, sigma: float, T: float,
                     option_type: str = 'call', criticality_adjustment: bool = False) -> float:
        """
        Black-Scholes pricing with optional criticality adjustment.
        
        Args:
            S: Spot price
            K: Strike price
            r: Risk-free rate
            sigma: Volatility
            T: Time to maturity
            option_type: 'call' or 'put'
            criticality_adjustment: Apply sigma_c adjustment
            
        Returns:
            Option price
        """
        if not _HAS_QUANTLIB:
            # Fallback to analytical formula
            return CriticalPricing._bs_analytical(S, K, r, sigma, T, option_type)
        
        # Setup QuantLib objects
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today
        
        # Option setup
        exercise = ql.EuropeanExercise(today + int(T * 365))
        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type == 'call' else ql.Option.Put,
            K
        )
        option = ql.VanillaOption(payoff, exercise)
        
        # Market data
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, r, ql.Actual365Fixed())
        )
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), sigma, ql.Actual365Fixed())
        )
        
        # Process
        bs_process = ql.BlackScholesProcess(spot_handle, flat_ts, flat_vol_ts)
        
        # Engine
        engine = ql.AnalyticEuropeanEngine(bs_process)
        option.setPricingEngine(engine)
        
        price = option.NPV()
        
        # Apply criticality adjustment if requested
        if criticality_adjustment:
            # Adjust for market criticality (simplified)
            sigma_c = CriticalPricing._estimate_market_criticality(sigma, T)
            adjustment_factor = 1.0 + 0.1 * sigma_c  # 10% max adjustment
            price *= adjustment_factor
        
        return float(price)
    
    @staticmethod
    def _bs_analytical(S: float, K: float, r: float, sigma: float, T: float, option_type: str) -> float:
        """
        Analytical Black-Scholes formula (fallback).
        """
        from scipy.stats import norm
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return float(price)
    
    @staticmethod
    def _estimate_market_criticality(sigma: float, T: float) -> float:
        """
        Estimate market criticality from volatility and time.
        
        Args:
            sigma: Volatility
            T: Time to maturity
            
        Returns:
            Criticality estimate (0-1)
        """
        # High volatility + short time = high criticality
        criticality = (sigma * np.sqrt(T)) / 0.5  # Normalize
        return float(np.clip(criticality, 0, 1))
