"""
Sigma-C Zipline Plugin
======================
Copyright (c) 2025 ForgottenForge.xyz

Zipline/Backtrader integration for criticality-aware trading strategies.
"""

from typing import Dict, Any, Optional
import numpy as np

try:
    from zipline.api import order_target_percent, record
    _HAS_ZIPLINE = True
except ImportError:
    _HAS_ZIPLINE = False


class CriticalStrategy:
    """
    Base class for criticality-aware trading strategies.
    
    Usage:
        from zipline.api import order_target_percent
        from sigma_c.trading.zipline import CriticalStrategy
        
        class MyStrategy(CriticalStrategy):
            def handle_data(self, context, data):
                sigma_c = self.get_market_criticality()
                if sigma_c > 0.8:
                    # Market near crash point
                    order_target_percent(spy, 0)
    """
    
    def __init__(self):
        self.sigma_c_history = []
        self.lookback = 20
    
    def get_market_criticality(self, prices: Optional[np.ndarray] = None) -> float:
        """
        Compute market criticality from price history.
        
        Args:
            prices: Price array (optional, uses internal history if None)
            
        Returns:
            Criticality value (0-1)
        """
        if prices is None:
            if len(self.sigma_c_history) < self.lookback:
                return 0.5  # Unknown
            prices = np.array(self.sigma_c_history[-self.lookback:])
        
        # Compute returns
        returns = np.diff(prices) / prices[:-1]
        
        # Criticality metrics
        volatility = np.std(returns)
        skewness = self._compute_skewness(returns)
        
        # High volatility + negative skew = high criticality
        sigma_c = (volatility * 10) * (1 + abs(min(skewness, 0)))
        
        return float(np.clip(sigma_c, 0, 1))
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of distribution."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        skew = np.mean(((data - mean) / std) ** 3)
        return float(skew)
    
    def update_price_history(self, price: float):
        """
        Update internal price history.
        
        Args:
            price: Current price
        """
        self.sigma_c_history.append(price)
        
        # Keep only recent history
        if len(self.sigma_c_history) > self.lookback * 2:
            self.sigma_c_history = self.sigma_c_history[-self.lookback:]
    
    def handle_data(self, context, data):
        """
        Override this method in subclasses.
        
        Args:
            context: Zipline context
            data: Market data
        """
        raise NotImplementedError("Subclasses must implement handle_data")


# Example strategy
class CrashAvoidanceStrategy(CriticalStrategy):
    """
    Example strategy that reduces exposure near critical points.
    """
    
    def __init__(self, symbol='SPY', critical_threshold=0.7):
        super().__init__()
        self.symbol = symbol
        self.critical_threshold = critical_threshold
    
    def handle_data(self, context, data):
        """
        Reduce exposure when criticality is high.
        """
        if not _HAS_ZIPLINE:
            return
        
        # Get current price
        price = data.current(self.symbol, 'price')
        self.update_price_history(price)
        
        # Compute criticality
        sigma_c = self.get_market_criticality()
        
        # Adjust position based on criticality
        if sigma_c > self.critical_threshold:
            # High criticality - reduce exposure
            target_percent = 0.2
        elif sigma_c < 0.3:
            # Low criticality - full exposure
            target_percent = 1.0
        else:
            # Medium criticality - moderate exposure
            target_percent = 0.6
        
        # Execute order
        order_target_percent(self.symbol, target_percent)
        
        # Record for analysis
        record(sigma_c=sigma_c, position=target_percent)
