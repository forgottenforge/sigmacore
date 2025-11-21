"""Sigma-C Finance Integrations"""

try:
    from .quantlib import CriticalPricing
    from .zipline import CriticalStrategy, CrashAvoidanceStrategy
    __all__ = ['CriticalPricing', 'CriticalStrategy', 'CrashAvoidanceStrategy']
except ImportError:
    __all__ = []
