"""Sigma-C Plugins - Framework Extensions"""

try:
    from .pennylane import SigmaCDevice
    __all__ = ['SigmaCDevice']
except ImportError:
    __all__ = []
