from .core.orchestrator import Universe
from .core.base import SigmaCAdapter

def __getattr__(name):
    """Lazy imports for optional domain adapters."""
    _adapter_map = {
        'QuantumAdapter': ('.adapters.quantum', 'QuantumAdapter'),
        'GPUAdapter': ('.adapters.gpu', 'GPUAdapter'),
        'FinancialAdapter': ('.adapters.financial', 'FinancialAdapter'),
        'MLAdapter': ('.adapters.ml', 'MLAdapter'),
        'ClimateAdapter': ('.adapters.climate', 'ClimateAdapter'),
        'SeismicAdapter': ('.adapters.seismic', 'SeismicAdapter'),
        'MagneticAdapter': ('.adapters.magnetic', 'MagneticAdapter'),
        'EdgeAdapter': ('.adapters.edge', 'EdgeAdapter'),
        'LLMCostAdapter': ('.adapters.llm_cost', 'LLMCostAdapter'),
    }
    if name in _adapter_map:
        module_path, attr = _adapter_map[name]
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr)
    raise AttributeError(f"module 'sigma_c' has no attribute {name!r}")

__version__ = "2.1.0"
__all__ = [
    "Universe", "SigmaCAdapter",
    "QuantumAdapter", "GPUAdapter", "FinancialAdapter", "MLAdapter",
    "ClimateAdapter", "SeismicAdapter", "MagneticAdapter",
    "EdgeAdapter", "LLMCostAdapter",
]
