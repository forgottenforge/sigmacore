from .factory import AdapterFactory

def __getattr__(name):
    """Lazy imports for adapter classes."""
    _map = {
        'QuantumAdapter': ('.quantum', 'QuantumAdapter'),
        'GPUAdapter': ('.gpu', 'GPUAdapter'),
        'FinancialAdapter': ('.financial', 'FinancialAdapter'),
        'MLAdapter': ('.ml', 'MLAdapter'),
        'ClimateAdapter': ('.climate', 'ClimateAdapter'),
        'SeismicAdapter': ('.seismic', 'SeismicAdapter'),
        'MagneticAdapter': ('.magnetic', 'MagneticAdapter'),
        'EdgeAdapter': ('.edge', 'EdgeAdapter'),
        'LLMCostAdapter': ('.llm_cost', 'LLMCostAdapter'),
    }
    if name in _map:
        import importlib
        module_path, attr = _map[name]
        module = importlib.import_module(module_path, package=__name__)
        return getattr(module, attr)
    raise AttributeError(f"module 'sigma_c.adapters' has no attribute {name!r}")

__all__ = [
    "AdapterFactory",
    "QuantumAdapter", "GPUAdapter", "FinancialAdapter", "MLAdapter",
    "ClimateAdapter", "SeismicAdapter", "MagneticAdapter",
    "EdgeAdapter", "LLMCostAdapter",
]
