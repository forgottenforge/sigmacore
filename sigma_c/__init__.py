from .core.orchestrator import Universe
from .core.base import SigmaCAdapter
from .adapters.quantum import QuantumAdapter
from .adapters.gpu import GPUAdapter
from .adapters.financial import FinancialAdapter
from .adapters.ml import MLAdapter
from .adapters.climate import ClimateAdapter
from .adapters.seismic import SeismicAdapter
from .adapters.magnetic import MagneticAdapter
from .adapters.edge import EdgeAdapter
from .adapters.llm_cost import LLMCostAdapter

__version__ = "2.0.0"
__all__ = ["Universe", "SigmaCAdapter", "EdgeAdapter", "LLMCostAdapter"]
