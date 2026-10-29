"""
sphcap: spherical cap discrepancy -- configurations, exact/approximate
computation, and gradient-based minimization.
"""
from .discrepancy import discrepancy, DiscrepancyResult
from .backend import Backend, get_default_backend
from . import configs
from . import utils
from . import balance, gram

__all__ = [
    "discrepancy", "DiscrepancyResult",
    "Backend", "get_default_backend",
    "configs", "utils", "balance", "gram",
]

__version__ = "0.2.0"
