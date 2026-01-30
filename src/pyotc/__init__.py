"""Top-level package for pyotc."""

__author__ = """Jay Hineman"""
__email__ = "jay.hineman@gmail.com"
__version__ = "0.3.0"

# Public API re-exports
from .otc import entropic_otc, exact_otc

__all__ = [
    "exact_otc",
    "entropic_otc",
    "__author__",
    "__email__",
    "__version__",
]
