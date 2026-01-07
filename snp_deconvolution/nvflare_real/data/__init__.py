"""
Federated data management utilities.

This module provides data splitting and federated data loading
functionality for distributing datasets across clients.
"""

# Import data_splitter by default (no heavy dependencies)
from .data_splitter import FederatedDataSplitter

# Lazy import for federated_data_module (requires pytorch_lightning)
def _lazy_import_data_module():
    """Lazy import to avoid requiring pytorch_lightning for basic usage."""
    try:
        from .federated_data_module import SNPFederatedDataModule
        return SNPFederatedDataModule
    except ImportError as e:
        raise ImportError(
            "SNPFederatedDataModule requires pytorch_lightning. "
            "Install with: pip install pytorch-lightning"
        ) from e

__all__ = ['FederatedDataSplitter', 'SNPFederatedDataModule']

# Make SNPFederatedDataModule available but only import when accessed
def __getattr__(name):
    if name == 'SNPFederatedDataModule':
        return _lazy_import_data_module()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
