"""
XGBoost integration for NVFlare federated learning.

This module provides data loaders and configurations for
running XGBoost models in a federated setting.
"""
from .data_loader import SNPXGBDataLoader, XGBDataLoader

__all__ = ["SNPXGBDataLoader", "XGBDataLoader"]
