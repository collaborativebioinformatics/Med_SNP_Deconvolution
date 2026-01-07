"""
NVFlare Base Module for SNP Deconvolution

Provides abstract base classes and wrappers for horizontal federated learning
with NVFlare 2.4+.

Architecture:
    - SNPDeconvExecutor: Abstract base for federated executors
    - XGBoostNVFlareExecutor: Wrapper for XGBoost models
    - DLNVFlareExecutor: Wrapper for PyTorch deep learning models
    - Serialization utilities for model sharing

Federated Learning Strategy:
    - Horizontal Federation: Each site has different samples, same SNP set
    - Aggregation: FedAvg for DL, ensemble/histogram for XGBoost
    - Privacy: Local data never leaves site, only model updates shared

Usage:
    >>> from snp_deconvolution.nvflare_base import XGBoostNVFlareExecutor
    >>> executor = XGBoostNVFlareExecutor(trainer, data_loader)
    >>> weights = executor.get_model_weights()  # For sharing
    >>> executor.set_model_weights(aggregated_weights)  # From server
"""

from .base_executor import SNPDeconvExecutor, ExecutorMetrics
from .xgb_nvflare_wrapper import XGBoostNVFlareExecutor
from .dl_nvflare_wrapper import DLNVFlareExecutor
from .model_shareable import (
    serialize_pytorch_weights,
    deserialize_pytorch_weights,
    serialize_xgboost_model,
    deserialize_xgboost_model,
    validate_model_weights,
)
from .aggregation import (
    federated_averaging,
    trimmed_mean_aggregation,
    median_aggregation,
    aggregate_weights,
    FedOptAggregator,
    AggregationResult,
)

__all__ = [
    'SNPDeconvExecutor',
    'ExecutorMetrics',
    'XGBoostNVFlareExecutor',
    'DLNVFlareExecutor',
    'serialize_pytorch_weights',
    'deserialize_pytorch_weights',
    'serialize_xgboost_model',
    'deserialize_xgboost_model',
    'validate_model_weights',
    'federated_averaging',
    'trimmed_mean_aggregation',
    'median_aggregation',
    'aggregate_weights',
    'FedOptAggregator',
    'AggregationResult',
]

__version__ = '0.1.0'
