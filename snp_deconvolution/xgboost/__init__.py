"""
XGBoost GPU-accelerated SNP Deconvolution Module.

This module provides GPU-accelerated XGBoost implementations for SNP analysis,
feature selection, and population classification tasks. Optimized for A100/H100 GPUs
and high-dimensional sparse genomic data.

Classes:
    XGBoostSNPTrainer: GPU-accelerated XGBoost trainer for SNP classification
    IterativeSNPSelector: Iterative feature selection using XGBoost importance

Utilities:
    evaluate_classification: Comprehensive classification evaluation
    save_feature_importance: Save importance scores to file
    load_feature_importance: Load importance scores from file
    filter_sparse_matrix_by_features: Filter matrix to selected features
    And more utility functions...
"""

from .xgb_trainer import XGBoostSNPTrainer
from .feature_selector import IterativeSNPSelector
from .utils import (
    evaluate_classification,
    print_evaluation_report,
    save_feature_importance,
    load_feature_importance,
    filter_sparse_matrix_by_features,
    compute_sparsity_statistics,
    create_stratified_split,
    get_top_k_indices,
    encode_populations,
    decode_populations,
    merge_importance_scores,
    compute_class_weights,
    log_dataset_info,
)

__all__ = [
    # Core classes
    'XGBoostSNPTrainer',
    'IterativeSNPSelector',
    # Utility functions
    'evaluate_classification',
    'print_evaluation_report',
    'save_feature_importance',
    'load_feature_importance',
    'filter_sparse_matrix_by_features',
    'compute_sparsity_statistics',
    'create_stratified_split',
    'get_top_k_indices',
    'encode_populations',
    'decode_populations',
    'merge_importance_scores',
    'compute_class_weights',
    'log_dataset_info',
]

__version__ = '0.1.0'
