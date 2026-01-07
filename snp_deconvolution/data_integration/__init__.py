"""
Data Integration Module for SNP Deconvolution

This module provides tools for loading and integrating data from haploblock
pipeline outputs, VCF files, and population labels for machine learning.

Unified Interface (recommended):
    - UnifiedFeatureLoader: Supports both Cluster and SNP modes
    - load_unified_features: Convenience function for quick loading

For Deep Learning (embedding-based):
    - ClusterFeatureLoader: Load cluster IDs for embedding layers

For XGBoost (sparse features):
    - SparseGenotypeMatrix: Load genotype data as sparse matrix
"""

from .haploblock_loader import HaploblockMLDataLoader
from .sparse_genotype_matrix import SparseGenotypeMatrix
from .feature_engineering import (
    encode_hashes_as_features,
    create_haploblock_features,
)
from .cluster_feature_loader import (
    ClusterFeatureLoader,
    load_cluster_features,
)
from .unified_loader import (
    UnifiedFeatureLoader,
    load_unified_features,
)

__all__ = [
    # Unified Interface (recommended)
    "UnifiedFeatureLoader",
    "load_unified_features",
    # Deep Learning
    "ClusterFeatureLoader",
    "load_cluster_features",
    # XGBoost / Legacy
    "HaploblockMLDataLoader",
    "SparseGenotypeMatrix",
    # Feature engineering
    "encode_hashes_as_features",
    "create_haploblock_features",
]
