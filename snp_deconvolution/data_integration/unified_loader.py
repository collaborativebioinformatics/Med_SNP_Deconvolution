"""
Unified Feature Loader for SNP Deconvolution

Provides a unified interface for loading both:
- Cluster ID features (privacy-preserving mode)
- Raw SNP features (baseline mode)

This loader ensures consistent data formats for both XGBoost and Deep Learning models,
enabling fair comparison between privacy-preserving and baseline approaches.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import scipy.sparse as sp

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .cluster_feature_loader import ClusterFeatureLoader

logger = logging.getLogger(__name__)


class UnifiedFeatureLoader:
    """
    Unified data loader supporting both Cluster ID and SNP feature modes.

    This class provides a consistent interface for loading data regardless of
    whether you're using the privacy-preserving Cluster mode or the baseline SNP mode.

    Attributes:
        pipeline_output_dir: Directory containing haploblock pipeline outputs
        vcf_path: Path to VCF file (optional, for SNP mode)
        use_cluster_features: Current feature mode
        cluster_loader: ClusterFeatureLoader instance

    Example:
        >>> # Privacy-preserving mode (default)
        >>> loader = UnifiedFeatureLoader('out_dir/TNFa')
        >>> X_train, y_train = loader.load_training_data(
        ...     population_files=['igsr-chb.tsv', 'igsr-gbr.tsv', 'igsr-pur.tsv'],
        ...     mode='cluster'
        ... )
        >>>
        >>> # Baseline mode
        >>> loader = UnifiedFeatureLoader('out_dir/TNFa', vcf_path='data/chr6.vcf.gz')
        >>> X_train, y_train = loader.load_training_data(
        ...     population_files=['igsr-chb.tsv', 'igsr-gbr.tsv', 'igsr-pur.tsv'],
        ...     mode='snp'
        ... )
    """

    def __init__(
        self,
        pipeline_output_dir: Union[str, Path],
        vcf_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize unified feature loader.

        Args:
            pipeline_output_dir: Directory containing haploblock pipeline outputs
            vcf_path: Optional path to VCF file for SNP mode
        """
        self.pipeline_output_dir = Path(pipeline_output_dir)
        self.vcf_path = Path(vcf_path) if vcf_path else None

        # Initialize cluster feature loader
        self.cluster_loader = ClusterFeatureLoader(self.pipeline_output_dir)

        # State
        self._cluster_data: Optional[Dict] = None
        self._vocab_sizes: Optional[List[int]] = None
        self._cluster_matrix: Optional[np.ndarray] = None
        self._snp_matrix: Optional[sp.csr_matrix] = None
        self._labels: Optional[np.ndarray] = None
        self._sample_ids: Optional[List[str]] = None

        logger.info(f"UnifiedFeatureLoader initialized: {pipeline_output_dir}")

    def load_cluster_features(self) -> Tuple[np.ndarray, List[int]]:
        """
        Load Cluster ID features from pipeline output.

        Returns:
            cluster_matrix: (n_samples, n_haploblocks) array of cluster IDs
            vocab_sizes: List of cluster counts per haploblock (for embeddings)
        """
        if self._cluster_matrix is not None:
            return self._cluster_matrix, self._vocab_sizes

        # Load cluster assignments
        self._cluster_data, self._vocab_sizes = self.cluster_loader.load_cluster_assignments()

        # Create cluster matrix
        self._cluster_matrix = self.cluster_loader.create_cluster_matrix(self._cluster_data)
        self._sample_ids = self.cluster_loader.sample_ids

        logger.info(
            f"Loaded cluster features: {self._cluster_matrix.shape}, "
            f"{len(self._vocab_sizes)} haploblocks"
        )

        return self._cluster_matrix, self._vocab_sizes

    def load_snp_features(self) -> sp.csr_matrix:
        """
        Load raw SNP features from VCF file.

        Returns:
            snp_matrix: (n_samples, n_snps) sparse matrix of genotypes

        Raises:
            ValueError: If vcf_path not provided
        """
        if self._snp_matrix is not None:
            return self._snp_matrix

        if self.vcf_path is None:
            raise ValueError(
                "VCF path not provided. Initialize UnifiedFeatureLoader with vcf_path "
                "to use SNP mode."
            )

        # Try to import sparse genotype matrix loader
        try:
            from .sparse_genotype_matrix import SparseGenotypeMatrix
            loader = SparseGenotypeMatrix()
            self._snp_matrix = loader.from_vcf(self.vcf_path)
            logger.info(f"Loaded SNP features from VCF: {self._snp_matrix.shape}")
        except ImportError:
            raise ImportError(
                "SparseGenotypeMatrix not available. "
                "Please ensure sparse_genotype_matrix.py is in data_integration/"
            )

        return self._snp_matrix

    def load_labels(
        self,
        population_files: List[Union[str, Path]]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Load population labels for samples.

        Args:
            population_files: List of population TSV files (igsr-*.tsv)

        Returns:
            labels: (n_samples,) array of population labels
            label_map: {sample_id: label}
        """
        # Ensure cluster data is loaded (to get sample_ids)
        if self._sample_ids is None:
            self.load_cluster_features()

        return self.cluster_loader.load_population_labels(
            [Path(p) for p in population_files]
        )

    def load_training_data(
        self,
        population_files: List[Union[str, Path]],
        mode: str = 'cluster',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
    ) -> Dict[str, Union[np.ndarray, sp.csr_matrix, List]]:
        """
        Load complete training data with train/val/test splits.

        Args:
            population_files: List of population TSV files
            mode: 'cluster' for privacy-preserving mode, 'snp' for baseline
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary containing:
            - X_train, X_val, X_test: Feature matrices
            - y_train, y_val, y_test: Label arrays
            - vocab_sizes: (cluster mode only) List of cluster counts
            - n_features: Number of features
            - n_classes: Number of population classes
            - mode: Feature mode used
        """
        # Load features based on mode
        if mode == 'cluster':
            X, vocab_sizes = self.load_cluster_features()
        elif mode == 'snp':
            X = self.load_snp_features()
            if sp.issparse(X):
                X = X.toarray()
            vocab_sizes = None
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'cluster' or 'snp'.")

        # Load labels
        labels, _ = self.load_labels(population_files)

        # Filter to labeled samples
        labeled_mask = labels >= 0
        X_labeled = X[labeled_mask]
        y_labeled = labels[labeled_mask]

        n_samples = len(X_labeled)
        logger.info(f"Preparing {mode} dataset: {n_samples} labeled samples")

        # Shuffle and split
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        dataset = {
            'X_train': X_labeled[train_idx],
            'X_val': X_labeled[val_idx],
            'X_test': X_labeled[test_idx],
            'y_train': y_labeled[train_idx],
            'y_val': y_labeled[val_idx],
            'y_test': y_labeled[test_idx],
            'n_features': X_labeled.shape[1],
            'n_classes': len(np.unique(y_labeled)),
            'mode': mode,
        }

        # Add vocab sizes for cluster mode
        if mode == 'cluster' and vocab_sizes is not None:
            dataset['vocab_sizes'] = [v + 1 for v in vocab_sizes]  # +1 for padding

        logger.info(
            f"Dataset prepared ({mode} mode): "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

        return dataset

    def load_for_xgboost(
        self,
        population_files: List[Union[str, Path]],
        use_cluster_features: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data formatted for XGBoost training.

        Args:
            population_files: List of population TSV files
            use_cluster_features: If True, use cluster mode; if False, use SNP mode
            **kwargs: Additional arguments passed to load_training_data

        Returns:
            X_train, y_train, X_val, y_val: Training and validation data
        """
        mode = 'cluster' if use_cluster_features else 'snp'
        dataset = self.load_training_data(population_files, mode=mode, **kwargs)

        return (
            dataset['X_train'],
            dataset['y_train'],
            dataset['X_val'],
            dataset['y_val']
        )

    def load_for_pytorch(
        self,
        population_files: List[Union[str, Path]],
        **kwargs
    ) -> Dict[str, 'torch.Tensor']:
        """
        Load data formatted for PyTorch training.

        Always uses cluster mode (embedding-based models require categorical input).

        Args:
            population_files: List of population TSV files
            **kwargs: Additional arguments passed to prepare_dataset

        Returns:
            Dataset dictionary with PyTorch tensors
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        return self.cluster_loader.prepare_dataset(
            [Path(p) for p in population_files],
            **kwargs
        )

    def get_feature_info(self, mode: str = 'cluster') -> Dict[str, any]:
        """
        Get information about the features for a given mode.

        Args:
            mode: 'cluster' or 'snp'

        Returns:
            Dictionary with feature information
        """
        if mode == 'cluster':
            if self._cluster_matrix is None:
                self.load_cluster_features()

            return {
                'mode': 'cluster',
                'n_haploblocks': self._cluster_matrix.shape[1],
                'n_samples': self._cluster_matrix.shape[0],
                'vocab_sizes': self._vocab_sizes,
                'feature_type': 'Cluster ID (categorical)',
                'privacy_level': 'high',
            }
        else:
            if self._snp_matrix is None:
                self.load_snp_features()

            return {
                'mode': 'snp',
                'n_snps': self._snp_matrix.shape[1],
                'n_samples': self._snp_matrix.shape[0],
                'feature_type': 'SNP genotype (0/1/2)',
                'privacy_level': 'low',
            }


def load_unified_features(
    pipeline_output_dir: str,
    population_files: List[str],
    mode: str = 'cluster',
    vcf_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Convenience function to load features with a single call.

    Args:
        pipeline_output_dir: Path to pipeline output
        population_files: List of population TSV file paths
        mode: 'cluster' for privacy-preserving, 'snp' for baseline
        vcf_path: Path to VCF file (required for SNP mode)
        **kwargs: Additional arguments for load_training_data

    Returns:
        Dataset dictionary ready for training

    Example:
        >>> dataset = load_unified_features(
        ...     'out_dir/TNFa',
        ...     ['igsr-chb.tsv', 'igsr-gbr.tsv', 'igsr-pur.tsv'],
        ...     mode='cluster'
        ... )
        >>> X_train, y_train = dataset['X_train'], dataset['y_train']
    """
    loader = UnifiedFeatureLoader(pipeline_output_dir, vcf_path=vcf_path)
    return loader.load_training_data(population_files, mode=mode, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("UnifiedFeatureLoader - unified data loading for XGBoost and DL")
    print()
    print("Usage:")
    print("  # Privacy-preserving mode (recommended)")
    print("  loader = UnifiedFeatureLoader('out_dir/TNFa')")
    print("  X_train, y_train, X_val, y_val = loader.load_for_xgboost(")
    print("      ['igsr-chb.tsv', 'igsr-gbr.tsv', 'igsr-pur.tsv'],")
    print("      use_cluster_features=True")
    print("  )")
    print()
    print("  # Baseline mode")
    print("  loader = UnifiedFeatureLoader('out_dir/TNFa', vcf_path='chr6.vcf.gz')")
    print("  X_train, y_train, X_val, y_val = loader.load_for_xgboost(")
    print("      ['igsr-chb.tsv', 'igsr-gbr.tsv', 'igsr-pur.tsv'],")
    print("      use_cluster_features=False")
    print("  )")
