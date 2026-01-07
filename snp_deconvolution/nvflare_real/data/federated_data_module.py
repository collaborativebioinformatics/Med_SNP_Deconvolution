"""
Federated Data Module for NVFlare Lightning

Lightning DataModule that loads site-specific data splits for federated learning.
Supports both 'cluster' and 'snp' feature types with automatic dimension detection.

Key Features:
- Site-specific data loading based on site_name
- Supports cluster features (haploblock clusters) and SNP features (raw genotypes)
- Automatic feature dimension detection
- Memory-efficient data loading with PyTorch DataLoader
- Compatible with NVFlare federated learning workflow

Data Format:
    Each site has its own .npz file: {site_name}_{feature_type}.npz

    Structure:
        - X_train: Training features [n_samples, n_features, encoding_dim]
        - y_train: Training labels [n_samples]
        - X_val: Validation features [n_samples, n_features, encoding_dim]
        - y_val: Validation labels [n_samples]
        - X_test: Test features [n_samples, n_features, encoding_dim]
        - y_test: Test labels [n_samples]

Example:
    # In client.py
    data_module = SNPFederatedDataModule(
        data_dir='./data',
        site_name='site1',
        feature_type='cluster',
        batch_size=128
    )
    data_module.setup()
    trainer.fit(model, datamodule=data_module)

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-07
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class SNPFederatedDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for federated SNP/cluster data.

    Loads site-specific data splits for federated learning. Each site has
    its own training/validation/test data stored in .npz format.

    Args:
        data_dir: Directory containing federated data splits
        site_name: Unique identifier for this site (e.g., 'site1', 'hospital_A')
        feature_type: Type of features - 'cluster' or 'snp'
        batch_size: Training batch size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        drop_last: Whether to drop last incomplete batch

    Attributes:
        train_dataset: Training TensorDataset
        val_dataset: Validation TensorDataset
        test_dataset: Test TensorDataset
        n_features: Number of features (clusters or SNPs)
        encoding_dim: Dimension of genotype encoding
        num_classes: Number of population classes

    Example:
        >>> dm = SNPFederatedDataModule(
        ...     data_dir='./data',
        ...     site_name='site1',
        ...     feature_type='cluster'
        ... )
        >>> dm.setup()
        >>> print(f"Features: {dm.n_features}, Classes: {dm.num_classes}")
        >>> train_loader = dm.train_dataloader()
    """

    def __init__(
        self,
        data_dir: str,
        site_name: str,
        feature_type: str = 'cluster',
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.site_name = site_name
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # Data attributes (set after setup)
        self.train_dataset: Optional[TensorDataset] = None
        self.val_dataset: Optional[TensorDataset] = None
        self.test_dataset: Optional[TensorDataset] = None
        self.n_features: Optional[int] = None
        self.encoding_dim: Optional[int] = None
        self.num_classes: Optional[int] = None

        # Validate feature type
        if feature_type not in ['cluster', 'snp']:
            raise ValueError(f"Invalid feature_type: {feature_type}. Must be 'cluster' or 'snp'")

        logger.info(f"Initialized DataModule for site: {site_name}")
        logger.info(f"  Feature type: {feature_type}")
        logger.info(f"  Data directory: {data_dir}")

    def _normalize_site_name(self, site_name: str) -> str:
        """
        Normalize site name to match data directory naming convention.

        NVFlare simulator creates clients with names like 'site-1', but
        data directories may use 'site1' format. This method tries both.

        Args:
            site_name: Original site name from NVFlare

        Returns:
            Normalized site name that matches existing data directory
        """
        # Try original name first
        if (self.data_dir / site_name).exists():
            return site_name

        # Try removing hyphens (site-1 -> site1)
        normalized = site_name.replace('-', '')
        if (self.data_dir / normalized).exists():
            logger.info(f"Normalized site name: {site_name} -> {normalized}")
            return normalized

        # Try adding hyphens (site1 -> site-1)
        import re
        hyphenated = re.sub(r'(site)(\d+)', r'\1-\2', site_name)
        if hyphenated != site_name and (self.data_dir / hyphenated).exists():
            logger.info(f"Normalized site name: {site_name} -> {hyphenated}")
            return hyphenated

        # Return original if no match found
        return site_name

    def _get_data_file_path(self) -> Path:
        """
        Get path to site-specific data file.

        Supports multiple data layouts:
        1. {data_dir}/{site_name}_{feature_type}.npz (legacy single file)
        2. {data_dir}/{site_name}/train_{feature_type}.npz + val_{feature_type}.npz (current format)
        3. {data_dir}/{feature_type}.npz (single dataset mode)

        Returns:
            Path to data directory or file

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        # Normalize site name to handle format differences (site-1 vs site1)
        normalized_site_name = self._normalize_site_name(self.site_name)

        # Try layout 2: site subdirectory with train/val files (current format from data_splitter)
        site_dir = self.data_dir / normalized_site_name
        train_file = site_dir / f"train_{self.feature_type}.npz"
        val_file = site_dir / f"val_{self.feature_type}.npz"

        if train_file.exists() and val_file.exists():
            logger.info(f"Using split files from: {site_dir}")
            return site_dir  # Return directory, load split files in _load_npz_data

        # Try layout 1: site-specific combined file
        data_file = self.data_dir / f"{normalized_site_name}_{self.feature_type}.npz"
        if data_file.exists():
            logger.info(f"Using combined data file: {data_file}")
            return data_file

        # Try layout 3: single dataset mode
        data_file = self.data_dir / f"{self.feature_type}.npz"
        if data_file.exists():
            logger.info(f"Using shared data file: {data_file}")
            return data_file

        raise FileNotFoundError(
            f"Data file not found for site '{self.site_name}' (normalized: '{normalized_site_name}') "
            f"with feature type '{self.feature_type}'\n"
            f"Tried:\n"
            f"  1. {site_dir}/train_{self.feature_type}.npz (split format)\n"
            f"  2. {self.data_dir}/{normalized_site_name}_{self.feature_type}.npz (combined format)\n"
            f"  3. {self.data_dir}/{self.feature_type}.npz (shared format)\n"
            f"Please run prepare_federated_data.py to prepare the data."
        )

    def _encode_cluster_ids(
        self,
        X: torch.Tensor,
        encoding_dim: int = 8,
    ) -> torch.Tensor:
        """
        Convert 2D cluster IDs to 3D encoded format for model input.

        The model expects input shape (batch, n_features, encoding_dim).
        If input is 2D (batch, n_features), we encode each cluster ID
        using a simple Gaussian encoding.

        Args:
            X: Input tensor, shape (n_samples, n_features) or (n_samples, n_features, encoding_dim)
            encoding_dim: Target encoding dimension (default 8)

        Returns:
            Encoded tensor with shape (n_samples, n_features, encoding_dim)
        """
        if X.dim() == 3:
            # Already 3D, return as is
            return X

        if X.dim() != 2:
            raise ValueError(f"Expected 2D or 3D tensor, got {X.dim()}D")

        n_samples, n_features = X.shape

        # Simple encoding: normalize cluster IDs and create encoding vectors
        # Use Gaussian-like encoding where position in encoding_dim encodes the value
        X_encoded = torch.zeros(n_samples, n_features, encoding_dim)

        # Normalize cluster IDs to [0, 1] range
        X_float = X.float()
        X_min = X_float.min(dim=0, keepdim=True)[0]
        X_max = X_float.max(dim=0, keepdim=True)[0]
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        X_norm = (X_float - X_min) / X_range

        # Create encoding: spread the normalized value across encoding_dim
        # This creates a simple positional encoding based on the cluster ID
        for i in range(encoding_dim):
            # Create a Gaussian bump centered at position i/(encoding_dim-1)
            center = i / (encoding_dim - 1) if encoding_dim > 1 else 0.5
            sigma = 0.5 / encoding_dim  # Width of the Gaussian
            X_encoded[:, :, i] = torch.exp(-((X_norm - center) ** 2) / (2 * sigma ** 2))

        logger.info(f"Encoded 2D cluster IDs {X.shape} -> 3D {X_encoded.shape}")
        return X_encoded

    def _load_npz_data(self, data_path: Path) -> Tuple[torch.Tensor, ...]:
        """
        Load data from .npz file(s).

        Supports two formats:
        1. Combined file: {path}.npz with X_train, y_train, X_val, y_val, X_test, y_test
        2. Split files: {path}/train_{feature_type}.npz + val_{feature_type}.npz

        Args:
            data_path: Path to .npz file or directory containing split files

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test) as tensors

        Raises:
            ValueError: If required fields are missing or have invalid shapes
        """
        logger.info(f"Loading data from: {data_path}")

        # Check if it's a directory (split file format) or single file
        if data_path.is_dir():
            # Load split files (format from data_splitter.py)
            train_file = data_path / f"train_{self.feature_type}.npz"
            val_file = data_path / f"val_{self.feature_type}.npz"

            try:
                train_data = np.load(train_file)
                val_data = np.load(val_file)
            except Exception as e:
                raise IOError(f"Failed to load split files from {data_path}: {e}")

            # Extract data from split files
            X_train = torch.from_numpy(train_data['X']).float()
            y_train = torch.from_numpy(train_data['y']).long()
            X_val = torch.from_numpy(val_data['X']).float()
            y_val = torch.from_numpy(val_data['y']).long()

            # Use validation set as test set if no separate test set
            X_test = X_val.clone()
            y_test = y_val.clone()

            logger.info("Loaded split format (train/val files)")
            logger.info("Note: Using validation set as test set (no separate test file)")

        else:
            # Load combined file
            try:
                data = np.load(data_path)
            except Exception as e:
                raise IOError(f"Failed to load {data_path}: {e}")

            # Check for required fields
            required_fields = ['X_train', 'y_train', 'X_val', 'y_val']
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                # Try alternative format: X, y in single file
                if 'X' in data and 'y' in data:
                    logger.warning("Found X, y format - splitting into train/val/test")
                    X = torch.from_numpy(data['X']).float()
                    y = torch.from_numpy(data['y']).long()
                    n = len(X)
                    train_end = int(n * 0.7)
                    val_end = int(n * 0.85)
                    X_train, y_train = X[:train_end], y[:train_end]
                    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
                    X_test, y_test = X[val_end:], y[val_end:]
                else:
                    raise ValueError(
                        f"Missing required fields in {data_path}: {missing_fields}\n"
                        f"Available fields: {list(data.keys())}"
                    )
            else:
                # Standard combined format
                X_train = torch.from_numpy(data['X_train']).float()
                y_train = torch.from_numpy(data['y_train']).long()
                X_val = torch.from_numpy(data['X_val']).float()
                y_val = torch.from_numpy(data['y_val']).long()

                # Test set is optional
                if 'X_test' in data and 'y_test' in data:
                    X_test = torch.from_numpy(data['X_test']).float()
                    y_test = torch.from_numpy(data['y_test']).long()
                else:
                    X_test = X_val.clone()
                    y_test = y_val.clone()
                    logger.info("No test set found, using validation set as test set")

        # Handle 2D data (encode to 3D with proper encoding_dim)
        if X_train.dim() == 2:
            logger.info("Data is 2D, encoding cluster IDs to 3D format...")
            encoding_dim = 8  # Standard encoding dimension for the model
            X_train = self._encode_cluster_ids(X_train, encoding_dim)
            X_val = self._encode_cluster_ids(X_val, encoding_dim)
            X_test = self._encode_cluster_ids(X_test, encoding_dim)

        # Validate shapes
        if X_train.dim() != 3:
            raise ValueError(
                f"Expected X_train to be 3D [n_samples, n_features, encoding_dim], "
                f"got shape {X_train.shape}"
            )

        if y_train.dim() != 1:
            raise ValueError(
                f"Expected y_train to be 1D [n_samples], got shape {y_train.shape}"
            )

        # Log statistics
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Train: {X_train.shape[0]} samples")
        logger.info(f"  Val:   {X_val.shape[0]} samples")
        logger.info(f"  Test:  {X_test.shape[0]} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Encoding dim: {X_train.shape[2]}")
        logger.info(f"  Classes: {y_train.unique().tolist()}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def setup(self, stage: Optional[str] = None):
        """
        Load and setup datasets.

        This method is called by Lightning before training/validation/testing.

        Args:
            stage: Optional stage ('fit', 'validate', 'test', or None)

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        # Skip if already setup
        if self.train_dataset is not None:
            logger.debug("Data already setup, skipping")
            return

        # Get data file path
        data_file = self._get_data_file_path()

        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self._load_npz_data(data_file)

        # Extract metadata
        self.n_features = X_train.shape[1]
        self.encoding_dim = X_train.shape[2]
        self.num_classes = len(torch.unique(y_train))

        # Create datasets
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)

        logger.info("="*60)
        logger.info(f"Site: {self.site_name} - Data Setup Complete")
        logger.info("="*60)
        logger.info(f"Feature type: {self.feature_type}")
        logger.info(f"Number of features: {self.n_features}")
        logger.info(f"Encoding dimension: {self.encoding_dim}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Train/Val/Test: {len(self.train_dataset)}/{len(self.val_dataset)}/{len(self.test_dataset)}")
        logger.info("="*60)

    def train_dataloader(self) -> DataLoader:
        """
        Create training DataLoader.

        Returns:
            DataLoader for training data
        """
        if self.train_dataset is None:
            self.setup()

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create validation DataLoader.

        Returns:
            DataLoader for validation data
        """
        if self.val_dataset is None:
            self.setup()

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create test DataLoader.

        Returns:
            DataLoader for test data
        """
        if self.test_dataset is None:
            self.setup()

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample batch for inspection.

        Useful for debugging and determining model input dimensions.

        Returns:
            Tuple of (X_batch, y_batch)
        """
        if self.train_dataset is None:
            self.setup()

        loader = self.train_dataloader()
        batch = next(iter(loader))
        return batch

    def get_class_distribution(self) -> dict:
        """
        Get class distribution statistics.

        Returns:
            Dictionary with class counts for train/val/test splits
        """
        if self.train_dataset is None:
            self.setup()

        def count_classes(dataset):
            labels = dataset.tensors[1]
            unique, counts = torch.unique(labels, return_counts=True)
            return {int(c): int(cnt) for c, cnt in zip(unique, counts)}

        return {
            'train': count_classes(self.train_dataset),
            'val': count_classes(self.val_dataset),
            'test': count_classes(self.test_dataset),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SNPFederatedDataModule(\n"
            f"  site_name={self.site_name},\n"
            f"  feature_type={self.feature_type},\n"
            f"  n_features={self.n_features},\n"
            f"  encoding_dim={self.encoding_dim},\n"
            f"  num_classes={self.num_classes},\n"
            f"  batch_size={self.batch_size},\n"
            f"  data_dir={self.data_dir}\n"
            f")"
        )


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
    )

    print("Testing SNPFederatedDataModule")
    print("="*60)

    # Test with dummy data
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        n_samples = 100
        n_features = 50
        encoding_dim = 8
        num_classes = 3

        dummy_data = {
            'X_train': np.random.randn(n_samples, n_features, encoding_dim).astype(np.float32),
            'y_train': np.random.randint(0, num_classes, n_samples).astype(np.int64),
            'X_val': np.random.randn(20, n_features, encoding_dim).astype(np.float32),
            'y_val': np.random.randint(0, num_classes, 20).astype(np.int64),
            'X_test': np.random.randn(20, n_features, encoding_dim).astype(np.float32),
            'y_test': np.random.randint(0, num_classes, 20).astype(np.int64),
        }

        data_file = Path(tmpdir) / "site1_cluster.npz"
        np.savez(data_file, **dummy_data)
        print(f"Created dummy data: {data_file}")

        # Test DataModule
        dm = SNPFederatedDataModule(
            data_dir=tmpdir,
            site_name='site1',
            feature_type='cluster',
            batch_size=32,
        )

        print("\nSetup DataModule...")
        dm.setup()

        print("\nDataModule Info:")
        print(dm)

        print("\nClass Distribution:")
        dist = dm.get_class_distribution()
        for split, counts in dist.items():
            print(f"  {split}: {counts}")

        print("\nSample Batch:")
        X_batch, y_batch = dm.get_sample_batch()
        print(f"  X shape: {X_batch.shape}")
        print(f"  y shape: {y_batch.shape}")

        print("\nDataLoaders:")
        print(f"  Train batches: {len(dm.train_dataloader())}")
        print(f"  Val batches: {len(dm.val_dataloader())}")
        print(f"  Test batches: {len(dm.test_dataloader())}")

        print("\n" + "="*60)
        print("Test completed successfully!")
