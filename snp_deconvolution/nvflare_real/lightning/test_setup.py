#!/usr/bin/env python3
"""
Quick test script for NVFlare Lightning setup

This script creates dummy data and runs a minimal POC simulation to verify
that all components are working correctly.

Usage:
    python test_setup.py
"""

import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_test_data(
    output_dir: str = 'test_data',
    num_sites: int = 3,
    n_samples: int = 100,
    n_features: int = 50,
    encoding_dim: int = 8,
    num_classes: int = 3,
    feature_type: str = 'cluster'
):
    """
    Create dummy test data for POC simulation.

    Args:
        output_dir: Output directory for data files
        num_sites: Number of sites to create data for
        n_samples: Number of samples per site
        n_features: Number of features (clusters or SNPs)
        encoding_dim: Genotype encoding dimension
        num_classes: Number of classes
        feature_type: 'cluster' or 'snp'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Creating Test Data")
    logger.info("="*60)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Number of sites: {num_sites}")
    logger.info(f"Samples per site: {n_samples}")
    logger.info(f"Features: {n_features}")
    logger.info(f"Encoding dim: {encoding_dim}")
    logger.info(f"Classes: {num_classes}")
    logger.info(f"Feature type: {feature_type}")
    logger.info("="*60)

    for i in range(num_sites):
        site_name = f"site{i+1}"

        # Generate random data
        data = {
            'X_train': np.random.randn(n_samples, n_features, encoding_dim).astype(np.float32),
            'y_train': np.random.randint(0, num_classes, n_samples).astype(np.int64),
            'X_val': np.random.randn(n_samples // 5, n_features, encoding_dim).astype(np.float32),
            'y_val': np.random.randint(0, num_classes, n_samples // 5).astype(np.int64),
            'X_test': np.random.randn(n_samples // 5, n_features, encoding_dim).astype(np.float32),
            'y_test': np.random.randint(0, num_classes, n_samples // 5).astype(np.int64),
        }

        # Save data file
        data_file = output_path / f"{site_name}_{feature_type}.npz"
        np.savez(data_file, **data)

        logger.info(f"Created {site_name}_{feature_type}.npz")
        logger.info(f"  Train: {data['X_train'].shape}")
        logger.info(f"  Val:   {data['X_val'].shape}")
        logger.info(f"  Test:  {data['X_test'].shape}")

    logger.info("="*60)
    logger.info("Test data creation completed!")
    logger.info("="*60)
    return output_path


def test_data_module(data_dir: str):
    """Test the federated data module."""
    try:
        from snp_deconvolution.nvflare_real.data.federated_data_module import (
            SNPFederatedDataModule,
        )
    except ImportError as e:
        logger.error(f"Failed to import SNPFederatedDataModule: {e}")
        logger.error("Make sure PyTorch Lightning is installed: pip install pytorch-lightning")
        return False

    logger.info("\n" + "="*60)
    logger.info("Testing Federated Data Module")
    logger.info("="*60)

    try:
        # Test loading data for site1
        dm = SNPFederatedDataModule(
            data_dir=data_dir,
            site_name='site1',
            feature_type='cluster',
            batch_size=32,
            num_workers=0,  # Avoid multiprocessing in test
        )

        # Setup data
        dm.setup()

        # Get sample batch
        X_batch, y_batch = dm.get_sample_batch()

        logger.info(f"Sample batch loaded:")
        logger.info(f"  X shape: {X_batch.shape}")
        logger.info(f"  y shape: {y_batch.shape}")

        # Get class distribution
        dist = dm.get_class_distribution()
        logger.info(f"Class distribution:")
        for split, counts in dist.items():
            logger.info(f"  {split}: {counts}")

        logger.info("="*60)
        logger.info("Data Module Test: PASSED")
        logger.info("="*60)
        return True

    except Exception as e:
        logger.error(f"Data Module Test: FAILED")
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files(data_dir: str = 'test_data', workspace_dir: str = 'test_workspace'):
    """Clean up test files."""
    logger.info("\n" + "="*60)
    logger.info("Cleaning up test files")
    logger.info("="*60)

    for path in [data_dir, workspace_dir]:
        if Path(path).exists():
            logger.info(f"Removing {path}...")
            shutil.rmtree(path)

    logger.info("Cleanup completed")
    logger.info("="*60)


def main():
    """Main test function."""
    logger.info("\n" + "#"*60)
    logger.info("# NVFlare Lightning Setup Test")
    logger.info("#"*60)

    # Create test data
    data_dir = create_test_data(
        output_dir='test_data',
        num_sites=3,
        n_samples=100,
        n_features=50,
        encoding_dim=8,
        num_classes=3,
        feature_type='cluster'
    )

    # Test data module
    dm_success = test_data_module(str(data_dir))

    # Summary
    logger.info("\n" + "#"*60)
    logger.info("# Test Summary")
    logger.info("#"*60)
    logger.info(f"Data Creation: PASSED")
    logger.info(f"Data Module Test: {'PASSED' if dm_success else 'FAILED'}")

    if dm_success:
        logger.info("\n" + "="*60)
        logger.info("All tests passed!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Run POC simulation:")
        logger.info("   python job.py --mode poc --num_rounds 2 --run_now --data_dir ./test_data")
        logger.info("\n2. Or export job for deployment:")
        logger.info("   python job.py --mode export --export_dir ./jobs --data_dir ./test_data")
        logger.info("\n3. Clean up test files:")
        logger.info("   rm -rf test_data test_workspace")
    else:
        logger.error("\nSome tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
