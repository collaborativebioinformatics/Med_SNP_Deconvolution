#!/usr/bin/env python3
"""
Test script for Label Skew data splitting functionality
"""
import numpy as np
import logging
from pathlib import Path
import tempfile
import shutil
from collections import Counter

# Import the splitter
import sys
sys.path.insert(0, str(Path(__file__).parent))

from snp_deconvolution.nvflare_real.data.data_splitter import FederatedDataSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_label_skew_basic():
    """Test basic label skew splitting with 3 classes and 3 sites"""
    logger.info("\n" + "="*80)
    logger.info("Test 1: Basic Label Skew - 3 classes, 3 sites, 2 labels per site")
    logger.info("="*80)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_label_skew_")

    try:
        # Generate synthetic data with balanced classes
        np.random.seed(42)
        n_samples_per_class = 100
        n_classes = 3
        n_features = 50

        X_list = []
        y_list = []

        for class_id in range(n_classes):
            X_class = np.random.randn(n_samples_per_class, n_features) + class_id
            y_class = np.full(n_samples_per_class, class_id)
            X_list.append(X_class)
            y_list.append(y_class)

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        logger.info(f"Generated data: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"Class distribution: {dict(Counter(y))}")

        # Create splitter
        splitter = FederatedDataSplitter(
            output_dir=temp_dir,
            num_sites=3,
            seed=42
        )

        # Perform label skew split
        stats = splitter.split_and_save(
            X=X,
            y=y,
            val_ratio=0.15,
            feature_type='test',
            split_method='label_skew',
            labels_per_site=2
        )

        # Verify results
        logger.info("\n" + "-"*80)
        logger.info("Verification:")
        logger.info("-"*80)

        for site_name, site_stats in stats.items():
            train_labels = set(site_stats['train_label_dist'].keys())
            val_labels = set(site_stats['val_label_dist'].keys())
            all_labels = train_labels.union(val_labels)

            logger.info(f"{site_name}:")
            logger.info(f"  Unique labels: {sorted(all_labels)}")
            logger.info(f"  Train samples: {site_stats['train_samples']}")
            logger.info(f"  Val samples: {site_stats['val_samples']}")
            logger.info(f"  Train label dist: {site_stats['train_label_dist']}")
            logger.info(f"  Val label dist: {site_stats['val_label_dist']}")

            # Each site should have exactly 2 classes
            assert len(all_labels) == 2, f"{site_name} should have 2 classes, got {len(all_labels)}"

        # Verify all classes are represented
        all_site_labels = set()
        for site_stats in stats.values():
            train_labels = set(site_stats['train_label_dist'].keys())
            val_labels = set(site_stats['val_label_dist'].keys())
            all_site_labels.update(train_labels.union(val_labels))

        assert len(all_site_labels) == n_classes, \
            f"All {n_classes} classes should be represented, got {len(all_site_labels)}"

        logger.info("\nTest 1 PASSED!")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


def test_label_skew_more_sites_than_classes():
    """Test label skew when there are more sites than classes"""
    logger.info("\n" + "="*80)
    logger.info("Test 2: More sites than classes - 3 classes, 5 sites, 2 labels per site")
    logger.info("="*80)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_label_skew_")

    try:
        # Generate synthetic data
        np.random.seed(42)
        n_samples_per_class = 80
        n_classes = 3
        n_features = 30

        X_list = []
        y_list = []

        for class_id in range(n_classes):
            X_class = np.random.randn(n_samples_per_class, n_features) + class_id
            y_class = np.full(n_samples_per_class, class_id)
            X_list.append(X_class)
            y_list.append(y_class)

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        logger.info(f"Generated data: X.shape={X.shape}, y.shape={y.shape}")

        # Create splitter with more sites than classes
        splitter = FederatedDataSplitter(
            output_dir=temp_dir,
            num_sites=5,
            seed=42
        )

        # Perform label skew split
        stats = splitter.split_and_save(
            X=X,
            y=y,
            val_ratio=0.15,
            feature_type='test',
            split_method='label_skew',
            labels_per_site=2
        )

        # Verify results
        logger.info("\n" + "-"*80)
        logger.info("Verification:")
        logger.info("-"*80)

        for site_name, site_stats in stats.items():
            train_labels = set(site_stats['train_label_dist'].keys())
            val_labels = set(site_stats['val_label_dist'].keys())
            all_labels = train_labels.union(val_labels)

            logger.info(f"{site_name}: labels={sorted(all_labels)}, "
                       f"train={site_stats['train_samples']}, val={site_stats['val_samples']}")

        logger.info("\nTest 2 PASSED!")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


def test_label_skew_edge_case_all_labels():
    """Test when labels_per_site >= num_classes (all sites get all labels)"""
    logger.info("\n" + "="*80)
    logger.info("Test 3: Edge case - labels_per_site >= num_classes")
    logger.info("="*80)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_label_skew_")

    try:
        # Generate synthetic data
        np.random.seed(42)
        n_samples_per_class = 60
        n_classes = 3
        n_features = 20

        X_list = []
        y_list = []

        for class_id in range(n_classes):
            X_class = np.random.randn(n_samples_per_class, n_features) + class_id
            y_class = np.full(n_samples_per_class, class_id)
            X_list.append(X_class)
            y_list.append(y_class)

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        logger.info(f"Generated data: X.shape={X.shape}, y.shape={y.shape}")

        # Create splitter
        splitter = FederatedDataSplitter(
            output_dir=temp_dir,
            num_sites=3,
            seed=42
        )

        # Perform label skew split with labels_per_site=4 (> num_classes=3)
        stats = splitter.split_and_save(
            X=X,
            y=y,
            val_ratio=0.15,
            feature_type='test',
            split_method='label_skew',
            labels_per_site=4  # More than num_classes
        )

        # Verify all sites get all labels
        logger.info("\n" + "-"*80)
        logger.info("Verification:")
        logger.info("-"*80)

        for site_name, site_stats in stats.items():
            train_labels = set(site_stats['train_label_dist'].keys())
            val_labels = set(site_stats['val_label_dist'].keys())
            all_labels = train_labels.union(val_labels)

            logger.info(f"{site_name}: labels={sorted(all_labels)}")

            # Each site should have all 3 classes
            assert len(all_labels) == n_classes, \
                f"{site_name} should have all {n_classes} classes, got {len(all_labels)}"

        logger.info("\nTest 3 PASSED!")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


def test_label_skew_single_label_per_site():
    """Test extreme skew: each site gets only 1 label"""
    logger.info("\n" + "="*80)
    logger.info("Test 4: Extreme skew - 1 label per site")
    logger.info("="*80)

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="test_label_skew_")

    try:
        # Generate synthetic data
        np.random.seed(42)
        n_samples_per_class = 50
        n_classes = 4
        n_features = 20

        X_list = []
        y_list = []

        for class_id in range(n_classes):
            X_class = np.random.randn(n_samples_per_class, n_features) + class_id
            y_class = np.full(n_samples_per_class, class_id)
            X_list.append(X_class)
            y_list.append(y_class)

        X = np.vstack(X_list)
        y = np.hstack(y_list)

        logger.info(f"Generated data: X.shape={X.shape}, y.shape={y.shape}")

        # Create splitter with 4 sites
        splitter = FederatedDataSplitter(
            output_dir=temp_dir,
            num_sites=4,
            seed=42
        )

        # Perform label skew split with labels_per_site=1
        stats = splitter.split_and_save(
            X=X,
            y=y,
            val_ratio=0.15,
            feature_type='test',
            split_method='label_skew',
            labels_per_site=1  # Each site gets only 1 label
        )

        # Verify results
        logger.info("\n" + "-"*80)
        logger.info("Verification:")
        logger.info("-"*80)

        for site_name, site_stats in stats.items():
            train_labels = set(site_stats['train_label_dist'].keys())
            val_labels = set(site_stats['val_label_dist'].keys())
            all_labels = train_labels.union(val_labels)

            logger.info(f"{site_name}: labels={sorted(all_labels)}, "
                       f"train={site_stats['train_samples']}, val={site_stats['val_samples']}")

            # Each site should have exactly 1 class
            assert len(all_labels) == 1, \
                f"{site_name} should have 1 class, got {len(all_labels)}"

        logger.info("\nTest 4 PASSED!")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    logger.info("\n" + "#"*80)
    logger.info("# Testing Label Skew Data Splitting")
    logger.info("#"*80 + "\n")

    # Run all tests
    test_label_skew_basic()
    test_label_skew_more_sites_than_classes()
    test_label_skew_edge_case_all_labels()
    test_label_skew_single_label_per_site()

    logger.info("\n" + "#"*80)
    logger.info("# ALL TESTS PASSED!")
    logger.info("#"*80 + "\n")
