#!/usr/bin/env python
"""
Test script for quantity skew data splitting
"""
import numpy as np
import logging
from snp_deconvolution.nvflare_real.data.data_splitter import FederatedDataSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_quantity_skew():
    """Test quantity skew splitting"""
    print("\n" + "="*80)
    print("Testing Quantity Skew Data Splitting")
    print("="*80 + "\n")

    # Create synthetic data
    n_samples = 1000
    n_features = 50
    n_classes = 3

    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    print(f"Generated synthetic data:")
    print(f"  - Total samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - Classes: {n_classes}")
    print(f"  - Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print()

    # Test 1: Balanced split (for comparison)
    print("\n" + "-"*80)
    print("Test 1: Balanced Split (Control)")
    print("-"*80)
    splitter_balanced = FederatedDataSplitter(
        output_dir='/tmp/federated_test_balanced',
        num_sites=3,
        seed=42
    )

    stats_balanced = splitter_balanced.split_and_save(
        X=X,
        y=y,
        val_ratio=0.15,
        feature_type='test',
        split_method='balanced'
    )

    # Test 2: Quantity skew with default min_ratio=0.1
    print("\n" + "-"*80)
    print("Test 2: Quantity Skew Split (min_ratio=0.1)")
    print("-"*80)
    splitter_skew1 = FederatedDataSplitter(
        output_dir='/tmp/federated_test_quantity_skew1',
        num_sites=3,
        seed=42
    )

    stats_skew1 = splitter_skew1.split_and_save(
        X=X,
        y=y,
        val_ratio=0.15,
        feature_type='test',
        split_method='quantity_skew',
        min_ratio=0.1
    )

    # Test 3: Quantity skew with min_ratio=0.05 (more skewed)
    print("\n" + "-"*80)
    print("Test 3: Quantity Skew Split (min_ratio=0.05, more skewed)")
    print("-"*80)
    splitter_skew2 = FederatedDataSplitter(
        output_dir='/tmp/federated_test_quantity_skew2',
        num_sites=3,
        seed=43  # Different seed for different distribution
    )

    stats_skew2 = splitter_skew2.split_and_save(
        X=X,
        y=y,
        val_ratio=0.15,
        feature_type='test',
        split_method='quantity_skew',
        min_ratio=0.05
    )

    # Test 4: Quantity skew with 5 sites
    print("\n" + "-"*80)
    print("Test 4: Quantity Skew Split (5 sites, min_ratio=0.08)")
    print("-"*80)
    splitter_skew3 = FederatedDataSplitter(
        output_dir='/tmp/federated_test_quantity_skew3',
        num_sites=5,
        seed=44
    )

    stats_skew3 = splitter_skew3.split_and_save(
        X=X,
        y=y,
        val_ratio=0.15,
        feature_type='test',
        split_method='quantity_skew',
        min_ratio=0.08
    )

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    print("\nBalanced Split:")
    for site_name, site_stat in stats_balanced.items():
        print(f"  {site_name}: {site_stat['total_samples']} samples "
              f"({site_stat['total_samples']/n_samples*100:.1f}%)")

    print("\nQuantity Skew Split (min_ratio=0.1):")
    for site_name, site_stat in stats_skew1.items():
        print(f"  {site_name}: {site_stat['total_samples']} samples "
              f"({site_stat['total_samples']/n_samples*100:.1f}%)")

    print("\nQuantity Skew Split (min_ratio=0.05):")
    for site_name, site_stat in stats_skew2.items():
        print(f"  {site_name}: {site_stat['total_samples']} samples "
              f"({site_stat['total_samples']/n_samples*100:.1f}%)")

    print("\nQuantity Skew Split (5 sites, min_ratio=0.08):")
    for site_name, site_stat in stats_skew3.items():
        print(f"  {site_name}: {site_stat['total_samples']} samples "
              f"({site_stat['total_samples']/n_samples*100:.1f}%)")

    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_quantity_skew()
