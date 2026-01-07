#!/usr/bin/env python3
"""
Example Usage of Data Integration Module

This script demonstrates how to use the data_integration module
for loading and preparing data for machine learning.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from snp_deconvolution.data_integration import (
    HaploblockMLDataLoader,
    SparseGenotypeMatrix,
    encode_hashes_as_features,
    create_haploblock_features,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_1_load_haploblock_features():
    """Example 1: Load haploblock features from pipeline outputs."""
    logger.info("=" * 60)
    logger.info("Example 1: Load Haploblock Features")
    logger.info("=" * 60)

    # This is a placeholder example - adjust paths to your data
    pipeline_dir = Path("/path/to/pipeline/output")

    try:
        loader = HaploblockMLDataLoader(pipeline_dir)

        # Load features (will look for variant_counts.tsv and haploblock_hashes.tsv)
        features_df = loader.load_haploblock_features()

        logger.info(f"Features shape: {features_df.shape}")
        logger.info(f"Sample IDs: {features_df.index[:5].tolist()}")
        logger.info(f"Feature columns: {features_df.columns[:5].tolist()}")

    except Exception as e:
        logger.error(f"Error in Example 1: {e}")


def example_2_load_population_labels():
    """Example 2: Load population labels from 1000 Genomes files."""
    logger.info("=" * 60)
    logger.info("Example 2: Load Population Labels")
    logger.info("=" * 60)

    # Use actual data files from the repository
    data_dir = Path(__file__).parent.parent.parent / "data"

    population_files = [
        data_dir / "igsr-chb.tsv.tsv",  # Chinese Han Beijing
        data_dir / "igsr-gbr.tsv.tsv",  # British
        data_dir / "igsr-pur.tsv.tsv",  # Puerto Rican
    ]

    # Check which files exist
    existing_files = [f for f in population_files if f.exists()]

    if not existing_files:
        logger.warning("No population files found. Skipping example.")
        return

    try:
        # Create a dummy loader (directory doesn't need to exist for this method)
        loader = HaploblockMLDataLoader(Path("."))

        labels = loader.load_population_labels(existing_files)

        logger.info(f"Total labeled samples: {len(labels)}")
        logger.info(f"Populations: {set(labels.values())}")

        # Count samples per population
        from collections import Counter
        pop_counts = Counter(labels.values())
        for pop_id, count in sorted(pop_counts.items()):
            pop_name = existing_files[pop_id].stem.split('-')[1].upper()
            logger.info(f"  Population {pop_id} ({pop_name}): {count} samples")

    except Exception as e:
        logger.error(f"Error in Example 2: {e}")


def example_3_create_ml_dataset():
    """Example 3: Create complete ML dataset."""
    logger.info("=" * 60)
    logger.info("Example 3: Create Complete ML Dataset")
    logger.info("=" * 60)

    # This demonstrates the full workflow
    data_dir = Path(__file__).parent.parent.parent / "data"
    pipeline_dir = Path("/path/to/pipeline/output")

    population_files = [
        data_dir / "igsr-chb.tsv.tsv",
        data_dir / "igsr-gbr.tsv.tsv",
        data_dir / "igsr-pur.tsv.tsv",
    ]

    existing_files = [f for f in population_files if f.exists()]

    if not existing_files:
        logger.warning("No population files found. Skipping example.")
        return

    try:
        loader = HaploblockMLDataLoader(pipeline_dir)

        dataset = loader.create_ml_dataset(
            population_files=existing_files
        )

        logger.info("Dataset contents:")
        for key, value in dataset.items():
            if isinstance(value, (pd.DataFrame, np.ndarray)):
                logger.info(f"  {key}: shape {value.shape}")
            elif isinstance(value, dict):
                logger.info(f"  {key}: {len(value)} entries")
            else:
                logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Error in Example 3: {e}")


def example_4_sparse_matrix_from_numpy():
    """Example 4: Convert numpy array to sparse matrix."""
    logger.info("=" * 60)
    logger.info("Example 4: Sparse Matrix from Numpy")
    logger.info("=" * 60)

    # Generate synthetic genotype data
    n_samples = 100
    n_variants = 10000
    genotypes = np.random.choice([0, 1, 2], size=(n_samples, n_variants))

    # Add some missing values
    missing_mask = np.random.random((n_samples, n_variants)) < 0.05
    genotypes = genotypes.astype(float)
    genotypes[missing_mask] = -1

    logger.info(f"Dense matrix shape: {genotypes.shape}")
    logger.info(f"Dense matrix size: {genotypes.nbytes / 1e6:.2f} MB")

    # Convert to sparse
    sparse_matrix = SparseGenotypeMatrix.from_numpy(genotypes)

    # Estimate sparse size (approximate)
    sparse_size = (sparse_matrix.data.nbytes +
                   sparse_matrix.indices.nbytes +
                   sparse_matrix.indptr.nbytes) / 1e6

    logger.info(f"Sparse matrix shape: {sparse_matrix.shape}")
    logger.info(f"Sparse matrix size: {sparse_size:.2f} MB")
    logger.info(f"Non-zero elements: {sparse_matrix.nnz}")
    logger.info(
        f"Sparsity: {100 * (1 - sparse_matrix.nnz / sparse_matrix.size):.2f}%"
    )


def example_5_feature_engineering():
    """Example 5: Feature engineering from haploblock data."""
    logger.info("=" * 60)
    logger.info("Example 5: Feature Engineering")
    logger.info("=" * 60)

    data_dir = Path(__file__).parent.parent.parent / "data"

    # Look for haploblock boundaries file
    boundaries_files = list(data_dir.glob("haploblock_boundaries*.tsv"))

    if not boundaries_files:
        logger.warning("No haploblock boundaries files found. Skipping example.")
        return

    boundaries_file = boundaries_files[0]
    logger.info(f"Using boundaries file: {boundaries_file.name}")

    try:
        # Create haploblock features
        features_df = create_haploblock_features(
            boundaries_file=boundaries_file,
            include_length=True,
        )

        logger.info(f"Haploblock features shape: {features_df.shape}")
        logger.info(f"Columns: {features_df.columns.tolist()}")
        logger.info("\nFirst 5 haploblocks:")
        logger.info(features_df.head().to_string())

        # Summary statistics
        logger.info("\nHaploblock length statistics:")
        logger.info(f"  Mean: {features_df['length'].mean():.0f} bp")
        logger.info(f"  Median: {features_df['length'].median():.0f} bp")
        logger.info(f"  Min: {features_df['length'].min():.0f} bp")
        logger.info(f"  Max: {features_df['length'].max():.0f} bp")

    except Exception as e:
        logger.error(f"Error in Example 5: {e}")


def example_6_synthetic_hash_encoding():
    """Example 6: Encode synthetic haploblock hashes."""
    logger.info("=" * 60)
    logger.info("Example 6: Hash Encoding (Synthetic Data)")
    logger.info("=" * 60)

    # Create synthetic hash data
    n_samples = 50
    n_haploblocks = 10

    # Generate random hex hashes
    hash_data = {}
    samples = [f"NA{18000 + i}" for i in range(n_samples)]

    for sample in samples:
        hashes = [hex(np.random.randint(0, 2**32)) for _ in range(n_haploblocks)]
        hash_data[sample] = hashes

    # Create DataFrame
    hash_df = pd.DataFrame(hash_data).T
    hash_df.columns = [f"haploblock_{i}" for i in range(n_haploblocks)]

    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        temp_file = Path(f.name)
        hash_df.to_csv(f, sep='\t')

    try:
        # Encode as integer features
        int_features = encode_hashes_as_features(temp_file, binary_encoding=False)
        logger.info(f"Integer features shape: {int_features.shape}")
        logger.info(f"Integer features dtype: {int_features.dtype}")

        # Encode as binary features
        binary_features = encode_hashes_as_features(temp_file, binary_encoding=True)
        logger.info(f"Binary features shape: {binary_features.shape}")
        logger.info(f"Binary features dtype: {binary_features.dtype}")

        # Show size comparison
        int_size = int_features.nbytes / 1024
        binary_size = binary_features.nbytes / 1024
        logger.info(f"\nMemory usage:")
        logger.info(f"  Integer encoding: {int_size:.2f} KB")
        logger.info(f"  Binary encoding: {binary_size:.2f} KB")

    except Exception as e:
        logger.error(f"Error in Example 6: {e}")
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def main():
    """Run all examples."""
    logger.info("Data Integration Module - Example Usage")
    logger.info("")

    examples = [
        # example_1_load_haploblock_features,  # Requires pipeline output
        example_2_load_population_labels,
        # example_3_create_ml_dataset,  # Requires pipeline output
        example_4_sparse_matrix_from_numpy,
        example_5_feature_engineering,
        example_6_synthetic_hash_encoding,
    ]

    for example_func in examples:
        try:
            example_func()
            logger.info("")
        except Exception as e:
            logger.error(f"Failed to run {example_func.__name__}: {e}")
            logger.info("")

    logger.info("=" * 60)
    logger.info("Examples completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
