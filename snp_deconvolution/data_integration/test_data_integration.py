#!/usr/bin/env python3
"""
Unit tests for data_integration module

Run with: pytest test_data_integration.py -v
Or: python test_data_integration.py
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from haploblock_loader import HaploblockMLDataLoader
from sparse_genotype_matrix import SparseGenotypeMatrix
from feature_engineering import (
    encode_hashes_as_features,
    create_haploblock_features,
)


class TestHaploblockMLDataLoader(unittest.TestCase):
    """Test HaploblockMLDataLoader class."""

    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def test_init_valid_directory(self):
        """Test initialization with valid directory."""
        loader = HaploblockMLDataLoader(self.temp_path)
        self.assertEqual(loader.pipeline_output_dir, self.temp_path)

    def test_init_invalid_directory(self):
        """Test initialization with invalid directory."""
        with self.assertRaises(ValueError):
            HaploblockMLDataLoader("/nonexistent/path")

    def test_load_population_labels(self):
        """Test loading population labels."""
        # Create test population files
        pop_files = []
        for pop_idx, pop_name in enumerate(["CHB", "GBR", "PUR"]):
            pop_file = self.temp_path / f"igsr-{pop_name.lower()}.tsv"
            with open(pop_file, "w") as f:
                f.write("Sample name\tPopulation code\n")
                for i in range(10):
                    f.write(f"NA{18000 + pop_idx * 10 + i}\t{pop_name}\n")
            pop_files.append(pop_file)

        loader = HaploblockMLDataLoader(self.temp_path)
        labels = loader.load_population_labels(pop_files)

        self.assertEqual(len(labels), 30)
        self.assertEqual(set(labels.values()), {0, 1, 2})
        self.assertEqual(labels["NA18000"], 0)
        self.assertEqual(labels["NA18010"], 1)
        self.assertEqual(labels["NA18020"], 2)


class TestSparseGenotypeMatrix(unittest.TestCase):
    """Test SparseGenotypeMatrix class."""

    def test_from_numpy_dense(self):
        """Test conversion from dense numpy array."""
        # Create test genotype matrix
        genotypes = np.random.randint(0, 3, size=(50, 100))

        sparse_matrix = SparseGenotypeMatrix.from_numpy(genotypes)

        self.assertEqual(sparse_matrix.shape, (50, 100))
        self.assertGreater(sparse_matrix.nnz, 0)

    def test_from_numpy_with_missing(self):
        """Test conversion with missing values."""
        genotypes = np.random.randint(0, 3, size=(50, 100)).astype(float)

        # Add missing values
        missing_mask = np.random.random((50, 100)) < 0.1
        genotypes[missing_mask] = -1

        sparse_matrix = SparseGenotypeMatrix.from_numpy(genotypes)

        self.assertEqual(sparse_matrix.shape, (50, 100))
        # Missing values should not be stored
        self.assertLess(sparse_matrix.nnz, 50 * 100)

    def test_save_load_npz(self):
        """Test saving and loading sparse matrix."""
        genotypes = np.random.randint(0, 3, size=(50, 100))
        sparse_matrix = SparseGenotypeMatrix.from_numpy(genotypes)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            temp_file = Path(f.name)

        try:
            SparseGenotypeMatrix.save_npz(sparse_matrix, temp_file)
            loaded_matrix = SparseGenotypeMatrix.load_npz(temp_file)

            self.assertEqual(sparse_matrix.shape, loaded_matrix.shape)
            self.assertEqual(sparse_matrix.nnz, loaded_matrix.nnz)
            np.testing.assert_array_equal(
                sparse_matrix.toarray(),
                loaded_matrix.toarray()
            )
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions."""

    def test_encode_hashes_as_features_integer(self):
        """Test integer encoding of hashes."""
        # Create test hash file
        hash_data = {
            "NA18000": ["0x12345678", "0xABCDEF12"],
            "NA18001": ["0x87654321", "0x21FEDCBA"],
        }
        hash_df = pd.DataFrame(hash_data).T
        hash_df.columns = ["haploblock_0", "haploblock_1"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            temp_file = Path(f.name)
            hash_df.to_csv(f, sep="\t")

        try:
            features = encode_hashes_as_features(temp_file, binary_encoding=False)

            self.assertEqual(features.shape, (2, 2))
            self.assertEqual(features.dtype, np.int64)
            # Check first hash value
            self.assertEqual(features[0, 0], int("12345678", 16))
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_encode_hashes_as_features_binary(self):
        """Test binary encoding of hashes."""
        hash_data = {
            "NA18000": ["0x12345678", "0xABCDEF12"],
            "NA18001": ["0x87654321", "0x21FEDCBA"],
        }
        hash_df = pd.DataFrame(hash_data).T
        hash_df.columns = ["haploblock_0", "haploblock_1"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            temp_file = Path(f.name)
            hash_df.to_csv(f, sep="\t")

        try:
            features = encode_hashes_as_features(temp_file, binary_encoding=True)

            # 2 samples x 2 haploblocks x 32 bits = (2, 64)
            self.assertEqual(features.shape, (2, 64))
            self.assertEqual(features.dtype, np.int8)
            # Values should be 0 or 1
            self.assertTrue(np.all((features == 0) | (features == 1)))
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_create_haploblock_features_basic(self):
        """Test basic haploblock feature creation."""
        # Create test boundaries file
        boundaries_data = "START\tEND\n1\t1000\n1000\t2000\n2000\t5000\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            temp_file = Path(f.name)
            f.write(boundaries_data)

        try:
            features_df = create_haploblock_features(
                boundaries_file=temp_file,
                include_length=True
            )

            self.assertEqual(len(features_df), 3)
            self.assertIn("length", features_df.columns)
            self.assertEqual(features_df["length"].iloc[0], 999)
            self.assertEqual(features_df["length"].iloc[1], 1000)
            self.assertEqual(features_df["length"].iloc[2], 3000)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_create_haploblock_features_with_counts(self):
        """Test haploblock features with variant counts."""
        # Create boundaries file
        boundaries_data = "START\tEND\n1\t1000\n1000\t2000\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            boundaries_file = Path(f.name)
            f.write(boundaries_data)

        # Create variant counts file
        counts_data = "sample\thaploblock_0\thaploblock_1\nNA18000\t10\t20\nNA18001\t15\t25\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
            counts_file = Path(f.name)
            f.write(counts_data.replace("\\t", "\t"))

        try:
            # Need to create proper TSV
            with open(counts_file, "w") as f:
                f.write("sample_id\t0\t1\n")
                f.write("NA18000\t10\t20\n")
                f.write("NA18001\t15\t25\n")

            features_df = create_haploblock_features(
                boundaries_file=boundaries_file,
                variant_counts_file=counts_file,
                include_statistics=True
            )

            self.assertEqual(len(features_df), 2)
            self.assertIn("mean_variants", features_df.columns)
            self.assertAlmostEqual(features_df["mean_variants"].iloc[0], 12.5)
            self.assertAlmostEqual(features_df["mean_variants"].iloc[1], 22.5)

        finally:
            if boundaries_file.exists():
                boundaries_file.unlink()
            if counts_file.exists():
                counts_file.unlink()


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHaploblockMLDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestSparseGenotypeMatrix))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
