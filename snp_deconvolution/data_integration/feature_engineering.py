"""
Feature Engineering Module

Utilities for creating and transforming features from haploblock data.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def encode_hashes_as_features(
    hash_file: Path,
    binary_encoding: bool = True
) -> np.ndarray:
    """
    Convert binary hashes to numerical features.

    Haploblock hashes are typically stored as hex strings representing
    binary fingerprints. This function converts them to numerical features
    suitable for machine learning.

    Args:
        hash_file: Path to TSV file with haploblock hashes
                  Format: sample_id\\thash1\\thash2\\t...
        binary_encoding: If True, converts each hash to binary vector.
                        If False, converts to integer values.

    Returns:
        Numpy array of shape (n_samples, n_features)
        - If binary_encoding=True: (n_samples, n_haploblocks * hash_bits)
        - If binary_encoding=False: (n_samples, n_haploblocks)

    Raises:
        FileNotFoundError: If hash_file does not exist
        ValueError: If file format is invalid

    Example:
        >>> features = encode_hashes_as_features("hashes.tsv", binary_encoding=True)
        >>> features.shape
        (100, 320)  # 100 samples, 10 haploblocks * 32 bits each
    """
    hash_file = Path(hash_file)
    if not hash_file.exists():
        raise FileNotFoundError(f"Hash file not found: {hash_file}")

    logger.info(f"Loading hashes from {hash_file}")

    try:
        # Read hash file
        hash_df = pd.read_csv(hash_file, sep="\t", index_col=0)

        if hash_df.empty:
            raise ValueError("Hash file is empty")

        n_samples, n_haploblocks = hash_df.shape
        logger.info(
            f"Loaded {n_samples} samples with {n_haploblocks} haploblock hashes"
        )

        if binary_encoding:
            # Convert each hash to binary representation
            binary_features = []

            for sample_idx in range(n_samples):
                sample_binary = []

                for col in hash_df.columns:
                    hash_val = hash_df.iloc[sample_idx][col]

                    # Convert hash to binary
                    if isinstance(hash_val, str):
                        # Assume hex string
                        try:
                            # Remove '0x' prefix if present
                            hash_val = hash_val.replace("0x", "").replace("0X", "")
                            # Convert hex to integer
                            int_val = int(hash_val, 16)
                        except ValueError:
                            logger.warning(
                                f"Invalid hash format at sample {sample_idx}, "
                                f"haploblock {col}: {hash_val}"
                            )
                            int_val = 0
                    else:
                        # Assume already numeric
                        int_val = int(hash_val) if not pd.isna(hash_val) else 0

                    # Convert to binary string (32-bit)
                    binary_str = format(int_val, "032b")
                    binary_array = np.array([int(b) for b in binary_str])
                    sample_binary.extend(binary_array)

                binary_features.append(sample_binary)

            features = np.array(binary_features, dtype=np.int8)
            logger.info(
                f"Encoded hashes to binary features: {features.shape}"
            )

        else:
            # Convert to integer values directly
            # Handle non-numeric values first
            if hash_df.dtypes.apply(lambda x: x == object).any():
                # Convert hex strings to integers
                features = np.zeros((n_samples, n_haploblocks), dtype=np.int64)
                for col_idx in range(n_haploblocks):
                    col = hash_df.columns[col_idx]
                    if hash_df[col].dtype == object:
                        features[:, col_idx] = hash_df[col].apply(
                            lambda x: int(str(x).replace("0x", "").replace("0X", ""), 16)
                            if isinstance(x, str) and x != ""
                            else 0
                        ).values
                    else:
                        features[:, col_idx] = hash_df[col].values
            else:
                features = hash_df.values.astype(np.int64)

            logger.info(
                f"Encoded hashes to integer features: {features.shape}"
            )

        return features

    except Exception as e:
        raise ValueError(f"Failed to encode hashes from {hash_file}: {e}")


def create_haploblock_features(
    boundaries_file: Path,
    variant_counts_file: Optional[Path] = None,
    include_length: bool = True,
    include_density: bool = True,
    include_statistics: bool = True
) -> pd.DataFrame:
    """
    Create haploblock-level features from boundaries and variant counts.

    Generates aggregate features for each haploblock including:
    - Length (END - START)
    - Variant density (variants per base pair)
    - Variant statistics (mean, std, min, max per sample)

    Args:
        boundaries_file: Path to haploblock_boundaries.tsv
                        Format: START\\tEND
        variant_counts_file: Optional path to variant_counts.tsv
                            Format: sample_id\\tcount1\\tcount2\\t...
        include_length: Include haploblock length features
        include_density: Include variant density features (requires variant_counts)
        include_statistics: Include variant count statistics (requires variant_counts)

    Returns:
        DataFrame with haploblock features.
        - If only boundaries_file: columns = ['haploblock_id', 'length']
        - If variant_counts_file provided: additional statistics columns

    Raises:
        FileNotFoundError: If required files do not exist
        ValueError: If file formats are invalid

    Example:
        >>> features = create_haploblock_features(
        ...     "boundaries.tsv",
        ...     "variant_counts.tsv"
        ... )
        >>> features.columns
        Index(['haploblock_id', 'length', 'mean_variants', 'std_variants', ...])
    """
    boundaries_file = Path(boundaries_file)
    if not boundaries_file.exists():
        raise FileNotFoundError(f"Boundaries file not found: {boundaries_file}")

    logger.info(f"Loading haploblock boundaries from {boundaries_file}")

    try:
        # Read boundaries file
        with open(boundaries_file, "r") as f:
            header = next(f)
            if not header.startswith("START\t"):
                raise ValueError(
                    "Boundaries file must have header: START\\tEND"
                )

            boundaries = []
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid boundary line: {line.strip()}"
                    )
                start, end = int(parts[0]), int(parts[1])
                boundaries.append((start, end))

        n_haploblocks = len(boundaries)
        logger.info(f"Loaded {n_haploblocks} haploblock boundaries")

        # Initialize feature DataFrame
        features_data = {
            "haploblock_id": list(range(n_haploblocks)),
            "start": [b[0] for b in boundaries],
            "end": [b[1] for b in boundaries],
        }

        # Add length features
        if include_length:
            features_data["length"] = [
                end - start for start, end in boundaries
            ]

        # Load variant counts if provided
        if variant_counts_file:
            variant_counts_file = Path(variant_counts_file)
            if not variant_counts_file.exists():
                raise FileNotFoundError(
                    f"Variant counts file not found: {variant_counts_file}"
                )

            logger.info(f"Loading variant counts from {variant_counts_file}")
            variant_df = pd.read_csv(variant_counts_file, sep="\t", index_col=0)

            if variant_df.shape[1] != n_haploblocks:
                raise ValueError(
                    f"Variant counts columns ({variant_df.shape[1]}) don't match "
                    f"number of haploblocks ({n_haploblocks})"
                )

            # Add density features
            if include_density:
                lengths = features_data["length"]
                mean_counts = variant_df.mean(axis=0).values

                features_data["variant_density"] = [
                    count / length if length > 0 else 0
                    for count, length in zip(mean_counts, lengths)
                ]

            # Add statistics features
            if include_statistics:
                features_data["mean_variants"] = variant_df.mean(axis=0).values
                features_data["std_variants"] = variant_df.std(axis=0).values
                features_data["min_variants"] = variant_df.min(axis=0).values
                features_data["max_variants"] = variant_df.max(axis=0).values
                features_data["median_variants"] = variant_df.median(axis=0).values

                # Add coefficient of variation (CV)
                means = variant_df.mean(axis=0).values
                stds = variant_df.std(axis=0).values
                features_data["cv_variants"] = np.where(
                    means > 0,
                    stds / means,
                    0
                )

        features_df = pd.DataFrame(features_data)

        logger.info(
            f"Created haploblock features: {len(features_df)} haploblocks, "
            f"{len(features_df.columns)} features"
        )

        return features_df

    except Exception as e:
        raise ValueError(f"Failed to create haploblock features: {e}")


def compute_pairwise_haploblock_distances(
    hash_file: Path,
    metric: str = "hamming"
) -> np.ndarray:
    """
    Compute pairwise distances between samples based on haploblock hashes.

    Args:
        hash_file: Path to haploblock_hashes.tsv
        metric: Distance metric ("hamming", "euclidean", "cosine")

    Returns:
        Pairwise distance matrix (n_samples x n_samples)

    Raises:
        FileNotFoundError: If hash_file does not exist
        ValueError: If metric is not supported
    """
    if metric not in ["hamming", "euclidean", "cosine"]:
        raise ValueError(
            f"Unsupported metric: {metric}. "
            f"Choose from: hamming, euclidean, cosine"
        )

    hash_file = Path(hash_file)
    if not hash_file.exists():
        raise FileNotFoundError(f"Hash file not found: {hash_file}")

    logger.info(f"Computing {metric} distances from {hash_file}")

    # Load hashes as binary features
    features = encode_hashes_as_features(hash_file, binary_encoding=True)
    n_samples = features.shape[0]

    logger.info(f"Computing pairwise distances for {n_samples} samples")

    if metric == "hamming":
        # Hamming distance: fraction of differing bits
        # Use broadcasting for efficiency
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            diff = features != features[i]
            distances[i] = diff.sum(axis=1) / features.shape[1]

    elif metric == "euclidean":
        # Euclidean distance
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(features, metric="euclidean"))

    elif metric == "cosine":
        # Cosine distance
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(features, metric="cosine"))

    logger.info(f"Computed {metric} distance matrix: {distances.shape}")

    return distances


def extract_haploblock_regions_from_vcf(
    vcf_path: Path,
    boundaries_file: Path,
    output_dir: Path,
    chromosome: str
) -> List[Path]:
    """
    Extract VCF regions corresponding to each haploblock.

    This creates separate VCF files for each haploblock region,
    useful for region-specific analysis.

    Args:
        vcf_path: Path to input VCF file
        boundaries_file: Path to haploblock_boundaries.tsv
        output_dir: Directory for output VCF files
        chromosome: Chromosome name (e.g., "chr6" or "6")

    Returns:
        List of paths to extracted VCF files (one per haploblock)

    Raises:
        FileNotFoundError: If input files don't exist
        RuntimeError: If bcftools commands fail
    """
    import subprocess

    vcf_path = Path(vcf_path)
    boundaries_file = Path(boundaries_file)
    output_dir = Path(output_dir)

    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")
    if not boundaries_file.exists():
        raise FileNotFoundError(f"Boundaries file not found: {boundaries_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize chromosome name
    if not chromosome.startswith("chr"):
        chromosome = f"chr{chromosome}"

    # Load boundaries
    logger.info(f"Loading boundaries from {boundaries_file}")
    boundaries = []
    with open(boundaries_file, "r") as f:
        next(f)  # Skip header
        for line in f:
            start, end = map(int, line.strip().split("\t"))
            boundaries.append((start, end))

    logger.info(f"Extracting {len(boundaries)} haploblock regions from VCF")

    output_files = []
    for idx, (start, end) in enumerate(boundaries):
        region = f"{chromosome}:{start}-{end}"
        output_vcf = output_dir / f"haploblock_{idx}_{chromosome}_{start}-{end}.vcf.gz"

        try:
            # Extract region
            subprocess.run(
                [
                    "bcftools", "view",
                    "-r", region,
                    "-o", str(output_vcf),
                    "-O", "z",
                    str(vcf_path)
                ],
                check=True,
                capture_output=True
            )

            # Index the output
            subprocess.run(
                ["bcftools", "index", str(output_vcf)],
                check=True,
                capture_output=True
            )

            output_files.append(output_vcf)

        except subprocess.CalledProcessError as e:
            logger.warning(
                f"Failed to extract region {region}: {e.stderr.decode()}"
            )

    logger.info(f"Extracted {len(output_files)} haploblock VCF files")

    return output_files
