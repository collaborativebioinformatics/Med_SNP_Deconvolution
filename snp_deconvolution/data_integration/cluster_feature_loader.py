"""
Cluster Feature Loader for Deep Learning

Extracts cluster IDs from haploblock pipeline output for embedding-based models.

Pipeline structure:
    - clusters/chr{chr}_{start}-{end}_cluster.tsv: representative → individual mapping
    - individual_hashes_{start}-{end}.tsv: individual → hash (contains cluster info)

Key insight: Cluster membership is the categorical feature for DL embedding.

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class ClusterFeatureLoader:
    """
    Load cluster assignments as categorical features for embedding-based models.

    Each sample has a cluster ID per haploblock. These IDs are used as
    input to embedding layers that learn representations.

    Supports two modes:
    1. Discovery mode (default): Auto-detect cluster files, only load haploblocks with data
    2. Boundaries mode: Load specific haploblocks from boundaries file

    Attributes:
        n_haploblocks: Number of haploblocks
        n_clusters_per_haploblock: List of cluster counts per haploblock
        sample_ids: List of sample IDs
    """

    def __init__(self, pipeline_output_dir: Path, auto_discover: bool = True):
        """
        Args:
            pipeline_output_dir: Directory containing pipeline outputs
                                (clusters/, individual_hashes_*.tsv)
            auto_discover: If True, automatically discover cluster files instead of
                          requiring a boundaries file (default: True)
        """
        self.pipeline_output_dir = Path(pipeline_output_dir)
        self.clusters_dir = self.pipeline_output_dir / "clusters"
        self.auto_discover = auto_discover

        self.n_haploblocks = 0
        self.n_clusters_per_haploblock: List[int] = []
        self.haploblock_boundaries: List[Tuple[int, int]] = []
        self.sample_ids: List[str] = []

        logger.info(f"ClusterFeatureLoader initialized: {pipeline_output_dir} (auto_discover={auto_discover})")

    def _find_boundaries_file(self) -> Optional[Path]:
        """Auto-find boundaries file with various naming patterns."""
        patterns = [
            "haploblock_boundaries.tsv",
            "haploblock_boundaries_*.tsv",
            "*_boundaries.tsv",
        ]
        for pattern in patterns:
            matches = list(self.pipeline_output_dir.glob(pattern))
            if matches:
                # Prefer shorter names (more specific)
                return sorted(matches, key=lambda p: len(p.name))[0]
        return None

    def load_haploblock_boundaries(self, boundaries_file: Optional[Path] = None) -> List[Tuple[int, int]]:
        """Load haploblock boundaries from file."""
        if boundaries_file is None:
            boundaries_file = self._find_boundaries_file()
            if boundaries_file is None:
                raise FileNotFoundError(
                    f"No boundaries file found in {self.pipeline_output_dir}. "
                    "Expected haploblock_boundaries*.tsv"
                )

        boundaries = []
        with open(boundaries_file, "r") as f:
            header = next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    start, end = int(parts[0]), int(parts[1])
                    boundaries.append((start, end))

        self.haploblock_boundaries = boundaries
        self.n_haploblocks = len(boundaries)
        logger.info(f"Loaded {self.n_haploblocks} haploblock boundaries from {boundaries_file.name}")
        return boundaries

    def discover_cluster_files(self) -> List[Tuple[Path, int, int]]:
        """
        Discover all cluster files in clusters/ directory.

        Returns:
            List of (file_path, start, end) tuples
        """
        if not self.clusters_dir.exists():
            logger.warning(f"Clusters directory not found: {self.clusters_dir}")
            return []

        cluster_files = []
        # Pattern: chr6_31480875-31603888_cluster.tsv or similar
        pattern = re.compile(r"(?:chr\d+_)?(\d+)-(\d+)_cluster\.tsv$")

        for f in self.clusters_dir.glob("*_cluster.tsv"):
            match = pattern.search(f.name)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                cluster_files.append((f, start, end))

        # Sort by start position
        cluster_files.sort(key=lambda x: x[1])
        logger.info(f"Discovered {len(cluster_files)} cluster files")
        return cluster_files

    def _parse_cluster_file(self, cluster_file: Path) -> Tuple[Dict[str, int], int]:
        """
        Parse a single cluster file.

        Returns:
            sample_to_cluster: {sample_id: cluster_id}
            n_clusters: Number of clusters
        """
        sample_to_cluster: Dict[str, int] = {}
        rep_to_cluster: Dict[str, int] = {}
        next_cluster = 0

        with open(cluster_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue

                rep, indiv = parts

                # Assign cluster ID to representative
                if rep not in rep_to_cluster:
                    rep_to_cluster[rep] = next_cluster
                    next_cluster += 1

                # Map individual to cluster
                # Extract sample name (e.g., "HG00096" from "HG00096_chr6_...")
                sample_name = indiv.split("_")[0]
                sample_to_cluster[sample_name] = rep_to_cluster[rep]

        return sample_to_cluster, next_cluster

    def load_cluster_assignments(self) -> Tuple[Dict[str, Dict[str, int]], List[int]]:
        """
        Load cluster assignments for all haploblocks.

        In auto_discover mode: only loads haploblocks that have cluster files.
        In boundaries mode: loads all haploblocks from boundaries file.

        Returns:
            cluster_data: {haploblock_key: {sample_id: cluster_id}}
            n_clusters_per_haploblock: List of cluster counts
        """
        cluster_data: Dict[str, Dict[str, int]] = {}
        n_clusters_per_haploblock = []
        all_samples = set()

        if self.auto_discover:
            # Discovery mode: find cluster files directly
            cluster_files = self.discover_cluster_files()

            if not cluster_files:
                raise FileNotFoundError(
                    f"No cluster files found in {self.clusters_dir}. "
                    "Run the clustering pipeline (step 4) first."
                )

            self.haploblock_boundaries = [(start, end) for _, start, end in cluster_files]

            for cluster_file, start, end in cluster_files:
                sample_to_cluster, n_clusters = self._parse_cluster_file(cluster_file)
                cluster_data[f"{start}-{end}"] = sample_to_cluster
                n_clusters_per_haploblock.append(n_clusters + 1)  # +1 for padding
                all_samples.update(sample_to_cluster.keys())

        else:
            # Boundaries mode: load from boundaries file
            if not self.haploblock_boundaries:
                self.load_haploblock_boundaries()

            n_found = 0
            n_missing = 0

            for start, end in self.haploblock_boundaries:
                # Look for cluster file
                cluster_file = None
                for pattern in [
                    f"chr*_{start}-{end}_cluster.tsv",
                    f"*_{start}-{end}_cluster.tsv",
                    f"{start}-{end}_cluster.tsv"
                ]:
                    matches = list(self.clusters_dir.glob(pattern))
                    if matches:
                        cluster_file = matches[0]
                        break

                if cluster_file is None:
                    n_missing += 1
                    cluster_data[f"{start}-{end}"] = {}
                    n_clusters_per_haploblock.append(1)  # Default 1 cluster
                    continue

                n_found += 1
                sample_to_cluster, n_clusters = self._parse_cluster_file(cluster_file)
                cluster_data[f"{start}-{end}"] = sample_to_cluster
                n_clusters_per_haploblock.append(n_clusters + 1)  # +1 for padding
                all_samples.update(sample_to_cluster.keys())

            if n_missing > 0:
                logger.warning(
                    f"Cluster files: {n_found} found, {n_missing} missing "
                    f"(use auto_discover=True to skip missing)"
                )

        self.n_haploblocks = len(self.haploblock_boundaries)
        self.n_clusters_per_haploblock = n_clusters_per_haploblock
        self.sample_ids = sorted(all_samples)

        logger.info(
            f"Loaded {len(cluster_data)} haploblocks, "
            f"{len(self.sample_ids)} samples, "
            f"clusters per haploblock: {n_clusters_per_haploblock}"
        )

        return cluster_data, n_clusters_per_haploblock

    def create_cluster_matrix(
        self,
        cluster_data: Dict[str, Dict[str, int]],
        sample_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Create cluster ID matrix for embedding input.

        Args:
            cluster_data: {haploblock_key: {sample_id: cluster_id}}
            sample_ids: List of sample IDs (uses self.sample_ids if None)

        Returns:
            (n_samples, n_haploblocks) array of cluster IDs
        """
        if sample_ids is None:
            sample_ids = self.sample_ids

        n_samples = len(sample_ids)
        n_haploblocks = len(self.haploblock_boundaries)

        # Initialize with 0 (padding/unknown)
        cluster_matrix = np.zeros((n_samples, n_haploblocks), dtype=np.int64)

        for hb_idx, (start, end) in enumerate(self.haploblock_boundaries):
            hb_key = f"{start}-{end}"
            hb_clusters = cluster_data.get(hb_key, {})

            for sample_idx, sample_id in enumerate(sample_ids):
                # Cluster IDs start from 1 (0 is padding)
                cluster_id = hb_clusters.get(sample_id, 0)
                cluster_matrix[sample_idx, hb_idx] = cluster_id + 1  # Shift by 1

        logger.info(f"Created cluster matrix: {cluster_matrix.shape}")
        return cluster_matrix

    def load_population_labels(
        self,
        population_files: List[Path]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Load population labels for samples.

        Args:
            population_files: List of population TSV files (igsr-*.tsv)

        Returns:
            labels: (n_samples,) array of population labels
            label_map: {sample_id: label}
        """
        label_map: Dict[str, int] = {}

        for pop_idx, pop_file in enumerate(population_files):
            pop_file = Path(pop_file)
            pop_df = pd.read_csv(pop_file, sep="\t")

            if "Sample name" in pop_df.columns:
                for sample in pop_df["Sample name"]:
                    label_map[sample] = pop_idx

        # Create aligned label array
        labels = np.array([
            label_map.get(s, -1) for s in self.sample_ids
        ])

        n_labeled = np.sum(labels >= 0)
        logger.info(f"Loaded labels: {n_labeled}/{len(labels)} samples labeled")

        return labels, label_map

    def load_strand_assignments(
        self,
        hash_files_pattern: str = "individual_hashes_*.tsv"
    ) -> Dict[str, int]:
        """
        Load strand assignments from hash files.

        Hash file format: sample_chr_strand_haploblock → hash
        Extract strand (0 or 1) from the sample identifier.

        Returns:
            sample_to_strand: {sample_id: strand (0 or 1)}
        """
        sample_to_strand: Dict[str, int] = {}

        # Find hash files
        hash_files = list(self.pipeline_output_dir.glob(hash_files_pattern))
        if not hash_files:
            logger.warning(f"No hash files found matching {hash_files_pattern}")
            return sample_to_strand

        for hash_file in hash_files:
            with open(hash_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        # Parse: HG00096_chr6_0_12345 (sample_chr_strand_haploblock)
                        sample_key = parts[0]
                        key_parts = sample_key.split("_")
                        if len(key_parts) >= 3:
                            sample_name = key_parts[0]
                            try:
                                strand = int(key_parts[2])  # 0 or 1
                                sample_to_strand[sample_name] = strand
                            except (ValueError, IndexError):
                                pass

        logger.info(f"Loaded strand info for {len(sample_to_strand)} samples")
        return sample_to_strand

    def prepare_dataset(
        self,
        population_files: List[Path],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        include_strand: bool = True,
        chromosome_id: int = 6
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare complete dataset for training.

        Args:
            population_files: List of population TSV files
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            random_seed: Random seed for reproducibility
            include_strand: Include strand IDs for composite embedding
            chromosome_id: Chromosome ID (default: 6)

        Returns:
            Dictionary with train/val/test splits:
            - X_train, X_val, X_test: (n, n_haploblocks) cluster ID tensors
            - y_train, y_val, y_test: (n,) label tensors
            - strand_train, strand_val, strand_test: (n,) strand IDs (if include_strand)
            - chr_ids: (n,) chromosome IDs (constant)
            - vocab_sizes: List of embedding vocab sizes per haploblock
        """
        # Load data
        cluster_data, vocab_sizes = self.load_cluster_assignments()
        cluster_matrix = self.create_cluster_matrix(cluster_data)
        labels, _ = self.load_population_labels(population_files)

        # Load strand assignments if requested
        strand_map = {}
        if include_strand:
            strand_map = self.load_strand_assignments()

        # Create strand array (default to 0 if not found)
        strand_array = np.array([
            strand_map.get(s, 0) for s in self.sample_ids
        ], dtype=np.int64)

        # Filter to labeled samples
        labeled_mask = labels >= 0
        X = cluster_matrix[labeled_mask]
        y = labels[labeled_mask]
        strands = strand_array[labeled_mask]

        n_samples = len(X)
        logger.info(f"Preparing dataset: {n_samples} labeled samples")

        # Shuffle and split
        np.random.seed(random_seed)
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        dataset = {
            'X_train': torch.from_numpy(X[train_idx]).long(),
            'X_val': torch.from_numpy(X[val_idx]).long(),
            'X_test': torch.from_numpy(X[test_idx]).long(),
            'y_train': torch.from_numpy(y[train_idx]).long(),
            'y_val': torch.from_numpy(y[val_idx]).long(),
            'y_test': torch.from_numpy(y[test_idx]).long(),
            'vocab_sizes': [v + 1 for v in vocab_sizes],  # +1 for padding
            'n_haploblocks': self.n_haploblocks,
            'n_classes': len(set(y)),
            'sample_ids': self.sample_ids,
            'chromosome_id': chromosome_id
        }

        # Add strand IDs
        if include_strand:
            dataset['strand_train'] = torch.from_numpy(strands[train_idx]).long()
            dataset['strand_val'] = torch.from_numpy(strands[val_idx]).long()
            dataset['strand_test'] = torch.from_numpy(strands[test_idx]).long()

        logger.info(
            f"Dataset prepared: train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )

        return dataset


def load_cluster_features(
    pipeline_output_dir: str,
    population_files: List[str],
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Convenience function to load cluster features for DL training.

    Args:
        pipeline_output_dir: Path to pipeline output
        population_files: List of population TSV file paths

    Returns:
        Dataset dictionary ready for training
    """
    loader = ClusterFeatureLoader(Path(pipeline_output_dir))
    return loader.prepare_dataset(
        [Path(p) for p in population_files],
        **kwargs
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    print("ClusterFeatureLoader - loads cluster IDs for embedding")
    print()
    print("Usage (auto-discover mode - recommended):")
    print("  loader = ClusterFeatureLoader('out_dir/TNFa')  # auto_discover=True by default")
    print("  cluster_data, vocab_sizes = loader.load_cluster_assignments()")
    print("  # Only loads haploblocks that have cluster files")
    print()
    print("Usage (boundaries mode):")
    print("  loader = ClusterFeatureLoader('out_dir/TNFa', auto_discover=False)")
    print("  cluster_data, vocab_sizes = loader.load_cluster_assignments()")
    print("  # Loads all haploblocks from boundaries file, warns about missing clusters")
    print()
    print("Full training pipeline:")
    print("  loader = ClusterFeatureLoader('out_dir/TNFa')")
    print("  dataset = loader.prepare_dataset(['igsr-chb.tsv', 'igsr-gbr.tsv', 'igsr-pur.tsv'])")
    print("  X_train = dataset['X_train']  # (n_samples, n_haploblocks) cluster IDs")
    print("  vocab_sizes = dataset['vocab_sizes']  # for embedding layers")
