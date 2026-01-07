"""
Haploblock ML Data Loader

Load and integrate data from haploblock pipeline outputs for machine learning.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HaploblockMLDataLoader:
    """
    Load ML data from haploblock pipeline outputs.

    This class provides methods to load variant counts, haploblock hashes,
    population labels, and cluster assignments for use in machine learning
    pipelines.
    """

    def __init__(self, pipeline_output_dir: Path):
        """
        Initialize loader with pipeline output directory.

        Args:
            pipeline_output_dir: Path to directory containing haploblock
                                pipeline outputs (variant_counts.tsv,
                                haploblock_hashes.tsv, etc.)

        Raises:
            ValueError: If directory does not exist
        """
        self.pipeline_output_dir = Path(pipeline_output_dir)
        if not self.pipeline_output_dir.exists():
            raise ValueError(
                f"Pipeline output directory does not exist: {pipeline_output_dir}"
            )

        logger.info(f"Initialized HaploblockMLDataLoader with dir: {self.pipeline_output_dir}")

    def load_haploblock_features(
        self,
        variant_counts_file: Optional[Path] = None,
        hashes_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Load haploblock features from variant_counts.tsv and haploblock_hashes.tsv.

        Args:
            variant_counts_file: Path to variant_counts.tsv. If None, looks in
                               pipeline_output_dir
            hashes_file: Path to haploblock_hashes.tsv. If None, looks in
                        pipeline_output_dir

        Returns:
            DataFrame with haploblock features indexed by sample ID.
            Columns include variant counts per haploblock and hash values.

        Raises:
            FileNotFoundError: If required files are not found
            ValueError: If files are malformed
        """
        # Locate variant counts file
        if variant_counts_file is None:
            variant_counts_file = self.pipeline_output_dir / "variant_counts.tsv"
        else:
            variant_counts_file = Path(variant_counts_file)

        # Locate hashes file
        if hashes_file is None:
            hashes_file = self.pipeline_output_dir / "haploblock_hashes.tsv"
        else:
            hashes_file = Path(hashes_file)

        # Load variant counts if available
        features_df = None
        if variant_counts_file.exists():
            try:
                features_df = pd.read_csv(
                    variant_counts_file,
                    sep="\t",
                    index_col=0
                )
                logger.info(
                    f"Loaded variant counts: {features_df.shape[0]} samples, "
                    f"{features_df.shape[1]} haploblocks"
                )
            except Exception as e:
                logger.warning(f"Failed to load variant counts: {e}")
                raise ValueError(f"Failed to parse variant_counts.tsv: {e}")

        # Load haploblock hashes if available
        if hashes_file.exists():
            try:
                hashes_df = pd.read_csv(
                    hashes_file,
                    sep="\t",
                    index_col=0
                )
                logger.info(
                    f"Loaded haploblock hashes: {hashes_df.shape[0]} samples, "
                    f"{hashes_df.shape[1]} haploblocks"
                )

                # Merge with variant counts if both exist
                if features_df is not None:
                    # Add prefix to distinguish hash columns from count columns
                    hashes_df = hashes_df.add_prefix("hash_")
                    features_df = features_df.join(hashes_df, how="outer")
                else:
                    features_df = hashes_df

            except Exception as e:
                logger.warning(f"Failed to load haploblock hashes: {e}")
                if features_df is None:
                    raise ValueError(f"Failed to parse haploblock_hashes.tsv: {e}")

        if features_df is None:
            raise FileNotFoundError(
                f"No feature files found in {self.pipeline_output_dir}"
            )

        return features_df

    def load_population_labels(
        self,
        population_files: List[Path]
    ) -> Dict[str, int]:
        """
        Load 1000 Genomes population groups as phenotype labels.

        Args:
            population_files: List of paths to population TSV files
                            (e.g., igsr-chb.tsv.tsv, igsr-gbr.tsv.tsv)
                            Files must have 'Sample name' column.

        Returns:
            Dictionary mapping sample name to population label (integer).
            Population labels are assigned sequentially: 0, 1, 2, ...

        Raises:
            FileNotFoundError: If any file does not exist
            ValueError: If file format is invalid
        """
        sample_to_label: Dict[str, int] = {}

        for pop_idx, pop_file in enumerate(population_files):
            pop_file = Path(pop_file)

            if not pop_file.exists():
                raise FileNotFoundError(f"Population file not found: {pop_file}")

            try:
                # Read TSV file
                pop_df = pd.read_csv(pop_file, sep="\t")

                # Check for required column
                if "Sample name" not in pop_df.columns:
                    raise ValueError(
                        f"'Sample name' column not found in {pop_file}"
                    )

                # Extract sample names
                samples = pop_df["Sample name"].tolist()

                # Validate sample names (1000 Genomes format)
                for sample in samples:
                    if not (sample.startswith("HG") or sample.startswith("NA")):
                        logger.warning(
                            f"Unexpected sample name format: {sample} in {pop_file}"
                        )
                    sample_to_label[sample] = pop_idx

                logger.info(
                    f"Loaded {len(samples)} samples from {pop_file.name} "
                    f"with label {pop_idx}"
                )

            except Exception as e:
                raise ValueError(f"Failed to parse {pop_file}: {e}")

        logger.info(
            f"Total samples with labels: {len(sample_to_label)} "
            f"across {len(population_files)} populations"
        )

        return sample_to_label

    def load_cluster_assignments(
        self,
        clusters_dir: Path
    ) -> Dict[str, Dict[str, int]]:
        """
        Load cluster assignments from clusters/*.tsv files.

        Each TSV file contains representative-to-individual mappings.
        Format: representative\\tindividual (no header)

        Args:
            clusters_dir: Path to directory containing cluster TSV files

        Returns:
            Dictionary mapping cluster file name to a dictionary of
            {sample_name: cluster_id}. Cluster IDs are assigned sequentially
            per file based on representative order.

        Raises:
            FileNotFoundError: If directory does not exist
            ValueError: If any file is malformed
        """
        clusters_dir = Path(clusters_dir)

        if not clusters_dir.exists():
            raise FileNotFoundError(f"Clusters directory not found: {clusters_dir}")

        all_clusters: Dict[str, Dict[str, int]] = {}

        # Find all TSV files in directory
        cluster_files = sorted(clusters_dir.glob("*.tsv"))

        if not cluster_files:
            logger.warning(f"No cluster TSV files found in {clusters_dir}")
            return all_clusters

        for cluster_file in cluster_files:
            try:
                # Parse cluster file using same logic as data_parser.py
                representative_to_cluster: Dict[str, int] = {}
                individual_to_cluster: Dict[str, int] = {}
                next_cluster = 0

                with open(cluster_file, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.rstrip()
                        if not line:
                            continue

                        parts = line.split("\t")
                        if len(parts) != 2:
                            raise ValueError(
                                f"Line {line_num}: expected 2 columns, got {len(parts)}"
                            )

                        rep, indiv = parts

                        # Assign cluster ID to representative on first occurrence
                        if rep not in representative_to_cluster:
                            representative_to_cluster[rep] = next_cluster
                            next_cluster += 1

                        # Map individual to cluster
                        individual_to_cluster[indiv] = representative_to_cluster[rep]

                all_clusters[cluster_file.stem] = individual_to_cluster

                logger.info(
                    f"Loaded {len(individual_to_cluster)} individuals in "
                    f"{len(representative_to_cluster)} clusters from {cluster_file.name}"
                )

            except Exception as e:
                raise ValueError(f"Failed to parse {cluster_file}: {e}")

        return all_clusters

    def create_ml_dataset(
        self,
        vcf_path: Optional[Path] = None,
        population_files: Optional[List[Path]] = None,
        clusters_dir: Optional[Path] = None,
        variant_counts_file: Optional[Path] = None,
        hashes_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create complete ML dataset with genotypes and labels.

        This method integrates haploblock features, population labels, and
        optionally cluster assignments into a single dataset ready for ML.

        Args:
            vcf_path: Optional path to VCF file for genotype loading
            population_files: List of population TSV files for labels
            clusters_dir: Optional directory containing cluster assignments
            variant_counts_file: Optional path to variant_counts.tsv
            hashes_file: Optional path to haploblock_hashes.tsv

        Returns:
            Dictionary containing:
                - 'features': pd.DataFrame with haploblock features (samples x features)
                - 'labels': np.ndarray with population labels (if population_files provided)
                - 'sample_ids': List of sample IDs
                - 'label_map': Dict mapping sample to label (if population_files provided)
                - 'clusters': Dict of cluster assignments (if clusters_dir provided)
                - 'metadata': Dict with dataset statistics

        Raises:
            ValueError: If required data cannot be loaded
        """
        dataset: Dict[str, Any] = {}

        # Load haploblock features
        logger.info("Loading haploblock features...")
        features_df = self.load_haploblock_features(
            variant_counts_file=variant_counts_file,
            hashes_file=hashes_file
        )
        dataset["features"] = features_df
        dataset["sample_ids"] = features_df.index.tolist()

        # Load population labels if provided
        if population_files:
            logger.info("Loading population labels...")
            label_map = self.load_population_labels(population_files)
            dataset["label_map"] = label_map

            # Create label array aligned with features
            labels = np.array([
                label_map.get(sample, -1)
                for sample in features_df.index
            ])
            dataset["labels"] = labels

            # Count labeled samples
            n_labeled = np.sum(labels >= 0)
            logger.info(f"Labeled samples: {n_labeled}/{len(labels)}")

        # Load cluster assignments if provided
        if clusters_dir:
            logger.info("Loading cluster assignments...")
            clusters = self.load_cluster_assignments(clusters_dir)
            dataset["clusters"] = clusters

        # Add metadata
        dataset["metadata"] = {
            "n_samples": len(features_df),
            "n_features": features_df.shape[1],
            "n_populations": len(set(dataset.get("label_map", {}).values())) if population_files else 0,
            "feature_names": features_df.columns.tolist(),
            "has_vcf": vcf_path is not None,
            "has_labels": population_files is not None,
            "has_clusters": clusters_dir is not None,
        }

        logger.info(
            f"Created ML dataset: {dataset['metadata']['n_samples']} samples, "
            f"{dataset['metadata']['n_features']} features"
        )

        return dataset
