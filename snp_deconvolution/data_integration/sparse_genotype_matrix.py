"""
Sparse Genotype Matrix Module

Memory-efficient sparse matrix representations for genotype data.
Supports loading from VCF files and conversion to various ML frameworks.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class SparseGenotypeMatrix:
    """
    Sparse representation for genotype data.

    Provides efficient storage and conversion methods for large-scale
    genotype matrices, supporting VCF input and conversion to ML frameworks
    like XGBoost and PyTorch.
    """

    @staticmethod
    def from_vcf(
        vcf_path: Path,
        samples: Optional[List[str]] = None,
        region: Optional[str] = None,
        maf_threshold: float = 0.01,
        max_missing: float = 0.1
    ) -> Tuple[sp.csr_matrix, List[str], List[Dict[str, Any]]]:
        """
        Load VCF as sparse CSR matrix.

        This method uses bcftools to extract genotypes and converts them
        to a sparse matrix representation for memory efficiency.

        Args:
            vcf_path: Path to VCF/BCF file (bgzipped and indexed)
            samples: Optional list of sample IDs to extract. If None, uses all.
            region: Optional genomic region (e.g., "chr6:1000-5000")
            maf_threshold: Minimum minor allele frequency (default: 0.01)
            max_missing: Maximum fraction of missing genotypes per variant (default: 0.1)

        Returns:
            Tuple of:
                - sparse_matrix: scipy CSR matrix (samples x variants)
                - sample_ids: List of sample IDs (rows)
                - snp_info: List of dicts with variant metadata (columns)
                  Each dict contains: {'chrom', 'pos', 'id', 'ref', 'alt', 'maf'}

        Raises:
            FileNotFoundError: If VCF file does not exist
            RuntimeError: If bcftools commands fail
            ValueError: If no variants pass filters
        """
        vcf_path = Path(vcf_path)
        if not vcf_path.exists():
            raise FileNotFoundError(f"VCF file not found: {vcf_path}")

        logger.info(f"Loading VCF from {vcf_path}")
        if region:
            logger.info(f"Region: {region}")
        if samples:
            logger.info(f"Extracting {len(samples)} samples")

        # Build bcftools command for extracting genotypes
        cmd = ["bcftools", "query"]

        # Add sample filter if specified
        if samples:
            cmd.extend(["-s", ",".join(samples)])

        # Add region filter if specified
        if region:
            cmd.extend(["-r", region])

        # Format: CHROM POS ID REF ALT [GT for each sample]
        # GT format: 0=ref, 1=alt1, 2=alt2, etc.
        cmd.extend([
            "-f", "%CHROM\t%POS\t%ID\t%REF\t%ALT[\t%GT]\n",
            str(vcf_path)
        ])

        try:
            # Execute bcftools
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            lines = result.stdout.strip().split("\n")

            if not lines or lines[0] == "":
                raise ValueError("No variants found in VCF")

            logger.info(f"Extracted {len(lines)} variants from VCF")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"bcftools query failed: {e.stderr}")

        # Get sample IDs if not provided
        if samples is None:
            cmd_samples = ["bcftools", "query", "-l", str(vcf_path)]
            result = subprocess.run(
                cmd_samples,
                check=True,
                capture_output=True,
                text=True
            )
            samples = result.stdout.strip().split("\n")
            logger.info(f"Found {len(samples)} samples in VCF")

        sample_ids = samples
        n_samples = len(sample_ids)

        # Parse variants and build sparse matrix
        row_indices = []  # Sample indices
        col_indices = []  # Variant indices
        data = []  # Genotype values
        snp_info = []

        for var_idx, line in enumerate(lines):
            fields = line.split("\t")

            # Parse variant info
            chrom = fields[0]
            pos = int(fields[1])
            var_id = fields[2] if fields[2] != "." else f"{chrom}:{pos}"
            ref = fields[3]
            alt = fields[4]

            # Parse genotypes (starting from field 5)
            genotypes = fields[5:]

            if len(genotypes) != n_samples:
                logger.warning(
                    f"Variant {var_id}: expected {n_samples} genotypes, "
                    f"got {len(genotypes)}"
                )
                continue

            # Convert genotypes to numeric values
            gt_values = []
            n_missing = 0

            for gt in genotypes:
                # Handle various genotype formats
                if gt == "." or gt == "./." or gt == ".|.":
                    gt_values.append(-1)  # Missing
                    n_missing += 1
                elif "/" in gt or "|" in gt:
                    # Diploid genotype (e.g., "0/1" or "0|1")
                    alleles = gt.replace("|", "/").split("/")
                    try:
                        # Sum alleles: 0/0=0, 0/1=1, 1/1=2
                        allele_sum = sum(int(a) if a != "." else 0 for a in alleles)
                        gt_values.append(allele_sum)
                    except ValueError:
                        gt_values.append(-1)
                        n_missing += 1
                else:
                    # Haploid or unknown format
                    try:
                        gt_values.append(int(gt))
                    except ValueError:
                        gt_values.append(-1)
                        n_missing += 1

            # Filter by missing rate
            missing_rate = n_missing / n_samples
            if missing_rate > max_missing:
                continue

            # Calculate MAF
            valid_gts = [g for g in gt_values if g >= 0]
            if not valid_gts:
                continue

            # Count alternate alleles (assuming diploid)
            alt_count = sum(valid_gts)
            total_alleles = len(valid_gts) * 2  # Diploid
            maf = min(alt_count / total_alleles, 1 - alt_count / total_alleles)

            # Filter by MAF
            if maf < maf_threshold:
                continue

            # Add to sparse matrix
            for sample_idx, gt_val in enumerate(gt_values):
                if gt_val >= 0:  # Skip missing values in sparse matrix
                    row_indices.append(sample_idx)
                    col_indices.append(len(snp_info))
                    data.append(gt_val)

            # Store variant info
            snp_info.append({
                "chrom": chrom,
                "pos": pos,
                "id": var_id,
                "ref": ref,
                "alt": alt,
                "maf": maf,
                "missing_rate": missing_rate,
            })

        if not snp_info:
            raise ValueError(
                f"No variants passed filters (MAF >= {maf_threshold}, "
                f"missing <= {max_missing})"
            )

        # Create sparse CSR matrix
        n_variants = len(snp_info)
        sparse_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_samples, n_variants),
            dtype=np.int8  # Genotypes are 0, 1, 2
        )

        logger.info(
            f"Created sparse matrix: {n_samples} samples x {n_variants} variants"
        )
        logger.info(
            f"Sparsity: {100 * (1 - sparse_matrix.nnz / (n_samples * n_variants)):.2f}%"
        )

        return sparse_matrix, sample_ids, snp_info

    @staticmethod
    def from_numpy(genotypes: np.ndarray) -> sp.csr_matrix:
        """
        Convert dense numpy array to sparse CSR matrix.

        Args:
            genotypes: Dense numpy array (samples x variants)
                      Missing values should be -1 or np.nan

        Returns:
            Sparse CSR matrix with missing values excluded

        Raises:
            ValueError: If input is not 2D
        """
        if genotypes.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {genotypes.shape}")

        # Replace NaN with -1 for masking
        if np.issubdtype(genotypes.dtype, np.floating):
            genotypes = np.nan_to_num(genotypes, nan=-1)

        # Create mask for valid (non-missing) values
        valid_mask = genotypes >= 0

        # Create sparse matrix (excluding missing values)
        sparse_matrix = sp.csr_matrix(
            np.where(valid_mask, genotypes, 0),
            dtype=np.int8
        )

        logger.info(
            f"Converted numpy array to sparse: {genotypes.shape} -> "
            f"{sparse_matrix.nnz} non-zero elements"
        )

        return sparse_matrix

    @staticmethod
    def to_xgboost_dmatrix(
        sparse_matrix: sp.csr_matrix,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Any:
        """
        Convert sparse matrix to XGBoost DMatrix.

        Args:
            sparse_matrix: Sparse CSR matrix (samples x features)
            labels: Optional target labels (samples,)
            feature_names: Optional list of feature names

        Returns:
            xgboost.DMatrix object

        Raises:
            ImportError: If xgboost is not installed
            ValueError: If labels shape doesn't match matrix rows
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

        if labels is not None and len(labels) != sparse_matrix.shape[0]:
            raise ValueError(
                f"Labels length ({len(labels)}) doesn't match matrix rows "
                f"({sparse_matrix.shape[0]})"
            )

        # Create DMatrix
        dmatrix = xgb.DMatrix(
            sparse_matrix,
            label=labels,
            feature_names=feature_names
        )

        logger.info(
            f"Created XGBoost DMatrix: {dmatrix.num_row()} samples, "
            f"{dmatrix.num_col()} features"
        )

        return dmatrix

    @staticmethod
    def to_pytorch_tensor(
        sparse_matrix: sp.csr_matrix,
        device: str = "cuda",
        keep_sparse: bool = False
    ) -> Any:
        """
        Convert sparse matrix to PyTorch tensor.

        Args:
            sparse_matrix: Sparse CSR matrix (samples x features)
            device: Target device ("cuda", "cpu", or "cuda:0", etc.)
            keep_sparse: If True, returns sparse tensor. If False, returns dense.

        Returns:
            torch.Tensor (sparse or dense depending on keep_sparse)

        Raises:
            ImportError: If PyTorch is not installed
            RuntimeError: If CUDA requested but not available
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is not installed. Install with: pip install torch"
            )

        # Check CUDA availability
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"

        if keep_sparse:
            # Convert to PyTorch sparse tensor (COO format)
            coo = sparse_matrix.tocoo()
            indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
            values = torch.FloatTensor(coo.data)
            shape = coo.shape

            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, shape, dtype=torch.float32
            )
            sparse_tensor = sparse_tensor.to(device)

            logger.info(
                f"Created PyTorch sparse tensor: {shape} on {device}"
            )
            return sparse_tensor
        else:
            # Convert to dense tensor
            dense_array = sparse_matrix.toarray()
            dense_tensor = torch.FloatTensor(dense_array).to(device)

            logger.info(
                f"Created PyTorch dense tensor: {dense_tensor.shape} on {device}"
            )
            return dense_tensor

    @staticmethod
    def save_npz(sparse_matrix: sp.csr_matrix, output_path: Path) -> None:
        """
        Save sparse matrix to NPZ file.

        Args:
            sparse_matrix: Sparse CSR matrix to save
            output_path: Path to output .npz file

        Raises:
            IOError: If file cannot be written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sp.save_npz(output_path, sparse_matrix)
        logger.info(f"Saved sparse matrix to {output_path}")

    @staticmethod
    def load_npz(input_path: Path) -> sp.csr_matrix:
        """
        Load sparse matrix from NPZ file.

        Args:
            input_path: Path to .npz file

        Returns:
            Sparse CSR matrix

        Raises:
            FileNotFoundError: If file does not exist
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")

        sparse_matrix = sp.load_npz(input_path)
        logger.info(
            f"Loaded sparse matrix from {input_path}: {sparse_matrix.shape}"
        )
        return sparse_matrix
