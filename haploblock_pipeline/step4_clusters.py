#!/usr/bin/env python3
import os
import logging
import pathlib
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import data_parser

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# MMseqs2 params
# ----------------------------------------------------------------------
def calculate_mmseq_params(variant_counts_file: pathlib.Path):
    """
    Calculate min sequence identify and coverage fraction for MMSeqs2
    
    returns:
    - haploblock2min_id: dict, key=(start, end), value=min sequence identify
    - haploblock2cov_fraction: dict, key=(start, end), value=coverage fraction
    """
    haploblock2min_id = {}
    haploblock2cov_fraction = {}

    with open(variant_counts_file, "r") as f:
        header = f.readline()
        if not header.startswith("START\t"):
            raise ValueError(f"Variant counts file missing header: {header.strip()}")

        for line in f:
            start, end, mean, stdev = line.strip().split("\t")
            start = int(start)
            end = int(end)
            hap_len = end - start
            haploblock2min_id[(start, end)] = 1 - (float(mean) / hap_len)
            haploblock2cov_fraction[(start, end)] = 1 - (682 / hap_len)
        
    return(haploblock2min_id, haploblock2cov_fraction)


# ----------------------------------------------------------------------
# Run clustering per FASTA
# ----------------------------------------------------------------------
def compute_clusters(input_fasta: str, out: str, min_seq_id: float, cov_fraction: float, cov_mode: int,
                     chrom: str, start: str, end: str, threads_per_task: int = 1):

    # Resolve EVERYTHING to absolute paths
    input_fasta = str(pathlib.Path(input_fasta).resolve())

    output_prefix = pathlib.Path(out) / "clusters" / f"chr{chrom}_{start}-{end}"
    output_prefix = str(output_prefix.resolve())

    tmp_dir = pathlib.Path(out) / "tmp"
    tmp_dir = str(tmp_dir.resolve())

    # run MMSeqs2
    cmd = [
        "mmseqs", "easy-cluster",
        input_fasta,
        output_prefix,
        tmp_dir,
        "--min-seq-id", str(min_seq_id),
        "-c", str(cov_fraction),
        "--cov-mode", str(cov_mode),
        "--threads", str(threads_per_task),
        "--remove-tmp-files", "1"
    ]

    logger.debug("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)


# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------
def run_clusters(boundaries_file: pathlib.Path,
                 merged_consensus_dir: pathlib.Path,
                 variant_counts_file: pathlib.Path,
                 chrom: str,
                 out_dir: pathlib.Path,
                 cov_mode: int,
                 threads: None):

    # Create output and temporary directories if they don't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "clusters").mkdir(exist_ok=True)
    (out_dir / "tmp").mkdir(exist_ok=True)

    haploblock_boundaries = data_parser.parse_haploblock_boundaries(boundaries_file)
    logger.info("Found %d haploblocks", len(haploblock_boundaries))
    (haploblock2min_id, haploblock2cov_fraction) = calculate_mmseq_params(variant_counts_file)

    # Parallel strategy: limit concurrent tasks, each gets multiple threads
    cpu = os.cpu_count() or 8
    max_workers = min(8, cpu)  # at most 8 parallel MMseqs2 tasks
    threads_per_task = max(1, cpu // max_workers)
    logger.info("Computing clusters: %d parallel tasks × %d threads each (total cores: %d)",
                max_workers, threads_per_task, cpu)

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for start, end in haploblock_boundaries:
            input_fasta = merged_consensus_dir / f"chr{chrom}_region_{start}-{end}.fa"
            futures.append(
                executor.submit(
                    compute_clusters,
                    str(input_fasta),
                    str(out_dir),
                    haploblock2min_id[(start, end)],
                    haploblock2cov_fraction[(start, end)],
                    cov_mode,
                    chrom, start, end,
                    threads_per_task
                )
            )

        # Progress bar
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Clustering", unit="block"):
            fut.result()  # propagate exceptions if any

    logger.info("All clusters computed successfully.")


# ----------------------------------------------------------------------
# Pipeline wrapper
# ----------------------------------------------------------------------
def run(boundaries_file, merged_consensus_dir, variant_counts, chr, out, cov_mode=2, threads=None):
    run_clusters(
        pathlib.Path(boundaries_file),
        pathlib.Path(merged_consensus_dir),
        pathlib.Path(variant_counts),
        str(chr),
        pathlib.Path(out),
        cov_mode=cov_mode,
        threads=threads
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO
    )
    parser = argparse.ArgumentParser(description="Cluster haploblock consensus sequences using MMseqs2")
    parser.add_argument("--boundaries_file", type=pathlib.Path, required=True,
                        help="TSV file with header (START\tEND) and 2 columns: start end")
    parser.add_argument("--merged_consensus_dir", type=pathlib.Path, required=True,
                        help="Path to folder with merged phased FASTA files")
    parser.add_argument("--variant_counts", type=pathlib.Path, required=True,
                        help="TSV file with 4 columns: START, END, MEAN, STDEV")
    parser.add_argument("--chr", type=str, required=True,
                        help="Chromosome number")
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="Output folder path")
    parser.add_argument("--cov_mode", type=int, default=0,
                        help="alignment coverage, see MMSeqs2 documentation")
    parser.add_argument("--threads", type=int, default=None,
                        help="TODO")

    args = parser.parse_args()

    run(
        boundaries_file=args.boundaries_file,
        merged_consensus_dir=args.merged_consensus_dir,
        variant_counts=args.variant_counts,
        chr=args.chr,
        out=args.out,
        cov_mode=args.cov_mode,
        threads=args.threads
    )

