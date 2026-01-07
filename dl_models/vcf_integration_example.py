"""
Integration Example: Deep Learning Models with VCF Data

This script demonstrates how to:
1. Load genotype data from VCF files
2. Apply deep learning models for causal SNP identification
3. Integrate with the haploblock pipeline
4. Export results for downstream analysis

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from dataclasses import dataclass

# Import our models
from snp_interpretable_models import InterpretableSNPModel, GenotypeEncoder
from training_pipeline import SNPDataset, SNPModelTrainer, ModelEvaluator
from torch.utils.data import DataLoader


# ============================================================================
# VCF DATA LOADING
# ============================================================================

@dataclass
class SNPInfo:
    """Store SNP metadata"""
    chrom: str
    pos: int
    ref: str
    alt: str
    rsid: str
    maf: float


class VCFGenotypeLoader:
    """
    Load genotype matrix from VCF files.

    Handles:
    - Missing data imputation
    - Quality filtering
    - MAF filtering
    - LD pruning (optional)
    """

    def __init__(self,
                 vcf_path: Path,
                 min_maf: float = 0.01,
                 max_missing_rate: float = 0.1,
                 quality_threshold: float = 30.0):
        self.vcf_path = Path(vcf_path)
        self.min_maf = min_maf
        self.max_missing_rate = max_missing_rate
        self.quality_threshold = quality_threshold

        self.genotypes = None
        self.samples = None
        self.snp_info = []

    def load_with_pysam(self) -> Tuple[np.ndarray, List[str], List[SNPInfo]]:
        """
        Load VCF using pysam (production method)

        Returns:
            genotypes: (n_samples, n_snps) matrix with 0/1/2 encoding
            samples: List of sample IDs
            snp_info: List of SNP metadata
        """
        try:
            import pysam
        except ImportError:
            raise ImportError("pysam not installed. Run: pip install pysam")

        vcf = pysam.VariantFile(str(self.vcf_path))

        # Get samples
        samples = list(vcf.header.samples)
        n_samples = len(samples)

        genotype_list = []
        snp_info_list = []

        print(f"Loading VCF: {self.vcf_path}")
        print(f"Samples: {n_samples}")

        # Process variants
        n_variants = 0
        n_filtered = 0

        for record in vcf.fetch():
            # Quality filter
            if record.qual is not None and record.qual < self.quality_threshold:
                n_filtered += 1
                continue

            # Get genotypes for all samples
            gts = []
            missing_count = 0

            for sample in samples:
                gt = record.samples[sample]['GT']

                if None in gt:
                    gts.append(-1)  # Missing
                    missing_count += 1
                else:
                    # Convert to additive encoding
                    gts.append(sum(gt))

            # Missing rate filter
            missing_rate = missing_count / n_samples
            if missing_rate > self.max_missing_rate:
                n_filtered += 1
                continue

            # MAF filter
            valid_gts = [g for g in gts if g >= 0]
            if len(valid_gts) > 0:
                allele_freq = sum(valid_gts) / (2 * len(valid_gts))
                maf = min(allele_freq, 1 - allele_freq)

                if maf < self.min_maf:
                    n_filtered += 1
                    continue
            else:
                maf = 0.0

            # Store variant
            genotype_list.append(gts)
            snp_info_list.append(SNPInfo(
                chrom=record.chrom,
                pos=record.pos,
                ref=record.ref,
                alt=','.join(record.alts) if record.alts else '.',
                rsid=record.id if record.id else f"{record.chrom}:{record.pos}",
                maf=maf
            ))

            n_variants += 1

            if n_variants % 10000 == 0:
                print(f"  Processed {n_variants} variants, filtered {n_filtered}...")

        vcf.close()

        print(f"\nLoaded {n_variants} variants after filtering")
        print(f"Filtered out {n_filtered} variants")

        # Convert to numpy array
        genotypes = np.array(genotype_list).T  # (n_samples, n_snps)

        # Impute missing data (mean imputation)
        genotypes = self._impute_missing(genotypes)

        return genotypes, samples, snp_info_list

    def load_from_plink(self, bed_prefix: Path) -> Tuple[np.ndarray, List[str], List[SNPInfo]]:
        """
        Load from PLINK BED format

        Args:
            bed_prefix: Prefix for .bed/.bim/.fam files
        """
        try:
            from pandas_plink import read_plink
        except ImportError:
            raise ImportError("pandas-plink not installed. Run: pip install pandas-plink")

        # Read PLINK files
        (bim, fam, bed) = read_plink(str(bed_prefix))

        # Extract genotypes
        genotypes = bed.T.compute()  # (n_samples, n_snps)

        # Get sample IDs
        samples = fam['iid'].tolist()

        # Get SNP info
        snp_info_list = []
        for _, row in bim.iterrows():
            # Calculate MAF
            valid_gts = genotypes[:, _][genotypes[:, _] >= 0]
            if len(valid_gts) > 0:
                allele_freq = valid_gts.mean() / 2
                maf = min(allele_freq, 1 - allele_freq)
            else:
                maf = 0.0

            snp_info_list.append(SNPInfo(
                chrom=row['chrom'],
                pos=row['pos'],
                ref=row['a0'],
                alt=row['a1'],
                rsid=row['snp'],
                maf=maf
            ))

        # Apply filters
        genotypes, snp_info_list = self._apply_filters(genotypes, snp_info_list)

        # Impute missing
        genotypes = self._impute_missing(genotypes)

        return genotypes, samples, snp_info_list

    def _impute_missing(self, genotypes: np.ndarray) -> np.ndarray:
        """Impute missing genotypes (-1) with mean"""
        for j in range(genotypes.shape[1]):
            col = genotypes[:, j]
            missing_mask = col == -1

            if missing_mask.any():
                valid_values = col[~missing_mask]
                if len(valid_values) > 0:
                    mean_value = valid_values.mean()
                    genotypes[missing_mask, j] = np.round(mean_value)

        return genotypes

    def _apply_filters(self,
                       genotypes: np.ndarray,
                       snp_info: List[SNPInfo]) -> Tuple[np.ndarray, List[SNPInfo]]:
        """Apply MAF and missing rate filters"""
        keep_indices = []

        for i, info in enumerate(snp_info):
            # Missing rate
            col = genotypes[:, i]
            missing_rate = (col == -1).mean()

            if missing_rate > self.max_missing_rate:
                continue

            # MAF
            if info.maf < self.min_maf:
                continue

            keep_indices.append(i)

        filtered_genotypes = genotypes[:, keep_indices]
        filtered_snp_info = [snp_info[i] for i in keep_indices]

        return filtered_genotypes, filtered_snp_info


# ============================================================================
# PHENOTYPE LOADING
# ============================================================================

class PhenotypeLoader:
    """Load phenotype data from various formats"""

    @staticmethod
    def load_from_csv(csv_path: Path,
                     sample_col: str = 'IID',
                     phenotype_col: str = 'phenotype') -> pd.DataFrame:
        """
        Load phenotypes from CSV

        Expected format:
        IID,phenotype
        sample1,0
        sample2,1
        """
        df = pd.read_csv(csv_path)
        return df[[sample_col, phenotype_col]]

    @staticmethod
    def load_from_plink_pheno(pheno_path: Path) -> pd.DataFrame:
        """
        Load from PLINK phenotype file

        Format: FID IID phenotype
        """
        df = pd.read_csv(pheno_path, sep=r'\s+',
                        names=['FID', 'IID', 'phenotype'])
        return df[['IID', 'phenotype']]

    @staticmethod
    def match_samples(genotype_samples: List[str],
                     phenotype_df: pd.DataFrame,
                     sample_col: str = 'IID') -> Tuple[np.ndarray, List[int]]:
        """
        Match phenotypes to genotype samples

        Returns:
            phenotypes: Matched phenotype array
            indices: Indices of genotype samples to keep
        """
        # Create mapping
        pheno_dict = dict(zip(
            phenotype_df[sample_col],
            phenotype_df['phenotype']
        ))

        # Match samples
        matched_phenotypes = []
        matched_indices = []

        for i, sample in enumerate(genotype_samples):
            if sample in pheno_dict:
                matched_phenotypes.append(pheno_dict[sample])
                matched_indices.append(i)

        phenotypes = np.array(matched_phenotypes)

        return phenotypes, matched_indices


# ============================================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================================

class CausalSNPAnalysisPipeline:
    """
    End-to-end pipeline for causal SNP identification
    """

    def __init__(self,
                 output_dir: Path,
                 architecture: str = 'cnn_transformer',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.architecture = architecture
        self.device = device

        self.model = None
        self.genotypes = None
        self.phenotypes = None
        self.snp_info = None
        self.samples = None

    def load_data(self,
                  vcf_path: Optional[Path] = None,
                  bed_prefix: Optional[Path] = None,
                  phenotype_path: Path = None,
                  min_maf: float = 0.01):
        """Load genotype and phenotype data"""

        print("="*80)
        print("LOADING DATA")
        print("="*80)

        # Load genotypes
        loader = VCFGenotypeLoader(
            vcf_path=vcf_path if vcf_path else bed_prefix,
            min_maf=min_maf
        )

        if vcf_path:
            self.genotypes, self.samples, self.snp_info = loader.load_with_pysam()
        elif bed_prefix:
            self.genotypes, self.samples, self.snp_info = loader.load_from_plink(bed_prefix)
        else:
            raise ValueError("Must provide either vcf_path or bed_prefix")

        # Load phenotypes
        pheno_loader = PhenotypeLoader()

        if phenotype_path.suffix == '.csv':
            pheno_df = pheno_loader.load_from_csv(phenotype_path)
        else:
            pheno_df = pheno_loader.load_from_plink_pheno(phenotype_path)

        # Match samples
        self.phenotypes, keep_indices = pheno_loader.match_samples(
            self.samples, pheno_df
        )

        # Filter genotypes to matched samples
        self.genotypes = self.genotypes[keep_indices, :]
        self.samples = [self.samples[i] for i in keep_indices]

        print(f"\nFinal dataset:")
        print(f"  Samples: {len(self.samples)}")
        print(f"  SNPs: {len(self.snp_info)}")
        print(f"  Phenotype distribution: {np.bincount(self.phenotypes)}")

    def train_model(self,
                   num_epochs: int = 100,
                   batch_size: int = 32,
                   val_split: float = 0.15,
                   test_split: float = 0.15):
        """Train deep learning model"""

        print("\n" + "="*80)
        print("TRAINING MODEL")
        print("="*80)

        # Create dataset
        dataset = SNPDataset(
            self.genotypes,
            self.phenotypes,
            encoding='haplotype',
            normalize=True
        )

        # Split data
        n_total = len(dataset)
        n_test = int(n_total * test_split)
        n_val = int(n_total * val_split)
        n_train = n_total - n_test - n_val

        from torch.utils.data import random_split
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Create model
        n_snps = len(self.snp_info)
        num_classes = len(np.unique(self.phenotypes))

        self.model = InterpretableSNPModel(
            n_snps=n_snps,
            encoding_dim=8,
            num_classes=num_classes,
            architecture=self.architecture,
            dropout=0.2
        )

        # Train
        trainer = SNPModelTrainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            learning_rate=1e-4,
            use_focal_loss=True
        )

        checkpoint_dir = self.output_dir / "checkpoints"
        history = trainer.train(
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            verbose=True
        )

        # Save training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.output_dir / "training_history.csv", index=False)

        return history

    def evaluate_and_identify_snps(self, top_k: int = 100):
        """Evaluate model and identify causal SNPs"""

        print("\n" + "="*80)
        print("EVALUATION AND SNP IDENTIFICATION")
        print("="*80)

        evaluator = ModelEvaluator(self.model, device=self.device)

        # Evaluate
        metrics = evaluator.evaluate(self.test_loader)

        print("\nTest Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.output_dir / "test_metrics.csv", index=False)

        # Identify top SNPs
        print(f"\nIdentifying top {top_k} causal SNPs...")
        top_snps = evaluator.identify_top_snps(
            self.test_loader,
            top_k=top_k,
            method='ensemble'
        )

        # Create results dataframe
        results = []
        for snp_idx, importance_score in top_snps.items():
            info = self.snp_info[snp_idx]
            results.append({
                'snp_index': snp_idx,
                'rsid': info.rsid,
                'chromosome': info.chrom,
                'position': info.pos,
                'ref': info.ref,
                'alt': info.alt,
                'maf': info.maf,
                'importance_score': importance_score
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('importance_score', ascending=False)

        # Save results
        output_path = self.output_dir / "causal_snps.tsv"
        results_df.to_csv(output_path, sep='\t', index=False)

        print(f"\nResults saved to: {output_path}")
        print(f"\nTop 10 Causal SNPs:")
        print(results_df.head(10).to_string(index=False))

        # Visualize
        evaluator.visualize_snp_importance(
            top_snps,
            save_path=self.output_dir / "snp_importance.png"
        )

        return results_df

    def export_for_downstream_analysis(self, results_df: pd.DataFrame):
        """Export results in various formats for downstream tools"""

        print("\n" + "="*80)
        print("EXPORTING RESULTS")
        print("="*80)

        # 1. BED format for genome browsers
        bed_df = results_df[['chromosome', 'position', 'rsid', 'importance_score']].copy()
        bed_df['start'] = bed_df['position'] - 1
        bed_df['end'] = bed_df['position']
        bed_df = bed_df[['chromosome', 'start', 'end', 'rsid', 'importance_score']]
        bed_path = self.output_dir / "causal_snps.bed"
        bed_df.to_csv(bed_path, sep='\t', index=False, header=False)
        print(f"BED file: {bed_path}")

        # 2. VCF subset for extraction
        snp_list_path = self.output_dir / "causal_snps_list.txt"
        results_df['rsid'].to_csv(snp_list_path, index=False, header=False)
        print(f"SNP list: {snp_list_path}")

        # 3. JSON for web visualization
        import json
        json_data = results_df.to_dict(orient='records')
        json_path = self.output_dir / "causal_snps.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON file: {json_path}")

        print("\nExport complete!")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Complete pipeline example"""

    # Configuration
    vcf_path = Path("data/genotypes.vcf.gz")  # Your VCF file
    phenotype_path = Path("data/phenotypes.csv")  # Your phenotype file
    output_dir = Path("results/causal_snp_analysis")

    # Initialize pipeline
    pipeline = CausalSNPAnalysisPipeline(
        output_dir=output_dir,
        architecture='cnn_transformer',  # Best for causal SNP identification
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load data
    # NOTE: This example assumes you have these files
    # Uncomment when you have real data:
    # pipeline.load_data(
    #     vcf_path=vcf_path,
    #     phenotype_path=phenotype_path,
    #     min_maf=0.01
    # )

    # For demonstration, create synthetic data
    print("DEMO MODE: Using synthetic data")
    print("For real analysis, uncomment the load_data() call above")

    n_samples = 500
    n_snps = 1000
    pipeline.genotypes = np.random.randint(0, 3, size=(n_samples, n_snps))
    pipeline.phenotypes = np.random.randint(0, 2, size=n_samples)
    pipeline.samples = [f"sample_{i}" for i in range(n_samples)]
    pipeline.snp_info = [
        SNPInfo(
            chrom=f"chr{(i % 22) + 1}",
            pos=100000 + i * 1000,
            ref='A',
            alt='G',
            rsid=f"rs{i}",
            maf=np.random.uniform(0.05, 0.5)
        )
        for i in range(n_snps)
    ]

    # Train model
    history = pipeline.train_model(
        num_epochs=20,  # Use 100+ for real analysis
        batch_size=32
    )

    # Evaluate and identify causal SNPs
    results_df = pipeline.evaluate_and_identify_snps(top_k=50)

    # Export results
    pipeline.export_for_downstream_analysis(results_df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
