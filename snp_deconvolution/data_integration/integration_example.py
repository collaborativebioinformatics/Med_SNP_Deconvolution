#!/usr/bin/env python3
"""
Complete Integration Example: From Raw Data to ML-Ready Dataset

This example demonstrates the complete workflow:
1. Load haploblock features and population labels
2. Load genotypes from VCF as sparse matrix
3. Align samples across datasets
4. Prepare data for XGBoost or PyTorch
5. Train a simple classifier

Note: This is a template - adjust paths to your actual data.
"""

import logging
from pathlib import Path

import numpy as np

from snp_deconvolution.data_integration import (
    HaploblockMLDataLoader,
    SparseGenotypeMatrix,
    create_haploblock_features,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_integrate_data(
    pipeline_output_dir: Path,
    vcf_path: Path,
    population_files: list,
    region: str = None,
    maf_threshold: float = 0.01
):
    """
    Load and integrate all data sources.

    Args:
        pipeline_output_dir: Directory with haploblock pipeline outputs
        vcf_path: Path to VCF/BCF file
        population_files: List of population TSV files
        region: Optional genomic region (e.g., "chr6:25000000-35000000")
        maf_threshold: Minimum minor allele frequency

    Returns:
        Dictionary with integrated dataset
    """
    logger.info("=" * 60)
    logger.info("Step 1: Load Haploblock Features and Labels")
    logger.info("=" * 60)

    # Initialize loader
    loader = HaploblockMLDataLoader(pipeline_output_dir)

    # Create ML dataset from haploblock features
    dataset = loader.create_ml_dataset(population_files=population_files)

    logger.info(f"Haploblock features: {dataset['features'].shape}")
    logger.info(f"Labeled samples: {len(dataset['label_map'])}")
    logger.info(f"Populations: {dataset['metadata']['n_populations']}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: Load Genotypes from VCF")
    logger.info("=" * 60)

    # Load VCF as sparse matrix
    sparse_matrix, sample_ids, snp_info = SparseGenotypeMatrix.from_vcf(
        vcf_path=vcf_path,
        region=region,
        maf_threshold=maf_threshold
    )

    logger.info(f"Genotype matrix: {sparse_matrix.shape}")
    logger.info(f"Variants: {len(snp_info)}")
    logger.info(f"Samples in VCF: {len(sample_ids)}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: Align Samples Across Datasets")
    logger.info("=" * 60)

    # Find common samples
    haploblock_samples = set(dataset['features'].index)
    vcf_samples = set(sample_ids)
    labeled_samples = set(dataset['label_map'].keys())

    common_samples = haploblock_samples & vcf_samples & labeled_samples
    logger.info(f"Common samples across all datasets: {len(common_samples)}")

    if len(common_samples) == 0:
        raise ValueError("No common samples found across datasets!")

    # Sort for reproducibility
    common_samples = sorted(common_samples)

    # Extract and align data
    # Haploblock features
    X_haploblock = dataset['features'].loc[common_samples].values

    # Genotypes
    sample_to_idx = {s: i for i, s in enumerate(sample_ids)}
    vcf_indices = [sample_to_idx[s] for s in common_samples]
    X_genotypes = sparse_matrix[vcf_indices, :]

    # Labels
    y = np.array([dataset['label_map'][s] for s in common_samples])

    logger.info(f"Aligned haploblock features: {X_haploblock.shape}")
    logger.info(f"Aligned genotypes: {X_genotypes.shape}")
    logger.info(f"Labels: {y.shape}")

    # Count samples per population
    from collections import Counter
    pop_counts = Counter(y)
    for pop_id, count in sorted(pop_counts.items()):
        logger.info(f"  Population {pop_id}: {count} samples")

    return {
        'X_haploblock': X_haploblock,
        'X_genotypes': X_genotypes,
        'y': y,
        'sample_ids': common_samples,
        'snp_info': snp_info,
        'feature_names': dataset['features'].columns.tolist(),
    }


def prepare_for_xgboost(integrated_data, use_genotypes=True, use_haploblock=True):
    """
    Prepare data for XGBoost training.

    Args:
        integrated_data: Output from load_and_integrate_data
        use_genotypes: Include genotype features
        use_haploblock: Include haploblock features

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 4: Prepare Data for XGBoost")
    logger.info("=" * 60)

    import scipy.sparse as sp
    from sklearn.model_selection import train_test_split

    # Combine features based on flags
    features_list = []
    feature_names = []

    if use_genotypes:
        features_list.append(integrated_data['X_genotypes'])
        feature_names.extend([f"snp_{i}" for i in range(integrated_data['X_genotypes'].shape[1])])
        logger.info(f"Using genotype features: {integrated_data['X_genotypes'].shape[1]}")

    if use_haploblock:
        # Convert haploblock features to sparse
        haploblock_sparse = sp.csr_matrix(integrated_data['X_haploblock'])
        features_list.append(haploblock_sparse)
        feature_names.extend(integrated_data['feature_names'])
        logger.info(f"Using haploblock features: {integrated_data['X_haploblock'].shape[1]}")

    # Combine horizontally
    if len(features_list) > 1:
        X = sp.hstack(features_list, format='csr')
    else:
        X = features_list[0]

    y = integrated_data['y']

    logger.info(f"Combined feature matrix: {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, feature_names


def train_xgboost_classifier(X_train, X_test, y_train, y_test):
    """
    Train XGBoost classifier.

    Args:
        X_train, X_test: Feature matrices (sparse)
        y_train, y_test: Target labels

    Returns:
        Trained model and evaluation metrics
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 5: Train XGBoost Classifier")
    logger.info("=" * 60)

    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        logger.error("XGBoost not installed. Install with: pip install xgboost scikit-learn")
        return None, None

    # Create DMatrix
    dtrain = SparseGenotypeMatrix.to_xgboost_dmatrix(X_train, y_train)
    dtest = SparseGenotypeMatrix.to_xgboost_dmatrix(X_test, y_test)

    # Set parameters
    n_classes = len(np.unique(y_train))
    params = {
        'objective': 'multi:softmax',
        'num_class': n_classes,
        'max_depth': 6,
        'eta': 0.3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
    }

    logger.info(f"Training XGBoost with {n_classes} classes")
    logger.info(f"Parameters: {params}")

    # Train model
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        verbose_eval=10
    )

    # Evaluate
    y_pred = model.predict(dtest)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info("")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("")
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))

    return model, {'accuracy': accuracy}


def main_example_workflow():
    """
    Main workflow example (template - adjust paths).
    """
    # ADJUST THESE PATHS TO YOUR DATA
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"

    # Configuration
    config = {
        'pipeline_output_dir': project_root / "output" / "pipeline",
        'vcf_path': data_dir / "chr6.vcf.gz",  # Adjust to your VCF
        'population_files': [
            data_dir / "igsr-chb.tsv.tsv",
            data_dir / "igsr-gbr.tsv.tsv",
            data_dir / "igsr-pur.tsv.tsv",
        ],
        'region': "chr6:25000000-35000000",  # MHC region
        'maf_threshold': 0.05,
    }

    # Verify files exist
    existing_pop_files = [f for f in config['population_files'] if f.exists()]

    if not existing_pop_files:
        logger.error("No population files found. Please adjust paths.")
        return

    if not config['vcf_path'].exists():
        logger.warning(f"VCF file not found: {config['vcf_path']}")
        logger.warning("This example requires a VCF file. Exiting.")
        return

    if not config['pipeline_output_dir'].exists():
        logger.warning(f"Pipeline output directory not found: {config['pipeline_output_dir']}")
        logger.warning("Please run haploblock pipeline first or adjust path.")
        return

    # Run workflow
    try:
        # Step 1-3: Load and integrate data
        integrated_data = load_and_integrate_data(
            pipeline_output_dir=config['pipeline_output_dir'],
            vcf_path=config['vcf_path'],
            population_files=existing_pop_files,
            region=config['region'],
            maf_threshold=config['maf_threshold']
        )

        # Step 4: Prepare for ML
        X_train, X_test, y_train, y_test, feature_names = prepare_for_xgboost(
            integrated_data,
            use_genotypes=True,
            use_haploblock=True
        )

        # Step 5: Train model
        model, metrics = train_xgboost_classifier(
            X_train, X_test, y_train, y_test
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("Workflow completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)


if __name__ == "__main__":
    logger.info("Complete Integration Example")
    logger.info("")

    # Run the example workflow
    main_example_workflow()
