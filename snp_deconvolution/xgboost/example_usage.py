"""
Example usage of XGBoost GPU module for SNP deconvolution.

This script demonstrates how to use the XGBoostSNPTrainer and IterativeSNPSelector
for population classification from genomic data.
"""

import logging
import numpy as np
import scipy.sparse as sp
from pathlib import Path

from snp_deconvolution.xgboost import XGBoostSNPTrainer, IterativeSNPSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 50000,
    n_informative: int = 100,
    n_classes: int = 3,
    random_state: int = 42,
) -> tuple:
    """
    Generate synthetic sparse SNP data for demonstration.

    Args:
        n_samples: Number of samples
        n_features: Total number of SNP features
        n_informative: Number of informative SNPs
        n_classes: Number of population classes
        random_state: Random seed

    Returns:
        (X, y): Sparse feature matrix and labels
    """
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features")

    np.random.seed(random_state)

    # Generate sparse SNP matrix (mostly 0s with some 1s and 2s)
    density = 0.1  # 10% non-zero
    X = sp.random(n_samples, n_features, density=density, format='csr', random_state=random_state)
    X.data = np.random.choice([0, 1, 2], size=X.data.shape, p=[0.5, 0.3, 0.2])

    # Generate labels
    y = np.random.randint(0, n_classes, size=n_samples)

    # Make some features informative by correlating with labels
    for i in range(n_informative):
        feature_idx = np.random.randint(0, n_features)
        for class_idx in range(n_classes):
            mask = y == class_idx
            # Increase SNP values for this class
            X[mask, feature_idx] = np.random.choice([1, 2], size=mask.sum())

    logger.info(f"Generated data shape: {X.shape}, sparsity: {X.nnz / (X.shape[0] * X.shape[1]):.4f}")

    return X, y


def example_basic_training():
    """Example 1: Basic XGBoost training with GPU acceleration."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 1: Basic XGBoost Training")
    logger.info("=" * 80)

    # Generate synthetic data
    X_train, y_train = generate_synthetic_data(n_samples=800, n_features=10000)
    X_val, y_val = generate_synthetic_data(n_samples=200, n_features=10000)

    # Initialize trainer
    trainer = XGBoostSNPTrainer(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        gpu_id=0,
        early_stopping_rounds=10,
        num_class=3,
    )

    # Train
    logger.info("Training model...")
    trainer.fit(X_train, y_train, X_val, y_val, verbose=True)

    # Predict
    logger.info("Making predictions...")
    y_pred = trainer.predict(X_val)
    y_proba = trainer.predict_proba(X_val)

    # Evaluate
    accuracy = (y_pred == y_val).mean()
    logger.info(f"Validation accuracy: {accuracy:.4f}")
    logger.info(f"Prediction probabilities shape: {y_proba.shape}")

    # Get feature importance
    logger.info("Getting feature importance...")
    importance = trainer.get_feature_importance(importance_type='gain', top_k=10)
    logger.info(f"Top 10 SNPs by importance:")
    for snp_idx, score in list(importance.items())[:10]:
        logger.info(f"  SNP_{snp_idx}: {score:.4f}")

    # Save model
    model_path = Path("/tmp/xgboost_snp_model")
    logger.info(f"Saving model to {model_path}...")
    trainer.save_model(model_path)

    # Load model
    logger.info("Loading model...")
    new_trainer = XGBoostSNPTrainer()
    new_trainer.load_model(model_path)
    logger.info("Model loaded successfully")

    # Export for NVFlare
    logger.info("Exporting for NVFlare...")
    nvflare_dict = trainer.export_for_nvflare()
    logger.info(f"Exported model type: {nvflare_dict['model_type']}")
    logger.info(f"Exported features: {nvflare_dict['n_features']}")


def example_feature_selection():
    """Example 2: Iterative feature selection."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: Iterative Feature Selection")
    logger.info("=" * 80)

    # Generate synthetic data with many features
    X_train, y_train = generate_synthetic_data(n_samples=800, n_features=50000, n_informative=100)
    X_val, y_val = generate_synthetic_data(n_samples=200, n_features=50000, n_informative=100)

    # Initialize selector
    selector = IterativeSNPSelector(
        initial_k=10000,
        reduction_factor=0.5,
        min_snps=100,
        max_iterations=5,
        importance_type='gain',
        gpu_id=0,
        metric='accuracy',
    )

    # Perform selection
    logger.info("Performing iterative feature selection...")
    selected_indices, importance_scores = selector.select_features(
        X_train, y_train, X_val, y_val
    )

    logger.info(f"Selected {len(selected_indices)} SNPs")
    logger.info(f"Top 10 selected SNPs:")
    for snp_idx in selected_indices[:10]:
        score = importance_scores.get(snp_idx, 0.0)
        logger.info(f"  SNP_{snp_idx}: {score:.4f}")

    # Get selection history
    history = selector.get_selection_history()
    logger.info(f"\nSelection history ({len(history)} iterations):")
    for h in history:
        logger.info(
            f"  Iteration {h['iteration']}: {h['n_features']} features, "
            f"train={h['train_score']:.4f}, val={h['val_score']:.4f}"
        )

    best_info = selector.get_best_iteration_info()
    logger.info(f"\nBest iteration: {best_info['iteration']}")
    logger.info(f"  Features: {best_info['n_features']}")
    logger.info(f"  Train score: {best_info['train_score']:.4f}")
    logger.info(f"  Val score: {best_info['val_score']:.4f}")

    # Use selected features for final training
    logger.info("\nTraining final model with selected features...")
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]

    final_trainer = XGBoostSNPTrainer(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        gpu_id=0,
        num_class=3,
    )
    final_trainer.fit(X_train_selected, y_train, X_val_selected, y_val)

    y_pred = final_trainer.predict(X_val_selected)
    final_accuracy = (y_pred == y_val).mean()
    logger.info(f"Final model accuracy with selected features: {final_accuracy:.4f}")


def example_custom_parameters():
    """Example 3: Custom XGBoost parameters for different scenarios."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: Custom Parameters")
    logger.info("=" * 80)

    X_train, y_train = generate_synthetic_data(n_samples=800, n_features=10000)
    X_val, y_val = generate_synthetic_data(n_samples=200, n_features=10000)

    # Scenario 1: High regularization for preventing overfitting
    logger.info("\nScenario 1: High regularization")
    trainer_reg = XGBoostSNPTrainer(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        gpu_id=0,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=10.0,
        min_child_weight=5,
    )
    trainer_reg.fit(X_train, y_train, X_val, y_val, verbose=False)
    logger.info(f"Best iteration: {trainer_reg._best_iteration}")
    logger.info(f"Best score: {trainer_reg.model.best_score:.4f}")

    # Scenario 2: Deeper trees for complex patterns
    logger.info("\nScenario 2: Deeper trees")
    trainer_deep = XGBoostSNPTrainer(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        gpu_id=0,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
    )
    trainer_deep.fit(X_train, y_train, X_val, y_val, verbose=False)
    logger.info(f"Best iteration: {trainer_deep._best_iteration}")
    logger.info(f"Best score: {trainer_deep.model.best_score:.4f}")

    # Scenario 3: Fast training with fewer trees
    logger.info("\nScenario 3: Fast training")
    trainer_fast = XGBoostSNPTrainer(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.3,
        gpu_id=0,
        early_stopping_rounds=5,
    )
    trainer_fast.fit(X_train, y_train, X_val, y_val, verbose=False)
    logger.info(f"Best iteration: {trainer_fast._best_iteration}")
    logger.info(f"Best score: {trainer_fast.model.best_score:.4f}")


def example_real_world_workflow():
    """Example 4: Complete workflow for real-world SNP analysis."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: Real-World Workflow")
    logger.info("=" * 80)

    # Step 1: Load data (simulated)
    logger.info("Step 1: Loading genomic data...")
    X_train, y_train = generate_synthetic_data(
        n_samples=2000, n_features=100000, n_informative=500
    )
    X_val, y_val = generate_synthetic_data(
        n_samples=500, n_features=100000, n_informative=500
    )
    X_test, y_test = generate_synthetic_data(
        n_samples=500, n_features=100000, n_informative=500
    )

    # Step 2: Feature selection
    logger.info("\nStep 2: Performing feature selection...")
    selector = IterativeSNPSelector(
        initial_k=20000,
        reduction_factor=0.6,
        min_snps=500,
        max_iterations=4,
        importance_type='gain',
        gpu_id=0,
        xgb_params={
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.1,
        },
    )

    selected_indices, importance_scores = selector.select_features(
        X_train, y_train, X_val, y_val
    )
    logger.info(f"Selected {len(selected_indices)} informative SNPs")

    # Step 3: Train final model
    logger.info("\nStep 3: Training final optimized model...")
    X_train_selected = X_train[:, selected_indices]
    X_val_selected = X_val[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]

    final_trainer = XGBoostSNPTrainer(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        gpu_id=0,
        early_stopping_rounds=30,
        num_class=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=5.0,
    )

    final_trainer.fit(
        X_train_selected, y_train,
        X_val_selected, y_val,
        verbose=True
    )

    # Step 4: Evaluate on test set
    logger.info("\nStep 4: Evaluating on test set...")
    y_test_pred = final_trainer.predict(X_test_selected)
    y_test_proba = final_trainer.predict_proba(X_test_selected)

    test_accuracy = (y_test_pred == y_test).mean()
    logger.info(f"Test accuracy: {test_accuracy:.4f}")

    # Per-class accuracy
    for class_idx in range(3):
        mask = y_test == class_idx
        class_accuracy = (y_test_pred[mask] == y_test[mask]).mean()
        logger.info(f"  Class {class_idx} accuracy: {class_accuracy:.4f}")

    # Step 5: Save everything
    logger.info("\nStep 5: Saving model and results...")
    output_dir = Path("/tmp/snp_analysis_output")
    output_dir.mkdir(exist_ok=True)

    final_trainer.save_model(output_dir / "final_model")

    # Save selected features
    np.save(output_dir / "selected_snp_indices.npy", selected_indices)
    logger.info(f"Saved selected SNP indices to {output_dir / 'selected_snp_indices.npy'}")

    # Save importance scores
    with open(output_dir / "importance_scores.txt", 'w') as f:
        f.write("SNP_Index\tImportance_Score\n")
        for idx, score in sorted(importance_scores.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{idx}\t{score:.6f}\n")
    logger.info(f"Saved importance scores to {output_dir / 'importance_scores.txt'}")

    logger.info("\nWorkflow complete!")


if __name__ == "__main__":
    logger.info("XGBoost GPU SNP Deconvolution - Example Usage")
    logger.info("=" * 80)

    try:
        # Run examples
        example_basic_training()
        example_feature_selection()
        example_custom_parameters()
        example_real_world_workflow()

        logger.info("\n" + "=" * 80)
        logger.info("All examples completed successfully!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise
