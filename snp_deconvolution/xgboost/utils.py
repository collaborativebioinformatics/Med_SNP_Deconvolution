"""
Utility functions for XGBoost SNP analysis.

Helper functions for data preprocessing, evaluation, and visualization
specific to genomic SNP classification tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# Configure logging
logger = logging.getLogger(__name__)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Comprehensive evaluation of classification results.

    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes), optional
        class_names: List of class names for reporting

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Per-class precision
            - recall: Per-class recall
            - f1: Per-class F1 score
            - confusion_matrix: Confusion matrix
            - classification_report: Full text report
            - auc: AUC score (if probabilities provided and binary)
    """
    n_classes = len(np.unique(y_true))

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
    }

    # Compute AUC if probabilities provided
    if y_proba is not None:
        if n_classes == 2:
            # Binary classification
            try:
                auc = roc_auc_score(y_true, y_proba[:, 1])
                results['auc'] = auc
            except ValueError as e:
                logger.warning(f"Could not compute AUC: {e}")
        else:
            # Multi-class (one-vs-rest)
            try:
                auc_ovr = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                )
                results['auc_ovr'] = auc_ovr
            except ValueError as e:
                logger.warning(f"Could not compute multi-class AUC: {e}")

    logger.info(f"Evaluation: Accuracy={accuracy:.4f}, Macro F1={np.mean(f1):.4f}")

    return results


def print_evaluation_report(
    results: Dict[str, Union[float, np.ndarray]],
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Pretty print evaluation results.

    Args:
        results: Results dictionary from evaluate_classification
        class_names: List of class names
    """
    print("\n" + "=" * 80)
    print("CLASSIFICATION EVALUATION REPORT")
    print("=" * 80)

    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")

    if 'auc' in results:
        print(f"AUC: {results['auc']:.4f}")
    if 'auc_ovr' in results:
        print(f"AUC (One-vs-Rest): {results['auc_ovr']:.4f}")

    print("\nPer-Class Metrics:")
    print("-" * 80)

    if class_names is None:
        n_classes = len(results['precision'])
        class_names = [f"Class {i}" for i in range(n_classes)]

    for i, name in enumerate(class_names):
        print(f"{name:>15} | Precision: {results['precision'][i]:.4f} | "
              f"Recall: {results['recall'][i]:.4f} | F1: {results['f1'][i]:.4f} | "
              f"Support: {results['support'][i]}")

    print("\nConfusion Matrix:")
    print("-" * 80)
    print(results['confusion_matrix'])

    print("\nDetailed Report:")
    print("-" * 80)
    print(results['classification_report'])
    print("=" * 80 + "\n")


def save_feature_importance(
    importance_dict: Dict[int, float],
    output_path: Union[str, Path],
    snp_names: Optional[Dict[int, str]] = None,
    top_k: Optional[int] = None,
) -> None:
    """
    Save feature importance scores to file.

    Args:
        importance_dict: Dictionary mapping SNP index to importance score
        output_path: Path to save file
        snp_names: Optional mapping from SNP index to SNP name/ID
        top_k: Save only top K features (None for all)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by importance
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        sorted_importance = sorted_importance[:top_k]

    with open(output_path, 'w') as f:
        f.write("Rank\tSNP_Index\tSNP_Name\tImportance_Score\n")
        for rank, (idx, score) in enumerate(sorted_importance, 1):
            snp_name = snp_names.get(idx, f"SNP_{idx}") if snp_names else f"SNP_{idx}"
            f.write(f"{rank}\t{idx}\t{snp_name}\t{score:.6f}\n")

    logger.info(f"Saved {len(sorted_importance)} feature importance scores to {output_path}")


def load_feature_importance(
    input_path: Union[str, Path],
) -> Tuple[Dict[int, float], Dict[int, str]]:
    """
    Load feature importance scores from file.

    Args:
        input_path: Path to importance file

    Returns:
        (importance_dict, snp_names_dict): Tuple of:
            - importance_dict: SNP index to importance score
            - snp_names_dict: SNP index to SNP name
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Importance file not found: {input_path}")

    importance_dict = {}
    snp_names_dict = {}

    with open(input_path, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                rank, idx_str, snp_name, score_str = parts[:4]
                idx = int(idx_str)
                score = float(score_str)
                importance_dict[idx] = score
                snp_names_dict[idx] = snp_name

    logger.info(f"Loaded {len(importance_dict)} feature importance scores from {input_path}")

    return importance_dict, snp_names_dict


def filter_sparse_matrix_by_features(
    X: sp.csr_matrix,
    feature_indices: np.ndarray,
) -> sp.csr_matrix:
    """
    Filter sparse matrix to keep only specified features.

    Args:
        X: Input sparse matrix (n_samples, n_features)
        feature_indices: Indices of features to keep

    Returns:
        Filtered sparse matrix (n_samples, len(feature_indices))
    """
    if not sp.issparse(X):
        raise ValueError("X must be a sparse matrix")

    if not isinstance(feature_indices, np.ndarray):
        feature_indices = np.array(feature_indices)

    # Validate indices
    if np.any(feature_indices < 0) or np.any(feature_indices >= X.shape[1]):
        raise ValueError(f"Feature indices out of range [0, {X.shape[1]})")

    X_filtered = X[:, feature_indices]

    logger.info(
        f"Filtered matrix from {X.shape[1]} to {X_filtered.shape[1]} features "
        f"({X_filtered.nnz / (X_filtered.shape[0] * X_filtered.shape[1]):.4f} density)"
    )

    return X_filtered


def compute_sparsity_statistics(X: sp.csr_matrix) -> Dict[str, float]:
    """
    Compute sparsity statistics for SNP matrix.

    Args:
        X: Sparse SNP matrix

    Returns:
        Dictionary with statistics:
            - sparsity: Fraction of zero elements
            - density: Fraction of non-zero elements
            - nnz: Number of non-zero elements
            - mean_nnz_per_row: Average non-zeros per sample
            - std_nnz_per_row: Std dev of non-zeros per sample
    """
    if not sp.issparse(X):
        raise ValueError("X must be a sparse matrix")

    n_elements = X.shape[0] * X.shape[1]
    nnz = X.nnz
    density = nnz / n_elements
    sparsity = 1.0 - density

    # Per-row statistics
    nnz_per_row = np.diff(X.indptr)
    mean_nnz = np.mean(nnz_per_row)
    std_nnz = np.std(nnz_per_row)

    stats = {
        'sparsity': sparsity,
        'density': density,
        'nnz': nnz,
        'mean_nnz_per_row': mean_nnz,
        'std_nnz_per_row': std_nnz,
    }

    logger.info(
        f"Sparsity stats: {sparsity:.4f} sparsity, "
        f"{mean_nnz:.1f} ± {std_nnz:.1f} non-zeros per sample"
    )

    return stats


def create_stratified_split(
    X: sp.csr_matrix,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Create stratified train/test split.

    Args:
        X: Feature matrix
        y: Labels
        test_size: Fraction of data for test set
        random_state: Random seed

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info(
        f"Split data: train={X_train.shape[0]} samples, test={X_test.shape[0]} samples"
    )

    return X_train, X_test, y_train, y_test


def get_top_k_indices(
    importance_dict: Dict[int, float],
    k: int,
) -> np.ndarray:
    """
    Get indices of top K features by importance.

    Args:
        importance_dict: Dictionary mapping feature index to importance
        k: Number of top features to return

    Returns:
        Array of top K feature indices
    """
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_k_items = sorted_items[:k]
    return np.array([idx for idx, score in top_k_items])


def encode_populations(
    population_labels: List[str],
    population_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode population labels to integers.

    Args:
        population_labels: List of population labels (e.g., ['CHB', 'GBR', 'PUR'])
        population_mapping: Optional existing mapping to use

    Returns:
        (encoded_labels, mapping): Tuple of:
            - encoded_labels: Integer array
            - mapping: Dictionary mapping population to integer
    """
    if population_mapping is None:
        unique_pops = sorted(set(population_labels))
        population_mapping = {pop: idx for idx, pop in enumerate(unique_pops)}

    encoded_labels = np.array([population_mapping[pop] for pop in population_labels])

    logger.info(f"Encoded {len(population_labels)} samples with {len(population_mapping)} populations")

    return encoded_labels, population_mapping


def decode_populations(
    encoded_labels: np.ndarray,
    population_mapping: Dict[str, int],
) -> List[str]:
    """
    Decode integer labels back to population names.

    Args:
        encoded_labels: Integer array of labels
        population_mapping: Mapping from population to integer

    Returns:
        List of population names
    """
    # Reverse mapping
    reverse_mapping = {v: k for k, v in population_mapping.items()}

    decoded_labels = [reverse_mapping[label] for label in encoded_labels]

    return decoded_labels


def merge_importance_scores(
    importance_dicts: List[Dict[int, float]],
    method: str = 'mean',
) -> Dict[int, float]:
    """
    Merge multiple importance score dictionaries.

    Useful for ensemble models or cross-validation folds.

    Args:
        importance_dicts: List of importance dictionaries
        method: Merge method ('mean', 'median', 'max', 'min', 'sum')

    Returns:
        Merged importance dictionary
    """
    if not importance_dicts:
        raise ValueError("importance_dicts cannot be empty")

    valid_methods = {'mean', 'median', 'max', 'min', 'sum'}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}")

    # Collect all feature indices
    all_indices = set()
    for d in importance_dicts:
        all_indices.update(d.keys())

    # Merge scores
    merged = {}
    for idx in all_indices:
        scores = [d.get(idx, 0.0) for d in importance_dicts]

        if method == 'mean':
            merged[idx] = np.mean(scores)
        elif method == 'median':
            merged[idx] = np.median(scores)
        elif method == 'max':
            merged[idx] = np.max(scores)
        elif method == 'min':
            merged[idx] = np.min(scores)
        elif method == 'sum':
            merged[idx] = np.sum(scores)

    logger.info(f"Merged {len(importance_dicts)} importance dicts using {method} method")

    return merged


def compute_class_weights(
    y: np.ndarray,
    method: str = 'balanced',
) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.

    Args:
        y: Label array
        method: Weighting method ('balanced', 'inverse')

    Returns:
        Dictionary mapping class to weight
    """
    from collections import Counter

    class_counts = Counter(y)
    n_samples = len(y)
    n_classes = len(class_counts)

    weights = {}
    for class_idx, count in class_counts.items():
        if method == 'balanced':
            # sklearn-style balanced weights
            weights[class_idx] = n_samples / (n_classes * count)
        elif method == 'inverse':
            # Simple inverse frequency
            weights[class_idx] = 1.0 / count
        else:
            raise ValueError(f"Unknown method: {method}")

    logger.info(f"Computed class weights: {weights}")

    return weights


def log_dataset_info(
    X: sp.csr_matrix,
    y: np.ndarray,
    name: str = "Dataset",
) -> None:
    """
    Log comprehensive dataset information.

    Args:
        X: Feature matrix
        y: Labels
        name: Dataset name for logging
    """
    from collections import Counter

    logger.info(f"\n{'=' * 80}")
    logger.info(f"{name} Information")
    logger.info(f"{'=' * 80}")
    logger.info(f"Samples: {X.shape[0]}")
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Sparsity: {1.0 - X.nnz / (X.shape[0] * X.shape[1]):.4f}")
    logger.info(f"Memory usage: {X.data.nbytes + X.indices.nbytes + X.indptr.nbytes / 1e6:.2f} MB")

    class_counts = Counter(y)
    logger.info(f"Classes: {len(class_counts)}")
    for class_idx, count in sorted(class_counts.items()):
        logger.info(f"  Class {class_idx}: {count} samples ({count / len(y) * 100:.1f}%)")

    logger.info(f"{'=' * 80}\n")


__all__ = [
    'evaluate_classification',
    'print_evaluation_report',
    'save_feature_importance',
    'load_feature_importance',
    'filter_sparse_matrix_by_features',
    'compute_sparsity_statistics',
    'create_stratified_split',
    'get_top_k_indices',
    'encode_populations',
    'decode_populations',
    'merge_importance_scores',
    'compute_class_weights',
    'log_dataset_info',
]
