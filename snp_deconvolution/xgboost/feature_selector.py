"""
Iterative feature selection for SNP analysis using XGBoost importance.

This module implements an iterative feature selection strategy that progressively
reduces the feature space by selecting the most informative SNPs based on
XGBoost importance scores, optimizing for both performance and interpretability.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score

from .xgb_trainer import XGBoostSNPTrainer

# Configure logging
logger = logging.getLogger(__name__)


class IterativeSNPSelector:
    """
    Iterative feature selection using XGBoost importance.

    Strategy:
    1. Train on all SNPs
    2. Select top-K by importance
    3. Retrain on selected SNPs
    4. Repeat until convergence or stopping criteria met

    This approach progressively refines the SNP set, removing noise and
    focusing on the most discriminative markers for population classification.

    Attributes:
        selected_features_: Indices of selected features after fitting
        importance_scores_: Final importance scores for selected features
        selection_history_: History of each iteration (n_features, metrics)
        best_iteration_: Iteration with best validation performance

    Example:
        >>> selector = IterativeSNPSelector(
        ...     initial_k=10000,
        ...     reduction_factor=0.5,
        ...     min_snps=100
        ... )
        >>> selected_indices, scores = selector.select_features(
        ...     X_train, y_train, X_val, y_val
        ... )
        >>> X_train_selected = X_train[:, selected_indices]
    """

    def __init__(
        self,
        initial_k: int = 10000,
        reduction_factor: float = 0.5,
        min_snps: int = 100,
        max_iterations: int = 5,
        importance_type: str = 'gain',
        gpu_id: int = 0,
        xgb_params: Optional[Dict] = None,
        convergence_threshold: float = 0.001,
        metric: str = 'accuracy',
    ):
        """
        Initialize iterative SNP selector.

        Args:
            initial_k: Initial number of top SNPs to select (if fewer than total features)
            reduction_factor: Factor to reduce features by each iteration (0.0-1.0)
            min_snps: Minimum number of SNPs to retain
            max_iterations: Maximum number of iterations
            importance_type: XGBoost importance metric ('gain', 'weight', 'cover')
            gpu_id: GPU device ID for XGBoost training
            xgb_params: Optional custom XGBoost parameters (merged with defaults)
            convergence_threshold: Stop if improvement < threshold
            metric: Metric to optimize ('accuracy', 'f1_macro', 'f1_weighted')

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0.0 < reduction_factor < 1.0:
            raise ValueError("reduction_factor must be between 0.0 and 1.0")

        if initial_k < min_snps:
            raise ValueError("initial_k must be >= min_snps")

        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

        valid_metrics = {'accuracy', 'f1_macro', 'f1_weighted'}
        if metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")

        self.initial_k = initial_k
        self.reduction_factor = reduction_factor
        self.min_snps = min_snps
        self.max_iterations = max_iterations
        self.importance_type = importance_type
        self.gpu_id = gpu_id
        self.xgb_params = xgb_params or {}
        self.convergence_threshold = convergence_threshold
        self.metric = metric

        # State variables
        self.selected_features_: Optional[np.ndarray] = None
        self.importance_scores_: Optional[Dict[int, float]] = None
        self.selection_history_: List[Dict] = []
        self.best_iteration_: Optional[int] = None
        self._best_score: float = -np.inf

        logger.info(
            f"Initialized IterativeSNPSelector: initial_k={initial_k}, "
            f"reduction_factor={reduction_factor}, min_snps={min_snps}, "
            f"max_iterations={max_iterations}, metric={metric}"
        )

    def _compute_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Compute evaluation metric.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Metric score
        """
        if self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif self.metric == 'f1_weighted':
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _train_and_evaluate(
        self,
        X_train: sp.csr_matrix,
        y_train: np.ndarray,
        X_val: Optional[sp.csr_matrix],
        y_val: Optional[np.ndarray],
        feature_indices: np.ndarray,
    ) -> Tuple[XGBoostSNPTrainer, float, Optional[float]]:
        """
        Train model and evaluate performance.

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            feature_indices: Indices of features to use

        Returns:
            (trained_model, train_score, val_score)
        """
        # Select features
        X_train_subset = X_train[:, feature_indices]

        if X_val is not None:
            X_val_subset = X_val[:, feature_indices]
        else:
            X_val_subset = None

        # Setup XGBoost parameters
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'early_stopping_rounds': 30,
            'gpu_id': self.gpu_id,
        }
        default_params.update(self.xgb_params)

        # Train model
        trainer = XGBoostSNPTrainer(**default_params)

        feature_names = [f"SNP_{idx}" for idx in feature_indices]
        trainer.fit(
            X_train_subset,
            y_train,
            X_val_subset,
            y_val,
            verbose=False,
            feature_names=feature_names,
        )

        # Evaluate
        y_train_pred = trainer.predict(X_train_subset)
        train_score = self._compute_metric(y_train, y_train_pred)

        val_score = None
        if X_val is not None and y_val is not None:
            y_val_pred = trainer.predict(X_val_subset)
            val_score = self._compute_metric(y_val, y_val_pred)

        return trainer, train_score, val_score

    def select_features(
        self,
        X: sp.csr_matrix,
        y: np.ndarray,
        X_val: Optional[sp.csr_matrix] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[int, float]]:
        """
        Perform iterative feature selection.

        Args:
            X: Training data sparse matrix (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Optional validation data for performance monitoring
            y_val: Optional validation labels

        Returns:
            (selected_snp_indices, importance_scores): Tuple of:
                - selected_snp_indices: Array of selected SNP indices
                - importance_scores: Dict mapping SNP index to importance score

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if not sp.issparse(X):
            raise ValueError("X must be a sparse matrix (scipy.sparse.csr_matrix)")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have inconsistent shapes: X={X.shape}, y={y.shape}")

        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    f"X_val and y_val have inconsistent shapes: "
                    f"X_val={X_val.shape}, y_val={y_val.shape}"
                )
            if X_val.shape[1] != X.shape[1]:
                raise ValueError(
                    f"X_val must have same number of features as X: "
                    f"X_val.shape[1]={X_val.shape[1]}, X.shape[1]={X.shape[1]}"
                )

        n_total_features = X.shape[1]
        logger.info(
            f"Starting iterative feature selection with {n_total_features} total features"
        )

        # Initialize with all features or subset
        current_features = np.arange(n_total_features)
        if n_total_features > self.initial_k:
            logger.info(f"Initial subset: selecting top {self.initial_k} features")
            # Do initial training to get importance
            trainer, train_score, val_score = self._train_and_evaluate(
                X, y, X_val, y_val, current_features
            )
            importance = trainer.get_feature_importance(
                importance_type=self.importance_type,
                top_k=self.initial_k,
            )
            current_features = np.array(list(importance.keys()))
            logger.info(f"Selected {len(current_features)} features after initial pass")

        # Reset history
        self.selection_history_ = []
        self._best_score = -np.inf
        self.best_iteration_ = 0

        # Iterative selection
        for iteration in range(self.max_iterations):
            logger.info(
                f"\n=== Iteration {iteration + 1}/{self.max_iterations} ==="
            )
            logger.info(f"Current feature count: {len(current_features)}")

            # Train and evaluate
            try:
                trainer, train_score, val_score = self._train_and_evaluate(
                    X, y, X_val, y_val, current_features
                )
            except Exception as e:
                logger.error(f"Training failed at iteration {iteration + 1}: {e}")
                break

            # Use validation score if available, otherwise training score
            current_score = val_score if val_score is not None else train_score

            # Log scores
            score_str = f"train_{self.metric}={train_score:.4f}"
            if val_score is not None:
                score_str += f", val_{self.metric}={val_score:.4f}"
            logger.info(score_str)

            # Store iteration history
            iteration_info = {
                'iteration': iteration + 1,
                'n_features': len(current_features),
                'train_score': train_score,
                'val_score': val_score,
                'selected_features': current_features.copy(),
            }
            self.selection_history_.append(iteration_info)

            # Check if this is the best iteration
            if current_score > self._best_score:
                improvement = current_score - self._best_score
                logger.info(
                    f"New best score: {current_score:.4f} "
                    f"(improvement: {improvement:.4f})"
                )
                self._best_score = current_score
                self.best_iteration_ = iteration
                self.selected_features_ = current_features.copy()
                self.importance_scores_ = trainer.get_feature_importance(
                    importance_type=self.importance_type
                )

                # Check convergence
                if iteration > 0 and improvement < self.convergence_threshold:
                    logger.info(
                        f"Converged: improvement {improvement:.4f} < "
                        f"threshold {self.convergence_threshold}"
                    )
                    break
            else:
                logger.info(
                    f"No improvement: current={current_score:.4f}, "
                    f"best={self._best_score:.4f}"
                )

            # Check if we've reached minimum
            if len(current_features) <= self.min_snps:
                logger.info(f"Reached minimum SNPs: {self.min_snps}")
                break

            # Calculate next feature count
            next_n_features = int(len(current_features) * self.reduction_factor)
            next_n_features = max(next_n_features, self.min_snps)

            if next_n_features >= len(current_features):
                logger.info("Cannot reduce features further")
                break

            # Select top features for next iteration
            logger.info(
                f"Selecting top {next_n_features} features "
                f"(reduction factor: {self.reduction_factor})"
            )

            importance = trainer.get_feature_importance(
                importance_type=self.importance_type,
                top_k=next_n_features,
            )

            # Map back to original indices
            selected_local_indices = np.array(list(importance.keys()))
            current_features = current_features[selected_local_indices]

        # Final logging
        logger.info("\n=== Feature Selection Complete ===")
        logger.info(
            f"Best iteration: {self.best_iteration_ + 1} "
            f"with {len(self.selected_features_)} features"
        )
        logger.info(f"Best {self.metric}: {self._best_score:.4f}")

        if self.selected_features_ is None or self.importance_scores_ is None:
            raise RuntimeError("Feature selection failed - no valid iteration found")

        # Map importance scores back to original indices
        mapped_importance = {}
        for local_idx, score in self.importance_scores_.items():
            original_idx = self.selected_features_[local_idx]
            mapped_importance[original_idx] = score

        return self.selected_features_, mapped_importance

    def get_selection_history(self) -> List[Dict]:
        """
        Get history of each iteration.

        Returns:
            List of dictionaries containing:
                - iteration: Iteration number
                - n_features: Number of features
                - train_score: Training score
                - val_score: Validation score (or None)
                - selected_features: Array of selected feature indices
        """
        return self.selection_history_

    def get_best_iteration_info(self) -> Dict:
        """
        Get information about the best iteration.

        Returns:
            Dictionary with best iteration details

        Raises:
            ValueError: If selection has not been performed
        """
        if self.best_iteration_ is None or not self.selection_history_:
            raise ValueError("No selection history available")

        return self.selection_history_[self.best_iteration_]

    def plot_selection_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot feature selection curve showing performance vs number of features.

        Args:
            save_path: Optional path to save plot

        Raises:
            ValueError: If selection has not been performed
            ImportError: If matplotlib is not available
        """
        if not self.selection_history_:
            raise ValueError("No selection history available")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        iterations = [h['iteration'] for h in self.selection_history_]
        n_features = [h['n_features'] for h in self.selection_history_]
        train_scores = [h['train_score'] for h in self.selection_history_]
        val_scores = [h['val_score'] for h in self.selection_history_ if h['val_score'] is not None]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Score vs Iteration
        ax1.plot(iterations, train_scores, 'o-', label='Train', linewidth=2, markersize=8)
        if val_scores:
            ax1.plot(iterations[:len(val_scores)], val_scores, 's-', label='Validation',
                    linewidth=2, markersize=8)

        # Mark best iteration
        best_iter = self.best_iteration_ + 1
        best_val_score = (self.selection_history_[self.best_iteration_]['val_score']
                         if self.selection_history_[self.best_iteration_]['val_score'] is not None
                         else self.selection_history_[self.best_iteration_]['train_score'])

        ax1.axvline(best_iter, color='red', linestyle='--', alpha=0.7, label='Best')
        ax1.scatter([best_iter], [best_val_score], color='red', s=200, zorder=5, marker='*')

        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel(f'{self.metric.capitalize()}', fontsize=12)
        ax1.set_title('Feature Selection: Score vs Iteration', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Score vs Number of Features
        ax2.plot(n_features, train_scores, 'o-', label='Train', linewidth=2, markersize=8)
        if val_scores:
            ax2.plot(n_features[:len(val_scores)], val_scores, 's-', label='Validation',
                    linewidth=2, markersize=8)

        best_n_features = self.selection_history_[self.best_iteration_]['n_features']
        ax2.axvline(best_n_features, color='red', linestyle='--', alpha=0.7, label='Best')
        ax2.scatter([best_n_features], [best_val_score], color='red', s=200, zorder=5, marker='*')

        ax2.set_xlabel('Number of Features', fontsize=12)
        ax2.set_ylabel(f'{self.metric.capitalize()}', fontsize=12)
        ax2.set_title('Feature Selection: Score vs N Features', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved selection curve to {save_path}")

        plt.show()

    def __repr__(self) -> str:
        """String representation of the selector."""
        if self.selected_features_ is not None:
            status = f"fitted, {len(self.selected_features_)} features selected"
        else:
            status = "not fitted"

        return (
            f"IterativeSNPSelector(initial_k={self.initial_k}, "
            f"reduction_factor={self.reduction_factor}, "
            f"min_snps={self.min_snps}, {status})"
        )
