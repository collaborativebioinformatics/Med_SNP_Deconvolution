"""
GPU-accelerated XGBoost trainer for SNP analysis.

This module provides a production-ready XGBoost implementation optimized for
GPU acceleration on A100/H100 hardware, designed specifically for high-dimensional
sparse genomic data classification tasks.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import json
import numpy as np
import scipy.sparse as sp
import xgboost as xgb

# Configure logging
logger = logging.getLogger(__name__)


class XGBoostSNPTrainer:
    """
    GPU-accelerated XGBoost for SNP analysis.

    Uses tree_method='gpu_hist' for GPU acceleration on A100/H100 GPUs.
    Optimized for high-dimensional sparse genomic data with efficient
    memory handling and early stopping.

    Attributes:
        model: Trained XGBoost model
        feature_names: List of feature names (SNP identifiers)
        classes_: Array of class labels
        n_features_in_: Number of features seen during fit

    Example:
        >>> import scipy.sparse as sp
        >>> X_train = sp.csr_matrix(...)  # Sparse SNP matrix
        >>> y_train = np.array([0, 1, 2, ...])  # Population labels
        >>>
        >>> trainer = XGBoostSNPTrainer(
        ...     n_estimators=1000,
        ...     max_depth=6,
        ...     gpu_id=0
        ... )
        >>> trainer.fit(X_train, y_train, X_val, y_val)
        >>> predictions = trainer.predict(X_test)
        >>> probabilities = trainer.predict_proba(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        gpu_id: int = 0,
        early_stopping_rounds: int = 50,
        objective: str = 'multi:softprob',
        num_class: int = 3,
        random_state: int = 42,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        max_bin: int = 512,
        scale_pos_weight: Optional[float] = None,
    ):
        """
        Initialize XGBoost with GPU parameters.

        GPU-specific configuration for A100/H100:
        - tree_method: 'gpu_hist' for GPU histogram-based training
        - predictor: 'gpu_predictor' for GPU-accelerated prediction
        - max_bin: 512 for higher precision on large GPU memory

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (6-10 recommended for genomic data)
            learning_rate: Step size shrinkage (0.01-0.3 typical range)
            gpu_id: GPU device ID (0 for first GPU)
            early_stopping_rounds: Stop if no improvement for N rounds
            objective: Loss function ('multi:softprob', 'multi:softmax', 'binary:logistic')
            num_class: Number of classes (required for multi-class)
            random_state: Random seed for reproducibility
            subsample: Row sampling ratio per tree (0.5-1.0)
            colsample_bytree: Column sampling ratio per tree (0.5-1.0)
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            min_child_weight: Minimum sum of instance weight in child
            max_bin: Maximum number of bins for histogram building
            scale_pos_weight: Balance of positive/negative weights (for binary)

        Raises:
            ImportError: If xgboost is not installed
            RuntimeError: If GPU is not available
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.gpu_id = gpu_id
        self.early_stopping_rounds = early_stopping_rounds
        self.objective = objective
        self.num_class = num_class
        self.random_state = random_state
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.max_bin = max_bin
        self.scale_pos_weight = scale_pos_weight

        # Model state
        self.model: Optional[xgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self._best_iteration: Optional[int] = None
        self._evals_result: Dict[str, Dict[str, List[float]]] = {}

        # Verify GPU availability
        self._verify_gpu_availability()

        logger.info(
            f"Initialized XGBoostSNPTrainer with GPU {gpu_id}, "
            f"n_estimators={n_estimators}, max_depth={max_depth}"
        )

    def _verify_gpu_availability(self) -> None:
        """
        Verify that GPU is available for XGBoost.

        Raises:
            RuntimeError: If GPU is not available or improperly configured
        """
        try:
            # Test GPU availability by creating a small DMatrix
            test_data = sp.csr_matrix(np.random.rand(10, 5))
            test_dmatrix = xgb.DMatrix(test_data)

            # Try to build a tree on GPU
            test_params = {
                'tree_method': 'gpu_hist',
                'gpu_id': self.gpu_id,
            }
            xgb.train(test_params, test_dmatrix, num_boost_round=1, verbose_eval=False)

            logger.info(f"GPU {self.gpu_id} is available and functional")

        except Exception as e:
            logger.error(f"GPU verification failed: {e}")
            raise RuntimeError(
                f"GPU {self.gpu_id} is not available or XGBoost GPU support "
                f"is not properly configured. Error: {e}"
            ) from e

    def _get_params(self) -> Dict[str, Any]:
        """
        Get XGBoost parameters dictionary.

        Returns:
            Dictionary of XGBoost parameters
        """
        params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'gpu_id': self.gpu_id,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'random_state': self.random_state,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'min_child_weight': self.min_child_weight,
            'max_bin': self.max_bin,
        }

        if self.num_class is not None and self.num_class > 2:
            params['num_class'] = self.num_class

        if self.scale_pos_weight is not None:
            params['scale_pos_weight'] = self.scale_pos_weight

        return params

    def fit(
        self,
        X: sp.csr_matrix,
        y: np.ndarray,
        X_val: Optional[sp.csr_matrix] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
        feature_names: Optional[List[str]] = None,
    ) -> 'XGBoostSNPTrainer':
        """
        Train XGBoost model with optional validation for early stopping.

        Args:
            X: Training data sparse matrix (n_samples, n_features)
            y: Training labels (n_samples,)
            X_val: Optional validation data for early stopping
            y_val: Optional validation labels
            verbose: Whether to print training progress
            feature_names: Optional list of feature names (SNP IDs)

        Returns:
            self: Fitted estimator

        Raises:
            ValueError: If input shapes are invalid or inconsistent
            RuntimeError: If training fails
        """
        # Validate inputs
        if not sp.issparse(X):
            raise ValueError("X must be a sparse matrix (scipy.sparse.csr_matrix)")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent shapes: X={X.shape}, y={y.shape}"
            )

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

        # Store metadata
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.feature_names = feature_names or [f"SNP_{i}" for i in range(X.shape[1])]

        logger.info(
            f"Training on {X.shape[0]} samples with {X.shape[1]} features, "
            f"{len(self.classes_)} classes"
        )

        try:
            # Create DMatrix for training
            dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)

            # Setup evaluation list and parameters
            evals = [(dtrain, 'train')]
            params = self._get_params()

            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
                evals.append((dval, 'validation'))
                logger.info(f"Using validation set with {X_val.shape[0]} samples")

            # Configure verbosity
            verbose_eval = 50 if verbose else False

            # Train model
            logger.info("Starting XGBoost training on GPU...")
            self._evals_result = {}

            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds,
                evals_result=self._evals_result,
                verbose_eval=verbose_eval,
            )

            self._best_iteration = self.model.best_iteration

            logger.info(
                f"Training completed. Best iteration: {self._best_iteration}, "
                f"Best score: {self.model.best_score:.4f}"
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"XGBoost training failed: {e}") from e

        return self

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input sparse matrix (n_samples, n_features)

        Returns:
            Predicted class labels (n_samples,)

        Raises:
            ValueError: If model is not fitted or input shape is invalid
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        try:
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
            predictions = self.model.predict(
                dtest,
                iteration_range=(0, self._best_iteration + 1)
            )

            # For multi-class, predictions are probabilities
            if self.num_class is not None and self.num_class > 2:
                return np.argmax(predictions, axis=1)
            else:
                # Binary classification
                return (predictions > 0.5).astype(int)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input sparse matrix (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)

        Raises:
            ValueError: If model is not fitted or input shape is invalid
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        try:
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
            probas = self.model.predict(
                dtest,
                iteration_range=(0, self._best_iteration + 1)
            )

            # For multi-class, predictions are already probabilities
            if self.num_class is not None and self.num_class > 2:
                return probas
            else:
                # Binary classification - return both negative and positive class probabilities
                return np.vstack([1 - probas, probas]).T

        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise RuntimeError(f"Probability prediction failed: {e}") from e

    def get_feature_importance(
        self,
        importance_type: str = 'gain',
        top_k: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Get SNP importance scores.

        Args:
            importance_type: Type of importance metric:
                - 'gain': Average gain across all splits
                - 'weight': Number of times feature is used
                - 'cover': Average coverage across all splits
                - 'total_gain': Total gain across all splits
                - 'total_cover': Total coverage across all splits
            top_k: Return only top K features (None for all)

        Returns:
            Dictionary mapping SNP index to importance score, sorted by importance

        Raises:
            ValueError: If model is not fitted or invalid importance_type
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importance")

        valid_types = {'gain', 'weight', 'cover', 'total_gain', 'total_cover'}
        if importance_type not in valid_types:
            raise ValueError(
                f"Invalid importance_type '{importance_type}'. "
                f"Must be one of {valid_types}"
            )

        try:
            # Get importance scores from model
            importance_dict = self.model.get_score(importance_type=importance_type)

            # Convert feature names to indices and sort by importance
            importance_scores = {}
            for feat_name, score in importance_dict.items():
                if feat_name.startswith('SNP_'):
                    idx = int(feat_name.split('_')[1])
                else:
                    # Try to find index in feature_names
                    try:
                        idx = self.feature_names.index(feat_name)
                    except (ValueError, AttributeError):
                        logger.warning(f"Could not map feature name '{feat_name}' to index")
                        continue

                importance_scores[idx] = score

            # Sort by importance (descending)
            sorted_importance = dict(
                sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            )

            # Return top K if specified
            if top_k is not None:
                sorted_importance = dict(list(sorted_importance.items())[:top_k])

            logger.info(
                f"Retrieved {len(sorted_importance)} feature importance scores "
                f"(type={importance_type})"
            )

            return sorted_importance

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            raise RuntimeError(f"Failed to get feature importance: {e}") from e

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save model to file.

        Saves both the XGBoost model and metadata (feature names, classes, etc.)
        in separate files.

        Args:
            path: Path to save model (without extension)

        Raises:
            ValueError: If model is not fitted
            IOError: If save operation fails
        """
        if self.model is None:
            raise ValueError("Model must be fitted before saving")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save XGBoost model
            model_path = path.with_suffix('.json')
            self.model.save_model(str(model_path))
            logger.info(f"Saved XGBoost model to {model_path}")

            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'classes': self.classes_.tolist() if self.classes_ is not None else None,
                'n_features_in': self.n_features_in_,
                'best_iteration': self._best_iteration,
                'evals_result': self._evals_result,
                'params': {
                    'n_estimators': self.n_estimators,
                    'max_depth': self.max_depth,
                    'learning_rate': self.learning_rate,
                    'objective': self.objective,
                    'num_class': self.num_class,
                    'random_state': self.random_state,
                },
            }

            metadata_path = path.with_suffix('.meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise IOError(f"Failed to save model: {e}") from e

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load model from file.

        Args:
            path: Path to load model from (without extension)

        Raises:
            FileNotFoundError: If model files don't exist
            IOError: If load operation fails
        """
        path = Path(path)
        model_path = path.with_suffix('.json')
        metadata_path = path.with_suffix('.meta.json')

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        try:
            # Load XGBoost model
            self.model = xgb.Booster()
            self.model.load_model(str(model_path))
            logger.info(f"Loaded XGBoost model from {model_path}")

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.feature_names = metadata['feature_names']
            self.classes_ = np.array(metadata['classes']) if metadata['classes'] else None
            self.n_features_in_ = metadata['n_features_in']
            self._best_iteration = metadata['best_iteration']
            self._evals_result = metadata['evals_result']

            # Restore parameters
            params = metadata['params']
            self.n_estimators = params['n_estimators']
            self.max_depth = params['max_depth']
            self.learning_rate = params['learning_rate']
            self.objective = params['objective']
            self.num_class = params['num_class']
            self.random_state = params['random_state']

            logger.info(f"Loaded metadata from {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise IOError(f"Failed to load model: {e}") from e

    def export_for_nvflare(self) -> Dict[str, Any]:
        """
        Export model in NVFlare-compatible format.

        Creates a dictionary containing all necessary information for
        federated learning deployment with NVIDIA FLARE.

        Returns:
            Dictionary with model weights, configuration, and metadata

        Raises:
            ValueError: If model is not fitted
        """
        if self.model is None:
            raise ValueError("Model must be fitted before export")

        try:
            # Get model as JSON string
            model_json = self.model.save_raw(raw_format='json')

            export_dict = {
                'model_type': 'xgboost',
                'model_json': model_json.decode('utf-8'),
                'feature_names': self.feature_names,
                'n_features': self.n_features_in_,
                'num_class': self.num_class,
                'classes': self.classes_.tolist() if self.classes_ is not None else None,
                'best_iteration': self._best_iteration,
                'params': self._get_params(),
                'training_history': self._evals_result,
            }

            logger.info("Exported model for NVFlare")
            return export_dict

        except Exception as e:
            logger.error(f"Failed to export for NVFlare: {e}")
            raise RuntimeError(f"Failed to export for NVFlare: {e}") from e

    def get_training_history(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Get training history (evaluation results).

        Returns:
            Dictionary with training/validation metrics per iteration
        """
        return self._evals_result

    def __repr__(self) -> str:
        """String representation of the trainer."""
        fitted = "fitted" if self.model is not None else "not fitted"
        return (
            f"XGBoostSNPTrainer(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, gpu_id={self.gpu_id}, {fitted})"
        )
