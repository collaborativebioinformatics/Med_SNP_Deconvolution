"""
XGBoost NVFlare Executor for SNP Deconvolution

Wraps XGBoost SNP trainer for horizontal federated learning with NVFlare.

Federated XGBoost Strategies:
    1. Tree Ensemble: Each site trains, server ensembles models
    2. Histogram Aggregation: Aggregate gradient histograms (requires NVFlare XGBoost plugin)
    3. Secure Boost: Privacy-preserving boosting (advanced)

This implementation uses the ensemble approach for compatibility and simplicity.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass

from .base_executor import SNPDeconvExecutor, ExecutorMetrics
from .model_shareable import (
    serialize_xgboost_model,
    deserialize_xgboost_model,
    validate_model_weights,
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    DMatrix = xgb.DMatrix
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    DMatrix = Any  # Type placeholder when XGBoost not available

logger = logging.getLogger(__name__)


class XGBoostNVFlareExecutor(SNPDeconvExecutor):
    """
    NVFlare wrapper for XGBoost SNP trainer.

    Federated XGBoost strategy:
        - Each site trains local model on private data
        - Export as JSON tree structure
        - Server aggregates via ensemble (weighted voting)
        - Optional: Histogram aggregation for better performance

    Attributes:
        trainer: XGBoostSNPTrainer instance
        data_loader: Data loader providing local training data
        model: XGBoost Booster object
        _num_snps: Number of SNP features
        _num_populations: Number of population classes

    Example:
        >>> from snp_deconvolution.xgboost import XGBoostSNPTrainer
        >>> trainer = XGBoostSNPTrainer(num_snps=10000, num_populations=5)
        >>> executor = XGBoostNVFlareExecutor(trainer, data_loader)
        >>>
        >>> # Federated learning round
        >>> metrics = executor.local_train(num_epochs=10)
        >>> weights = executor.get_model_weights()
        >>> # Server aggregates weights from all sites
        >>> executor.set_model_weights(aggregated_weights)
    """

    def __init__(
        self,
        trainer: Any,  # XGBoostSNPTrainer
        data_loader: Any = None,
        num_snps: Optional[int] = None,
        num_populations: Optional[int] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize XGBoost NVFlare executor.

        Args:
            trainer: XGBoostSNPTrainer instance or None
            data_loader: Data loader for local training data
            num_snps: Number of SNP features (inferred if not provided)
            num_populations: Number of populations (inferred if not provided)
            xgb_params: XGBoost parameters override

        Raises:
            ImportError: If XGBoost not installed
            ValueError: If trainer and dimensions both None
        """
        super().__init__()

        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost not installed. Install with: pip install xgboost"
            )

        self.trainer = trainer
        self.data_loader = data_loader
        self.model: Optional[xgb.Booster] = None

        # Infer dimensions from trainer if available
        if trainer is not None:
            self._num_snps = getattr(trainer, 'num_snps', num_snps)
            self._num_populations = getattr(trainer, 'num_populations', num_populations)
        else:
            if num_snps is None or num_populations is None:
                raise ValueError(
                    "Must provide num_snps and num_populations when trainer is None"
                )
            self._num_snps = num_snps
            self._num_populations = num_populations

        # Default XGBoost parameters optimized for SNP data
        self.xgb_params = xgb_params or {
            'objective': 'multi:softprob',
            'num_class': self._num_populations,
            'tree_method': 'gpu_hist',  # GPU acceleration
            'device': 'cuda',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'eval_metric': 'mlogloss',
        }

        self._dtrain: Optional[DMatrix] = None
        self._dval: Optional[DMatrix] = None
        self._feature_names: Optional[List[str]] = None

        logger.info(
            f"Initialized XGBoostNVFlareExecutor: "
            f"{self._num_snps} SNPs, {self._num_populations} populations"
        )

    @property
    def model_type(self) -> str:
        """Return model type identifier"""
        return 'xgboost'

    @property
    def num_snps(self) -> int:
        """Return number of SNP features"""
        return self._num_snps

    @property
    def num_populations(self) -> int:
        """Return number of target populations"""
        return self._num_populations

    def prepare_data(self, X: np.ndarray, y: np.ndarray, is_validation: bool = False) -> None:
        """
        Prepare XGBoost DMatrix from data.

        Args:
            X: SNP genotype matrix (n_samples, n_snps)
            y: Population labels (n_samples,)
            is_validation: Whether this is validation data
        """
        # Create feature names
        if self._feature_names is None:
            self._feature_names = [f"SNP_{i}" for i in range(X.shape[1])]

        dmatrix = xgb.DMatrix(
            X,
            label=y,
            feature_names=self._feature_names,
            enable_categorical=False,
        )

        if is_validation:
            self._dval = dmatrix
            logger.info(f"Prepared validation data: {X.shape[0]} samples")
        else:
            self._dtrain = dmatrix
            logger.info(f"Prepared training data: {X.shape[0]} samples")

    def get_model_weights(self) -> Dict[str, Any]:
        """
        Export XGBoost model as JSON.

        Returns:
            Dictionary containing:
                - model_json: JSON tree structure
                - config: Booster configuration
                - feature_names: SNP feature names
                - num_snps: Number of SNP features
                - num_populations: Number of populations
                - metadata: Training info

        Raises:
            RuntimeError: If model not trained
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Train first.")

        try:
            serialized = serialize_xgboost_model(self.model, include_metadata=True)

            # Add SNP-specific metadata
            serialized.update({
                'num_snps': self._num_snps,
                'num_populations': self._num_populations,
                'model_type': 'xgboost',
                'xgb_params': self.xgb_params,
                'round': self._round,
            })

            logger.info(
                f"Exported XGBoost model: "
                f"{serialized['metadata']['num_boosted_rounds']} trees"
            )

            return serialized

        except Exception as e:
            raise RuntimeError(f"Failed to export model weights: {e}")

    def set_model_weights(self, weights: Dict[str, Any]) -> None:
        """
        Load aggregated model weights from server.

        Args:
            weights: Serialized model from get_model_weights()

        Raises:
            ValueError: If weights incompatible
        """
        # Validate compatibility
        if not self.validate_weights_compatibility(weights):
            raise ValueError("Incompatible model weights")

        if not validate_model_weights(
            weights,
            expected_model_type='xgboost',
            expected_num_snps=self._num_snps,
        ):
            raise ValueError("Model weight validation failed")

        try:
            self.model = deserialize_xgboost_model(weights, verify_checksum=True)

            # Update parameters if provided
            if 'xgb_params' in weights:
                self.xgb_params.update(weights['xgb_params'])

            logger.info("Loaded aggregated XGBoost model from server")

        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

    def local_train(
        self,
        num_epochs: int = 10,
        early_stopping_rounds: Optional[int] = None,
        **kwargs
    ) -> ExecutorMetrics:
        """
        Perform local training on private data.

        Args:
            num_epochs: Number of boosting rounds
            early_stopping_rounds: Stop if no improvement for N rounds
            **kwargs: Additional XGBoost training parameters

        Returns:
            ExecutorMetrics with training results

        Raises:
            RuntimeError: If training data not prepared
        """
        if self._dtrain is None:
            raise RuntimeError(
                "Training data not prepared. Call prepare_data() first."
            )

        try:
            # Prepare evaluation list
            evals = [(self._dtrain, 'train')]
            if self._dval is not None:
                evals.append((self._dval, 'validation'))

            # Update parameters with kwargs
            train_params = self.xgb_params.copy()
            train_params.update(kwargs)

            # Training callbacks
            callbacks = []
            if early_stopping_rounds:
                callbacks.append(
                    xgb.callback.EarlyStopping(
                        rounds=early_stopping_rounds,
                        save_best=True,
                    )
                )

            # Train model
            logger.info(f"Starting local training: {num_epochs} rounds")

            # Continue training from existing model if available
            evals_result = {}
            self.model = xgb.train(
                train_params,
                self._dtrain,
                num_boost_round=num_epochs,
                evals=evals,
                evals_result=evals_result,
                xgb_model=self.model,  # Continue from existing model
                callbacks=callbacks,
                verbose_eval=False,
            )

            # Extract metrics
            train_loss = evals_result['train']['mlogloss'][-1]
            train_acc = self._compute_accuracy(self._dtrain)

            # Validation metrics if available
            val_metrics = {}
            if self._dval is not None:
                val_loss = evals_result['validation']['mlogloss'][-1]
                val_acc = self._compute_accuracy(self._dval)
                val_metrics = {
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                }

            num_samples = self._dtrain.num_row()

            metrics = ExecutorMetrics(
                loss=train_loss,
                accuracy=train_acc,
                num_samples=num_samples,
                additional_metrics=val_metrics,
            )

            # Update history
            self._training_history.append(metrics.to_dict())

            logger.info(
                f"Training complete: loss={train_loss:.4f}, "
                f"accuracy={train_acc:.4f}, samples={num_samples}"
            )

            return metrics

        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

    def validate(self) -> ExecutorMetrics:
        """
        Validate model on local validation data.

        Returns:
            ExecutorMetrics with validation results

        Raises:
            RuntimeError: If no validation data or model
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        if self._dval is None:
            raise RuntimeError("Validation data not prepared")

        try:
            # Predict
            y_pred_proba = self.model.predict(self._dval)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = self._dval.get_label()

            # Compute metrics
            accuracy = np.mean(y_pred == y_true)

            # Multi-class log loss
            n_samples = len(y_true)
            epsilon = 1e-15
            y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

            # One-hot encode true labels
            y_true_onehot = np.zeros_like(y_pred_proba)
            y_true_onehot[np.arange(n_samples), y_true.astype(int)] = 1

            loss = -np.sum(y_true_onehot * np.log(y_pred_proba)) / n_samples

            metrics = ExecutorMetrics(
                loss=loss,
                accuracy=accuracy,
                num_samples=n_samples,
            )

            logger.info(
                f"Validation: loss={loss:.4f}, accuracy={accuracy:.4f}"
            )

            return metrics

        except Exception as e:
            raise RuntimeError(f"Validation failed: {e}")

    def get_feature_importance(self) -> Dict[int, float]:
        """
        Get SNP importance scores from local model.

        Returns:
            Dictionary mapping SNP index to normalized importance score

        Raises:
            RuntimeError: If model not trained
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        try:
            # Get gain-based importance
            importance_dict = self.model.get_score(importance_type='gain')

            # Convert feature names to indices
            importance_by_index = {}
            for feature_name, score in importance_dict.items():
                # Extract index from "SNP_123" format
                if feature_name.startswith('SNP_'):
                    idx = int(feature_name.split('_')[1])
                    importance_by_index[idx] = score
                else:
                    # Try direct index
                    try:
                        idx = int(feature_name)
                        importance_by_index[idx] = score
                    except ValueError:
                        logger.warning(f"Could not parse feature name: {feature_name}")

            # Normalize to sum to 1.0
            total = sum(importance_by_index.values())
            if total > 0:
                importance_by_index = {
                    idx: score / total
                    for idx, score in importance_by_index.items()
                }

            logger.info(
                f"Computed feature importance for {len(importance_by_index)} SNPs"
            )

            return importance_by_index

        except Exception as e:
            raise RuntimeError(f"Failed to compute feature importance: {e}")

    def _compute_accuracy(self, dmatrix: DMatrix) -> float:
        """Compute classification accuracy"""
        y_pred_proba = self.model.predict(dmatrix)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = dmatrix.get_label()
        return float(np.mean(y_pred == y_true))

    def get_tree_ensemble_size(self) -> int:
        """Get number of trees in ensemble"""
        if self.model is None:
            return 0
        return self.model.num_boosted_rounds()

    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        if self.model is None:
            return {'status': 'not_trained'}

        return {
            'status': 'trained',
            'num_trees': self.get_tree_ensemble_size(),
            'num_features': self.model.num_features(),
            'num_snps': self._num_snps,
            'num_populations': self._num_populations,
            'parameters': self.xgb_params,
            'round': self._round,
        }

    def __repr__(self) -> str:
        """String representation"""
        trees = self.get_tree_ensemble_size()
        return (
            f"XGBoostNVFlareExecutor("
            f"snps={self._num_snps}, "
            f"populations={self._num_populations}, "
            f"trees={trees}, "
            f"round={self._round})"
        )
