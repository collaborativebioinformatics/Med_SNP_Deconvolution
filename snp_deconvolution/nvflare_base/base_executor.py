"""
Abstract Base Executor for NVFlare SNP Deconvolution

Defines the interface for federated learning executors compatible with NVFlare 2.4+.
Designed for horizontal federated learning where each site has different samples
but the same SNP feature set.

Architecture:
    - Abstract base class defining required methods for federated learning
    - Compatible with NVFlare's Executor API (extends when NVFlare is installed)
    - Supports both XGBoost and PyTorch models
    - Handles model serialization, training, and validation

Federated Learning Flow:
    1. Server sends global model weights
    2. Site loads weights via set_model_weights()
    3. Site trains locally via local_train()
    4. Site exports weights via get_model_weights()
    5. Server aggregates weights from all sites
    6. Repeat until convergence
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExecutorMetrics:
    """Metrics returned from training/validation"""
    loss: float
    accuracy: float
    num_samples: int
    additional_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'loss': float(self.loss),
            'accuracy': float(self.accuracy),
            'num_samples': int(self.num_samples),
        }
        if self.additional_metrics:
            result.update(self.additional_metrics)
        return result


class SNPDeconvExecutor(ABC):
    """
    Abstract base class for NVFlare-compatible SNP deconvolution executors.

    Designed for horizontal federated learning:
        - Each site has different samples (rows)
        - All sites have same SNP set (columns)
        - FedAvg/FedProx aggregation for DL
        - Ensemble/histogram aggregation for XGBoost

    NVFlare Integration:
        - Extends nvflare.apis.executor.Executor (when NVFlare installed)
        - Implements execute() for federated tasks
        - Exports/imports model weights via Shareable objects
        - Maintains privacy by never sharing raw data

    Attributes:
        model_type: Type of model ('xgboost' or 'pytorch')
        num_snps: Number of SNP features
        num_populations: Number of target populations
        _round: Current federated learning round

    Example:
        >>> executor = XGBoostNVFlareExecutor(trainer, data_loader)
        >>> # Training round
        >>> metrics = executor.local_train(num_epochs=5)
        >>> weights = executor.get_model_weights()
        >>> # After server aggregation
        >>> executor.set_model_weights(aggregated_weights)
    """

    def __init__(self):
        """Initialize base executor"""
        self._round = 0
        self._best_accuracy = 0.0
        self._training_history: List[Dict[str, float]] = []

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return model type identifier.

        Returns:
            'xgboost' or 'pytorch'
        """
        pass

    @property
    @abstractmethod
    def num_snps(self) -> int:
        """Return number of SNP features"""
        pass

    @property
    @abstractmethod
    def num_populations(self) -> int:
        """Return number of target populations"""
        pass

    @abstractmethod
    def get_model_weights(self) -> Dict[str, Any]:
        """
        Export model weights in shareable format.

        This method serializes the current model state for transmission
        to the NVFlare server. The format depends on the model type:

        For XGBoost:
            - JSON tree structure
            - Booster configuration
            - Feature names

        For PyTorch:
            - state_dict as numpy arrays
            - Model architecture config
            - Optimizer state (optional)

        Returns:
            Dictionary containing:
                - weights: Serialized model weights
                - model_type: 'xgboost' or 'pytorch'
                - model_config: Architecture configuration
                - num_snps: Number of SNP features
                - num_populations: Number of populations
                - metadata: Additional information (version, timestamp, etc.)

        Raises:
            RuntimeError: If model is not initialized
        """
        pass

    @abstractmethod
    def set_model_weights(self, weights: Dict[str, Any]) -> None:
        """
        Load aggregated model weights from server.

        This method receives the global model weights aggregated by the
        NVFlare server and updates the local model.

        Args:
            weights: Dictionary with same structure as get_model_weights()

        Raises:
            ValueError: If weights format is incompatible
            RuntimeError: If model architecture mismatch
        """
        pass

    @abstractmethod
    def local_train(self, num_epochs: int = 1, **kwargs) -> ExecutorMetrics:
        """
        Perform local training on private data.

        Trains the model on the site's local dataset without sharing
        raw data with other sites or the server.

        Args:
            num_epochs: Number of training epochs
            **kwargs: Additional training parameters (learning_rate, etc.)

        Returns:
            ExecutorMetrics with training results:
                - loss: Training loss
                - accuracy: Training accuracy
                - num_samples: Number of samples used
                - additional_metrics: Optional metrics (AUC, F1, etc.)

        Raises:
            RuntimeError: If training fails
        """
        pass

    @abstractmethod
    def validate(self) -> ExecutorMetrics:
        """
        Validate model on local validation data.

        Returns:
            ExecutorMetrics with validation results

        Raises:
            RuntimeError: If validation fails or no validation data
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[int, float]:
        """
        Get SNP importance scores from local model.

        Feature importance helps identify which SNPs are most informative
        for population prediction. The aggregation of importance across
        sites can reveal globally relevant SNPs.

        Returns:
            Dictionary mapping SNP index to importance score.
            Scores are normalized to sum to 1.0.

        Example:
            {0: 0.15, 1: 0.08, 2: 0.12, ...}

        Note:
            For XGBoost: Uses gain-based importance
            For PyTorch: Uses attention weights or gradient-based methods
        """
        pass

    def get_training_history(self) -> List[Dict[str, float]]:
        """
        Get training history across rounds.

        Returns:
            List of metric dictionaries, one per round
        """
        return self._training_history

    def increment_round(self) -> None:
        """Increment the federated learning round counter"""
        self._round += 1
        logger.info(f"Advanced to round {self._round}")

    def get_current_round(self) -> int:
        """Get current federated learning round"""
        return self._round

    def save_checkpoint(self, path: Path) -> None:
        """
        Save executor checkpoint including weights and metadata.

        Args:
            path: Path to save checkpoint
        """
        import pickle

        checkpoint = {
            'weights': self.get_model_weights(),
            'round': self._round,
            'best_accuracy': self._best_accuracy,
            'training_history': self._training_history,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        """
        Load executor checkpoint.

        Args:
            path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        import pickle

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.set_model_weights(checkpoint['weights'])
        self._round = checkpoint['round']
        self._best_accuracy = checkpoint['best_accuracy']
        self._training_history = checkpoint['training_history']
        logger.info(f"Loaded checkpoint from {path}, round {self._round}")

    def validate_weights_compatibility(self, weights: Dict[str, Any]) -> bool:
        """
        Check if received weights are compatible with local model.

        Args:
            weights: Weights dictionary from server

        Returns:
            True if compatible, False otherwise
        """
        try:
            # Check model type
            if weights.get('model_type') != self.model_type:
                logger.error(
                    f"Model type mismatch: expected {self.model_type}, "
                    f"got {weights.get('model_type')}"
                )
                return False

            # Check dimensions
            if weights.get('num_snps') != self.num_snps:
                logger.error(
                    f"SNP dimension mismatch: expected {self.num_snps}, "
                    f"got {weights.get('num_snps')}"
                )
                return False

            if weights.get('num_populations') != self.num_populations:
                logger.error(
                    f"Population dimension mismatch: expected {self.num_populations}, "
                    f"got {weights.get('num_populations')}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating weights: {e}")
            return False

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"{self.__class__.__name__}("
            f"model_type={self.model_type}, "
            f"num_snps={self.num_snps}, "
            f"num_populations={self.num_populations}, "
            f"round={self._round})"
        )
