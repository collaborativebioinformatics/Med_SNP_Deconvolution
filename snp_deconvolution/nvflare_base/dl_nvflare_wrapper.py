"""
Deep Learning NVFlare Executor for SNP Deconvolution

Wraps PyTorch SNP models for horizontal federated learning with NVFlare.

Federated Deep Learning Strategies:
    1. FedAvg: Average model weights from all sites (default)
    2. FedProx: Add proximal term to prevent drift from global model
    3. FedOpt: Server-side optimizer (Adam, AdaGrad, etc.)

This implementation supports FedAvg and FedProx with GPU acceleration.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging
from dataclasses import dataclass
from copy import deepcopy

from .base_executor import SNPDeconvExecutor, ExecutorMetrics
from .model_shareable import (
    serialize_pytorch_weights,
    deserialize_pytorch_weights,
    validate_model_weights,
)

logger = logging.getLogger(__name__)


class DLNVFlareExecutor(SNPDeconvExecutor):
    """
    NVFlare wrapper for PyTorch SNP models.

    Federated DL strategy:
        - FedAvg: Weighted average of model weights
        - FedProx: Add proximal term ||w - w_global||^2

    Supports:
        - Multi-GPU training via DDP
        - Mixed precision (bf16) training
        - Gradient clipping and accumulation
        - Custom loss functions and optimizers

    Attributes:
        model: PyTorch model (SNPAttentionNet, etc.)
        trainer: MultiGPUSNPTrainer or custom trainer
        aggregation_strategy: 'fedavg' or 'fedprox'
        fedprox_mu: Proximal term weight for FedProx

    Example:
        >>> from snp_deconvolution.attention_dl import SNPAttentionNet
        >>> model = SNPAttentionNet(num_snps=10000, num_populations=5)
        >>> executor = DLNVFlareExecutor(
        ...     model=model,
        ...     trainer=trainer,
        ...     aggregation_strategy='fedavg'
        ... )
        >>>
        >>> # Federated learning round
        >>> metrics = executor.local_train(num_epochs=5)
        >>> weights = executor.get_model_weights()
    """

    def __init__(
        self,
        model: nn.Module,
        trainer: Any = None,  # MultiGPUSNPTrainer
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
        aggregation_strategy: str = 'fedavg',
        fedprox_mu: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize DL NVFlare executor.

        Args:
            model: PyTorch model
            trainer: Optional trainer instance
            optimizer: PyTorch optimizer (created if None)
            loss_fn: Loss function (CrossEntropyLoss if None)
            aggregation_strategy: 'fedavg' or 'fedprox'
            fedprox_mu: Proximal term coefficient for FedProx
            device: Torch device (auto-detect if None)

        Raises:
            ValueError: If aggregation_strategy invalid
        """
        super().__init__()

        if aggregation_strategy not in ['fedavg', 'fedprox']:
            raise ValueError(
                f"Unknown aggregation strategy: {aggregation_strategy}. "
                f"Use 'fedavg' or 'fedprox'"
            )

        self.model = model
        self.trainer = trainer
        self.aggregation_strategy = aggregation_strategy
        self.fedprox_mu = fedprox_mu

        # Device setup
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        # Optimizer setup
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=1e-5,
            )
        else:
            self.optimizer = optimizer

        # Loss function
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        # Store global model weights for FedProx
        self._global_weights: Optional[Dict[str, torch.Tensor]] = None

        # Infer model dimensions
        self._num_snps = self._infer_num_snps()
        self._num_populations = self._infer_num_populations()

        # Training state
        self._train_loader: Optional[Any] = None
        self._val_loader: Optional[Any] = None

        logger.info(
            f"Initialized DLNVFlareExecutor: "
            f"{self._num_snps} SNPs, {self._num_populations} populations, "
            f"strategy={aggregation_strategy}, device={self.device}"
        )

    @property
    def model_type(self) -> str:
        """Return model type identifier"""
        return 'pytorch'

    @property
    def num_snps(self) -> int:
        """Return number of SNP features"""
        return self._num_snps

    @property
    def num_populations(self) -> int:
        """Return number of target populations"""
        return self._num_populations

    def _infer_num_snps(self) -> int:
        """Infer number of SNPs from model architecture"""
        # Try to get from model attributes
        if hasattr(self.model, 'num_snps'):
            return self.model.num_snps

        # Try to infer from first layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                return module.in_features
            elif isinstance(module, nn.Conv1d):
                return module.in_channels

        logger.warning("Could not infer num_snps, using 0")
        return 0

    def _infer_num_populations(self) -> int:
        """Infer number of populations from model architecture"""
        # Try to get from model attributes
        if hasattr(self.model, 'num_populations'):
            return self.model.num_populations
        if hasattr(self.model, 'num_classes'):
            return self.model.num_classes

        # Try to infer from last layer
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Linear):
                return module.out_features

        logger.warning("Could not infer num_populations, using 0")
        return 0

    def set_data_loaders(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None
    ) -> None:
        """
        Set PyTorch data loaders.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        self._train_loader = train_loader
        self._val_loader = val_loader
        logger.info(
            f"Set data loaders: train={len(train_loader)}, "
            f"val={len(val_loader) if val_loader else 0}"
        )

    def get_model_weights(self) -> Dict[str, Any]:
        """
        Export PyTorch state_dict as numpy arrays.

        Returns:
            Dictionary containing:
                - weights: state_dict as numpy arrays
                - model_type: 'pytorch'
                - model_config: Architecture details
                - num_snps: Number of SNP features
                - num_populations: Number of populations
                - optimizer_state: Optimizer state (optional)

        Raises:
            RuntimeError: If model not initialized
        """
        try:
            # Get model state dict
            state_dict = self.model.state_dict()

            # Serialize to numpy
            serialized = serialize_pytorch_weights(
                state_dict,
                compress=False,
                include_metadata=True,
            )

            # Add model-specific metadata
            serialized.update({
                'model_type': 'pytorch',
                'num_snps': self._num_snps,
                'num_populations': self._num_populations,
                'aggregation_strategy': self.aggregation_strategy,
                'round': self._round,
                'model_class': self.model.__class__.__name__,
            })

            # Optional: Include optimizer state for warm restarts
            # (Not typically shared in federated learning)
            # serialized['optimizer_state'] = self.optimizer.state_dict()

            logger.info(
                f"Exported PyTorch model: "
                f"{serialized.get('total_parameters', 0):.2e} parameters, "
                f"{serialized.get('total_size_mb', 0):.2f} MB"
            )

            return serialized

        except Exception as e:
            raise RuntimeError(f"Failed to export model weights: {e}")

    def set_model_weights(self, weights: Dict[str, Any]) -> None:
        """
        Load aggregated weights from server.

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
            expected_model_type='pytorch',
            expected_num_snps=self._num_snps,
            expected_num_populations=self._num_populations,
        ):
            raise ValueError("Model weight validation failed")

        try:
            # Deserialize weights
            state_dict = deserialize_pytorch_weights(
                weights,
                device=self.device,
                verify_checksum=True,
            )

            # Load into model
            self.model.load_state_dict(state_dict, strict=True)

            # Store as global weights for FedProx
            if self.aggregation_strategy == 'fedprox':
                self._global_weights = deepcopy(state_dict)
                logger.info("Stored global weights for FedProx")

            logger.info("Loaded aggregated model weights from server")

        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

    def local_train(
        self,
        num_epochs: int = 1,
        use_mixed_precision: bool = False,
        gradient_clip_norm: Optional[float] = None,
        **kwargs
    ) -> ExecutorMetrics:
        """
        Local training with optional FedProx.

        Args:
            num_epochs: Number of training epochs
            use_mixed_precision: Use bf16 mixed precision
            gradient_clip_norm: Gradient clipping threshold
            **kwargs: Additional training parameters

        Returns:
            ExecutorMetrics with training results

        Raises:
            RuntimeError: If training data not set
        """
        if self._train_loader is None:
            raise RuntimeError("Training data loader not set. Call set_data_loaders() first.")

        if self.aggregation_strategy == 'fedprox' and self._global_weights is not None:
            return self._train_with_fedprox(
                num_epochs,
                use_mixed_precision,
                gradient_clip_norm,
                **kwargs
            )
        else:
            return self._train_with_fedavg(
                num_epochs,
                use_mixed_precision,
                gradient_clip_norm,
                **kwargs
            )

    def _train_with_fedavg(
        self,
        num_epochs: int,
        use_mixed_precision: bool,
        gradient_clip_norm: Optional[float],
        **kwargs
    ) -> ExecutorMetrics:
        """Standard FedAvg training"""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

        logger.info(f"Starting FedAvg training: {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (X, y) in enumerate(self._train_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass with optional mixed precision
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(X)
                        loss = self.loss_fn(outputs, y)
                else:
                    outputs = self.model(X)
                    loss = self.loss_fn(outputs, y)

                # Backward pass
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                    if gradient_clip_norm:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            gradient_clip_norm
                        )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            gradient_clip_norm
                        )
                    self.optimizer.step()

                # Metrics
                batch_size = X.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size

                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    epoch_correct += (predicted == y).sum().item()

            # Epoch summary
            epoch_loss /= epoch_samples
            epoch_acc = epoch_correct / epoch_samples

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"loss={epoch_loss:.4f}, accuracy={epoch_acc:.4f}"
            )

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples = epoch_samples  # Same for all epochs

        # Average across epochs
        avg_loss = total_loss / num_epochs
        avg_acc = total_correct / (num_epochs * total_samples)

        # Validation if available
        val_metrics = {}
        if self._val_loader is not None:
            val_result = self.validate()
            val_metrics = {
                'val_loss': val_result.loss,
                'val_accuracy': val_result.accuracy,
            }

        metrics = ExecutorMetrics(
            loss=avg_loss,
            accuracy=avg_acc,
            num_samples=total_samples,
            additional_metrics=val_metrics,
        )

        # Update history
        self._training_history.append(metrics.to_dict())

        logger.info(
            f"Training complete: loss={avg_loss:.4f}, accuracy={avg_acc:.4f}"
        )

        return metrics

    def _train_with_fedprox(
        self,
        num_epochs: int,
        use_mixed_precision: bool,
        gradient_clip_norm: Optional[float],
        **kwargs
    ) -> ExecutorMetrics:
        """
        Training with FedProx proximal term.

        FedProx adds a regularization term to the loss:
            L_fedprox = L_standard + (mu / 2) * ||w - w_global||^2

        This prevents local models from drifting too far from the global model.
        """
        if self._global_weights is None:
            logger.warning("Global weights not set, falling back to FedAvg")
            return self._train_with_fedavg(
                num_epochs,
                use_mixed_precision,
                gradient_clip_norm,
                **kwargs
            )

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

        logger.info(
            f"Starting FedProx training: {num_epochs} epochs, mu={self.fedprox_mu}"
        )

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for batch_idx, (X, y) in enumerate(self._train_loader):
                X = X.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                if use_mixed_precision:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(X)
                        loss = self.loss_fn(outputs, y)

                        # Add proximal term
                        proximal_term = 0.0
                        for name, param in self.model.named_parameters():
                            if name in self._global_weights:
                                proximal_term += torch.sum(
                                    (param - self._global_weights[name]) ** 2
                                )
                        loss += (self.fedprox_mu / 2.0) * proximal_term
                else:
                    outputs = self.model(X)
                    loss = self.loss_fn(outputs, y)

                    # Add proximal term
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in self._global_weights:
                            proximal_term += torch.sum(
                                (param - self._global_weights[name]) ** 2
                            )
                    loss += (self.fedprox_mu / 2.0) * proximal_term

                # Backward pass
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                    if gradient_clip_norm:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            gradient_clip_norm
                        )
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_norm:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            gradient_clip_norm
                        )
                    self.optimizer.step()

                # Metrics
                batch_size = X.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size

                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    epoch_correct += (predicted == y).sum().item()

            # Epoch summary
            epoch_loss /= epoch_samples
            epoch_acc = epoch_correct / epoch_samples

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"loss={epoch_loss:.4f}, accuracy={epoch_acc:.4f}"
            )

            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples = epoch_samples

        # Average across epochs
        avg_loss = total_loss / num_epochs
        avg_acc = total_correct / (num_epochs * total_samples)

        # Validation if available
        val_metrics = {}
        if self._val_loader is not None:
            val_result = self.validate()
            val_metrics = {
                'val_loss': val_result.loss,
                'val_accuracy': val_result.accuracy,
            }

        metrics = ExecutorMetrics(
            loss=avg_loss,
            accuracy=avg_acc,
            num_samples=total_samples,
            additional_metrics=val_metrics,
        )

        # Update history
        self._training_history.append(metrics.to_dict())

        logger.info(
            f"FedProx training complete: loss={avg_loss:.4f}, accuracy={avg_acc:.4f}"
        )

        return metrics

    def validate(self) -> ExecutorMetrics:
        """
        Validate model on local validation data.

        Returns:
            ExecutorMetrics with validation results

        Raises:
            RuntimeError: If no validation data
        """
        if self._val_loader is None:
            raise RuntimeError("Validation data loader not set")

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for X, y in self._val_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)

                batch_size = X.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == y).sum().item()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        metrics = ExecutorMetrics(
            loss=avg_loss,
            accuracy=accuracy,
            num_samples=total_samples,
        )

        logger.info(
            f"Validation: loss={avg_loss:.4f}, accuracy={accuracy:.4f}"
        )

        return metrics

    def get_feature_importance(self) -> Dict[int, float]:
        """
        Get SNP importance from attention weights or gradients.

        Returns:
            Dictionary mapping SNP index to importance score

        Note:
            Implementation depends on model architecture:
            - Attention models: Use attention weights
            - CNN models: Use gradient-based methods
            - MLP models: Use integrated gradients
        """
        try:
            # Try to get from attention mechanism
            if hasattr(self.model, 'get_attention_weights'):
                attention_weights = self.model.get_attention_weights()
                return self._normalize_importance(attention_weights)

            # Fall back to gradient-based importance
            logger.info("Using gradient-based feature importance")
            return self._gradient_based_importance()

        except Exception as e:
            logger.error(f"Failed to compute feature importance: {e}")
            return {}

    def _gradient_based_importance(self) -> Dict[int, float]:
        """Compute gradient-based feature importance"""
        if self._val_loader is None:
            logger.warning("No validation data for importance computation")
            return {}

        self.model.eval()

        # Accumulate gradients
        importance_scores = None

        for X, y in self._val_loader:
            X = X.to(self.device)
            X.requires_grad = True
            y = y.to(self.device)

            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)
            loss.backward()

            # Accumulate absolute gradients
            if importance_scores is None:
                importance_scores = torch.abs(X.grad).sum(dim=0)
            else:
                importance_scores += torch.abs(X.grad).sum(dim=0)

            break  # Use first batch only for efficiency

        # Convert to dictionary
        importance_dict = {
            i: float(score)
            for i, score in enumerate(importance_scores.cpu().numpy())
        }

        return self._normalize_importance(importance_dict)

    def _normalize_importance(self, importance: Dict[int, float]) -> Dict[int, float]:
        """Normalize importance scores to sum to 1.0"""
        total = sum(importance.values())
        if total > 0:
            return {idx: score / total for idx, score in importance.items()}
        return importance

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DLNVFlareExecutor("
            f"model={self.model.__class__.__name__}, "
            f"snps={self._num_snps}, "
            f"populations={self._num_populations}, "
            f"strategy={self.aggregation_strategy}, "
            f"round={self._round})"
        )
