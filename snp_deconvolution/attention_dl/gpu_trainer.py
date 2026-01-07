"""
Multi-GPU Trainer with bf16 Support for SNP Models

Production-ready trainer with:
- bfloat16 (bf16) mixed precision for A100/H100
- Multi-GPU training via DataParallel
- Focal loss for class imbalance
- Learning rate scheduling
- Early stopping
- Checkpoint management
- Comprehensive logging

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import time
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Used in genomic prediction models to focus on hard-to-classify examples.

    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'

    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Class labels (batch_size,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# MULTI-GPU TRAINER
# ============================================================================

class MultiGPUSNPTrainer:
    """
    GPU trainer with bf16 AMP support for A100/H100.

    Features:
    - bfloat16 mixed precision (native A100/H100 support)
    - Multi-GPU via DataParallel
    - Gradient accumulation
    - Learning rate scheduling (Cosine annealing + ReduceLROnPlateau)
    - Early stopping
    - Checkpoint management
    - Focal loss for imbalanced data
    - Comprehensive metrics logging

    Note on bf16:
        - bf16 has same dynamic range as fp32 (8-bit exponent)
        - No gradient scaling (GradScaler) needed
        - Better stability than fp16
        - Native support on A100/H100

    Args:
        model: SNP model (GPUOptimizedSNPModel)
        train_loader: Training data loader
        val_loader: Validation data loader
        gpu_ids: List of GPU IDs to use
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        use_amp: Enable automatic mixed precision
        amp_dtype: Data type for AMP (default: torch.bfloat16)
        use_focal_loss: Use focal loss instead of cross-entropy
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
        gradient_accumulation_steps: Number of steps to accumulate gradients

    Example:
        >>> trainer = MultiGPUSNPTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     gpu_ids=[0, 1],
        ...     use_amp=True,
        ...     amp_dtype=torch.bfloat16
        ... )
        >>> history = trainer.train(num_epochs=100)
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 gpu_ids: List[int] = [0],
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 use_amp: bool = True,
                 amp_dtype: torch.dtype = torch.bfloat16,
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 gradient_accumulation_steps: int = 1):
        """
        Initialize trainer.

        Note: For bf16 on A100/H100, GradScaler is NOT needed.
        bf16 has same dynamic range as fp32.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gpu_ids = gpu_ids
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_ids[0]}')
            self.model = self.model.to(self.device)

            # Multi-GPU if requested
            if len(gpu_ids) > 1:
                from .gpu_optimized_models import DataParallelSNPModel
                self.model = DataParallelSNPModel(self.model, device_ids=gpu_ids)
                logger.info(f"Multi-GPU training on GPUs: {gpu_ids}")
            else:
                logger.info(f"Single GPU training on GPU: {gpu_ids[0]}")
        else:
            self.device = torch.device('cpu')
            logger.warning("CUDA not available. Using CPU.")

        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            self.criterion = nn.CrossEntropyLoss()
            logger.info("Using Cross-Entropy Loss")

        # Optimizer (AdamW with weight decay)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate schedulers
        self.scheduler_cosine = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7
        )
        self.scheduler_plateau = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.history = defaultdict(list)

        # Validate amp_dtype
        if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logger.warning("bf16 not supported. Falling back to fp16.")
            self.amp_dtype = torch.float16

        logger.info(
            f"Trainer initialized: lr={learning_rate}, "
            f"weight_decay={weight_decay}, "
            f"AMP={use_amp} (dtype={self.amp_dtype})"
        )

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train one epoch with bf16 AMP.

        Returns:
            (average_loss, accuracy) for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with AMP
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Backward pass
            # Note: No GradScaler needed for bf16
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_loss = epoch_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate model.

        Returns:
            (average_loss, accuracy) on validation set
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with AMP
            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self,
              num_epochs: int = 100,
              early_stopping_patience: int = 15,
              checkpoint_dir: Optional[Path] = None,
              save_best_only: bool = True,
              verbose: int = 1) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Args:
            num_epochs: Maximum number of epochs
            early_stopping_patience: Epochs to wait before early stopping
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save checkpoints when validation improves
            verbose: Logging verbosity (0: silent, 1: progress, 2: detailed)

        Returns:
            Training history dictionary
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Training
            train_loss, train_acc = self.train_epoch()

            # Validation
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler_cosine.step()
            self.scheduler_plateau.step(val_loss)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - epoch_start

            # Logging
            if verbose >= 1:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s) - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                improved = True

            # Save checkpoint
            if checkpoint_dir is not None and (improved or not save_best_only):
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)

                if improved:
                    best_path = checkpoint_dir / "best_model.pt"
                    self.save_checkpoint(best_path)
                    logger.info(f"Best model saved (val_loss: {val_loss:.4f}, val_acc: {val_acc:.2f}%)")

            # Early stopping
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.2f}s - "
            f"Best Val Loss: {self.best_val_loss:.4f}, Best Val Acc: {self.best_val_acc:.2f}%"
        )

        return dict(self.history)

    def save_checkpoint(self, path: Path) -> None:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        # Get model state dict (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_cosine_state_dict': self.scheduler_cosine.state_dict(),
            'scheduler_plateau_state_dict': self.scheduler_plateau.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'history': dict(self.history)
        }

        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state dict (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler_cosine.load_state_dict(checkpoint['scheduler_cosine_state_dict'])
        self.scheduler_plateau.load_state_dict(checkpoint['scheduler_plateau_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = defaultdict(list, checkpoint['history'])

        logger.info(f"Checkpoint loaded: {path} (epoch {self.current_epoch})")

    @torch.no_grad()
    def get_snp_importance(self,
                          data_loader: DataLoader,
                          method: str = 'attention',
                          aggregate: str = 'mean') -> Dict[int, float]:
        """
        Extract SNP importance across dataset.

        Args:
            data_loader: Data loader for samples
            method: Importance extraction method ('attention', 'gradient')
            aggregate: How to aggregate across samples ('mean', 'max', 'median')

        Returns:
            Dictionary mapping SNP index to importance score
        """
        self.model.eval()
        importance_scores = []

        for inputs, _ in data_loader:
            inputs = inputs.to(self.device)

            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    if isinstance(self.model, nn.DataParallel):
                        scores = self.model.module.get_snp_importance(inputs, method=method)
                    else:
                        scores = self.model.get_snp_importance(inputs, method=method)
            else:
                if isinstance(self.model, nn.DataParallel):
                    scores = self.model.module.get_snp_importance(inputs, method=method)
                else:
                    scores = self.model.get_snp_importance(inputs, method=method)

            importance_scores.append(scores.cpu())

        # Aggregate across all samples
        all_scores = torch.cat(importance_scores, dim=0)  # (total_samples, n_snps)

        if aggregate == 'mean':
            aggregated = all_scores.mean(dim=0)
        elif aggregate == 'max':
            aggregated = all_scores.max(dim=0)[0]
        elif aggregate == 'median':
            aggregated = all_scores.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")

        # Convert to dictionary
        n_snps = aggregated.shape[0]
        importance_dict = {i: aggregated[i].item() for i in range(n_snps)}

        return importance_dict


if __name__ == "__main__":
    """Example usage and testing"""

    logging.basicConfig(level=logging.INFO)

    # Test focal loss
    print("Testing Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    inputs = torch.randn(8, 2)  # (batch_size, num_classes)
    targets = torch.randint(0, 2, (8,))  # (batch_size,)

    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")

    # Test trainer (mock)
    print("\nTrainer class loaded successfully.")
    print("For full testing, create model and data loaders.")
