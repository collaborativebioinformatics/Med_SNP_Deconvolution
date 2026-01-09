"""
PyTorch Lightning Trainer for SNP Models

Simplified training using PyTorch Lightning:
- Automatic bf16 mixed precision (no manual AMP)
- Multi-GPU training via Lightning strategies
- Built-in early stopping and checkpointing
- Clean, maintainable code

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, '/Users/saltfish/Files/Coding/Haploblock_Clusters_ElixirBH25')
from dl_models.snp_interpretable_models import InterpretableSNPModel

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance in population classification.

    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class SNPLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for SNP classification.

    Lightning handles bf16, multi-GPU, gradient clipping automatically.

    Args:
        n_snps: Number of SNPs
        encoding_dim: Dimension of genotype encoding (default: 8)
        num_classes: Number of population classes (default: 3)
        architecture: Model architecture ('cnn', 'cnn_transformer', 'gnn')
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        use_focal_loss: Use focal loss for class imbalance
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        scheduler_type: 'cosine', 'plateau', or 'none'
        mu: FedProx proximal coefficient (default: 0.0 for FedAvg)

    Example:
        >>> model = SNPLightningModule(n_snps=10000, num_classes=3)
        >>> trainer = pl.Trainer(precision="bf16-mixed", max_epochs=100)
        >>> trainer.fit(model, train_loader, val_loader)
    """

    def __init__(
        self,
        n_snps: int,
        encoding_dim: int = 8,
        num_classes: int = 3,
        architecture: str = 'cnn_transformer',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        scheduler_type: str = 'cosine',
        mu: float = 0.0,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create base model
        self.model = InterpretableSNPModel(
            n_snps=n_snps,
            encoding_dim=encoding_dim,
            num_classes=num_classes,
            architecture=architecture,
            **model_kwargs
        )

        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # FedProx: Store global model weights for proximal term
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        return self.model(x, return_attention=return_attention)

    def _compute_proximal_term(self) -> torch.Tensor:
        """
        Compute FedProx proximal term: (mu/2) * ||w - w_global||^2

        This term penalizes local model drift from the global model,
        helping convergence in heterogeneous federated settings.

        Returns:
            Proximal regularization term (scalar tensor with gradient tracking)
        """
        if self.global_model_state is None or self.hparams.mu == 0.0:
            return torch.tensor(0.0, device=self.device)

        # Initialize as tensor to maintain gradient tracking through the computation
        prox_term = torch.tensor(0.0, device=self.device)
        for name, param in self.named_parameters():
            if name in self.global_model_state:
                global_param = self.global_model_state[name].to(param.device)
                # This maintains gradient flow through param
                prox_term = prox_term + torch.sum((param - global_param) ** 2)

        return (self.hparams.mu / 2.0) * prox_term

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # FedProx: Add proximal term to loss
        prox_term = self._compute_proximal_term()
        total_loss = loss + prox_term

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)

        if self.hparams.mu > 0.0:
            self.log('train_prox_term', prox_term, prog_bar=False, sync_dist=True)
            self.log('train_total_loss', total_loss, prog_bar=True, sync_dist=True)

        self.training_step_outputs.append({
            'loss': loss.detach(),
            'acc': acc.detach(),
            'prox_term': prox_term.detach() if isinstance(prox_term, torch.Tensor) else torch.tensor(0.0)
        })
        return total_loss

    def on_train_epoch_end(self):
        if self.training_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            avg_acc = torch.stack([x['acc'] for x in self.training_step_outputs]).mean()
            self.log('train_epoch_loss', avg_loss)
            self.log('train_epoch_acc', avg_acc)
        self.training_step_outputs.clear()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'acc': acc.detach()
        })
        return {'val_loss': loss, 'val_acc': acc}

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            avg_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
            self.log('val_epoch_loss', avg_loss)
            self.log('val_epoch_acc', avg_acc)
        self.validation_step_outputs.clear()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', acc, sync_dist=True)

        return {'test_loss': loss, 'test_acc': acc, 'preds': preds, 'targets': y}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        if self.hparams.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
            }
        elif self.hparams.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}
            }
        return optimizer

    @torch.no_grad()
    def get_snp_importance(self, x: torch.Tensor, method: str = 'attention') -> torch.Tensor:
        """Extract SNP importance scores."""
        self.eval()
        _ = self(x, return_attention=True)

        if hasattr(self.model.model, 'attention_weights'):
            return self.model.model.attention_weights
        elif hasattr(self.model.model, 'get_snp_importance'):
            return self.model.model.get_snp_importance()
        raise ValueError(f"Model does not support importance extraction via {method}")

    def identify_causal_snps(
        self, x: torch.Tensor, top_k: int = 100, method: str = 'attention'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Identify top causal SNPs."""
        return self.model.identify_causal_snps(x, top_k=top_k, method=method)


def create_lightning_trainer(
    output_dir: str = 'results/snp_lightning',
    max_epochs: int = 100,
    precision: str = 'bf16-mixed',
    accelerator: str = 'gpu',
    devices: int = 1,
    early_stopping_patience: int = 15,
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    enable_tensorboard: bool = True,
    **kwargs
) -> pl.Trainer:
    """
    Create configured Lightning Trainer.

    Args:
        output_dir: Directory for checkpoints and logs
        max_epochs: Maximum training epochs
        precision: 'bf16-mixed' for A100/H100, '16-mixed' for older GPUs
        accelerator: 'gpu' or 'cpu'
        devices: Number of GPUs
        early_stopping_patience: Epochs before early stopping
        gradient_clip_val: Gradient clipping value

    Returns:
        Configured pl.Trainer
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=early_stopping_patience, mode='min', verbose=True),
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='snp-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss', mode='min', save_top_k=3, save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]

    loggers = []
    if enable_tensorboard:
        loggers.append(TensorBoardLogger(save_dir=output_dir, name='tensorboard'))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision=precision,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        logger=loggers if loggers else None,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=True,
        log_every_n_steps=10,
        **kwargs
    )

    logger.info(f"Trainer: precision={precision}, devices={devices}, max_epochs={max_epochs}")
    return trainer


def train_snp_model(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    n_snps: int,
    num_classes: int = 3,
    batch_size: int = 128,
    max_epochs: int = 100,
    output_dir: str = 'results/snp_lightning',
    **kwargs
) -> Tuple[SNPLightningModule, pl.Trainer]:
    """
    Train SNP model with Lightning (simplified API).

    Args:
        train_data: (X_train, y_train) tensors
        val_data: (X_val, y_val) tensors
        n_snps: Number of SNPs
        num_classes: Number of population classes
        batch_size: Batch size
        max_epochs: Maximum epochs
        output_dir: Output directory

    Returns:
        (trained_model, trainer)
    """
    X_train, y_train = train_data
    X_val, y_val = val_data

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model_kwargs = {k: v for k, v in kwargs.items()
                    if k in ['encoding_dim', 'architecture', 'learning_rate',
                             'weight_decay', 'use_focal_loss', 'scheduler_type']}
    model = SNPLightningModule(n_snps=n_snps, num_classes=num_classes, **model_kwargs)

    trainer_kwargs = {k: v for k, v in kwargs.items()
                      if k in ['precision', 'devices', 'early_stopping_patience',
                               'gradient_clip_val', 'accumulate_grad_batches']}
    trainer = create_lightning_trainer(output_dir=output_dir, max_epochs=max_epochs, **trainer_kwargs)

    trainer.fit(model, train_loader, val_loader)
    return model, trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("PyTorch Lightning SNP Trainer")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("No GPU available")
