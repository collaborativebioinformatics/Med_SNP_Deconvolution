"""
Attention-based Deep Learning Module for SNP Deconvolution

GPU-optimized implementation with PyTorch Lightning for A100/H100 GPUs.
Uses bf16 precision via Lightning's automatic mixed precision.

This module provides:
- SNPLightningModule: Main training module (RECOMMENDED)
- NVFlare + Lightning integration
- Memory management utilities
"""

# PyTorch Lightning (recommended)
from .lightning_trainer import (
    SNPLightningModule,
    FocalLoss,
    create_lightning_trainer,
    train_snp_model,
)

# NVFlare integration
from .nvflare_lightning import (
    SNPDataModule,
    run_standalone,
    run_federated_client,
)

# Memory utilities
from .memory_manager import GPUMemoryManager

# Legacy (use Lightning instead)
from .gpu_optimized_models import GPUOptimizedSNPModel
from .gpu_trainer import MultiGPUSNPTrainer

__all__ = [
    'SNPLightningModule',
    'FocalLoss',
    'create_lightning_trainer',
    'train_snp_model',
    'SNPDataModule',
    'run_standalone',
    'run_federated_client',
    'GPUMemoryManager',
    'GPUOptimizedSNPModel',
    'MultiGPUSNPTrainer',
]

__version__ = '2.0.0'
