#!/usr/bin/env python3
"""
NVFlare Lightning Client - Official Client API Mode

This client script is automatically invoked by NVFlare for federated learning.
It uses the official Client API with flare.patch() for seamless Lightning integration.

Reference: https://nvflare.readthedocs.io/en/2.7.0/hello-world/hello-lightning/

Usage:
    # Automatically invoked by NVFlare (no manual execution needed)
    # The client receives global model, trains locally, and sends updates back

Architecture:
    1. flare.init() - Initialize NVFlare client
    2. flare.get_site_name() - Get unique site identifier
    3. flare.patch(trainer) - Integrate Lightning trainer with NVFlare
    4. while flare.is_running() - FL training loop
    5. flare.receive() - Receive global model from server
    6. trainer.fit() - Local training
    7. Model updates automatically sent back to server

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-07
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from snp_deconvolution.attention_dl.lightning_trainer import (
    SNPLightningModule,
    FocalLoss,
)
from snp_deconvolution.nvflare_real.data.federated_data_module import (
    SNPFederatedDataModule,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='NVFlare Lightning Client for SNP Deconvolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/saltfish/Files/Coding/Med_SNP_Deconvolution/snp_deconvolution/nvflare_real/data',
        help='Directory containing federated data splits'
    )
    parser.add_argument(
        '--feature_type',
        type=str,
        choices=['cluster', 'snp'],
        default='cluster',
        help='Feature type: cluster or snp'
    )

    # Model arguments
    parser.add_argument(
        '--architecture',
        type=str,
        default='cnn_transformer',
        choices=['cnn', 'cnn_transformer', 'gnn'],
        help='Model architecture'
    )
    parser.add_argument(
        '--encoding_dim',
        type=int,
        default=8,
        help='Dimension of genotype encoding'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=3,
        help='Number of population classes'
    )

    # Training arguments
    parser.add_argument(
        '--local_epochs',
        type=int,
        default=1,
        help='Number of local training epochs per FL round'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
        help='L2 regularization weight decay'
    )

    # Loss function arguments
    parser.add_argument(
        '--use_focal_loss',
        action='store_true',
        default=True,
        help='Use focal loss for class imbalance'
    )
    parser.add_argument(
        '--focal_alpha',
        type=float,
        default=0.25,
        help='Focal loss alpha parameter'
    )
    parser.add_argument(
        '--focal_gamma',
        type=float,
        default=2.0,
        help='Focal loss gamma parameter'
    )

    # Hardware arguments
    parser.add_argument(
        '--precision',
        type=str,
        default='bf16-mixed',
        choices=['32', '16-mixed', 'bf16-mixed'],
        help='Training precision (bf16-mixed for A100/H100)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )

    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Directory to save local checkpoints'
    )
    parser.add_argument(
        '--save_top_k',
        type=int,
        default=1,
        help='Save top K best models'
    )

    return parser.parse_args()


def create_model(
    n_features: int,
    encoding_dim: int,
    num_classes: int,
    architecture: str,
    learning_rate: float,
    weight_decay: float,
    use_focal_loss: bool,
    focal_alpha: float,
    focal_gamma: float,
) -> SNPLightningModule:
    """
    Create SNP Lightning model.

    Args:
        n_features: Number of features (SNPs or clusters)
        encoding_dim: Genotype encoding dimension
        num_classes: Number of classes
        architecture: Model architecture
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_focal_loss: Whether to use focal loss
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma

    Returns:
        SNPLightningModule instance
    """
    model = SNPLightningModule(
        n_snps=n_features,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        architecture=architecture,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        scheduler_type='cosine',
    )
    logger.info(f"Created model: arch={architecture}, n_features={n_features}, classes={num_classes}")
    return model


def run_federated_client(args):
    """
    Run federated learning client using NVFlare Client API.

    This function implements the official NVFlare + Lightning integration pattern:
    1. Initialize NVFlare client with flare.init()
    2. Get site identifier with flare.get_site_name()
    3. Patch Lightning trainer with flare.patch()
    4. Training loop: receive model -> train -> send updates

    Args:
        args: Command line arguments
    """
    try:
        import nvflare.client as flare
    except ImportError:
        logger.error("NVFlare not installed. Install with: pip install nvflare")
        raise RuntimeError("NVFlare package is required for federated training")

    # Step 1: Initialize NVFlare client
    logger.info("Initializing NVFlare client...")
    flare.init()

    # Step 2: Get site name (unique identifier for this client)
    site_name = flare.get_site_name()
    logger.info(f"=== NVFlare Client Started: {site_name} ===")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Feature type: {args.feature_type}")
    logger.info(f"Local epochs per round: {args.local_epochs}")

    # Step 3: Create data module for this site
    data_module = SNPFederatedDataModule(
        data_dir=args.data_dir,
        site_name=site_name,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()

    # Get actual feature dimensions from data
    sample_batch = next(iter(data_module.train_dataloader()))
    X_sample = sample_batch[0]
    n_features = X_sample.shape[1]
    actual_encoding_dim = X_sample.shape[2]

    logger.info(f"Data loaded: n_features={n_features}, encoding_dim={actual_encoding_dim}")
    logger.info(f"Train samples: {len(data_module.train_dataset)}, Val samples: {len(data_module.val_dataset)}")

    # Step 4: Create model
    model = create_model(
        n_features=n_features,
        encoding_dim=actual_encoding_dim,
        num_classes=args.num_classes,
        architecture=args.architecture,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )

    # Step 5: Create Lightning trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / site_name,
        filename='snp-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True,
    )

    # Determine accelerator
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accelerator = 'cpu'
        devices = 'auto'
        logger.info("Using CPU")

    trainer = pl.Trainer(
        max_epochs=args.local_epochs,
        precision=args.precision,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    # Step 6: CRITICAL - Patch trainer with NVFlare
    # This enables federated learning by intercepting model updates
    logger.info("Patching Lightning trainer with NVFlare...")
    flare.patch(trainer)
    logger.info("Trainer patched successfully - federated learning enabled")

    # Step 7: Federated learning loop
    logger.info("Starting federated learning loop...")
    round_num = 0

    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        round_num = input_model.current_round
        logger.info(f"\n{'='*60}")
        logger.info(f"FL Round {round_num}: Received global model from server")
        logger.info(f"{'='*60}")

        # Optional: Validate with current global model before training
        logger.info(f"Round {round_num}: Validating global model...")
        val_results = trainer.validate(model, datamodule=data_module)
        if val_results:
            logger.info(f"Round {round_num}: Global model val_loss={val_results[0].get('val_loss', 'N/A'):.4f}")

        # Local training
        logger.info(f"Round {round_num}: Starting local training ({args.local_epochs} epochs)...")
        trainer.fit(model, datamodule=data_module)
        logger.info(f"Round {round_num}: Local training completed")

        # Optional: Test with updated model
        logger.info(f"Round {round_num}: Testing updated model...")
        test_results = trainer.test(model, datamodule=data_module, ckpt_path="best")
        if test_results:
            logger.info(f"Round {round_num}: Test acc={test_results[0].get('test_acc', 'N/A'):.4f}")

        # Model updates are automatically sent back to server via flare.patch()
        logger.info(f"Round {round_num}: Sending model updates to server...")

    logger.info(f"\n{'='*60}")
    logger.info(f"Federated learning completed after {round_num} rounds")
    logger.info(f"Client {site_name} finished successfully")
    logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("="*60)
    logger.info("NVFlare Lightning Client for SNP Deconvolution")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Lightning version: {pl.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    logger.info("="*60)

    # Run federated client
    run_federated_client(args)


if __name__ == "__main__":
    main()
