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
    HaploblockLightningModule,
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

    # Data arguments - use relative path from project root as default
    default_data_dir = str(project_root / 'data' / 'federated')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=default_data_dir,
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
        '--model_type',
        type=str,
        default='haploblock',
        choices=['snp', 'haploblock'],
        help='Model type: snp (3D input) or haploblock (2D cluster IDs)'
    )
    parser.add_argument(
        '--architecture',
        type=str,
        default='transformer',
        choices=['cnn', 'cnn_transformer', 'transformer'],
        help='Model architecture'
    )
    parser.add_argument(
        '--encoding_dim',
        type=int,
        default=8,
        help='Dimension of genotype encoding (for SNP model)'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=32,
        help='Embedding dimension (for Haploblock model)'
    )
    parser.add_argument(
        '--transformer_dim',
        type=int,
        default=128,
        help='Transformer hidden dimension'
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

    # Federated learning strategy arguments
    parser.add_argument(
        '--strategy',
        type=str,
        default='fedavg',
        choices=['fedavg', 'fedprox', 'scaffold', 'fedopt'],
        help='Federated learning strategy (fedavg, fedprox, scaffold, or fedopt)'
    )
    parser.add_argument(
        '--mu',
        type=float,
        default=0.01,
        help='FedProx proximal term coefficient (only used with --strategy fedprox)'
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
    model_type: str,
    n_features: int,
    vocab_sizes: list,
    encoding_dim: int,
    embedding_dim: int,
    transformer_dim: int,
    num_classes: int,
    architecture: str,
    learning_rate: float,
    weight_decay: float,
    use_focal_loss: bool,
    focal_alpha: float,
    focal_gamma: float,
    strategy: str = 'fedavg',
    mu: float = 0.0,
) -> pl.LightningModule:
    """
    Create Lightning model based on model type.

    Args:
        model_type: 'snp' (3D input) or 'haploblock' (2D cluster IDs)
        n_features: Number of features (SNPs or haploblocks)
        vocab_sizes: List of vocabulary sizes per haploblock (for haploblock model)
        encoding_dim: Genotype encoding dimension (for SNP model)
        embedding_dim: Embedding dimension (for haploblock model)
        transformer_dim: Transformer hidden dimension
        num_classes: Number of classes
        architecture: Model architecture
        learning_rate: Learning rate
        weight_decay: Weight decay
        use_focal_loss: Whether to use focal loss
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
        strategy: Federated learning strategy
        mu: FedProx proximal coefficient

    Returns:
        Lightning module instance
    """
    # Set mu based on strategy
    effective_mu = mu if strategy == 'fedprox' else 0.0

    if model_type == 'haploblock':
        # Use HaploblockLightningModule for 2D cluster ID input
        model = HaploblockLightningModule(
            n_haploblocks=n_features,
            vocab_sizes=vocab_sizes,
            embedding_dim=embedding_dim,
            transformer_dim=transformer_dim,
            num_classes=num_classes,
            architecture=architecture if architecture in ['transformer', 'cnn_transformer'] else 'transformer',
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            scheduler_type='cosine',
            mu=effective_mu,
        )
        logger.info(f"Created HaploblockLightningModule: arch={architecture}, n_haploblocks={n_features}")
        logger.info(f"Vocab sizes: {vocab_sizes[:5]}... (total {len(vocab_sizes)})")
    else:
        # Use SNPLightningModule for 3D input
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
            mu=effective_mu,
        )
        logger.info(f"Created SNPLightningModule: arch={architecture}, n_snps={n_features}")

    logger.info(f"Strategy: {strategy}, mu={effective_mu}")
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
        from nvflare.app_common.abstract.fl_model import ParamsType
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

    # Determine if data is 2D (haploblock) or 3D (SNP)
    if len(X_sample.shape) == 2:
        # 2D data: (batch, n_haploblocks) - cluster IDs
        logger.info(f"Detected 2D data (haploblock cluster IDs): shape={X_sample.shape}")
        actual_encoding_dim = None

        # Compute vocab sizes from all training data
        all_data = data_module.train_dataset.tensors[0]
        vocab_sizes = []
        for i in range(n_features):
            max_val = int(all_data[:, i].max().item()) + 1
            vocab_sizes.append(max(max_val + 1, 2))  # +1 for padding, min 2
        logger.info(f"Computed vocab sizes: {vocab_sizes[:5]}... max={max(vocab_sizes)}")
    else:
        # 3D data: (batch, n_snps, encoding_dim)
        actual_encoding_dim = X_sample.shape[2]
        vocab_sizes = [1] * n_features  # Not used for SNP model
        logger.info(f"Detected 3D data (SNP): shape={X_sample.shape}")

    logger.info(f"Data loaded: n_features={n_features}")
    logger.info(f"Train samples: {len(data_module.train_dataset)}, Val samples: {len(data_module.val_dataset)}")

    # Step 4: Create model with federated learning strategy
    model = create_model(
        model_type=args.model_type,
        n_features=n_features,
        vocab_sizes=vocab_sizes,
        encoding_dim=actual_encoding_dim or args.encoding_dim,
        embedding_dim=args.embedding_dim,
        transformer_dim=args.transformer_dim,
        num_classes=args.num_classes,
        architecture=args.architecture,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        strategy=args.strategy,
        mu=args.mu,
    )

    # Step 5: Configure accelerator
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accelerator = 'cpu'
        devices = 'auto'
        logger.info("Using CPU")

    # Step 6: Federated learning loop (using standard NVFlare Client API)
    # Reference: https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type/client_api
    logger.info("Starting federated learning loop...")
    round_num = 0
    total_steps = 0

    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        round_num = input_model.current_round if hasattr(input_model, 'current_round') else round_num
        logger.info(f"\n{'='*60}")
        logger.info(f"FL Round {round_num}: Received global model from server")
        logger.info(f"{'='*60}")

        # CRITICAL: Load global model weights into local model
        if input_model.params is not None and len(input_model.params) > 0:
            logger.info(f"Round {round_num}: Loading global model weights ({len(input_model.params)} parameters)...")
            try:
                # FedProx: Store global model state BEFORE loading into local model
                # This ensures we store the actual server weights, not the local model's weights
                if args.strategy == 'fedprox':
                    logger.info(f"Round {round_num}: Storing global model state for FedProx (mu={args.mu})...")
                    model.global_model_state = {
                        name: param.clone().detach().cpu()
                        for name, param in input_model.params.items()
                    }
                    logger.info(f"Round {round_num}: Global model state stored ({len(model.global_model_state)} parameters)")

                model.load_state_dict(input_model.params)
                logger.info(f"Round {round_num}: Global model weights loaded successfully")

            except RuntimeError as e:
                logger.error(f"Round {round_num}: Error loading state_dict: {e}")
                if round_num == 0:
                    logger.warning(f"Round {round_num}: Ignoring load error for initial round, using local initialization.")
                else:
                    raise e
        else:
            logger.info(f"Round {round_num}: No global weights received (or empty params), using initial/local weights")

        # Create Trainer for THIS round
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.checkpoint_dir) / site_name,
            filename=f'snp-round{round_num}-{{epoch:02d}}-{{val_loss:.4f}}',
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

        # Local training
        logger.info(f"Round {round_num}: Starting local training ({args.local_epochs} epochs)...")
        trainer.fit(model, datamodule=data_module)
        logger.info(f"Round {round_num}: Local training completed")

        # Calculate steps for this round
        steps_this_round = args.local_epochs * len(data_module.train_dataloader())
        total_steps += steps_this_round

        # Create output FLModel with updated weights
        # Reference: https://nvflare.readthedocs.io/en/main/release_notes/flare_240
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            params_type=ParamsType.FULL,
            metrics={"NUM_STEPS_CURRENT_ROUND": steps_this_round},
        )

        # Send model back to server
        logger.info(f"Round {round_num}: Sending model updates to server...")
        flare.send(output_model)
        logger.info(f"Round {round_num}: Model updates sent successfully")

        # Clear global_model_state to prevent memory leak
        if hasattr(model, 'global_model_state') and model.global_model_state is not None:
            model.global_model_state = None
            logger.debug(f"Round {round_num}: Cleared global_model_state to free memory")

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

