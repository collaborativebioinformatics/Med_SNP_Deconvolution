#!/usr/bin/env python3
"""
NVFlare Lightning Client with Scaffold Algorithm

Scaffold (Stochastic Controlled Averaging for Federated Learning) uses control variates
to correct for client drift in heterogeneous federated learning settings.

Algorithm Overview:
1. Each client maintains control variates c_i (same shape as model parameters)
2. Server maintains global control variate c
3. Client update: gradient_corrected = gradient + c - c_i
4. After local training: c_i_new = c_i - c + (1/K*lr) * (w_global - w_i)
5. Send delta_c = c_i_new - c_i to server
6. Server aggregates: c_new = c + (1/n) * sum(delta_c_i)

Reference:
    Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    ICML 2020. https://arxiv.org/abs/1910.06378

NVFlare Integration:
    - Uses AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL for global control variate c
    - Uses AlgorithmConstants.SCAFFOLD_CTRL_DIFF for client control variate difference delta_c
    - Compatible with NVFlare Client API mode

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
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


class ScaffoldOptimizer:
    """
    Wrapper optimizer that applies Scaffold control variate corrections.

    This optimizer wraps any base optimizer (e.g., AdamW, SGD) and applies
    the Scaffold correction: gradient_corrected = gradient + c - c_i

    Args:
        base_optimizer: The underlying optimizer (e.g., torch.optim.AdamW)
        params: Model parameters to optimize
        global_control: Global control variate c (from server)
        local_control: Local control variate c_i (client-specific)

    Example:
        >>> base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scaffold_opt = ScaffoldOptimizer(base_opt, model.parameters(), c_global, c_local)
        >>> scaffold_opt.step()  # Applies Scaffold correction before optimization
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        params: Any,
        global_control: Dict[str, torch.Tensor],
        local_control: Dict[str, torch.Tensor]
    ):
        self.base_optimizer = base_optimizer
        self.params = list(params)
        self.global_control = global_control
        self.local_control = local_control

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients of base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """
        Apply Scaffold correction and perform optimization step.

        For each parameter p with gradient g:
            g_corrected = g + c[p] - c_i[p]

        Then call base_optimizer.step() with corrected gradients.
        """
        # Apply Scaffold correction: gradient + c - c_i
        with torch.no_grad():
            for param_group in self.base_optimizer.param_groups:
                for p in param_group['params']:
                    if p.grad is None:
                        continue

                    # Find parameter name for control variates
                    param_name = None
                    for name, param in zip(self._get_param_names(), self.params):
                        if p is param:
                            param_name = name
                            break

                    if param_name is None:
                        continue

                    # Apply correction: g = g + c - c_i
                    if param_name in self.global_control and param_name in self.local_control:
                        c_global = self.global_control[param_name].to(p.device)
                        c_local = self.local_control[param_name].to(p.device)
                        p.grad.add_(c_global - c_local)

        # Perform optimization with corrected gradients
        loss = self.base_optimizer.step(closure)
        return loss

    def _get_param_names(self):
        """Get parameter names from the model."""
        names = []
        for param_group in self.base_optimizer.param_groups:
            for p in param_group['params']:
                # Try to find the parameter name
                for name, param in self._named_parameters.items():
                    if p is param:
                        names.append(name)
                        break
        return names

    def state_dict(self):
        """Get optimizer state dict."""
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state dict."""
        self.base_optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Access base optimizer param groups."""
        return self.base_optimizer.param_groups


class ScaffoldLightningModule(SNPLightningModule):
    """
    SNP Lightning Module with Scaffold algorithm support.

    Extends SNPLightningModule to:
    1. Maintain local control variates c_i
    2. Apply control variate corrections during training
    3. Compute control variate updates after training

    Args:
        Same as SNPLightningModule, plus:
        learning_rate_for_scaffold: Learning rate for Scaffold control variate updates
            (should match the effective learning rate used in training)
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
        learning_rate_for_scaffold: Optional[float] = None,
        **model_kwargs
    ):
        super().__init__(
            n_snps=n_snps,
            encoding_dim=encoding_dim,
            num_classes=num_classes,
            architecture=architecture,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            scheduler_type=scheduler_type,
            mu=0.0,  # Scaffold doesn't use FedProx proximal term
            **model_kwargs
        )

        # Scaffold-specific parameters
        self.scaffold_lr = learning_rate_for_scaffold or learning_rate

        # Control variates (initialized to zeros)
        self.local_control: Dict[str, torch.Tensor] = {}
        self.global_control: Dict[str, torch.Tensor] = {}

        # Store global model weights for computing delta_c
        self.global_model_weights: Optional[Dict[str, torch.Tensor]] = None

        # Number of local training steps (K in the algorithm)
        self.num_local_steps = 0

        self._initialize_control_variates()

    def _initialize_control_variates(self):
        """Initialize local control variates c_i to zeros."""
        logger.info("Initializing Scaffold control variates to zeros...")
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.local_control[name] = torch.zeros_like(
                    param.data, device='cpu'
                )
        logger.info(f"Initialized {len(self.local_control)} control variates")

    def load_global_control(self, global_control: Dict[str, torch.Tensor]):
        """
        Load global control variate c from server.

        Args:
            global_control: Global control variate dictionary {param_name: tensor}
        """
        logger.info(f"Loading global control variates ({len(global_control)} parameters)...")
        self.global_control = {
            name: tensor.clone().cpu()
            for name, tensor in global_control.items()
        }
        logger.info("Global control variates loaded successfully")

    def store_global_weights(self):
        """Store current model weights as global weights (before local training)."""
        logger.info("Storing global model weights for Scaffold...")
        self.global_model_weights = {
            name: param.clone().detach().cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        logger.info(f"Stored {len(self.global_model_weights)} global model weights")

    def compute_control_variate_update(
        self
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute updated local control variates and their difference.

        Formula: c_i_new = c_i - c + (1/(K*lr)) * (w_global - w_local)
        where K is the number of local training steps

        Returns:
            (new_local_control, delta_control):
                - new_local_control: Updated c_i
                - delta_control: c_i_new - c_i (to send to server)
        """
        if self.global_model_weights is None:
            logger.warning("Global model weights not stored, cannot compute control variate update")
            return self.local_control, {}

        if self.num_local_steps == 0:
            logger.warning("No local training steps recorded, using num_local_steps=1")
            self.num_local_steps = 1

        logger.info(f"Computing Scaffold control variate update (K={self.num_local_steps}, lr={self.scaffold_lr})...")

        new_local_control = {}
        delta_control = {}

        # Move model to CPU for computation
        current_weights = {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

        for name in self.local_control.keys():
            if name not in self.global_model_weights or name not in current_weights:
                logger.warning(f"Skipping parameter {name} - not found in weights")
                new_local_control[name] = self.local_control[name]
                delta_control[name] = torch.zeros_like(self.local_control[name])
                continue

            c_i = self.local_control[name]
            c = self.global_control.get(name, torch.zeros_like(c_i))
            w_global = self.global_model_weights[name]
            w_local = current_weights[name]

            # c_i_new = c_i - c + (1/(K*lr)) * (w_global - w_local)
            weight_diff = w_global - w_local
            correction = weight_diff / (self.num_local_steps * self.scaffold_lr)
            c_i_new = c_i - c + correction

            new_local_control[name] = c_i_new
            delta_control[name] = c_i_new - c_i

        logger.info(f"Computed control variate updates for {len(delta_control)} parameters")

        # Update local control variates
        self.local_control = new_local_control

        # Reset step counter
        self.num_local_steps = 0

        return new_local_control, delta_control

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Track number of local training steps."""
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.num_local_steps += 1

    def configure_optimizers(self):
        """
        Configure optimizer with Scaffold control variate corrections.

        Note: We create the base optimizer but will wrap it with ScaffoldOptimizer
        during training when global control variates are available.
        """
        # Create base optimizer (same as parent class)
        base_optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Store reference to base optimizer for Scaffold wrapping
        self.base_optimizer = base_optimizer

        # Scheduler configuration (same as parent)
        if self.hparams.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                base_optimizer, T_0=10, T_mult=2, eta_min=1e-7
            )
            return {
                'optimizer': base_optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
            }
        elif self.hparams.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                base_optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
            )
            return {
                'optimizer': base_optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}
            }

        return base_optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None
    ):
        """
        Override optimizer step to apply Scaffold correction.

        This is called by Lightning during training. We apply the Scaffold
        correction: gradient = gradient + c - c_i before the optimizer step.
        """
        # Apply Scaffold correction to gradients
        if self.global_control and self.local_control:
            with torch.no_grad():
                for name, param in self.named_parameters():
                    if param.grad is None or not param.requires_grad:
                        continue

                    if name in self.global_control and name in self.local_control:
                        c_global = self.global_control[name].to(param.device)
                        c_local = self.local_control[name].to(param.device)
                        # Apply correction: g = g + c - c_i
                        param.grad.add_(c_global - c_local)

        # Call base optimizer step
        optimizer.step(closure=optimizer_closure)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='NVFlare Lightning Client with Scaffold Algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
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


def create_scaffold_model(
    n_features: int,
    encoding_dim: int,
    num_classes: int,
    architecture: str,
    learning_rate: float,
    weight_decay: float,
    use_focal_loss: bool,
    focal_alpha: float,
    focal_gamma: float,
) -> ScaffoldLightningModule:
    """
    Create Scaffold-enabled SNP Lightning model.

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
        ScaffoldLightningModule instance
    """
    model = ScaffoldLightningModule(
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
        learning_rate_for_scaffold=learning_rate,
    )
    logger.info(f"Created Scaffold model: arch={architecture}, n_features={n_features}, classes={num_classes}")
    return model


def run_scaffold_client(args):
    """
    Run federated learning client with Scaffold algorithm.

    Scaffold Algorithm Flow:
    1. Initialize local control variates c_i to zeros
    2. Receive global model weights w and global control variate c from server
    3. Store global weights for later delta_c computation
    4. Train locally with gradient correction: g = g + c - c_i
    5. Compute control variate update: c_i_new = c_i - c + (1/K*lr) * (w_global - w_local)
    6. Send updated weights w_local and delta_c = c_i_new - c_i to server

    Args:
        args: Command line arguments
    """
    try:
        import nvflare.client as flare
        from nvflare.app_common.abstract.fl_model import ParamsType
        from nvflare.app_common.app_constant import AlgorithmConstants
    except ImportError:
        logger.error("NVFlare not installed. Install with: pip install nvflare")
        raise RuntimeError("NVFlare package is required for federated training")

    # Step 1: Initialize NVFlare client
    logger.info("Initializing NVFlare Scaffold client...")
    flare.init()

    # Step 2: Get site name
    site_name = flare.get_site_name()
    logger.info(f"=== NVFlare Scaffold Client Started: {site_name} ===")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Feature type: {args.feature_type}")
    logger.info(f"Local epochs per round: {args.local_epochs}")

    # Step 3: Create data module
    data_module = SNPFederatedDataModule(
        data_dir=args.data_dir,
        site_name=site_name,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup()

    # Get feature dimensions
    sample_batch = next(iter(data_module.train_dataloader()))
    X_sample = sample_batch[0]
    n_features = X_sample.shape[1]
    actual_encoding_dim = X_sample.shape[2]

    logger.info(f"Data loaded: n_features={n_features}, encoding_dim={actual_encoding_dim}")
    logger.info(f"Train samples: {len(data_module.train_dataset)}, Val samples: {len(data_module.val_dataset)}")

    # Step 4: Create Scaffold model
    model = create_scaffold_model(
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

    # Step 5: Configure accelerator
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        accelerator = 'cpu'
        devices = 'auto'
        logger.info("Using CPU")

    # Step 6: Scaffold federated learning loop
    logger.info("Starting Scaffold federated learning loop...")
    round_num = 0

    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        round_num = input_model.current_round if hasattr(input_model, 'current_round') else round_num
        logger.info(f"\n{'='*60}")
        logger.info(f"Scaffold Round {round_num}: Received global model from server")
        logger.info(f"{'='*60}")

        # Load global model weights
        if input_model.params is not None and len(input_model.params) > 0:
            logger.info(f"Round {round_num}: Loading global model weights...")
            try:
                model.load_state_dict(input_model.params)
                logger.info(f"Round {round_num}: Global model weights loaded successfully")
            except RuntimeError as e:
                logger.error(f"Round {round_num}: Error loading state_dict: {e}")
                if round_num == 0:
                    logger.warning(f"Round {round_num}: Using local initialization.")
                else:
                    raise e

        # Scaffold: Load global control variate c from server metadata
        if hasattr(input_model, 'meta') and input_model.meta is not None:
            if AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL in input_model.meta:
                global_control = input_model.meta[AlgorithmConstants.SCAFFOLD_CTRL_GLOBAL]
                logger.info(f"Round {round_num}: Loading global control variate...")
                model.load_global_control(global_control)
            else:
                logger.info(f"Round {round_num}: No global control variate in metadata (first round?)")
                # Initialize global control to zeros if not provided
                model.global_control = {
                    name: torch.zeros_like(tensor)
                    for name, tensor in model.local_control.items()
                }

        # Scaffold: Store global weights before local training
        model.store_global_weights()

        # Create Trainer for this round
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.checkpoint_dir) / site_name,
            filename=f'scaffold-round{round_num}-{{epoch:02d}}-{{val_loss:.4f}}',
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

        # Local training with Scaffold gradient correction
        logger.info(f"Round {round_num}: Starting Scaffold local training...")
        trainer.fit(model, datamodule=data_module)
        logger.info(f"Round {round_num}: Local training completed")

        # Scaffold: Compute control variate update
        logger.info(f"Round {round_num}: Computing Scaffold control variate update...")
        new_local_control, delta_control = model.compute_control_variate_update()
        logger.info(f"Round {round_num}: Control variate update computed")

        # Create output FLModel with updated weights and delta_c
        steps_this_round = args.local_epochs * len(data_module.train_dataloader())

        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            params_type=ParamsType.FULL,
            metrics={
                "NUM_STEPS_CURRENT_ROUND": steps_this_round,
            },
            meta={
                AlgorithmConstants.SCAFFOLD_CTRL_DIFF: delta_control,
            }
        )

        # Send model back to server
        logger.info(f"Round {round_num}: Sending model updates and control variate diff to server...")
        flare.send(output_model)
        logger.info(f"Round {round_num}: Model and Scaffold updates sent successfully")

    logger.info(f"\n{'='*60}")
    logger.info(f"Scaffold federated learning completed after {round_num} rounds")
    logger.info(f"Client {site_name} finished successfully")
    logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("="*60)
    logger.info("NVFlare Scaffold Client for SNP Deconvolution")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Lightning version: {pl.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    logger.info("="*60)

    # Run Scaffold federated client
    run_scaffold_client(args)


if __name__ == "__main__":
    main()
