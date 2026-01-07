"""
NVFlare + PyTorch Lightning Integration for SNP Deconvolution

Federated learning using NVFlare's native Lightning support.
Reference: https://nvflare.readthedocs.io/en/2.7.0/hello-world/hello-lightning/

Usage:
    # Standalone
    python nvflare_lightning.py --standalone

    # Federated (via NVFlare job submission)
    # See job_config/ for deployment configuration
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping
from typing import Optional
import logging
import argparse
from pathlib import Path
import sys

sys.path.insert(0, '/Users/saltfish/Files/Coding/Haploblock_Clusters_ElixirBH25')
from snp_deconvolution.attention_dl.lightning_trainer import (
    SNPLightningModule,
    create_lightning_trainer
)

logger = logging.getLogger(__name__)


class SNPDataModule(pl.LightningDataModule):
    """
    LightningDataModule for SNP data.
    Each site has its own subset in federated learning.
    """

    def __init__(
        self,
        data_dir: str = 'data/snp_preprocessed',
        batch_size: int = 128,
        num_workers: int = 4,
        site_name: Optional[str] = None
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.site_name = site_name
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: Optional[str] = None):
        """Load data for current site."""
        if self.site_name:
            data_file = self.data_dir / f"{self.site_name}_data.pt"
            if data_file.exists():
                data = torch.load(data_file)
                self.train_data = (data['X_train'], data['y_train'])
                self.val_data = (data['X_val'], data['y_val'])
                self.test_data = (data['X_test'], data['y_test'])
                return

        data_file = self.data_dir / "snp_data.pt"
        if data_file.exists():
            data = torch.load(data_file)
            self.train_data = (data['X_train'], data['y_train'])
            self.val_data = (data['X_val'], data['y_val'])
            self.test_data = (data['X_test'], data['y_test'])
        else:
            # Dummy data for testing
            n_snps, encoding_dim = 1000, 8
            self.train_data = (torch.randn(800, n_snps, encoding_dim), torch.randint(0, 3, (800,)))
            self.val_data = (torch.randn(100, n_snps, encoding_dim), torch.randint(0, 3, (100,)))
            self.test_data = (torch.randn(100, n_snps, encoding_dim), torch.randint(0, 3, (100,)))

    def train_dataloader(self):
        if self.train_data is None:
            self.setup()
        return DataLoader(
            TensorDataset(*self.train_data),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        if self.val_data is None:
            self.setup()
        return DataLoader(
            TensorDataset(*self.val_data),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        if self.test_data is None:
            self.setup()
        return DataLoader(
            TensorDataset(*self.test_data),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )


def run_federated_client(
    data_dir: str,
    n_snps: int = 10000,
    num_classes: int = 3,
    local_epochs: int = 1,
    batch_size: int = 128
):
    """
    Run federated learning client with NVFlare.
    Uses flare.patch(trainer) for FL integration.
    """
    try:
        import nvflare.client.lightning as flare
    except ImportError:
        logger.error("NVFlare not installed. Run: pip install nvflare")
        raise

    flare.init()
    site_name = flare.get_site_name()
    logger.info(f"NVFlare client: {site_name}")

    dm = SNPDataModule(data_dir=data_dir, batch_size=batch_size, site_name=site_name)
    dm.setup()

    X_sample = dm.train_data[0]
    actual_n_snps = X_sample.shape[1]
    encoding_dim = X_sample.shape[2]

    model = SNPLightningModule(
        n_snps=actual_n_snps,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        architecture='cnn_transformer',
        learning_rate=1e-4
    )

    trainer = pl.Trainer(
        max_epochs=local_epochs,
        precision='bf16-mixed',
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=False,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, mode='min')]
    )

    flare.patch(trainer)

    while flare.is_running():
        input_model = flare.receive()
        logger.info(f"Round {input_model.current_round}: received global model")
        trainer.validate(model, datamodule=dm)
        trainer.fit(model, datamodule=dm)
        trainer.test(ckpt_path="best", datamodule=dm)

    logger.info("Federated training completed")


def run_standalone(
    data_dir: str = 'data/snp_preprocessed',
    output_dir: str = 'results/snp_lightning',
    n_snps: int = 10000,
    num_classes: int = 3,
    max_epochs: int = 100,
    batch_size: int = 128
):
    """Run standalone (non-federated) training."""
    logger.info("Running standalone training")

    dm = SNPDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.setup()

    X_sample = dm.train_data[0]
    actual_n_snps = X_sample.shape[1]
    encoding_dim = X_sample.shape[2]

    model = SNPLightningModule(
        n_snps=actual_n_snps,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        architecture='cnn_transformer',
        learning_rate=1e-4
    )

    trainer = create_lightning_trainer(
        output_dir=output_dir,
        max_epochs=max_epochs,
        precision='bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed',
        devices=1 if torch.cuda.is_available() else 'auto',
        early_stopping_patience=15
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    logger.info(f"Training complete. Results: {output_dir}")
    return model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SNP Training with Lightning')
    parser.add_argument('--standalone', action='store_true', help='Run standalone')
    parser.add_argument('--data-dir', type=str, default='data/snp_preprocessed')
    parser.add_argument('--output-dir', type=str, default='results/snp_lightning')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.standalone:
        run_standalone(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size
        )
    else:
        run_federated_client(data_dir=args.data_dir, batch_size=args.batch_size)
