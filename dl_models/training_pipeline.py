"""
Training Pipeline for Interpretable SNP Models

This module provides a complete training pipeline with:
- Data loading and preprocessing
- Model training with early stopping
- Validation and evaluation
- Interpretability analysis
- Model versioning for A/B testing

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from snp_interpretable_models import (
    InterpretableSNPModel,
    GenotypeEncoder,
    FocalLoss,
    create_ld_adjacency_matrix
)


# ============================================================================
# 1. DATASET AND DATA LOADING
# ============================================================================

class SNPDataset(Dataset):
    """Dataset for SNP/genotype data"""

    def __init__(self,
                 genotypes: np.ndarray,
                 phenotypes: np.ndarray,
                 encoding: str = 'haplotype',
                 normalize: bool = True):
        """
        Args:
            genotypes: (n_samples, n_snps) genotype matrix
            phenotypes: (n_samples,) phenotype labels
            encoding: Encoding method ('additive', 'onehot', 'haplotype')
            normalize: Whether to normalize genotypes
        """
        self.genotypes = genotypes
        self.phenotypes = phenotypes
        self.encoding = encoding
        self.normalize = normalize

        self.encoder = GenotypeEncoder()
        self._preprocess()

    def _preprocess(self):
        """Encode and normalize genotypes"""
        if self.encoding == 'additive':
            self.encoded = self.encoder.additive_encoding(self.genotypes)
        elif self.encoding == 'onehot':
            self.encoded = self.encoder.onehot_encoding(self.genotypes)
        elif self.encoding == 'haplotype':
            self.encoded = self.encoder.haplotype_encoding(self.genotypes)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        if self.normalize:
            self.encoded = self.encoder.normalize_genotypes(self.encoded, method='standardize')

        self.labels = torch.from_numpy(self.phenotypes).long()

    def __len__(self) -> int:
        return len(self.genotypes)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded[idx], self.labels[idx]


# ============================================================================
# 2. TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class MetricsTracker:
    """Track training metrics"""

    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

    def plot(self, save_path: Optional[Path] = None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()

        plt.close()


# ============================================================================
# 3. TRAINER CLASS
# ============================================================================

class SNPModelTrainer:
    """Complete training pipeline for SNP models with bf16 AMP support"""

    def __init__(self,
                 model: InterpretableSNPModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 use_amp: bool = True,
                 amp_dtype: torch.dtype = torch.bfloat16):
        """
        Initialize trainer with optional AMP (Automatic Mixed Precision).

        Args:
            model: SNP model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            use_focal_loss: Use focal loss for class imbalance
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            use_amp: Enable Automatic Mixed Precision
            amp_dtype: AMP dtype (torch.bfloat16 for A100/H100, torch.float16 for older GPUs)

        Note:
            For bf16 (bfloat16) on A100/H100:
            - Native hardware support, no loss scaling needed
            - Same dynamic range as fp32, better numerical stability
            - GradScaler is NOT needed for bf16
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # AMP configuration
        self.use_amp = use_amp and device != 'cpu'
        self.amp_dtype = amp_dtype

        # Note: GradScaler is only needed for float16, not bfloat16
        # bf16 has same dynamic range as fp32, so no scaling needed
        self.use_grad_scaler = self.use_amp and amp_dtype == torch.float16
        if self.use_grad_scaler:
            self.scaler = torch.amp.GradScaler()
        else:
            self.scaler = None

        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Tracking
        self.metrics = MetricsTracker()
        self.early_stopping = EarlyStopping(patience=15)
        self.best_model_state = None
        self.best_val_loss = float('inf')

        # Log AMP configuration
        if self.use_amp:
            dtype_name = 'bfloat16' if amp_dtype == torch.bfloat16 else 'float16'
            print(f"AMP enabled with {dtype_name} precision")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with optional AMP (bf16/fp16)"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                # AMP forward pass (bf16 or fp16)
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                if self.use_grad_scaler:
                    # fp16: use GradScaler
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # bf16: no scaling needed
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
            else:
                # Standard fp32 training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """Validate the model with optional AMP"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self,
             num_epochs: int = 100,
             checkpoint_dir: Optional[Path] = None,
             verbose: bool = True) -> Dict:
        """
        Complete training loop

        Args:
            num_epochs: Maximum number of epochs
            checkpoint_dir: Directory to save checkpoints
            verbose: Print progress
        Returns:
            Training history
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        print(f"Training on device: {self.device}")
        print(f"Model architecture: {self.model.architecture}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*80)

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Track metrics
            self.metrics.update({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'learning_rate': current_lr
            })

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()

                if checkpoint_dir:
                    checkpoint_path = checkpoint_dir / "best_model.pt"
                    self.save_checkpoint(checkpoint_path, epoch, val_loss, val_acc)

            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                      f"LR: {current_lr:.2e} | Time: {epoch_time:.2f}s")

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        # Training complete
        total_time = time.time() - start_time
        print("="*80)
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        # Plot training curves
        if checkpoint_dir:
            self.metrics.plot(checkpoint_dir / "training_curves.png")

        return self.metrics.history

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'metrics_history': self.metrics.history,
            'model_config': {
                'n_snps': self.model.n_snps,
                'encoding_dim': self.model.encoding_dim,
                'architecture': self.model.architecture
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint


# ============================================================================
# 4. EVALUATION AND INTERPRETABILITY
# ============================================================================

class ModelEvaluator:
    """Evaluate model and extract interpretability"""

    def __init__(self, model: InterpretableSNPModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive evaluation

        Returns:
            Dictionary with accuracy, precision, recall, F1
        """
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                results = self.model.predict_with_interpretation(
                    inputs,
                    methods=['attention']
                )
                all_preds.append(results['predictions'].cpu())
                all_targets.append(targets)
                all_probs.append(results['probabilities'].cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        probs = torch.cat(all_probs).numpy()

        # Compute metrics
        accuracy = (preds == targets).mean() * 100
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average='weighted'
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # AUC for binary classification
        if probs.shape[1] == 2:
            metrics['auc'] = roc_auc_score(targets, probs[:, 1])

        return metrics

    def identify_top_snps(self,
                         data_loader: DataLoader,
                         top_k: int = 50,
                         method: str = 'ensemble') -> Dict[int, float]:
        """
        Identify most important SNPs across dataset

        Returns:
            Dictionary mapping SNP index to importance score
        """
        snp_importance_scores = {}

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)

                # Get importance for this batch
                indices, scores = self.model.identify_causal_snps(
                    inputs,
                    top_k=top_k,
                    method=method
                )

                # Aggregate scores
                for idx, score in zip(indices.cpu().numpy(), scores.cpu().numpy()):
                    if idx not in snp_importance_scores:
                        snp_importance_scores[idx] = []
                    snp_importance_scores[idx].append(score)

        # Average scores across batches
        avg_scores = {
            idx: np.mean(scores)
            for idx, scores in snp_importance_scores.items()
        }

        # Sort by importance
        sorted_snps = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_snps[:top_k])

    def visualize_snp_importance(self,
                                snp_importance: Dict[int, float],
                                snp_positions: Optional[Dict[int, int]] = None,
                                save_path: Optional[Path] = None):
        """
        Visualize SNP importance scores

        Args:
            snp_importance: Dict mapping SNP index to importance
            snp_positions: Dict mapping SNP index to genomic position
            save_path: Path to save figure
        """
        indices = list(snp_importance.keys())
        scores = list(snp_importance.values())

        fig, ax = plt.subplots(figsize=(12, 6))

        if snp_positions:
            positions = [snp_positions.get(idx, idx) for idx in indices]
            ax.scatter(positions, scores, alpha=0.6, s=50)
            ax.set_xlabel('Genomic Position')
        else:
            ax.bar(range(len(indices)), scores, alpha=0.7)
            ax.set_xlabel('SNP Index')

        ax.set_ylabel('Importance Score')
        ax.set_title('Top SNP Importance Scores')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SNP importance plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


# ============================================================================
# 5. MODEL VERSIONING FOR A/B TESTING
# ============================================================================

class ModelRegistry:
    """
    Manage multiple model versions for A/B testing and rollback
    """

    def __init__(self, registry_dir: Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_dir / "registry.json"
        self._load_metadata()

    def _load_metadata(self):
        """Load registry metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'models': {}}

    def _save_metadata(self):
        """Save registry metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def register_model(self,
                      model_path: Path,
                      version: str,
                      metrics: Dict[str, float],
                      config: Dict,
                      description: str = ""):
        """Register a new model version"""
        model_id = f"model_v{version}"
        timestamp = datetime.now().isoformat()

        self.metadata['models'][model_id] = {
            'version': version,
            'path': str(model_path),
            'timestamp': timestamp,
            'metrics': metrics,
            'config': config,
            'description': description,
            'status': 'candidate'  # candidate, production, archived
        }

        self._save_metadata()
        print(f"Model {model_id} registered successfully")

    def promote_to_production(self, version: str):
        """Promote model to production"""
        model_id = f"model_v{version}"

        # Demote current production model
        for mid, info in self.metadata['models'].items():
            if info['status'] == 'production':
                info['status'] = 'archived'

        # Promote new model
        self.metadata['models'][model_id]['status'] = 'production'
        self._save_metadata()
        print(f"Model {model_id} promoted to production")

    def get_production_model(self) -> Optional[Dict]:
        """Get current production model info"""
        for model_id, info in self.metadata['models'].items():
            if info['status'] == 'production':
                return {model_id: info}
        return None

    def list_models(self) -> Dict:
        """List all registered models"""
        return self.metadata['models']


# ============================================================================
# 6. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Complete training pipeline example"""

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Configuration
    n_samples = 1000
    n_snps = 500
    num_classes = 2
    encoding_dim = 8
    batch_size = 32

    print("="*80)
    print("SNP MODEL TRAINING PIPELINE")
    print("="*80)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic genotype data...")
    genotypes = np.random.randint(0, 3, size=(n_samples, n_snps))
    phenotypes = np.random.randint(0, num_classes, size=n_samples)

    # Create dataset
    dataset = SNPDataset(genotypes, phenotypes, encoding='haplotype', normalize=True)

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Train samples: {train_size}, Val samples: {val_size}, Test samples: {test_size}")

    # 2. Create model
    print("\n2. Creating interpretable SNP model...")
    model = InterpretableSNPModel(
        n_snps=n_snps,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        architecture='cnn_transformer',
        dropout=0.2
    )

    # 3. Train model
    print("\n3. Training model...")
    checkpoint_dir = Path("./checkpoints/snp_model_experiment_1")
    trainer = SNPModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        use_focal_loss=True
    )

    history = trainer.train(
        num_epochs=50,
        checkpoint_dir=checkpoint_dir,
        verbose=True
    )

    # 4. Evaluate model
    print("\n4. Evaluating model...")
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(test_loader)

    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # 5. Identify causal SNPs
    print("\n5. Identifying top causal SNPs...")
    top_snps = evaluator.identify_top_snps(test_loader, top_k=20, method='ensemble')

    print("\nTop 20 Causal SNPs:")
    for idx, (snp_idx, score) in enumerate(list(top_snps.items())[:20], 1):
        print(f"  {idx:2d}. SNP {snp_idx:4d}: {score:.6f}")

    # Visualize
    evaluator.visualize_snp_importance(
        top_snps,
        save_path=checkpoint_dir / "snp_importance.png"
    )

    # 6. Register model
    print("\n6. Registering model in registry...")
    registry = ModelRegistry(Path("./model_registry"))
    registry.register_model(
        model_path=checkpoint_dir / "best_model.pt",
        version="1.0.0",
        metrics=metrics,
        config={
            'architecture': 'cnn_transformer',
            'n_snps': n_snps,
            'encoding': 'haplotype',
            'encoding_dim': encoding_dim
        },
        description="Initial CNN-Transformer model for SNP analysis"
    )

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel saved to: {checkpoint_dir}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Test accuracy: {metrics['accuracy']:.2f}%")
