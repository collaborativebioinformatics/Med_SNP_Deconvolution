"""
Example Usage: Complete Training Pipeline with bf16 Support

Demonstrates:
1. Model creation with GPU optimization
2. Data preparation and loading
3. Training with bf16 mixed precision
4. SNP importance extraction
5. Model evaluation and interpretation

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
from pathlib import Path
import logging

# Import attention_dl module
from gpu_optimized_models import create_gpu_optimized_model, create_multi_gpu_model
from gpu_trainer import MultiGPUSNPTrainer
from memory_manager import GPUMemoryManager, MemoryEfficientDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 1000,
    n_snps: int = 10000,
    encoding_dim: int = 8,
    num_classes: int = 2
) -> tuple:
    """
    Generate synthetic SNP data for testing.

    Args:
        n_samples: Number of samples
        n_snps: Number of SNPs
        encoding_dim: Encoding dimension
        num_classes: Number of classes

    Returns:
        (X_train, y_train, X_val, y_val)
    """
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_snps} SNPs")

    # Generate random genotype data (encoded)
    X = torch.randn(n_samples, n_snps, encoding_dim)
    y = torch.randint(0, num_classes, (n_samples,))

    # Split into train/val
    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    return X_train, y_train, X_val, y_val


def example_basic_training():
    """
    Example 1: Basic training with GPU optimization and bf16
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Training with bf16 Support")
    print("="*80)

    # Configuration
    n_snps = 5000
    encoding_dim = 8
    num_classes = 2
    num_epochs = 10

    # Generate data
    X_train, y_train, X_val, y_val = generate_synthetic_data(
        n_samples=500,
        n_snps=n_snps,
        encoding_dim=encoding_dim
    )

    # Calculate optimal batch size
    batch_size = GPUMemoryManager.get_optimal_batch_size(
        n_snps=n_snps,
        encoding_dim=encoding_dim,
        dtype=torch.bfloat16
    )
    logger.info(f"Using batch size: {batch_size}")

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = MemoryEfficientDataLoader.create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = MemoryEfficientDataLoader.create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Create GPU-optimized model
    logger.info("Creating GPU-optimized model...")
    model = create_gpu_optimized_model(
        n_snps=n_snps,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        architecture='cnn_transformer',
        use_amp=True,
        amp_dtype=torch.bfloat16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Log memory before training
    GPUMemoryManager.log_memory_usage("Before training")

    # Create trainer
    trainer = MultiGPUSNPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gpu_ids=[0],
        learning_rate=1e-4,
        use_amp=True,
        amp_dtype=torch.bfloat16,
        use_focal_loss=True
    )

    # Train
    logger.info("Starting training...")
    checkpoint_dir = Path("/tmp/snp_checkpoints")
    history = trainer.train(
        num_epochs=num_epochs,
        early_stopping_patience=5,
        checkpoint_dir=checkpoint_dir
    )

    # Log memory after training
    GPUMemoryManager.log_memory_usage("After training")

    # Print results
    print("\nTraining Results:")
    print(f"  Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"  Best Val Acc: {trainer.best_val_acc:.2f}%")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final Val Loss: {history['val_loss'][-1]:.4f}")


def example_multi_gpu_training():
    """
    Example 2: Multi-GPU training
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-GPU Training")
    print("="*80)

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        logger.warning("Multi-GPU not available. Skipping example.")
        return

    # Configuration
    n_snps = 10000
    encoding_dim = 8
    gpu_ids = [0, 1]  # Use 2 GPUs

    # Generate data
    X_train, y_train, X_val, y_val = generate_synthetic_data(
        n_samples=1000,
        n_snps=n_snps
    )

    # Create data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = MemoryEfficientDataLoader.create_dataloader(
        train_dataset, batch_size=batch_size
    )
    val_loader = MemoryEfficientDataLoader.create_dataloader(
        val_dataset, batch_size=batch_size
    )

    # Create model
    model = create_gpu_optimized_model(
        n_snps=n_snps,
        encoding_dim=encoding_dim,
        architecture='cnn_transformer',
        use_amp=True,
        device='cuda:0'
    )

    # Wrap for multi-GPU
    model = create_multi_gpu_model(model, gpu_ids=gpu_ids)

    # Train
    trainer = MultiGPUSNPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gpu_ids=gpu_ids,
        learning_rate=1e-4,
        use_amp=True
    )

    history = trainer.train(num_epochs=5)

    print(f"\nMulti-GPU Training Complete:")
    print(f"  Best Val Loss: {trainer.best_val_loss:.4f}")


def example_snp_importance_extraction():
    """
    Example 3: Extract SNP importance scores
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: SNP Importance Extraction")
    print("="*80)

    # Configuration
    n_snps = 1000
    encoding_dim = 8

    # Generate data
    X_train, y_train, X_val, y_val = generate_synthetic_data(
        n_samples=200,
        n_snps=n_snps
    )

    # Create and train model (simplified)
    model = create_gpu_optimized_model(
        n_snps=n_snps,
        encoding_dim=encoding_dim,
        architecture='cnn_transformer'
    )

    # Create data loader
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = MemoryEfficientDataLoader.create_dataloader(
        val_dataset, batch_size=16
    )

    # Create trainer (just for importance extraction)
    trainer = MultiGPUSNPTrainer(
        model=model,
        train_loader=val_loader,  # Dummy
        val_loader=val_loader,
        gpu_ids=[0] if torch.cuda.is_available() else None
    )

    # Extract SNP importance
    logger.info("Extracting SNP importance scores...")
    importance_dict = trainer.get_snp_importance(
        data_loader=val_loader,
        method='attention',
        aggregate='mean'
    )

    # Get top SNPs
    sorted_snps = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_snps[:10]

    print("\nTop 10 Most Important SNPs:")
    for idx, (snp_idx, score) in enumerate(top_10, 1):
        print(f"  {idx}. SNP {snp_idx}: {score:.4f}")


def example_memory_profiling():
    """
    Example 4: Memory profiling and optimization
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Memory Profiling")
    print("="*80)

    if not torch.cuda.is_available():
        logger.warning("GPU not available. Skipping memory profiling.")
        return

    # Test different SNP counts
    snp_counts = [1000, 5000, 10000, 50000]

    print("\nOptimal Batch Sizes for Different SNP Counts:")
    print(f"{'SNPs':<10} {'Min Batch':<12} {'Optimal':<12} {'Max Batch':<12} {'Memory Est':<15}")
    print("-" * 70)

    for n_snps in snp_counts:
        min_bs, opt_bs, max_bs = GPUMemoryManager.suggest_batch_size_range(
            n_snps=n_snps,
            encoding_dim=8,
            dtype=torch.bfloat16
        )

        mem_est = GPUMemoryManager.estimate_memory_usage(
            batch_size=opt_bs,
            n_snps=n_snps,
            encoding_dim=8,
            dtype=torch.bfloat16
        )

        print(f"{n_snps:<10} {min_bs:<12} {opt_bs:<12} {max_bs:<12} {mem_est['total_gb']:.2f}GB")

    # Profile model forward pass
    print("\nProfiling Model Forward Pass:")
    n_snps = 5000
    model = create_gpu_optimized_model(n_snps=n_snps, encoding_dim=8)
    x = torch.randn(16, n_snps, 8).cuda()

    def forward_pass():
        with torch.no_grad():
            return model(x)

    result, mem_before, mem_after = GPUMemoryManager.profile_memory_usage(
        forward_pass
    )

    print(f"  Memory before: {mem_before.allocated_gb:.2f}GB")
    print(f"  Memory after: {mem_after.allocated_gb:.2f}GB")
    print(f"  Memory delta: {mem_after.allocated_gb - mem_before.allocated_gb:.2f}GB")
    print(f"  Peak memory: {mem_after.max_allocated_gb:.2f}GB")


def example_different_architectures():
    """
    Example 5: Compare different architectures
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Comparing Architectures")
    print("="*80)

    n_snps = 1000
    encoding_dim = 8
    architectures = ['cnn', 'cnn_transformer', 'gnn']

    print(f"\n{'Architecture':<20} {'Parameters':<15} {'Memory (bf16)':<15}")
    print("-" * 50)

    for arch in architectures:
        try:
            model = create_gpu_optimized_model(
                n_snps=n_snps,
                encoding_dim=encoding_dim,
                architecture=arch,
                use_amp=True
            )

            n_params = sum(p.numel() for p in model.parameters())

            mem_est = GPUMemoryManager.estimate_memory_usage(
                batch_size=32,
                n_snps=n_snps,
                encoding_dim=encoding_dim,
                model_params=n_params,
                dtype=torch.bfloat16
            )

            print(f"{arch:<20} {n_params:>12,}   {mem_est['total_gb']:>10.2f}GB")

        except Exception as e:
            print(f"{arch:<20} Error: {e}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("SNP DECONVOLUTION: ATTENTION DL MODULE EXAMPLES")
    print("bf16 (bfloat16) Support for A100/H100 GPUs")
    print("="*80)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
        total_gb, free_gb = GPUMemoryManager.get_gpu_memory_gb()
        print(f"Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
    else:
        print("\nNo GPU available. Running on CPU.")

    try:
        # Run examples
        example_basic_training()
        example_snp_importance_extraction()
        example_memory_profiling()
        example_different_architectures()
        # example_multi_gpu_training()  # Uncomment if you have multiple GPUs

        print("\n" + "="*80)
        print("All examples completed successfully!")
        print("="*80)

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
