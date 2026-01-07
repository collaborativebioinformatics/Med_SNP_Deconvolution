"""
Configuration for Attention-Based Deep Learning Module

Default configurations for different use cases and hardware setups.

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    n_snps: int
    encoding_dim: int = 8
    num_classes: int = 2
    architecture: str = 'cnn_transformer'  # 'cnn', 'cnn_transformer', 'gnn'

    # Architecture-specific parameters
    # CNN parameters
    cnn_channels: List[int] = None
    kernel_sizes: List[int] = None

    # Transformer parameters
    transformer_dim: int = 128
    num_transformer_layers: int = 4
    num_heads: int = 8

    # General parameters
    dropout: float = 0.2
    use_attention: bool = True

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128] if self.architecture == 'cnn' else [32, 64]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 5, 7] if self.architecture == 'cnn' else [5]


@dataclass
class GPUConfig:
    """GPU optimization configuration"""
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.bfloat16  # bf16 for A100/H100
    use_gradient_checkpointing: bool = False
    compile_model: bool = False
    gpu_ids: List[int] = None

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0] if torch.cuda.is_available() else []

        # Fallback to fp16 if bf16 not supported
        if self.amp_dtype == torch.bfloat16 and torch.cuda.is_available():
            if not torch.cuda.is_bf16_supported():
                print("bf16 not supported. Using fp16.")
                self.amp_dtype = torch.float16


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 100
    batch_size: Optional[int] = None  # Auto-calculate if None
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # Loss function
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Optimization
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 15

    # Scheduling
    use_cosine_scheduler: bool = True
    use_plateau_scheduler: bool = True

    # Checkpointing
    checkpoint_dir: Optional[str] = './checkpoints'
    save_best_only: bool = True

    # Logging
    verbose: int = 1


@dataclass
class DataConfig:
    """Data loading configuration"""
    num_workers: Optional[int] = None  # Auto-calculate if None
    pin_memory: bool = True
    prefetch_factor: int = 2
    shuffle_train: bool = True


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class PresetConfigs:
    """Preset configurations for common scenarios"""

    @staticmethod
    def small_dataset_cpu() -> dict:
        """
        Configuration for small datasets on CPU.

        Use case: < 10K SNPs, < 1K samples, CPU-only
        """
        return {
            'model': ModelConfig(
                n_snps=5000,
                encoding_dim=8,
                architecture='cnn',
                cnn_channels=[32, 64],
                dropout=0.3
            ),
            'gpu': GPUConfig(
                use_amp=False,
                compile_model=False,
                gpu_ids=[]
            ),
            'training': TrainingConfig(
                num_epochs=50,
                batch_size=16,
                learning_rate=1e-3,
                early_stopping_patience=10
            ),
            'data': DataConfig(
                num_workers=2,
                pin_memory=False
            )
        }

    @staticmethod
    def medium_dataset_single_gpu() -> dict:
        """
        Configuration for medium datasets on single GPU.

        Use case: 10K-50K SNPs, 1K-10K samples, single A100/H100
        """
        return {
            'model': ModelConfig(
                n_snps=25000,
                encoding_dim=8,
                architecture='cnn_transformer',
                transformer_dim=128,
                num_transformer_layers=4,
                dropout=0.2
            ),
            'gpu': GPUConfig(
                use_amp=True,
                amp_dtype=torch.bfloat16,
                use_gradient_checkpointing=False,
                compile_model=True,
                gpu_ids=[0]
            ),
            'training': TrainingConfig(
                num_epochs=100,
                batch_size=None,  # Auto-calculate
                learning_rate=1e-4,
                gradient_accumulation_steps=1,
                early_stopping_patience=15
            ),
            'data': DataConfig(
                num_workers=None,  # Auto-calculate
                pin_memory=True
            )
        }

    @staticmethod
    def large_dataset_multi_gpu() -> dict:
        """
        Configuration for large datasets on multiple GPUs.

        Use case: >50K SNPs, >10K samples, multiple A100/H100
        """
        return {
            'model': ModelConfig(
                n_snps=100000,
                encoding_dim=8,
                architecture='cnn_transformer',
                transformer_dim=256,
                num_transformer_layers=6,
                num_heads=16,
                dropout=0.2
            ),
            'gpu': GPUConfig(
                use_amp=True,
                amp_dtype=torch.bfloat16,
                use_gradient_checkpointing=True,  # Memory-efficient
                compile_model=True,
                gpu_ids=[0, 1, 2, 3]  # 4 GPUs
            ),
            'training': TrainingConfig(
                num_epochs=150,
                batch_size=None,  # Auto-calculate per GPU
                learning_rate=5e-5,
                gradient_accumulation_steps=2,
                early_stopping_patience=20
            ),
            'data': DataConfig(
                num_workers=None,  # Auto-calculate
                pin_memory=True,
                prefetch_factor=4
            )
        }

    @staticmethod
    def memory_constrained() -> dict:
        """
        Configuration for memory-constrained scenarios.

        Use case: Large SNP count but limited GPU memory
        """
        return {
            'model': ModelConfig(
                n_snps=50000,
                encoding_dim=8,
                architecture='cnn_transformer',
                transformer_dim=64,  # Smaller dimension
                num_transformer_layers=3,
                dropout=0.3
            ),
            'gpu': GPUConfig(
                use_amp=True,
                amp_dtype=torch.bfloat16,
                use_gradient_checkpointing=True,  # Save memory
                compile_model=False,
                gpu_ids=[0]
            ),
            'training': TrainingConfig(
                num_epochs=100,
                batch_size=8,  # Small batch
                learning_rate=1e-4,
                gradient_accumulation_steps=4,  # Effective batch = 32
                early_stopping_patience=15
            ),
            'data': DataConfig(
                num_workers=2,
                pin_memory=True
            )
        }

    @staticmethod
    def fast_prototyping() -> dict:
        """
        Configuration for fast prototyping and testing.

        Use case: Quick experiments, small subset
        """
        return {
            'model': ModelConfig(
                n_snps=1000,
                encoding_dim=8,
                architecture='cnn',
                cnn_channels=[16, 32],
                dropout=0.2
            ),
            'gpu': GPUConfig(
                use_amp=True,
                amp_dtype=torch.bfloat16,
                compile_model=False,
                gpu_ids=[0]
            ),
            'training': TrainingConfig(
                num_epochs=10,
                batch_size=32,
                learning_rate=1e-3,
                early_stopping_patience=5,
                verbose=2
            ),
            'data': DataConfig(
                num_workers=4,
                pin_memory=True
            )
        }

    @staticmethod
    def production_inference() -> dict:
        """
        Configuration for production inference.

        Use case: Deployed model, optimized for inference speed
        """
        return {
            'model': ModelConfig(
                n_snps=25000,
                encoding_dim=8,
                architecture='cnn_transformer',
                dropout=0.0  # Disable dropout for inference
            ),
            'gpu': GPUConfig(
                use_amp=True,
                amp_dtype=torch.bfloat16,
                compile_model=True,  # Optimize for speed
                gpu_ids=[0]
            ),
            'training': TrainingConfig(
                batch_size=64  # Larger batch for inference
            ),
            'data': DataConfig(
                num_workers=8,
                pin_memory=True,
                shuffle_train=False
            )
        }


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def get_preset_config(preset_name: str) -> dict:
    """
    Get preset configuration by name.

    Args:
        preset_name: Name of preset configuration
            - 'small_cpu': Small dataset on CPU
            - 'medium_gpu': Medium dataset on single GPU
            - 'large_multi_gpu': Large dataset on multiple GPUs
            - 'memory_constrained': Memory-constrained scenario
            - 'fast_prototype': Fast prototyping
            - 'production': Production inference

    Returns:
        Dictionary with model, gpu, training, and data configs

    Example:
        >>> config = get_preset_config('medium_gpu')
        >>> model_config = config['model']
        >>> gpu_config = config['gpu']
    """
    presets = {
        'small_cpu': PresetConfigs.small_dataset_cpu,
        'medium_gpu': PresetConfigs.medium_dataset_single_gpu,
        'large_multi_gpu': PresetConfigs.large_dataset_multi_gpu,
        'memory_constrained': PresetConfigs.memory_constrained,
        'fast_prototype': PresetConfigs.fast_prototyping,
        'production': PresetConfigs.production_inference
    }

    if preset_name not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")

    return presets[preset_name]()


def print_config(config: dict) -> None:
    """
    Pretty-print configuration.

    Args:
        config: Configuration dictionary
    """
    print("="*80)
    print("CONFIGURATION")
    print("="*80)

    for category, cfg in config.items():
        print(f"\n{category.upper()}:")
        for key, value in cfg.__dict__.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    """Example usage"""

    print("Available preset configurations:")
    presets = [
        'small_cpu',
        'medium_gpu',
        'large_multi_gpu',
        'memory_constrained',
        'fast_prototype',
        'production'
    ]

    for preset in presets:
        print(f"\n{'='*80}")
        print(f"PRESET: {preset}")
        print('='*80)
        config = get_preset_config(preset)
        print_config(config)
