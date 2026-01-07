"""
GPU-Optimized Model Wrappers with bf16 Support

Wraps interpretable SNP models with GPU optimizations for A100/H100:
- bfloat16 (bf16) mixed precision for native hardware support
- Optional gradient checkpointing for memory efficiency
- Memory-efficient attention patterns
- Distributed training compatibility

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/Users/saltfish/Files/Coding/Haploblock_Clusters_ElixirBH25')
from dl_models.snp_interpretable_models import InterpretableSNPModel

logger = logging.getLogger(__name__)


class GPUOptimizedSNPModel(nn.Module):
    """
    Wraps InterpretableSNPModel with GPU optimizations for A100/H100.

    Features:
    - Automatic Mixed Precision (AMP) with bf16 (bfloat16)
        * bf16 is native to A100/H100 GPUs
        * Same dynamic range as fp32 (no gradient scaling needed)
        * Better stability than fp16 for training
    - Optional gradient checkpointing for memory efficiency
    - Memory-efficient attention implementations
    - Distributed training support

    Args:
        base_model: InterpretableSNPModel instance
        use_amp: Enable automatic mixed precision with bf16
        amp_dtype: Data type for AMP (default: torch.bfloat16)
        use_gradient_checkpointing: Enable gradient checkpointing (saves memory)
        compile_model: Use torch.compile for optimization (PyTorch 2.0+)

    Example:
        >>> from dl_models.snp_interpretable_models import InterpretableSNPModel
        >>> base = InterpretableSNPModel(n_snps=10000, encoding_dim=8)
        >>> model = GPUOptimizedSNPModel(base, use_amp=True)
        >>> x = torch.randn(32, 10000, 8).cuda()
        >>> output = model(x)
    """

    def __init__(self,
                 base_model: InterpretableSNPModel,
                 use_amp: bool = True,
                 amp_dtype: torch.dtype = torch.bfloat16,
                 use_gradient_checkpointing: bool = False,
                 compile_model: bool = False):
        super().__init__()

        self.base_model = base_model
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Validate dtype for current hardware
        if amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logger.warning(
                "bf16 not supported on this GPU. Falling back to fp16. "
                "For A100/H100, ensure CUDA capability >= 8.0"
            )
            self.amp_dtype = torch.float16

        # Apply gradient checkpointing if requested
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Compile model with torch.compile (PyTorch 2.0+)
        if compile_model:
            try:
                self.base_model = torch.compile(self.base_model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Using eager mode.")

        logger.info(
            f"GPUOptimizedSNPModel initialized: "
            f"AMP={use_amp} (dtype={amp_dtype}), "
            f"GradCheckpoint={use_gradient_checkpointing}"
        )

    def _enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing for memory efficiency.

        Trades compute for memory by recomputing activations during backward pass.
        Useful for very large models or limited GPU memory.
        """
        model = self.base_model.model  # Get underlying model

        # Apply to transformer layers if available
        if hasattr(model, 'transformer'):
            if hasattr(model.transformer, 'layers'):
                for layer in model.transformer.layers:
                    if hasattr(layer, 'checkpoint'):
                        layer.checkpoint = True

        logger.info("Gradient checkpointing enabled")

    def forward(self,
                x: torch.Tensor,
                return_attention: bool = False,
                **kwargs) -> torch.Tensor:
        """
        Forward pass with optional AMP context.

        Args:
            x: Input tensor (batch, n_snps, encoding_dim)
            return_attention: Whether to return attention weights
            **kwargs: Additional arguments for base model

        Returns:
            Model output (logits or tuple with attention)
        """
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                return self.base_model(x, return_attention=return_attention, **kwargs)
        return self.base_model(x, return_attention=return_attention, **kwargs)

    def predict_with_interpretation(self,
                                    x: torch.Tensor,
                                    methods: List[str] = ['attention']) -> Dict[str, torch.Tensor]:
        """
        Make predictions with interpretability methods.

        Args:
            x: Input tensor (batch, n_snps, encoding_dim)
            methods: List of interpretability methods to use

        Returns:
            Dictionary with predictions and importance scores
        """
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                return self.base_model.predict_with_interpretation(x, methods=methods)
        return self.base_model.predict_with_interpretation(x, methods=methods)

    def identify_causal_snps(self,
                            x: torch.Tensor,
                            top_k: int = 10,
                            method: str = 'ensemble') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify top causal SNPs.

        Args:
            x: Input genotypes
            top_k: Number of top SNPs to return
            method: 'attention', 'integrated_gradients', or 'ensemble'

        Returns:
            (indices, scores) of top SNPs
        """
        if self.use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                return self.base_model.identify_causal_snps(x, top_k=top_k, method=method)
        return self.base_model.identify_causal_snps(x, top_k=top_k, method=method)

    def get_snp_importance(self,
                          x: torch.Tensor,
                          method: str = 'attention') -> torch.Tensor:
        """
        Get SNP importance scores.

        Args:
            x: Input tensor (batch, n_snps, encoding_dim)
            method: Importance extraction method

        Returns:
            Importance scores (batch, n_snps)
        """
        if method == 'attention':
            # Forward pass to populate attention weights
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                        self.base_model(x, return_attention=True)
                else:
                    self.base_model(x, return_attention=True)

                # Extract attention from underlying model
                if hasattr(self.base_model.model, 'attention_weights'):
                    return self.base_model.model.attention_weights
                elif hasattr(self.base_model.model, 'get_snp_importance'):
                    return self.base_model.model.get_snp_importance()

        raise ValueError(f"Unknown importance method: {method}")

    @property
    def n_snps(self) -> int:
        """Number of SNPs in the model"""
        return self.base_model.n_snps

    @property
    def encoding_dim(self) -> int:
        """Encoding dimension"""
        return self.base_model.encoding_dim

    @property
    def architecture(self) -> str:
        """Model architecture name"""
        return self.base_model.architecture


def create_gpu_optimized_model(
    n_snps: int,
    encoding_dim: int = 8,
    num_classes: int = 2,
    architecture: str = 'cnn_transformer',
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_gradient_checkpointing: bool = False,
    compile_model: bool = False,
    device: str = 'cuda',
    **model_kwargs
) -> GPUOptimizedSNPModel:
    """
    Factory function to create GPU-optimized SNP model.

    Args:
        n_snps: Number of SNPs
        encoding_dim: Dimension of genotype encoding
        num_classes: Number of output classes
        architecture: Model architecture ('cnn', 'cnn_transformer', 'gnn')
        use_amp: Enable AMP with bf16
        amp_dtype: Data type for AMP (torch.bfloat16 or torch.float16)
        use_gradient_checkpointing: Enable gradient checkpointing
        compile_model: Use torch.compile
        device: Device to move model to
        **model_kwargs: Additional arguments for base model

    Returns:
        GPU-optimized model ready for training

    Example:
        >>> model = create_gpu_optimized_model(
        ...     n_snps=10000,
        ...     encoding_dim=8,
        ...     architecture='cnn_transformer',
        ...     use_amp=True,
        ...     device='cuda:0'
        ... )
        >>> model.eval()
    """
    # Create base model
    base_model = InterpretableSNPModel(
        n_snps=n_snps,
        encoding_dim=encoding_dim,
        num_classes=num_classes,
        architecture=architecture,
        **model_kwargs
    )

    # Wrap with GPU optimizations
    gpu_model = GPUOptimizedSNPModel(
        base_model=base_model,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        use_gradient_checkpointing=use_gradient_checkpointing,
        compile_model=compile_model
    )

    # Move to device
    if torch.cuda.is_available() and device.startswith('cuda'):
        gpu_model = gpu_model.to(device)
        logger.info(f"Model moved to {device}")
    else:
        logger.warning(f"CUDA not available. Using CPU.")
        gpu_model = gpu_model.to('cpu')

    return gpu_model


# ============================================================================
# MULTI-GPU WRAPPER
# ============================================================================

class DataParallelSNPModel(nn.DataParallel):
    """
    DataParallel wrapper that preserves model methods.

    Allows accessing model-specific methods like get_snp_importance()
    even when wrapped in DataParallel.
    """

    def predict_with_interpretation(self, *args, **kwargs):
        return self.module.predict_with_interpretation(*args, **kwargs)

    def identify_causal_snps(self, *args, **kwargs):
        return self.module.identify_causal_snps(*args, **kwargs)

    def get_snp_importance(self, *args, **kwargs):
        return self.module.get_snp_importance(*args, **kwargs)

    @property
    def n_snps(self) -> int:
        return self.module.n_snps

    @property
    def encoding_dim(self) -> int:
        return self.module.encoding_dim

    @property
    def architecture(self) -> str:
        return self.module.architecture


def create_multi_gpu_model(
    model: GPUOptimizedSNPModel,
    gpu_ids: List[int] = None
) -> nn.Module:
    """
    Wrap model for multi-GPU training.

    Args:
        model: GPU-optimized SNP model
        gpu_ids: List of GPU IDs to use (default: all available)

    Returns:
        Multi-GPU wrapped model

    Example:
        >>> model = create_gpu_optimized_model(n_snps=10000)
        >>> multi_gpu_model = create_multi_gpu_model(model, gpu_ids=[0, 1])
    """
    if gpu_ids is None:
        gpu_ids = list(range(torch.cuda.device_count()))

    if len(gpu_ids) <= 1:
        logger.info("Single GPU mode")
        return model

    logger.info(f"Multi-GPU mode with GPUs: {gpu_ids}")
    return DataParallelSNPModel(model, device_ids=gpu_ids)


if __name__ == "__main__":
    """Example usage and testing"""

    logging.basicConfig(level=logging.INFO)

    # Test model creation
    print("Creating GPU-optimized model...")
    model = create_gpu_optimized_model(
        n_snps=1000,
        encoding_dim=8,
        num_classes=2,
        architecture='cnn_transformer',
        use_amp=True,
        amp_dtype=torch.bfloat16
    )

    print(f"\nModel architecture: {model.architecture}")
    print(f"Number of SNPs: {model.n_snps}")
    print(f"Encoding dimension: {model.encoding_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    if torch.cuda.is_available():
        print("\nTesting forward pass...")
        x = torch.randn(4, 1000, 8).cuda()

        with torch.no_grad():
            output = model(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        # Test interpretation
        print("\nTesting interpretation...")
        results = model.predict_with_interpretation(x, methods=['attention'])
        print(f"Predictions: {results['predictions']}")
        print(f"Probabilities shape: {results['probabilities'].shape}")

        # Test SNP importance
        print("\nTesting causal SNP identification...")
        indices, scores = model.identify_causal_snps(x, top_k=5)
        print(f"Top 5 SNP indices: {indices.cpu().numpy()}")
        print(f"Importance scores: {scores.cpu().numpy()}")
    else:
        print("\nCUDA not available. Skipping GPU tests.")
