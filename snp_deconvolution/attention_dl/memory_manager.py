"""
GPU Memory Management Utilities for SNP Models

Tools for optimizing GPU memory usage with large SNP datasets:
- Automatic batch size calculation
- Memory profiling and monitoring
- Cache management
- Memory-efficient data loading

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import torch
import gc
import logging
from typing import Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Container for GPU memory statistics"""
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    free_gb: float
    total_gb: float
    utilization_pct: float


class GPUMemoryManager:
    """
    GPU memory management for large SNP matrices.

    Provides utilities for:
    - Calculating optimal batch sizes
    - Monitoring memory usage
    - Clearing cache
    - Memory profiling

    Optimized for A100/H100 GPUs with bf16 precision.
    """

    @staticmethod
    def get_gpu_memory_gb(device: int = 0) -> Tuple[float, float]:
        """
        Get total and free GPU memory in GB.

        Args:
            device: GPU device index

        Returns:
            (total_memory_gb, free_memory_gb)
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0

        torch.cuda.set_device(device)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        free = total - allocated

        return total, free

    @staticmethod
    def get_optimal_batch_size(
        n_snps: int,
        encoding_dim: int = 8,
        model_params: int = 1_500_000,
        gpu_memory_gb: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
        safety_margin: float = 0.75
    ) -> int:
        """
        Calculate optimal batch size for A100/H100.

        Estimates memory usage considering:
        - Input tensor size
        - Model parameters
        - Activation memory
        - Gradient memory
        - Optimizer state

        For 40GB+ GPU, can use larger batches.
        bf16 uses 2 bytes per element.

        Args:
            n_snps: Number of SNPs
            encoding_dim: Encoding dimension (typically 8)
            model_params: Number of model parameters
            gpu_memory_gb: Available GPU memory in GB (auto-detect if None)
            dtype: Data type (torch.bfloat16 or torch.float16)
            safety_margin: Fraction of memory to use (0.75 = 75%)

        Returns:
            Optimal batch size

        Example:
            >>> batch_size = GPUMemoryManager.get_optimal_batch_size(
            ...     n_snps=50000,
            ...     encoding_dim=8,
            ...     gpu_memory_gb=40.0
            ... )
            >>> print(f"Optimal batch size: {batch_size}")
        """
        # Get available GPU memory
        if gpu_memory_gb is None:
            total_gb, free_gb = GPUMemoryManager.get_gpu_memory_gb()
            if total_gb == 0:
                logger.warning("No GPU detected. Using small batch size.")
                return 8
            available_gb = free_gb
        else:
            available_gb = gpu_memory_gb

        # Apply safety margin
        usable_gb = available_gb * safety_margin

        # Bytes per element
        if dtype in (torch.bfloat16, torch.float16):
            bytes_per_element = 2
        elif dtype == torch.float32:
            bytes_per_element = 4
        else:
            bytes_per_element = 4

        # Memory estimates (in bytes)
        # 1. Input tensor per sample
        input_size_per_sample = n_snps * encoding_dim * bytes_per_element

        # 2. Model parameters (constant)
        model_size = model_params * bytes_per_element

        # 3. Activations (rough estimate: 3x model params per sample)
        # This accounts for intermediate activations during forward pass
        activation_size_per_sample = model_params * 3 * bytes_per_element

        # 4. Gradients (same as model params, constant across batch)
        gradient_size = model_params * bytes_per_element

        # 5. Optimizer state (AdamW: 2x model params for momentum and variance)
        optimizer_state = model_params * 2 * bytes_per_element

        # Total constant memory
        constant_memory_bytes = model_size + gradient_size + optimizer_state

        # Memory per sample
        per_sample_memory_bytes = input_size_per_sample + activation_size_per_sample

        # Available memory for batch
        usable_bytes = usable_gb * (1024**3)
        available_for_batch = usable_bytes - constant_memory_bytes

        # Calculate batch size
        if available_for_batch <= 0:
            logger.warning("Model too large for available memory. Using batch_size=1")
            return 1

        batch_size = int(available_for_batch / per_sample_memory_bytes)

        # Clamp to reasonable range
        batch_size = max(1, min(batch_size, 512))

        logger.info(
            f"Optimal batch size calculated: {batch_size} "
            f"(n_snps={n_snps}, encoding_dim={encoding_dim}, "
            f"gpu_memory={available_gb:.1f}GB, dtype={dtype})"
        )

        return batch_size

    @staticmethod
    def estimate_memory_usage(
        batch_size: int,
        n_snps: int,
        encoding_dim: int = 8,
        model_params: int = 1_500_000,
        dtype: torch.dtype = torch.bfloat16
    ) -> Dict[str, float]:
        """
        Estimate memory usage for given configuration.

        Args:
            batch_size: Batch size
            n_snps: Number of SNPs
            encoding_dim: Encoding dimension
            model_params: Number of model parameters
            dtype: Data type

        Returns:
            Dictionary with memory estimates in GB
        """
        # Bytes per element
        if dtype in (torch.bfloat16, torch.float16):
            bytes_per_element = 2
        elif dtype == torch.float32:
            bytes_per_element = 4
        else:
            bytes_per_element = 4

        # Calculate components
        input_gb = (batch_size * n_snps * encoding_dim * bytes_per_element) / (1024**3)
        model_gb = (model_params * bytes_per_element) / (1024**3)
        activations_gb = (batch_size * model_params * 3 * bytes_per_element) / (1024**3)
        gradients_gb = (model_params * bytes_per_element) / (1024**3)
        optimizer_gb = (model_params * 2 * bytes_per_element) / (1024**3)

        total_gb = input_gb + model_gb + activations_gb + gradients_gb + optimizer_gb

        return {
            'input_gb': input_gb,
            'model_gb': model_gb,
            'activations_gb': activations_gb,
            'gradients_gb': gradients_gb,
            'optimizer_gb': optimizer_gb,
            'total_gb': total_gb
        }

    @staticmethod
    def clear_cache() -> None:
        """
        Clear GPU cache and run garbage collection.

        Useful between training runs or when memory is fragmented.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info("GPU cache cleared")

    @staticmethod
    def log_memory_usage(tag: str = "", device: int = 0) -> None:
        """
        Log current GPU memory usage.

        Args:
            tag: Optional tag for logging
            device: GPU device index
        """
        if not torch.cuda.is_available():
            logger.info(f"{tag} - No GPU available")
            return

        stats = GPUMemoryManager.get_memory_stats(device)

        logger.info(
            f"{tag} - GPU {device} Memory: "
            f"Allocated: {stats.allocated_gb:.2f}GB, "
            f"Reserved: {stats.reserved_gb:.2f}GB, "
            f"Free: {stats.free_gb:.2f}GB, "
            f"Total: {stats.total_gb:.2f}GB "
            f"({stats.utilization_pct:.1f}% used)"
        )

    @staticmethod
    def get_memory_stats(device: int = 0) -> MemoryStats:
        """
        Get detailed GPU memory statistics.

        Args:
            device: GPU device index

        Returns:
            MemoryStats object with detailed memory info
        """
        if not torch.cuda.is_available():
            return MemoryStats(
                allocated_gb=0.0,
                reserved_gb=0.0,
                max_allocated_gb=0.0,
                free_gb=0.0,
                total_gb=0.0,
                utilization_pct=0.0
            )

        torch.cuda.set_device(device)

        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        free = total - allocated
        utilization = (allocated / total) * 100 if total > 0 else 0.0

        return MemoryStats(
            allocated_gb=allocated,
            reserved_gb=reserved,
            max_allocated_gb=max_allocated,
            free_gb=free,
            total_gb=total,
            utilization_pct=utilization
        )

    @staticmethod
    def reset_peak_memory_stats(device: int = 0) -> None:
        """
        Reset peak memory statistics.

        Useful for profiling specific code sections.

        Args:
            device: GPU device index
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            logger.debug(f"Peak memory stats reset for GPU {device}")

    @staticmethod
    def profile_memory_usage(func, *args, device: int = 0, **kwargs) -> Tuple[any, MemoryStats, MemoryStats]:
        """
        Profile memory usage of a function.

        Args:
            func: Function to profile
            *args: Function arguments
            device: GPU device index
            **kwargs: Function keyword arguments

        Returns:
            (function_result, memory_before, memory_after)

        Example:
            >>> result, before, after = GPUMemoryManager.profile_memory_usage(
            ...     model, input_tensor
            ... )
            >>> print(f"Memory delta: {after.allocated_gb - before.allocated_gb:.2f}GB")
        """
        GPUMemoryManager.clear_cache()
        GPUMemoryManager.reset_peak_memory_stats(device)

        # Before
        memory_before = GPUMemoryManager.get_memory_stats(device)

        # Run function
        result = func(*args, **kwargs)

        # After
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        memory_after = GPUMemoryManager.get_memory_stats(device)

        logger.info(
            f"Memory profiling - Before: {memory_before.allocated_gb:.2f}GB, "
            f"After: {memory_after.allocated_gb:.2f}GB, "
            f"Delta: {memory_after.allocated_gb - memory_before.allocated_gb:.2f}GB, "
            f"Peak: {memory_after.max_allocated_gb:.2f}GB"
        )

        return result, memory_before, memory_after

    @staticmethod
    def suggest_batch_size_range(
        n_snps: int,
        encoding_dim: int = 8,
        model_params: int = 1_500_000,
        gpu_memory_gb: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[int, int, int]:
        """
        Suggest a range of batch sizes to test.

        Returns:
            (min_batch_size, optimal_batch_size, max_batch_size)

        Example:
            >>> min_bs, opt_bs, max_bs = GPUMemoryManager.suggest_batch_size_range(
            ...     n_snps=50000
            ... )
            >>> print(f"Try batch sizes: {min_bs}, {opt_bs}, {max_bs}")
        """
        optimal = GPUMemoryManager.get_optimal_batch_size(
            n_snps=n_snps,
            encoding_dim=encoding_dim,
            model_params=model_params,
            gpu_memory_gb=gpu_memory_gb,
            dtype=dtype,
            safety_margin=0.75
        )

        # Conservative (80% safety margin)
        min_batch = GPUMemoryManager.get_optimal_batch_size(
            n_snps=n_snps,
            encoding_dim=encoding_dim,
            model_params=model_params,
            gpu_memory_gb=gpu_memory_gb,
            dtype=dtype,
            safety_margin=0.60
        )

        # Aggressive (90% safety margin)
        max_batch = GPUMemoryManager.get_optimal_batch_size(
            n_snps=n_snps,
            encoding_dim=encoding_dim,
            model_params=model_params,
            gpu_memory_gb=gpu_memory_gb,
            dtype=dtype,
            safety_margin=0.85
        )

        return min_batch, optimal, max_batch


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader for large SNP datasets.

    Features:
    - Pinned memory for faster GPU transfer
    - Prefetching
    - Optimal worker configuration
    """

    @staticmethod
    def get_optimal_num_workers() -> int:
        """
        Get optimal number of data loader workers.

        Returns:
            Number of workers (typically num_cpus - 2)
        """
        try:
            import os
            num_cpus = os.cpu_count() or 4
            # Use num_cpus - 2, but at least 2
            return max(2, min(num_cpus - 2, 8))
        except:
            return 4

    @staticmethod
    def create_dataloader(
        dataset,
        batch_size: int,
        shuffle: bool = True,
        pin_memory: bool = True,
        num_workers: Optional[int] = None,
        prefetch_factor: int = 2
    ):
        """
        Create optimized DataLoader.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            pin_memory: Use pinned memory for faster GPU transfer
            num_workers: Number of workers (auto if None)
            prefetch_factor: Batches to prefetch per worker

        Returns:
            Optimized DataLoader

        Example:
            >>> from torch.utils.data import TensorDataset
            >>> dataset = TensorDataset(inputs, targets)
            >>> loader = MemoryEfficientDataLoader.create_dataloader(
            ...     dataset, batch_size=32
            ... )
        """
        from torch.utils.data import DataLoader

        if num_workers is None:
            num_workers = MemoryEfficientDataLoader.get_optimal_num_workers()

        logger.info(
            f"Creating DataLoader: batch_size={batch_size}, "
            f"num_workers={num_workers}, pin_memory={pin_memory}"
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        )


if __name__ == "__main__":
    """Example usage and testing"""

    logging.basicConfig(level=logging.INFO)

    print("GPU Memory Manager Tests")
    print("=" * 80)

    # Test GPU detection
    if torch.cuda.is_available():
        print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        total_gb, free_gb = GPUMemoryManager.get_gpu_memory_gb()
        print(f"Total memory: {total_gb:.2f}GB")
        print(f"Free memory: {free_gb:.2f}GB")

        # Test memory stats
        stats = GPUMemoryManager.get_memory_stats()
        print(f"\nMemory Stats:")
        print(f"  Allocated: {stats.allocated_gb:.2f}GB")
        print(f"  Reserved: {stats.reserved_gb:.2f}GB")
        print(f"  Free: {stats.free_gb:.2f}GB")
        print(f"  Utilization: {stats.utilization_pct:.1f}%")

        # Test batch size calculation
        print("\nOptimal Batch Size Calculations:")
        for n_snps in [1000, 10000, 50000, 100000]:
            batch_size = GPUMemoryManager.get_optimal_batch_size(
                n_snps=n_snps,
                encoding_dim=8,
                dtype=torch.bfloat16
            )
            print(f"  n_snps={n_snps:6d} -> batch_size={batch_size}")

        # Test batch size range
        print("\nBatch Size Range for 50000 SNPs:")
        min_bs, opt_bs, max_bs = GPUMemoryManager.suggest_batch_size_range(
            n_snps=50000
        )
        print(f"  Conservative: {min_bs}")
        print(f"  Optimal: {opt_bs}")
        print(f"  Aggressive: {max_bs}")

        # Test memory estimation
        print("\nMemory Usage Estimation (batch_size=32, 10000 SNPs):")
        mem_est = GPUMemoryManager.estimate_memory_usage(
            batch_size=32,
            n_snps=10000,
            encoding_dim=8,
            dtype=torch.bfloat16
        )
        for key, value in mem_est.items():
            print(f"  {key}: {value:.2f}GB")

    else:
        print("\nNo GPU detected. Running CPU-only tests.")

    # Test data loader configuration
    print("\nOptimal DataLoader Configuration:")
    num_workers = MemoryEfficientDataLoader.get_optimal_num_workers()
    print(f"  Recommended num_workers: {num_workers}")
