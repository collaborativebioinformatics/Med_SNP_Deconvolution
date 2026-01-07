# Attention-Based Deep Learning Module for SNP Deconvolution

GPU-optimized deep learning implementation with **bfloat16 (bf16)** support for A100/H100 GPUs.

## Features

### GPU Optimization
- **bf16 (bfloat16) Mixed Precision**: Native A100/H100 support
  - Same dynamic range as fp32 (8-bit exponent)
  - No gradient scaling (GradScaler) needed
  - Better stability than fp16
- **Multi-GPU Training**: DataParallel support
- **Memory Management**: Automatic batch size calculation
- **Model Compilation**: torch.compile integration (PyTorch 2.0+)

### Model Architecture
- Wraps `InterpretableSNPModel` from `dl_models/snp_interpretable_models.py`
- Supports CNN, CNN-Transformer, and GNN architectures
- Attention-based interpretability for causal SNP identification
- Integrated Gradients support

### Training Features
- Focal Loss for class imbalance
- Learning rate scheduling (Cosine + ReduceLROnPlateau)
- Early stopping
- Gradient accumulation
- Checkpoint management
- Comprehensive logging

## Installation

```bash
# Install dependencies
pip install torch>=2.0.0
pip install numpy

# Verify bf16 support
python -c "import torch; print('bf16 supported:', torch.cuda.is_bf16_supported())"
```

## Quick Start

### 1. Create GPU-Optimized Model

```python
from gpu_optimized_models import create_gpu_optimized_model

model = create_gpu_optimized_model(
    n_snps=10000,
    encoding_dim=8,
    num_classes=2,
    architecture='cnn_transformer',
    use_amp=True,
    amp_dtype=torch.bfloat16,  # bf16 for A100/H100
    device='cuda:0'
)
```

### 2. Calculate Optimal Batch Size

```python
from memory_manager import GPUMemoryManager

batch_size = GPUMemoryManager.get_optimal_batch_size(
    n_snps=10000,
    encoding_dim=8,
    dtype=torch.bfloat16
)
print(f"Optimal batch size: {batch_size}")
```

### 3. Create Data Loaders

```python
from memory_manager import MemoryEfficientDataLoader
from torch.utils.data import TensorDataset

train_dataset = TensorDataset(X_train, y_train)
train_loader = MemoryEfficientDataLoader.create_dataloader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
```

### 4. Train Model

```python
from gpu_trainer import MultiGPUSNPTrainer

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

history = trainer.train(
    num_epochs=100,
    early_stopping_patience=15,
    checkpoint_dir='./checkpoints'
)
```

### 5. Extract SNP Importance

```python
# Get importance scores across dataset
importance_dict = trainer.get_snp_importance(
    data_loader=val_loader,
    method='attention',
    aggregate='mean'
)

# Get top SNPs
sorted_snps = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
top_10 = sorted_snps[:10]

for snp_idx, score in top_10:
    print(f"SNP {snp_idx}: {score:.4f}")
```

## Module Structure

```
attention_dl/
├── __init__.py                 # Module exports
├── gpu_optimized_models.py     # GPU-optimized model wrappers
├── gpu_trainer.py              # Multi-GPU trainer with bf16
├── memory_manager.py           # GPU memory utilities
├── example_usage.py            # Complete examples
├── test_attention_dl.py        # Unit tests
└── README.md                   # This file
```

## API Reference

### gpu_optimized_models.py

#### `GPUOptimizedSNPModel`
Wraps InterpretableSNPModel with GPU optimizations.

```python
model = GPUOptimizedSNPModel(
    base_model: InterpretableSNPModel,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_gradient_checkpointing: bool = False,
    compile_model: bool = False
)
```

#### `create_gpu_optimized_model`
Factory function to create GPU-optimized model.

```python
model = create_gpu_optimized_model(
    n_snps: int,
    encoding_dim: int = 8,
    num_classes: int = 2,
    architecture: str = 'cnn_transformer',
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    device: str = 'cuda',
    **model_kwargs
)
```

### gpu_trainer.py

#### `MultiGPUSNPTrainer`
GPU trainer with bf16 AMP support.

```python
trainer = MultiGPUSNPTrainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    gpu_ids: List[int] = [0],
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    gradient_accumulation_steps: int = 1
)
```

**Methods:**
- `train(num_epochs, early_stopping_patience, checkpoint_dir)`: Full training loop
- `train_epoch()`: Train one epoch
- `validate()`: Validate model
- `get_snp_importance(data_loader, method, aggregate)`: Extract SNP importance
- `save_checkpoint(path)`: Save checkpoint
- `load_checkpoint(path)`: Load checkpoint

#### `FocalLoss`
Focal loss for class imbalance.

```python
criterion = FocalLoss(alpha=0.25, gamma=2.0)
```

### memory_manager.py

#### `GPUMemoryManager`
GPU memory management utilities.

**Static Methods:**
- `get_optimal_batch_size(n_snps, encoding_dim, gpu_memory_gb, dtype)`: Calculate optimal batch size
- `estimate_memory_usage(batch_size, n_snps, encoding_dim, dtype)`: Estimate memory usage
- `suggest_batch_size_range(n_snps, encoding_dim, dtype)`: Suggest batch size range
- `get_memory_stats(device)`: Get detailed memory statistics
- `log_memory_usage(tag, device)`: Log current memory usage
- `clear_cache()`: Clear GPU cache
- `profile_memory_usage(func, *args, **kwargs)`: Profile function memory usage

#### `MemoryEfficientDataLoader`
Memory-efficient data loader utilities.

**Static Methods:**
- `get_optimal_num_workers()`: Get optimal number of workers
- `create_dataloader(dataset, batch_size, shuffle, pin_memory)`: Create optimized DataLoader

## Performance Considerations

### bf16 vs fp16 vs fp32

| Precision | Size | Range | Speed (A100) | Stability | Use Case |
|-----------|------|-------|--------------|-----------|----------|
| fp32 | 4 bytes | High | 1x | Best | Baseline |
| fp16 | 2 bytes | Low | 2x | Poor | Avoid for training |
| bf16 | 2 bytes | High | 2x | Good | **Recommended** |

**Why bf16?**
- Same dynamic range as fp32 (8-bit exponent)
- No gradient underflow/overflow issues
- No GradScaler needed
- Native hardware support on A100/H100

### Memory Optimization

**For Large SNP Datasets (>50K SNPs):**
1. Use bf16 precision (50% memory reduction)
2. Enable gradient checkpointing (trades compute for memory)
3. Use gradient accumulation for larger effective batch sizes
4. Clear cache between runs: `GPUMemoryManager.clear_cache()`

**Batch Size Guidelines:**
- 40GB GPU: 32-64 samples (50K SNPs, bf16)
- 80GB GPU: 64-128 samples (50K SNPs, bf16)

### Multi-GPU Training

```python
from gpu_optimized_models import create_multi_gpu_model

# Single GPU model
model = create_gpu_optimized_model(...)

# Wrap for multi-GPU
model = create_multi_gpu_model(model, gpu_ids=[0, 1, 2, 3])

# Train normally
trainer = MultiGPUSNPTrainer(model, ..., gpu_ids=[0, 1, 2, 3])
```

## Examples

### Example 1: Basic Training

```python
from gpu_optimized_models import create_gpu_optimized_model
from gpu_trainer import MultiGPUSNPTrainer
from memory_manager import GPUMemoryManager, MemoryEfficientDataLoader
import torch
from torch.utils.data import TensorDataset

# Data
X_train = torch.randn(1000, 10000, 8)  # (samples, SNPs, encoding_dim)
y_train = torch.randint(0, 2, (1000,))

# Calculate batch size
batch_size = GPUMemoryManager.get_optimal_batch_size(
    n_snps=10000, encoding_dim=8, dtype=torch.bfloat16
)

# Data loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = MemoryEfficientDataLoader.create_dataloader(
    train_dataset, batch_size=batch_size
)

# Model
model = create_gpu_optimized_model(
    n_snps=10000,
    encoding_dim=8,
    architecture='cnn_transformer',
    use_amp=True,
    amp_dtype=torch.bfloat16
)

# Train
trainer = MultiGPUSNPTrainer(
    model, train_loader, val_loader,
    use_amp=True, amp_dtype=torch.bfloat16
)
history = trainer.train(num_epochs=100)
```

### Example 2: SNP Importance Analysis

```python
# Extract importance scores
importance_dict = trainer.get_snp_importance(
    data_loader=test_loader,
    method='attention',
    aggregate='mean'
)

# Identify causal SNPs
causal_snps = [snp for snp, score in importance_dict.items() if score > threshold]
print(f"Found {len(causal_snps)} causal SNPs")
```

### Example 3: Memory Profiling

```python
from memory_manager import GPUMemoryManager

# Profile model
GPUMemoryManager.log_memory_usage("Before training")

# Train...

GPUMemoryManager.log_memory_usage("After training")

# Profile specific function
result, before, after = GPUMemoryManager.profile_memory_usage(
    model, input_tensor
)
print(f"Memory delta: {after.allocated_gb - before.allocated_gb:.2f}GB")
```

## Testing

Run unit tests:

```bash
python test_attention_dl.py
```

Run examples:

```bash
python example_usage.py
```

## Hardware Requirements

**Minimum:**
- CUDA-capable GPU (8GB+ VRAM)
- PyTorch 2.0+
- CUDA 11.8+

**Recommended:**
- NVIDIA A100 or H100 (bf16 native support)
- 40GB+ VRAM
- NVLink for multi-GPU

**Check bf16 support:**
```python
import torch
print("bf16 supported:", torch.cuda.is_bf16_supported())
# A100/H100: True (CUDA capability >= 8.0)
# V100/RTX 3090: False (use fp16 fallback)
```

## Troubleshooting

### OOM (Out of Memory) Errors

1. Reduce batch size:
   ```python
   batch_size = GPUMemoryManager.get_optimal_batch_size(
       n_snps=n_snps, safety_margin=0.60  # More conservative
   )
   ```

2. Enable gradient checkpointing:
   ```python
   model = create_gpu_optimized_model(
       ..., use_gradient_checkpointing=True
   )
   ```

3. Use gradient accumulation:
   ```python
   trainer = MultiGPUSNPTrainer(
       ..., gradient_accumulation_steps=4
   )
   ```

### bf16 Not Supported

If your GPU doesn't support bf16, the module automatically falls back to fp16:

```python
# Explicit fp16
model = create_gpu_optimized_model(
    ..., amp_dtype=torch.float16
)
```

### Slow Training

1. Check DataLoader workers:
   ```python
   num_workers = MemoryEfficientDataLoader.get_optimal_num_workers()
   ```

2. Enable model compilation (PyTorch 2.0+):
   ```python
   model = create_gpu_optimized_model(
       ..., compile_model=True
   )
   ```

3. Use pinned memory:
   ```python
   loader = MemoryEfficientDataLoader.create_dataloader(
       ..., pin_memory=True
   )
   ```

## Citation

If you use this module, please cite:

```bibtex
@software{snp_attention_dl,
  title={Attention-Based Deep Learning for SNP Deconvolution},
  author={Haploblock Analysis Pipeline},
  year={2026},
  note={bf16-optimized for A100/H100 GPUs}
}
```

## License

Part of the Haploblock Analysis Pipeline.

## References

- DPCformer (2025): CNN + Self-Attention for genotype-phenotype modeling
- G2PT (2025): Graph-based Transformer for genotype-phenotype translation
- Focal Loss: Lin et al. "Focal Loss for Dense Object Detection" (2017)
- Mixed Precision Training: Micikevicius et al. (2018)
