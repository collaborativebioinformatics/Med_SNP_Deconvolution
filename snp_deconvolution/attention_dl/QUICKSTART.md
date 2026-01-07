# Quick Start Guide: Attention DL for SNP Deconvolution

Get started with GPU-optimized deep learning for SNP analysis in 5 minutes.

## Prerequisites

```bash
# Check PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check bf16 support (A100/H100)
python -c "import torch; print('bf16 supported:', torch.cuda.is_bf16_supported())"
```

## 1-Minute Example

```python
import torch
from torch.utils.data import TensorDataset
from snp_deconvolution.attention_dl import (
    GPUOptimizedSNPModel,
    MultiGPUSNPTrainer,
    GPUMemoryManager,
    MemoryEfficientDataLoader
)
from snp_deconvolution.attention_dl.gpu_optimized_models import create_gpu_optimized_model

# Your SNP data (samples, SNPs, encoding_dim)
X_train = torch.randn(1000, 10000, 8)
y_train = torch.randint(0, 2, (1000,))

# Calculate optimal batch size
batch_size = GPUMemoryManager.get_optimal_batch_size(
    n_snps=10000,
    encoding_dim=8,
    dtype=torch.bfloat16
)

# Create data loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = MemoryEfficientDataLoader.create_dataloader(
    train_dataset, batch_size=batch_size
)

# Create GPU-optimized model with bf16
model = create_gpu_optimized_model(
    n_snps=10000,
    encoding_dim=8,
    architecture='cnn_transformer',
    use_amp=True,
    amp_dtype=torch.bfloat16
)

# Train
trainer = MultiGPUSNPTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    use_amp=True,
    amp_dtype=torch.bfloat16
)
history = trainer.train(num_epochs=100)

# Extract SNP importance
importance = trainer.get_snp_importance(val_loader)
```

## Step-by-Step Guide

### Step 1: Prepare Your Data

```python
import numpy as np
import torch

# Load your genotype data
# Expected format: (n_samples, n_snps) with values 0/1/2
genotypes = np.load('genotypes.npy')  # Shape: (1000, 10000)
phenotypes = np.load('phenotypes.npy')  # Shape: (1000,)

# Encode genotypes (use method from dl_models)
from dl_models.snp_interpretable_models import GenotypeEncoder

encoder = GenotypeEncoder()
X = encoder.haplotype_encoding(genotypes)  # (1000, 10000, 8)
X = encoder.normalize_genotypes(X, method='standardize')

# Convert to torch tensors
X = torch.from_numpy(X).float()
y = torch.from_numpy(phenotypes).long()

# Split train/val
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]
```

### Step 2: Configure for Your Hardware

#### Option A: Use Preset Configuration

```python
from snp_deconvolution.attention_dl.config import get_preset_config

# Choose preset based on your setup
config = get_preset_config('medium_gpu')  # For single A100/H100

model_cfg = config['model']
gpu_cfg = config['gpu']
train_cfg = config['training']
data_cfg = config['data']

# Update n_snps for your data
model_cfg.n_snps = X_train.shape[1]
```

#### Option B: Manual Configuration

```python
# For A100 40GB with 10K SNPs
batch_size = GPUMemoryManager.get_optimal_batch_size(
    n_snps=10000,
    encoding_dim=8,
    gpu_memory_gb=40.0,
    dtype=torch.bfloat16
)
print(f"Optimal batch size: {batch_size}")
```

### Step 3: Create Data Loaders

```python
from torch.utils.data import TensorDataset
from snp_deconvolution.attention_dl import MemoryEfficientDataLoader

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Create optimized data loaders
train_loader = MemoryEfficientDataLoader.create_dataloader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)

val_loader = MemoryEfficientDataLoader.create_dataloader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True
)
```

### Step 4: Create Model

```python
from snp_deconvolution.attention_dl.gpu_optimized_models import create_gpu_optimized_model

model = create_gpu_optimized_model(
    n_snps=10000,
    encoding_dim=8,
    num_classes=2,
    architecture='cnn_transformer',  # Best for causal SNP identification
    use_amp=True,
    amp_dtype=torch.bfloat16,  # bf16 for A100/H100
    compile_model=True,  # PyTorch 2.0+ optimization
    device='cuda:0'
)

print(f"Model: {model.architecture}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 5: Train Model

```python
from snp_deconvolution.attention_dl import MultiGPUSNPTrainer
from pathlib import Path

# Create trainer
trainer = MultiGPUSNPTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    gpu_ids=[0],
    learning_rate=1e-4,
    weight_decay=1e-5,
    use_amp=True,
    amp_dtype=torch.bfloat16,
    use_focal_loss=True  # Good for imbalanced data
)

# Train with early stopping
checkpoint_dir = Path('./checkpoints')
history = trainer.train(
    num_epochs=100,
    early_stopping_patience=15,
    checkpoint_dir=checkpoint_dir,
    verbose=1
)

print(f"Best validation loss: {trainer.best_val_loss:.4f}")
print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
```

### Step 6: Extract SNP Importance

```python
# Get importance scores for all SNPs
importance_dict = trainer.get_snp_importance(
    data_loader=val_loader,
    method='attention',  # Use attention weights
    aggregate='mean'  # Average across samples
)

# Sort by importance
sorted_snps = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Get top 20 causal SNPs
top_20 = sorted_snps[:20]
print("\nTop 20 Causal SNPs:")
for rank, (snp_idx, score) in enumerate(top_20, 1):
    print(f"{rank:2d}. SNP {snp_idx:5d}: {score:.6f}")

# Save results
import json
with open('snp_importance.json', 'w') as f:
    json.dump(importance_dict, f, indent=2)
```

### Step 7: Inference on New Data

```python
# Load best model
best_model_path = checkpoint_dir / 'best_model.pt'
trainer.load_checkpoint(best_model_path)

# Put model in eval mode
model.eval()

# Predict on new data
with torch.no_grad():
    new_data = torch.randn(10, 10000, 8).cuda()
    predictions = model(new_data)
    probabilities = torch.softmax(predictions, dim=1)
    classes = torch.argmax(predictions, dim=1)

print(f"Predictions: {classes.cpu().numpy()}")
print(f"Probabilities shape: {probabilities.shape}")
```

## Common Use Cases

### Use Case 1: Large SNP Dataset (>50K SNPs)

```python
# Enable gradient checkpointing to save memory
model = create_gpu_optimized_model(
    n_snps=100000,
    use_gradient_checkpointing=True,  # Trades compute for memory
    amp_dtype=torch.bfloat16
)

# Use gradient accumulation for larger effective batch size
trainer = MultiGPUSNPTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    gradient_accumulation_steps=4,  # Effective batch = batch_size * 4
    use_amp=True
)
```

### Use Case 2: Multi-GPU Training

```python
from snp_deconvolution.attention_dl.gpu_optimized_models import create_multi_gpu_model

# Create single-GPU model
model = create_gpu_optimized_model(n_snps=25000, device='cuda:0')

# Wrap for multi-GPU
model = create_multi_gpu_model(model, gpu_ids=[0, 1, 2, 3])

# Train (will use all GPUs)
trainer = MultiGPUSNPTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    gpu_ids=[0, 1, 2, 3]
)
history = trainer.train(num_epochs=100)
```

### Use Case 3: Memory-Constrained Environment

```python
# Use preset configuration
from snp_deconvolution.attention_dl.config import get_preset_config

config = get_preset_config('memory_constrained')

# Small batch size + gradient accumulation
batch_size = 8
gradient_accumulation_steps = 4  # Effective batch = 32

# Enable gradient checkpointing
model = create_gpu_optimized_model(
    n_snps=config['model'].n_snps,
    use_gradient_checkpointing=True
)

trainer = MultiGPUSNPTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    gradient_accumulation_steps=gradient_accumulation_steps
)
```

### Use Case 4: Fast Prototyping

```python
# Use fast prototype preset
config = get_preset_config('fast_prototype')

# Small model, few epochs
model = create_gpu_optimized_model(
    n_snps=1000,
    architecture='cnn',  # Faster than transformer
    compile_model=False
)

trainer = MultiGPUSNPTrainer(model, train_loader, val_loader)
history = trainer.train(num_epochs=10)  # Quick experiment
```

## Performance Tips

### 1. Optimize Batch Size

```python
# Get batch size range
min_bs, opt_bs, max_bs = GPUMemoryManager.suggest_batch_size_range(
    n_snps=10000,
    encoding_dim=8,
    dtype=torch.bfloat16
)
print(f"Try batch sizes: {min_bs}, {opt_bs}, {max_bs}")

# Test different batch sizes
for bs in [min_bs, opt_bs, max_bs]:
    print(f"\nTesting batch_size={bs}")
    # Train and measure speed...
```

### 2. Monitor Memory Usage

```python
from snp_deconvolution.attention_dl import GPUMemoryManager

# Before training
GPUMemoryManager.log_memory_usage("Before training")

# Train...
history = trainer.train(num_epochs=100)

# After training
GPUMemoryManager.log_memory_usage("After training")

# Clear cache if needed
GPUMemoryManager.clear_cache()
```

### 3. Profile Memory

```python
# Profile specific operations
def train_step():
    trainer.train_epoch()

result, before, after = GPUMemoryManager.profile_memory_usage(train_step)
print(f"Memory delta: {after.allocated_gb - before.allocated_gb:.2f}GB")
```

## Troubleshooting

### Problem: Out of Memory

**Solution 1**: Reduce batch size
```python
batch_size = GPUMemoryManager.get_optimal_batch_size(
    n_snps=n_snps,
    safety_margin=0.60  # More conservative
)
```

**Solution 2**: Enable gradient checkpointing
```python
model = create_gpu_optimized_model(
    n_snps=n_snps,
    use_gradient_checkpointing=True
)
```

**Solution 3**: Use gradient accumulation
```python
trainer = MultiGPUSNPTrainer(
    ...,
    gradient_accumulation_steps=4
)
```

### Problem: Slow Training

**Solution 1**: Increase num_workers
```python
loader = MemoryEfficientDataLoader.create_dataloader(
    dataset,
    batch_size=batch_size,
    num_workers=8  # More workers
)
```

**Solution 2**: Enable model compilation
```python
model = create_gpu_optimized_model(
    ...,
    compile_model=True  # PyTorch 2.0+
)
```

**Solution 3**: Use pinned memory
```python
loader = MemoryEfficientDataLoader.create_dataloader(
    dataset,
    batch_size=batch_size,
    pin_memory=True
)
```

### Problem: bf16 Not Supported

**Solution**: Use fp16 fallback
```python
model = create_gpu_optimized_model(
    ...,
    amp_dtype=torch.float16  # Use fp16 instead of bf16
)
```

## Next Steps

1. Read the full [README.md](README.md) for detailed API documentation
2. Check [example_usage.py](example_usage.py) for more examples
3. Run [test_attention_dl.py](test_attention_dl.py) to verify installation
4. Explore [config.py](config.py) for preset configurations

## Support

For issues or questions:
1. Check the README.md troubleshooting section
2. Verify GPU and PyTorch installation
3. Review example_usage.py for working examples
