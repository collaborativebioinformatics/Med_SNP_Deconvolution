# NVFlare Lightning Integration for SNP Deconvolution

Official NVFlare + PyTorch Lightning integration using the Client API pattern.

## Overview

This implementation follows the official NVFlare Lightning pattern from the [hello-lightning tutorial](https://nvflare.readthedocs.io/en/2.7.0/hello-world/hello-lightning/).

### Key Components

1. **client.py** - NVFlare client script with `flare.patch()` integration
2. **job.py** - Job configuration using `FedAvgRecipe`
3. **federated_data_module.py** - Lightning DataModule for site-specific data

### Architecture Flow

```
Server (FedAvg)
    |
    v
flare.receive() -> Global Model
    |
    v
trainer.fit() -> Local Training
    |
    v
flare.patch() -> Send Updates
    |
    v
Server Aggregation
```

## Quick Start

### 1. POC Mode (Local Simulation)

Simulate federated learning locally with multiple clients:

```bash
# Basic POC simulation
python job.py --mode poc --num_rounds 10 --num_clients 3

# Run simulation immediately
python job.py --mode poc --num_rounds 5 --run_now

# Custom client names
python job.py --mode poc --clients site1,site2,site3 --num_rounds 10

# With custom parameters
python job.py \
  --mode poc \
  --num_rounds 20 \
  --num_clients 3 \
  --local_epochs 2 \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --feature_type cluster \
  --architecture cnn_transformer
```

### 2. Export Mode (Production Deployment)

Create job package for deployed NVFlare system:

```bash
# Export job
python job.py --mode export --export_dir ./jobs --num_rounds 50

# Submit to deployed system
nvflare job submit -j ./jobs/snp_fedavg

# Check status
nvflare job list
```

### 3. Manual Client Execution (Advanced)

If running clients manually (not typical):

```bash
python client.py \
  --data_dir ./data \
  --feature_type cluster \
  --architecture cnn_transformer \
  --local_epochs 1 \
  --batch_size 128 \
  --learning_rate 0.0001
```

## Data Preparation

### Data Format

Each site requires a `.npz` file with the following structure:

```python
# File: {site_name}_{feature_type}.npz
# Example: site1_cluster.npz

data = {
    'X_train': np.array([n_train, n_features, encoding_dim]),  # Training features
    'y_train': np.array([n_train]),                             # Training labels
    'X_val': np.array([n_val, n_features, encoding_dim]),      # Validation features
    'y_val': np.array([n_val]),                                 # Validation labels
    'X_test': np.array([n_test, n_features, encoding_dim]),    # Test features
    'y_test': np.array([n_test]),                               # Test labels
}
```

### Feature Types

- **cluster**: Haploblock cluster features (default)
- **snp**: Raw SNP genotype features

### Directory Structure

```
data/
├── site1_cluster.npz
├── site1_snp.npz
├── site2_cluster.npz
├── site2_snp.npz
├── site3_cluster.npz
└── site3_snp.npz
```

### Data Preparation Script

```python
import numpy as np

# Example: Create dummy data for testing
n_samples = 1000
n_features = 50  # clusters or SNPs
encoding_dim = 8
num_classes = 3

for site in ['site1', 'site2', 'site3']:
    data = {
        'X_train': np.random.randn(800, n_features, encoding_dim).astype(np.float32),
        'y_train': np.random.randint(0, num_classes, 800).astype(np.int64),
        'X_val': np.random.randn(100, n_features, encoding_dim).astype(np.float32),
        'y_val': np.random.randint(0, num_classes, 100).astype(np.int64),
        'X_test': np.random.randn(100, n_features, encoding_dim).astype(np.float32),
        'y_test': np.random.randint(0, num_classes, 100).astype(np.int64),
    }

    np.savez(f'data/{site}_cluster.npz', **data)
    print(f"Created {site}_cluster.npz")
```

## Configuration Options

### Job Configuration (job.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | `poc` | Mode: `poc` (simulation) or `export` (deployment) |
| `--num_rounds` | `10` | Number of FL rounds |
| `--num_clients` | `3` | Number of clients (POC mode) |
| `--clients` | Auto | Comma-separated client names |
| `--local_epochs` | `1` | Local training epochs per round |
| `--batch_size` | `128` | Training batch size |
| `--learning_rate` | `1e-4` | Learning rate |
| `--architecture` | `cnn_transformer` | Model: `cnn`, `cnn_transformer`, `gnn` |
| `--num_classes` | `3` | Number of population classes |
| `--feature_type` | `cluster` | Feature type: `cluster` or `snp` |
| `--min_clients` | All | Minimum clients required per round |

### Client Configuration (client.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `./data` | Data directory |
| `--feature_type` | `cluster` | Feature type |
| `--architecture` | `cnn_transformer` | Model architecture |
| `--encoding_dim` | `8` | Genotype encoding dimension |
| `--num_classes` | `3` | Number of classes |
| `--local_epochs` | `1` | Local epochs per FL round |
| `--batch_size` | `128` | Batch size |
| `--learning_rate` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-5` | L2 regularization |
| `--use_focal_loss` | `True` | Use focal loss for imbalance |
| `--focal_alpha` | `0.25` | Focal loss alpha |
| `--focal_gamma` | `2.0` | Focal loss gamma |
| `--precision` | `bf16-mixed` | Training precision |
| `--num_workers` | `4` | Data loading workers |

## NVFlare Client API Pattern

### Core Functions

```python
import nvflare.client as flare

# 1. Initialize client
flare.init()

# 2. Get site identifier
site_name = flare.get_site_name()

# 3. Patch Lightning trainer (enables FL)
flare.patch(trainer)

# 4. FL training loop
while flare.is_running():
    # Receive global model
    input_model = flare.receive()

    # Local training
    trainer.fit(model, datamodule=data_module)

    # Updates automatically sent via patch()
```

### Why `flare.patch()`?

The `flare.patch(trainer)` function intercepts PyTorch Lightning's internal hooks to:

1. Load global model weights before training
2. Extract local model updates after training
3. Send updates to server for aggregation
4. Handle FL communication transparently

No manual model loading/saving required!

## Example Workflows

### Complete POC Workflow

```bash
# 1. Prepare data (see Data Preparation section)
python prepare_data.py --num_sites 3 --feature_type cluster

# 2. Run POC simulation
python job.py \
  --mode poc \
  --num_rounds 10 \
  --num_clients 3 \
  --local_epochs 2 \
  --batch_size 128 \
  --run_now

# 3. Results will be in ./nvflare_workspace/
ls ./nvflare_workspace/
```

### Production Deployment Workflow

```bash
# 1. Create job package
python job.py \
  --mode export \
  --export_dir ./production_jobs \
  --num_rounds 100 \
  --clients hospital_A,hospital_B,hospital_C \
  --local_epochs 3

# 2. Submit to deployed NVFlare system
nvflare job submit -j ./production_jobs/snp_fedavg

# 3. Monitor training
nvflare job list
nvflare job show JOB_ID

# 4. Download results
nvflare job download JOB_ID -d ./results
```

## Model Architectures

### Available Models

1. **CNN** - Convolutional Neural Network
   - Fast, efficient
   - Good for local patterns

2. **CNN-Transformer** (default)
   - Combines CNN + self-attention
   - Best overall performance
   - Captures both local and global patterns

3. **GNN** - Graph Neural Network
   - Models SNP interactions
   - Requires graph structure

### Architecture Selection

```bash
# CNN
python job.py --architecture cnn

# CNN-Transformer (recommended)
python job.py --architecture cnn_transformer

# GNN
python job.py --architecture gnn
```

## Monitoring and Debugging

### Check Client Logs

```bash
# POC mode
tail -f ./nvflare_workspace/server/run_*/log.txt
tail -f ./nvflare_workspace/site1/run_*/log.txt

# Production
# Access via NVFlare admin console
```

### Common Issues

#### 1. Data Not Found

```
FileNotFoundError: Data file not found for site 'site1'
```

**Solution:** Ensure data files exist with correct naming:
```bash
ls data/
# Should show: site1_cluster.npz, site2_cluster.npz, etc.
```

#### 2. Import Errors

```
ModuleNotFoundError: No module named 'nvflare'
```

**Solution:** Install NVFlare:
```bash
pip install nvflare
```

#### 3. GPU Memory Issues

```
CUDA out of memory
```

**Solution:** Reduce batch size:
```bash
python job.py --batch_size 64  # or 32
```

#### 4. Model Shape Mismatch

```
RuntimeError: shape mismatch
```

**Solution:** Check feature dimensions match data:
```python
# Verify data shape
import numpy as np
data = np.load('data/site1_cluster.npz')
print(f"Features: {data['X_train'].shape[1]}")
print(f"Encoding: {data['X_train'].shape[2]}")
```

## Performance Tuning

### Hyperparameters

```bash
# Conservative (stable)
python job.py \
  --learning_rate 0.0001 \
  --batch_size 128 \
  --local_epochs 1

# Aggressive (faster convergence)
python job.py \
  --learning_rate 0.0005 \
  --batch_size 256 \
  --local_epochs 3

# For large datasets
python job.py \
  --batch_size 512 \
  --num_workers 8 \
  --precision bf16-mixed
```

### Hardware Optimization

```bash
# GPU (A100/H100)
python job.py --precision bf16-mixed

# GPU (V100/older)
python job.py --precision 16-mixed

# CPU
python job.py --precision 32
```

## Testing

### Test DataModule

```bash
cd ../data
python federated_data_module.py
```

### Test Client (Standalone)

```bash
# Create test data first
python -c "
import numpy as np
import os
os.makedirs('test_data', exist_ok=True)
for i in range(3):
    data = {
        'X_train': np.random.randn(100, 50, 8).astype(np.float32),
        'y_train': np.random.randint(0, 3, 100).astype(np.int64),
        'X_val': np.random.randn(20, 50, 8).astype(np.float32),
        'y_val': np.random.randint(0, 3, 20).astype(np.int64),
        'X_test': np.random.randn(20, 50, 8).astype(np.float32),
        'y_test': np.random.randint(0, 3, 20).astype(np.int64),
    }
    np.savez(f'test_data/site{i+1}_cluster.npz', **data)
"

# Run POC
python job.py --mode poc --num_rounds 2 --run_now --data_dir ./test_data
```

## References

- [NVFlare Hello Lightning Tutorial](https://nvflare.readthedocs.io/en/2.7.0/hello-world/hello-lightning/)
- [NVFlare Client API](https://nvflare.readthedocs.io/en/2.7.0/programming_guide/fl_client_api.html)
- [FedAvg Recipe](https://nvflare.readthedocs.io/en/2.7.0/programming_guide/fed_avg_recipe.html)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## License

Part of Med_SNP_Deconvolution project.

## Support

For issues and questions, please refer to the main project documentation.
