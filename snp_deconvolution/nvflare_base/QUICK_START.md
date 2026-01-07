# Quick Start Guide - NVFlare Base Module

## 5-Minute Introduction

### XGBoost Federated Learning

```python
from snp_deconvolution.nvflare_base import XGBoostNVFlareExecutor
import numpy as np

# 1. Initialize executor
executor = XGBoostNVFlareExecutor(
    trainer=None,
    num_snps=10000,
    num_populations=5,
)

# 2. Prepare data (local site data)
X_train = np.random.randint(0, 3, (1000, 10000))
y_train = np.random.randint(0, 5, 1000)
executor.prepare_data(X_train, y_train)

# 3. Train locally
metrics = executor.local_train(num_epochs=10)
print(f"Accuracy: {metrics.accuracy:.4f}")

# 4. Export weights for server
weights = executor.get_model_weights()

# 5. After server aggregation, load global model
# executor.set_model_weights(aggregated_weights)
```

### PyTorch Federated Learning

```python
from snp_deconvolution.nvflare_base import DLNVFlareExecutor
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. Define your model
class SNPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_snps = 10000
        self.num_populations = 5
        self.fc1 = nn.Linear(10000, 512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SNPNet()

# 2. Initialize executor
executor = DLNVFlareExecutor(
    model=model,
    aggregation_strategy='fedavg',  # or 'fedprox'
)

# 3. Prepare data loader
X = torch.randn(1000, 10000)
y = torch.randint(0, 5, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
executor.set_data_loaders(loader, loader)

# 4. Train locally
metrics = executor.local_train(
    num_epochs=5,
    use_mixed_precision=True,
)

# 5. Export/import weights
weights = executor.get_model_weights()
# executor.set_model_weights(aggregated_weights)
```

### Server-Side Aggregation

```python
from snp_deconvolution.nvflare_base import federated_averaging

# Collect weights from all sites
site_weights = [site1_weights, site2_weights, site3_weights]
site_sample_counts = [100, 150, 80]

# Aggregate using FedAvg
result = federated_averaging(site_weights, site_sample_counts)
global_weights = result.aggregated_weights

# Distribute to all sites
for site in sites:
    site.executor.set_model_weights(global_weights)
```

## Key Concepts

### Horizontal Federated Learning
- Each site has **different samples** (different patients)
- All sites have **same features** (same SNP set)
- Model weights are shared, not data

### Aggregation Strategies

**FedAvg** (Default):
```python
executor = DLNVFlareExecutor(model, aggregation_strategy='fedavg')
```
- Weighted average of model weights
- Simple and effective
- Best for IID data

**FedProx** (For non-IID data):
```python
executor = DLNVFlareExecutor(
    model,
    aggregation_strategy='fedprox',
    fedprox_mu=0.1,  # Proximal term strength
)
```
- Prevents drift from global model
- Better for heterogeneous data

### Feature Importance

```python
# After training
importance = executor.get_feature_importance()

# Get top 10 SNPs
top_snps = sorted(
    importance.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print(f"Top SNPs: {top_snps}")
```

## Common Workflows

### Workflow 1: Simple FedAvg

```python
# Server initialization
global_weights = None

for round in range(10):
    site_weights = []
    site_counts = []

    # Each site trains
    for site in sites:
        if global_weights:
            site.executor.set_model_weights(global_weights)

        metrics = site.executor.local_train(num_epochs=5)
        weights = site.executor.get_model_weights()

        site_weights.append(weights)
        site_counts.append(metrics.num_samples)

    # Server aggregates
    result = federated_averaging(site_weights, site_counts)
    global_weights = result.aggregated_weights
```

### Workflow 2: Robust Aggregation

```python
from snp_deconvolution.nvflare_base import trimmed_mean_aggregation

# Use trimmed mean to handle outliers
result = trimmed_mean_aggregation(
    site_weights,
    site_counts,
    trim_ratio=0.1,  # Remove top/bottom 10%
)
```

### Workflow 3: Server-Side Optimizer

```python
from snp_deconvolution.nvflare_base import FedOptAggregator

# Initialize FedAdam
aggregator = FedOptAggregator(
    optimizer_type='adam',
    learning_rate=1e-3,
)

for round in range(10):
    # ... sites train ...

    # Aggregate with server-side Adam
    result = aggregator.aggregate(
        global_weights,
        site_weights,
        site_counts,
    )
    global_weights = result.aggregated_weights
```

## Checkpointing

### Save Checkpoint

```python
from pathlib import Path

# Save after each round
checkpoint_path = Path(f'./checkpoints/round_{round_num}.pkl')
executor.save_checkpoint(checkpoint_path)
```

### Load Checkpoint

```python
# Resume from checkpoint
executor.load_checkpoint(checkpoint_path)
current_round = executor.get_current_round()
print(f"Resumed from round {current_round}")
```

## Validation

```python
# Validate on local data
val_metrics = executor.validate()
print(f"Validation accuracy: {val_metrics.accuracy:.4f}")

# Check training history
history = executor.get_training_history()
for i, metrics in enumerate(history):
    print(f"Round {i}: {metrics}")
```

## Error Handling

```python
try:
    metrics = executor.local_train(num_epochs=10)
except RuntimeError as e:
    print(f"Training failed: {e}")
    # Handle error (skip round, use previous model, etc.)

# Validate weights before loading
if executor.validate_weights_compatibility(weights):
    executor.set_model_weights(weights)
else:
    print("Incompatible weights received!")
```

## Performance Tips

### XGBoost
```python
xgb_params = {
    'tree_method': 'gpu_hist',  # Use GPU
    'device': 'cuda',
    'max_depth': 6,
    'eta': 0.1,  # Learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

executor = XGBoostNVFlareExecutor(
    trainer=None,
    num_snps=10000,
    num_populations=5,
    xgb_params=xgb_params,
)
```

### PyTorch
```python
# Use mixed precision training
metrics = executor.local_train(
    num_epochs=5,
    use_mixed_precision=True,  # bf16
    gradient_clip_norm=1.0,    # Stability
)
```

## Testing

Run the example simulation:
```bash
cd /path/to/nvflare_base
python example_usage.py
```

Run unit tests:
```bash
pytest test_nvflare_base.py -v
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for architecture details
3. Review [example_usage.py](example_usage.py) for complete simulations
4. Run tests to verify installation: `pytest test_nvflare_base.py`

## Common Issues

### Issue: XGBoost not found
```bash
pip install xgboost>=2.0.0
```

### Issue: PyTorch not found
```bash
pip install torch>=2.0.0
```

### Issue: GPU not detected
```python
# Use CPU instead
xgb_params = {'tree_method': 'hist', 'device': 'cpu'}

# For PyTorch
executor = DLNVFlareExecutor(model, device=torch.device('cpu'))
```

## Support

- Documentation: See README.md
- Examples: Run example_usage.py
- Tests: Run pytest test_nvflare_base.py -v
- Issues: Report via project issue tracker
