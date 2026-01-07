# NVFlare Base Module for SNP Deconvolution

This module provides NVFlare 2.4+ compatible executors for horizontal federated learning of SNP deconvolution models.

## Architecture

### Horizontal Federation
- **Data Distribution**: Each site has different samples (rows) but the same SNP features (columns)
- **Privacy**: Raw genetic data never leaves the site
- **Aggregation**: Model weights/updates are shared and aggregated at the server

### Components

```
nvflare_base/
├── __init__.py                  # Module exports
├── base_executor.py             # Abstract base class for executors
├── xgb_nvflare_wrapper.py      # XGBoost federated wrapper
├── dl_nvflare_wrapper.py       # PyTorch federated wrapper
└── model_shareable.py          # Serialization utilities
```

## Usage

### XGBoost Federated Learning

```python
from snp_deconvolution.nvflare_base import XGBoostNVFlareExecutor
import numpy as np

# Initialize executor
executor = XGBoostNVFlareExecutor(
    trainer=xgb_trainer,  # Your XGBoostSNPTrainer
    num_snps=10000,
    num_populations=5,
)

# Prepare local training data
X_train = np.random.randint(0, 3, (1000, 10000))  # Genotype matrix
y_train = np.random.randint(0, 5, 1000)           # Population labels
executor.prepare_data(X_train, y_train)

# Federated learning round
metrics = executor.local_train(num_epochs=10)
print(f"Training accuracy: {metrics.accuracy:.4f}")

# Export weights for server aggregation
weights = executor.get_model_weights()

# After server aggregation, load global model
executor.set_model_weights(aggregated_weights)

# Get feature importance
importance = executor.get_feature_importance()
top_snps = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
```

### PyTorch Federated Learning

```python
from snp_deconvolution.nvflare_base import DLNVFlareExecutor
import torch
import torch.nn as nn

# Define or load your model
class SimpleSNPNet(nn.Module):
    def __init__(self, num_snps, num_populations):
        super().__init__()
        self.num_snps = num_snps
        self.num_populations = num_populations
        self.fc1 = nn.Linear(num_snps, 512)
        self.fc2 = nn.Linear(512, num_populations)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleSNPNet(num_snps=10000, num_populations=5)

# Initialize executor with FedAvg
executor = DLNVFlareExecutor(
    model=model,
    aggregation_strategy='fedavg',  # or 'fedprox'
    device=torch.device('cuda'),
)

# Set data loaders
executor.set_data_loaders(train_loader, val_loader)

# Federated learning round
metrics = executor.local_train(
    num_epochs=5,
    use_mixed_precision=True,  # bf16 training
    gradient_clip_norm=1.0,
)

# Export and import weights
weights = executor.get_model_weights()
executor.set_model_weights(aggregated_weights)
```

### FedProx (PyTorch)

FedProx adds a proximal term to prevent local models from drifting too far from the global model:

```python
executor = DLNVFlareExecutor(
    model=model,
    aggregation_strategy='fedprox',
    fedprox_mu=0.1,  # Proximal term coefficient
)

# Load global model (required for FedProx)
executor.set_model_weights(global_weights)

# Train with proximal regularization
metrics = executor.local_train(num_epochs=5)
```

## Federated Learning Strategies

### XGBoost

**Tree Ensemble Aggregation**:
1. Each site trains a local XGBoost model
2. Models exported as JSON tree structures
3. Server creates ensemble by combining trees
4. Sites receive aggregated model

**Advantages**:
- Simple and robust
- No data distribution assumptions
- Works with heterogeneous data

**Limitations**:
- Ensemble size grows with number of sites
- May overfit if sites have very different distributions

### PyTorch - FedAvg

**Weight Averaging**:
1. Sites train local models
2. Export model weights as numpy arrays
3. Server computes weighted average: `w_global = Σ(n_i/N * w_i)`
4. Sites receive averaged weights

**Advantages**:
- Simple and effective
- Well-studied convergence properties
- Works with any PyTorch model

**Hyperparameters**:
- `learning_rate`: Local optimizer learning rate
- `num_epochs`: Local training epochs per round

### PyTorch - FedProx

**Proximal Regularization**:
- Adds term to loss: `L = L_standard + (μ/2) * ||w - w_global||²`
- Prevents excessive drift from global model
- Better for non-IID data

**Hyperparameters**:
- `fedprox_mu`: Proximal term coefficient (0.01-1.0)
  - Higher values: Stay closer to global model
  - Lower values: More freedom for local adaptation

## Serialization

Models are serialized for network transmission:

### PyTorch
- `state_dict` → numpy arrays
- Platform-independent
- Optional compression for large models
- Checksum verification

### XGBoost
- Booster → JSON tree structure
- Human-readable format
- Includes feature names and configuration
- Checksum verification

## Privacy Considerations

### What is Shared
- Model weights (XGBoost: trees, PyTorch: parameters)
- Training metrics (loss, accuracy, sample count)
- Feature importance scores (aggregated)

### What is Never Shared
- Raw genotype data
- Individual sample information
- Gradient information (in standard FedAvg)

### Security Enhancements (Future)
- Differential Privacy: Add noise to weights
- Secure Aggregation: Encrypt weights during transmission
- Homomorphic Encryption: Compute on encrypted data

## Performance Optimization

### XGBoost
```python
xgb_params = {
    'tree_method': 'gpu_hist',  # GPU acceleration
    'device': 'cuda',
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
}
executor = XGBoostNVFlareExecutor(trainer, xgb_params=xgb_params)
```

### PyTorch
```python
# Mixed precision training (bf16)
metrics = executor.local_train(
    num_epochs=5,
    use_mixed_precision=True,
    gradient_clip_norm=1.0,
)

# Multi-GPU training (requires DDP setup)
# See attention_dl module for MultiGPUSNPTrainer
```

## Checkpointing

Save and restore executor state:

```python
# Save checkpoint
executor.save_checkpoint(Path('./checkpoints/executor_round_10.pkl'))

# Load checkpoint
executor.load_checkpoint(Path('./checkpoints/executor_round_10.pkl'))

# Check current round
print(f"Current round: {executor.get_current_round()}")
```

## Validation and Monitoring

```python
# Validate on local data
val_metrics = executor.validate()
print(f"Validation accuracy: {val_metrics.accuracy:.4f}")

# Get training history
history = executor.get_training_history()
for round_num, metrics in enumerate(history):
    print(f"Round {round_num}: {metrics}")

# Feature importance
importance = executor.get_feature_importance()
print(f"Top 10 SNPs: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]}")
```

## Integration with NVFlare (Phase 2)

This module is designed for easy integration with NVFlare:

```python
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

class SNPDeconvNVFlareExecutor(Executor):
    def __init__(self, executor: SNPDeconvExecutor):
        super().__init__()
        self.executor = executor

    def execute(self, task_name: str, shareable: Shareable,
                fl_ctx: FLContext, abort_signal) -> Shareable:
        if task_name == "train":
            # Get global weights from shareable
            weights = shareable.get("weights")
            self.executor.set_model_weights(weights)

            # Local training
            metrics = self.executor.local_train(num_epochs=5)

            # Export weights
            local_weights = self.executor.get_model_weights()

            # Create response shareable
            response = Shareable()
            response["weights"] = local_weights
            response["metrics"] = metrics.to_dict()
            return response
```

## Error Handling

All executors implement comprehensive error handling:

```python
try:
    metrics = executor.local_train(num_epochs=10)
except RuntimeError as e:
    print(f"Training failed: {e}")

try:
    executor.set_model_weights(weights)
except ValueError as e:
    print(f"Incompatible weights: {e}")

# Validate before loading
if executor.validate_weights_compatibility(weights):
    executor.set_model_weights(weights)
```

## Testing

```python
# Test basic functionality
def test_xgboost_executor():
    executor = XGBoostNVFlareExecutor(
        trainer=None,
        num_snps=100,
        num_populations=3,
    )

    # Prepare dummy data
    X = np.random.randint(0, 3, (50, 100))
    y = np.random.randint(0, 3, 50)
    executor.prepare_data(X, y)

    # Train
    metrics = executor.local_train(num_epochs=5)
    assert metrics.accuracy > 0

    # Export/import
    weights = executor.get_model_weights()
    executor.set_model_weights(weights)

    # Validate
    val_metrics = executor.validate()
    assert val_metrics.loss > 0

# Similar for PyTorch executor
```

## Requirements

```
torch>=2.0.0
xgboost>=2.0.0
numpy>=1.24.0
```

Optional for NVFlare integration:
```
nvflare>=2.4.0
```

## References

1. **FedAvg**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data", AISTATS 2017
2. **FedProx**: Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020
3. **NVFlare**: NVIDIA FLARE Documentation, https://nvflare.readthedocs.io/

## License

See main project LICENSE file.
