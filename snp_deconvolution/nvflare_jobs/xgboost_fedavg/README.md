# XGBoost Federated Learning Job for SNP Deconvolution

This job implements federated XGBoost training for SNP-based population deconvolution using NVFlare's histogram-based federated learning approach.

## Overview

- **Algorithm**: Federated XGBoost with histogram-based aggregation
- **Task**: Multi-class classification (3 populations)
- **Data**: SNP genotype features with optional cluster features
- **Framework**: NVFlare 2.5.1+

## Architecture

```
Server (XGBFedController)
    ├── Aggregates histogram from all clients
    └── Builds global XGBoost trees

Clients (FedXGBHistogramExecutor)
    ├── Load local SNP data via SNPXGBDataLoader
    ├── Compute local histograms
    └── Send histograms to server (privacy-preserving)
```

## Directory Structure

```
xgboost_fedavg/
├── app/
│   └── config/
│       ├── config_fed_server.json       # Server config (CPU)
│       ├── config_fed_server_gpu.json   # Server config (GPU)
│       ├── config_fed_client.json       # Client config (CPU)
│       └── config_fed_client_gpu.json   # Client config (GPU)
├── meta.json                            # Job metadata
└── README.md                            # This file
```

## XGBoost Parameters

### Optimized for SNP Classification

```json
{
  "objective": "multi:softprob",     // Multi-class soft probability
  "num_class": 3,                    // 3 populations
  "max_depth": 6,                    // Tree depth (balance complexity)
  "eta": 0.1,                        // Learning rate (conservative)
  "subsample": 0.8,                  // Row sampling (prevent overfitting)
  "colsample_bytree": 0.8,           // Column sampling (feature sampling)
  "tree_method": "hist",             // Histogram-based (CPU) or "gpu_hist" (GPU)
  "eval_metric": "mlogloss"          // Multi-class log loss
}
```

### CPU vs GPU

| Config | tree_method | Use Case |
|--------|-------------|----------|
| CPU    | `hist`      | General purpose, no GPU required |
| GPU    | `gpu_hist`  | Faster training with CUDA-enabled GPU |

## Data Loader

The `SNPXGBDataLoader` class loads site-specific data and returns XGBoost DMatrix.

### Required Data Format

Each site should have:

```
data/
├── site1_train.csv              # Site 1 training data
├── site1_cluster_features.csv   # Site 1 cluster features (optional)
├── site2_train.csv              # Site 2 training data
└── site2_cluster_features.csv   # Site 2 cluster features (optional)
```

### Data File Format

**Training data** (`{site_name}_train.csv`):
```csv
snp_0,snp_1,snp_2,...,snp_n,label
0,1,2,...,1,0
1,0,1,...,2,1
...
```

**Cluster features** (`{site_name}_cluster_features.csv`):
```csv
cluster_0,cluster_1,cluster_2,...
0.5,0.3,0.2
0.7,0.1,0.2
...
```

### Data Loader Configuration

```json
{
  "id": "snp_data_loader",
  "path": "snp_deconvolution.nvflare_real.xgboost.data_loader.SNPXGBDataLoader",
  "args": {
    "data_dir": "/path/to/data",         // Data directory
    "site_name": "site1",                 // Site identifier
    "use_cluster_features": true,         // Include cluster features
    "validation_split": 0.2,              // 20% validation split
    "random_seed": 42,                    // Random seed
    "feature_prefix": "snp_",             // SNP column prefix
    "label_column": "label",              // Label column name
    "enable_categorical": false           // Categorical features
  }
}
```

## Running the Job

### 1. Prepare Data

Ensure each site has properly formatted data files:

```bash
# Example data structure
/data/federated_snp/
├── site1_train.csv
├── site1_cluster_features.csv
├── site2_train.csv
├── site2_cluster_features.csv
└── site3_train.csv
```

### 2. Configure Job

Edit `meta.json` to specify:
- Number of clients
- Data paths
- Training parameters

```json
{
  "name": "xgboost_fedavg_snp",
  "resource_spec": {
    "site1": {"data_dir": "/data/federated_snp"},
    "site2": {"data_dir": "/data/federated_snp"},
    "site3": {"data_dir": "/data/federated_snp"}
  },
  "deploy_map": {
    "app": ["@ALL"]
  },
  "min_clients": 2,
  "num_rounds": 100
}
```

### 3. Submit Job (Simulator)

For local testing with NVFlare simulator:

```bash
# CPU version
nvflare simulator xgboost_fedavg \
  -w /tmp/nvflare/xgb_workspace \
  -n 3 \
  -t 3

# GPU version (requires CUDA)
nvflare simulator xgboost_fedavg \
  -w /tmp/nvflare/xgb_workspace \
  -n 3 \
  -t 3 \
  -gpu 0,1,2
```

### 4. Submit Job (Production)

Using NVFlare Admin Console:

```bash
# Submit job
submit_job xgboost_fedavg

# Monitor progress
list_jobs
show_job xgboost_fedavg

# Download results
download_job xgboost_fedavg
```

## Training Process

### Federated XGBoost Workflow

1. **Initialization**: Server broadcasts XGBoost parameters
2. **Local Training**: Each client:
   - Loads local data via `SNPXGBDataLoader`
   - Computes local histogram
   - Sends histogram to server
3. **Global Aggregation**: Server:
   - Aggregates histograms from all clients
   - Builds global tree nodes
   - Broadcasts tree updates
4. **Iteration**: Repeat steps 2-3 for configured rounds
5. **Model Save**: Final model saved as `model.json`

### Privacy Preservation

- Only **histograms** are shared (not raw data)
- Histograms are aggregated statistics (privacy-preserving)
- Optional: Enable `secure_training: true` for encrypted aggregation

## Output

After training completion:

```
workspace/
└── xgboost_fedavg/
    ├── model.json              # Final XGBoost model
    ├── metrics/                # Training metrics
    │   ├── train_mlogloss.csv
    │   └── val_mlogloss.csv
    └── logs/                   # Execution logs
        ├── server.log
        ├── site1.log
        ├── site2.log
        └── site3.log
```

### Loading Trained Model

```python
import xgboost as xgb

# Load model
model = xgb.Booster()
model.load_model("model.json")

# Make predictions
dtest = xgb.DMatrix(X_test)
predictions = model.predict(dtest)
```

## Hyperparameter Tuning

### Key Parameters to Tune

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `eta` | 0.1 | [0.01, 0.3] | Learning rate (lower = slower, more robust) |
| `max_depth` | 6 | [3, 10] | Tree depth (higher = more complex) |
| `subsample` | 0.8 | [0.5, 1.0] | Row sampling (lower = more regularization) |
| `colsample_bytree` | 0.8 | [0.5, 1.0] | Feature sampling |
| `min_child_weight` | 1 | [1, 10] | Minimum leaf weight (higher = more conservative) |
| `gamma` | 0 | [0, 5] | Minimum loss reduction (higher = more conservative) |

### Early Stopping

```json
{
  "early_stopping_rounds": 10
}
```

Stops training if validation metric doesn't improve for 10 rounds.

## Troubleshooting

### Common Issues

#### 1. Data Loading Errors

```
FileNotFoundError: Could not find data file for site site1
```

**Solution**: Check data file naming and paths:
- File should be named `{site_name}_train.csv`
- Path in `data_dir` should be correct
- Verify permissions

#### 2. GPU Errors

```
XGBoostError: gpu_id is set but XGBoost is not compiled with CUDA
```

**Solution**: Use CPU config or install XGBoost with GPU support:
```bash
pip install xgboost[gpu]
```

#### 3. Memory Issues

```
MemoryError: Unable to allocate array
```

**Solution**:
- Reduce `validation_split`
- Use feature selection
- Increase available memory

#### 4. Convergence Issues

```
WARNING: Model not converging after 100 rounds
```

**Solution**:
- Increase `num_rounds`
- Adjust `eta` (learning rate)
- Check data quality and distribution

## Performance Optimization

### CPU Optimization

```json
{
  "tree_method": "hist",
  "nthread": -1              // Use all CPU cores
}
```

### GPU Optimization

```json
{
  "tree_method": "gpu_hist",
  "predictor": "gpu_predictor",
  "gpu_id": 0
}
```

### Distributed Training

For large datasets, use multiple clients:
- Split data across sites naturally (federated)
- Each site processes local data
- Faster convergence with more data diversity

## Security Considerations

### Data Privacy

- **Histogram Sharing**: Only aggregated histograms shared (not raw data)
- **Secure Aggregation**: Enable `secure_training: true` for encryption
- **Differential Privacy**: Can be added via custom filters

### Network Security

- Use TLS for client-server communication
- Authenticate clients with certificates
- Encrypt model updates in transit

## References

- [NVFlare XGBoost Documentation](https://nvflare.readthedocs.io/en/2.5.1/user_guide/federated_xgboost/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [Federated Learning Best Practices](https://nvflare.readthedocs.io/en/latest/programming_guide.html)

## License

See project root LICENSE file.
