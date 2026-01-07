# XGBoost GPU Module for SNP Deconvolution

GPU-accelerated XGBoost implementation for SNP analysis and population classification, optimized for A100/H100 GPUs and high-dimensional sparse genomic data.

## Overview

This module provides production-ready implementations for:

1. **XGBoostSNPTrainer** - GPU-accelerated XGBoost training for SNP classification
2. **IterativeSNPSelector** - Iterative feature selection using XGBoost importance scores

## Features

- **GPU Acceleration**: Uses `tree_method='gpu_hist'` for A100/H100 GPUs
- **Sparse Matrix Support**: Optimized for high-dimensional sparse genomic data
- **Early Stopping**: Prevents overfitting with validation-based early stopping
- **Feature Selection**: Iterative refinement to identify most informative SNPs
- **NVFlare Compatible**: Export models for federated learning
- **Comprehensive Logging**: Production-ready logging throughout
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Robust error handling and validation

## Installation

### Requirements

```bash
pip install xgboost>=2.0.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install scikit-learn>=1.3.0
```

### GPU Support

Ensure you have:
- CUDA 11.8+ or 12.x
- cuDNN 8.x
- XGBoost built with GPU support

Verify GPU availability:
```python
import xgboost as xgb
print(xgb.build_info()['USE_CUDA'])  # Should be True
```

## Quick Start

### Basic Training

```python
import scipy.sparse as sp
import numpy as np
from snp_deconvolution.xgboost import XGBoostSNPTrainer

# Load your data (sparse matrix format)
X_train = sp.csr_matrix(...)  # Shape: (n_samples, n_snps)
y_train = np.array(...)        # Shape: (n_samples,)
X_val = sp.csr_matrix(...)
y_val = np.array(...)

# Initialize trainer
trainer = XGBoostSNPTrainer(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    gpu_id=0,
    early_stopping_rounds=50,
    num_class=3,  # CHB, GBR, PUR
)

# Train
trainer.fit(X_train, y_train, X_val, y_val, verbose=True)

# Predict
predictions = trainer.predict(X_test)
probabilities = trainer.predict_proba(X_test)

# Get feature importance
importance = trainer.get_feature_importance(importance_type='gain', top_k=100)
```

### Feature Selection

```python
from snp_deconvolution.xgboost import IterativeSNPSelector

# Initialize selector
selector = IterativeSNPSelector(
    initial_k=10000,
    reduction_factor=0.5,
    min_snps=100,
    max_iterations=5,
    importance_type='gain',
    gpu_id=0,
)

# Perform selection
selected_indices, importance_scores = selector.select_features(
    X_train, y_train, X_val, y_val
)

print(f"Selected {len(selected_indices)} SNPs")

# Use selected features
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]
```

## API Reference

### XGBoostSNPTrainer

#### Parameters

- **n_estimators** (int): Number of boosting rounds (default: 1000)
- **max_depth** (int): Maximum tree depth (default: 6)
- **learning_rate** (float): Step size shrinkage (default: 0.1)
- **gpu_id** (int): GPU device ID (default: 0)
- **early_stopping_rounds** (int): Stop if no improvement for N rounds (default: 50)
- **objective** (str): Loss function (default: 'multi:softprob')
- **num_class** (int): Number of classes (default: 3)
- **subsample** (float): Row sampling ratio (default: 0.8)
- **colsample_bytree** (float): Column sampling ratio (default: 0.8)
- **reg_alpha** (float): L1 regularization (default: 0.0)
- **reg_lambda** (float): L2 regularization (default: 1.0)
- **max_bin** (int): Max bins for histogram (default: 512)

#### Methods

- **fit(X, y, X_val, y_val, verbose)**: Train model
- **predict(X)**: Predict class labels
- **predict_proba(X)**: Predict class probabilities
- **get_feature_importance(importance_type, top_k)**: Get SNP importance scores
- **save_model(path)**: Save model to file
- **load_model(path)**: Load model from file
- **export_for_nvflare()**: Export for federated learning

### IterativeSNPSelector

#### Parameters

- **initial_k** (int): Initial number of top SNPs (default: 10000)
- **reduction_factor** (float): Factor to reduce features by (default: 0.5)
- **min_snps** (int): Minimum SNPs to retain (default: 100)
- **max_iterations** (int): Maximum iterations (default: 5)
- **importance_type** (str): Importance metric (default: 'gain')
- **gpu_id** (int): GPU device ID (default: 0)
- **convergence_threshold** (float): Stop if improvement < threshold (default: 0.001)
- **metric** (str): Metric to optimize (default: 'accuracy')

#### Methods

- **select_features(X, y, X_val, y_val)**: Perform iterative selection
- **get_selection_history()**: Get iteration history
- **get_best_iteration_info()**: Get best iteration details
- **plot_selection_curve(save_path)**: Plot selection curve

## Usage Examples

### Example 1: Population Classification

```python
from snp_deconvolution.xgboost import XGBoostSNPTrainer

# Load 1000 Genomes data
X_train, y_train = load_1000g_data('train')  # Your data loading function
X_val, y_val = load_1000g_data('val')

# Train classifier for CHB, GBR, PUR populations
trainer = XGBoostSNPTrainer(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    gpu_id=0,
    num_class=3,
)

trainer.fit(X_train, y_train, X_val, y_val)

# Evaluate
y_pred = trainer.predict(X_val)
accuracy = (y_pred == y_val).mean()
print(f"Accuracy: {accuracy:.4f}")
```

### Example 2: Identify Top SNPs

```python
from snp_deconvolution.xgboost import IterativeSNPSelector

# Select most informative SNPs
selector = IterativeSNPSelector(
    initial_k=50000,
    reduction_factor=0.6,
    min_snps=500,
    max_iterations=5,
)

selected_indices, scores = selector.select_features(
    X_train, y_train, X_val, y_val
)

# Get top 10 SNPs
top_snps = list(scores.items())[:10]
for snp_idx, score in top_snps:
    print(f"SNP {snp_idx}: {score:.4f}")

# Plot selection curve
selector.plot_selection_curve(save_path='selection_curve.png')
```

### Example 3: Custom Hyperparameters

```python
# High regularization for preventing overfitting
trainer = XGBoostSNPTrainer(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=10.0,
    min_child_weight=5,
)

# Deeper trees for complex patterns
trainer = XGBoostSNPTrainer(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.1,
    min_child_weight=1,
)
```

### Example 4: Model Persistence

```python
from pathlib import Path

# Save model
trainer.save_model(Path('models/snp_classifier'))

# Load model later
new_trainer = XGBoostSNPTrainer()
new_trainer.load_model(Path('models/snp_classifier'))

# Use loaded model
predictions = new_trainer.predict(X_test)
```

### Example 5: NVFlare Export

```python
# Export for federated learning
nvflare_dict = trainer.export_for_nvflare()

# Contains:
# - model_json: Serialized XGBoost model
# - feature_names: SNP identifiers
# - params: Training parameters
# - training_history: Evaluation metrics

print(f"Model type: {nvflare_dict['model_type']}")
print(f"Features: {nvflare_dict['n_features']}")
```

## Performance Tips

### GPU Optimization

1. **Use sparse matrices**: CSR format is most efficient
2. **Batch size**: Process in batches if memory limited
3. **max_bin**: Use 512+ for A100/H100 GPUs
4. **Multiple GPUs**: Set `gpu_id` for different devices

### Feature Selection Strategy

1. **Start broad**: Use `initial_k` to capture many SNPs initially
2. **Gradual reduction**: Use `reduction_factor=0.5-0.7` for stable selection
3. **Monitor convergence**: Check `convergence_threshold` for early stopping
4. **Validate**: Always use validation set for selection

### Hyperparameter Tuning

| Scenario | max_depth | learning_rate | reg_lambda | subsample |
|----------|-----------|---------------|------------|-----------|
| Baseline | 6 | 0.1 | 1.0 | 0.8 |
| High regularization | 4 | 0.05 | 10.0 | 0.7 |
| Complex patterns | 10 | 0.1 | 1.0 | 0.8 |
| Fast training | 5 | 0.3 | 1.0 | 0.8 |

## Troubleshooting

### GPU Not Available

```python
# Error: GPU is not available
# Solution: Check CUDA installation
import torch
print(torch.cuda.is_available())  # Should be True
```

### Out of Memory

```python
# Reduce max_bin or process in batches
trainer = XGBoostSNPTrainer(max_bin=256)  # Lower from default 512
```

### Slow Training

```python
# Increase learning_rate or reduce n_estimators
trainer = XGBoostSNPTrainer(
    n_estimators=500,  # Down from 1000
    learning_rate=0.2,  # Up from 0.1
)
```

## Architecture Details

### GPU Histogram Algorithm

The module uses XGBoost's GPU histogram algorithm (`tree_method='gpu_hist'`):

1. **Binning**: Continuous features binned into discrete bins
2. **Histogram construction**: GPU builds histograms in parallel
3. **Split finding**: GPU finds best splits across all bins
4. **Tree building**: Constructs trees using gradient boosting

### Memory Layout

- **Input**: CSR sparse matrix (row-major)
- **GPU**: Data transferred to GPU memory
- **Output**: Dense predictions on GPU
- **Transfer**: Minimal CPU-GPU transfers

## Integration

### With Other Modules

```python
# Integration with data loaders
from snp_deconvolution.data_integration import load_1000g_sparse

X, y = load_1000g_sparse('chr22')

# Integration with evaluation
from snp_deconvolution.evaluation import evaluate_model

metrics = evaluate_model(trainer, X_test, y_test)
```

### With NVFlare

```python
# Export for federated learning
export_dict = trainer.export_for_nvflare()

# Use in NVFlare executor
class XGBoostExecutor(Executor):
    def execute(self, task):
        model_data = task.data['model']
        # Load and use exported model
```

## Testing

Run the example usage script:

```bash
python snp_deconvolution/xgboost/example_usage.py
```

This will run all examples with synthetic data.

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [GPU Algorithm Details](https://arxiv.org/abs/1603.02754)
- [Feature Selection Methods](https://scikit-learn.org/stable/modules/feature_selection.html)

## License

See main project LICENSE file.

## Contact

For issues or questions, please open a GitHub issue.
