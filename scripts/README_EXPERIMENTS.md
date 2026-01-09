# Automated Federated Learning Experiment Runner

This document describes how to use the `run_fl_experiments.py` script to run automated federated learning experiments with different configurations.

## Overview

The experiment runner automates the entire federated learning workflow:

1. **Data Preparation**: Creates federated data splits with various non-IID strategies
2. **Training**: Runs NVFlare POC simulations with different FL algorithms
3. **Metrics Collection**: Gathers accuracy, loss, and convergence metrics
4. **Analysis**: Generates summary reports comparing all configurations

## Quick Start

### Basic Usage

Run all experiments with default settings:
```bash
python scripts/run_fl_experiments.py
```

### Dry Run

See what would be executed without running experiments:
```bash
python scripts/run_fl_experiments.py --dry_run
```

### Custom Configuration

Run specific strategies and data splits:
```bash
python scripts/run_fl_experiments.py \
    --strategies fedavg,fedprox \
    --splits iid,dirichlet_0.5 \
    --num_rounds 100
```

## Experiment Matrix

### Data Splits

The script supports multiple data partitioning strategies:

- **`iid`**: Independent and Identically Distributed (baseline)
  - Each site has balanced data distribution
  - Best case scenario for federated learning

- **`dirichlet_0.5`**: Moderate non-IID distribution
  - Uses Dirichlet distribution with α=0.5
  - Realistic heterogeneity between sites

- **`dirichlet_0.1`**: Strong non-IID distribution
  - Uses Dirichlet distribution with α=0.1
  - High class imbalance across sites
  - Challenging scenario

- **`label_skew`**: Label skew distribution
  - Each site has only a subset of classes
  - Tests strategy robustness to extreme heterogeneity

You can also create custom Dirichlet splits:
```bash
# Custom alpha value
python scripts/run_fl_experiments.py --splits dirichlet_0.3,dirichlet_0.7
```

### FL Strategies

- **`fedavg`**: Federated Averaging (baseline)
  - Simple averaging of client models
  - Reference: McMahan et al., 2017

- **`fedprox`**: Federated Proximal
  - Adds proximal term to handle heterogeneity
  - Better convergence with non-IID data
  - Reference: Li et al., 2020

- **`scaffold`**: Stochastic Controlled Averaging
  - Uses control variates to correct client drift
  - Best for highly heterogeneous data
  - Reference: Karimireddy et al., 2020

- **`fedopt_adam`**: FedOpt with Adam
  - Server-side adaptive optimization
  - Often faster convergence

- **`fedopt_yogi`**: FedOpt with Yogi
  - Alternative server optimizer
  - More stable than Adam in some cases
  - Reference: Reddi et al., 2021

### Models

Currently supports:
- **`lightning`**: PyTorch Lightning model with CNN+Transformer architecture

Future support:
- **`xgboost`**: XGBoost with federated learning

## Command Line Arguments

### Data Arguments

- `--pipeline_output`: Path to Haploblock pipeline output
  - Default: `out_dir/TNFa`

- `--population_files`: List of population label files
  - Default: `data/igsr-chb.tsv.tsv data/igsr-gbr.tsv.tsv data/igsr-pur.tsv.tsv`

- `--data_dir`: Base directory for federated data
  - Default: `data/federated_experiments`

- `--output_dir`: Output directory for results
  - Default: `results/fl_experiments`

### Experiment Selection

- `--strategies`: Comma-separated strategies or "all"
  - Example: `--strategies fedavg,fedprox,scaffold`
  - Default: `all`

- `--splits`: Comma-separated splits or "all"
  - Example: `--splits iid,dirichlet_0.5`
  - Default: `all`

- `--models`: Comma-separated models or "all"
  - Example: `--models lightning`
  - Default: `lightning`

### Experiment Configuration

- `--num_rounds`: Number of FL rounds per experiment
  - Default: `50`
  - Recommendation: 50-100 for convergence analysis

- `--num_sites`: Number of federated sites
  - Default: `3`
  - Must match your data preparation

### Execution Options

- `--dry_run`: Print commands without execution
  - Useful for debugging and planning

- `--parallel`: Run experiments in parallel (not implemented yet)
  - Coming soon for faster execution

- `--verbose`: Enable debug logging
  - Shows detailed progress information

## Output Structure

The script creates the following directory structure:

```
results/fl_experiments/
├── experiment_report.json          # Full experiment report
├── experiment_summary.csv          # Summary statistics
├── experiment_results.json         # Successful experiments
├── failed_experiments.json         # Failed experiments
│
├── lightning_iid_fedavg/           # Individual experiment
│   ├── workspace/                  # NVFlare workspace
│   │   └── lightning_iid_fedavg/   # Job artifacts
│   │       ├── app_server/         # Server logs and models
│   │       └── app_site*/          # Client logs
│   └── results/                    # Processed metrics
│
├── lightning_dirichlet_0.5_fedprox/
│   └── ...
│
└── ...
```

## Output Files

### experiment_report.json

Complete experiment report with:
- Timestamp and configuration
- Success/failure counts
- Full results for all experiments
- Detailed error messages

### experiment_summary.csv

Tabular summary with columns:
- `experiment`: Experiment name
- `split_type`: Data split strategy
- `strategy`: FL algorithm
- `model`: Model type
- `num_rounds`: Number of rounds
- `status`: success/failed/dry_run
- `duration_seconds`: Execution time
- `val_accuracy`: Final validation accuracy (if available)
- `val_loss`: Final validation loss (if available)

### Individual Experiment Results

Each experiment directory contains:
- **workspace/**: Complete NVFlare workspace with logs
- **results/**: Processed metrics and visualizations (if implemented)

## Examples

### Example 1: Quick Comparison

Compare FedAvg and FedProx on IID and non-IID data:

```bash
python scripts/run_fl_experiments.py \
    --strategies fedavg,fedprox \
    --splits iid,dirichlet_0.5 \
    --num_rounds 50
```

### Example 2: Full Strategy Comparison

Test all strategies on moderate non-IID data:

```bash
python scripts/run_fl_experiments.py \
    --strategies all \
    --splits dirichlet_0.5 \
    --num_rounds 100
```

### Example 3: Data Heterogeneity Study

Compare different levels of non-IID:

```bash
python scripts/run_fl_experiments.py \
    --strategies fedavg \
    --splits iid,dirichlet_0.7,dirichlet_0.5,dirichlet_0.3,dirichlet_0.1 \
    --num_rounds 50
```

### Example 4: Custom Configuration

Run with custom paths and settings:

```bash
python scripts/run_fl_experiments.py \
    --pipeline_output /path/to/pipeline/output \
    --data_dir /path/to/federated/data \
    --output_dir /path/to/results \
    --strategies fedprox,scaffold \
    --splits dirichlet_0.1 \
    --num_rounds 200 \
    --num_sites 5
```

## Analyzing Results

### View Summary

After experiments complete:

```bash
# View CSV summary
cat results/fl_experiments/experiment_summary.csv

# Or use pandas
python -c "
import pandas as pd
df = pd.read_csv('results/fl_experiments/experiment_summary.csv')
print(df.to_string())
"
```

### Best Configurations

The script automatically identifies:
1. Best strategy for each data split
2. Fastest converging experiments
3. Most robust configurations

Check the console output or `experiment_report.json`.

### Visualization (Manual)

You can create visualizations using the collected data:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/fl_experiments/experiment_summary.csv')

# Plot accuracy by strategy
df.groupby('strategy')['val_accuracy'].mean().plot(kind='bar')
plt.title('Average Accuracy by Strategy')
plt.ylabel('Validation Accuracy')
plt.show()

# Compare convergence speed
df.plot(x='split_type', y='duration_seconds', kind='bar')
plt.title('Training Time by Data Split')
plt.ylabel('Duration (seconds)')
plt.show()
```

## Troubleshooting

### Experiment Fails

Check `failed_experiments.json` for error details:

```bash
cat results/fl_experiments/failed_experiments.json | python -m json.tool
```

Common issues:
1. **Data preparation fails**: Check pipeline output exists
2. **NVFlare simulation fails**: Check workspace logs
3. **Timeout**: Increase timeout in script or reduce num_rounds

### Metrics Not Collected

If metrics collection fails:
1. Check workspace directory exists
2. Verify NVFlare logs were created
3. Run with `--verbose` for debugging

### Memory Issues

For multiple parallel experiments:
1. Reduce `--num_sites` (fewer clients = less memory)
2. Run experiments sequentially (don't use `--parallel`)
3. Clear old workspaces between runs

## Advanced Usage

### Custom Split Parameters

Modify `EXPERIMENT_MATRIX` in the script:

```python
EXPERIMENT_MATRIX = {
    "data_splits": {
        "custom_split": {"alpha": 0.2, "labels_per_site": None},
        # Add your configuration
    },
    # ...
}
```

### Custom Strategy Parameters

Add new strategies:

```python
EXPERIMENT_MATRIX = {
    "strategies": {
        "fedprox_strong": {
            "client_script": "client.py",
            "args": {"--strategy": "fedprox", "--mu": "0.1"}
        },
        # Add your configuration
    },
    # ...
}
```

### Integrating New Algorithms

To add a new FL algorithm:

1. Create client script (e.g., `new_algorithm_client.py`)
2. Add to `EXPERIMENT_MATRIX["strategies"]`
3. If needed, create custom controller
4. Update `run_nvflare_simulation()` to handle special cases

## Performance Tips

1. **Use dry run first**: Verify configuration before running
2. **Start small**: Test with 10 rounds before full runs
3. **Monitor resources**: Watch CPU/GPU usage during experiments
4. **Clean up**: Remove old workspaces to save disk space
5. **Parallel execution**: Coming soon for faster throughput

## Citation

If you use this experiment runner in your research:

```bibtex
@software{fl_experiment_runner,
  title={Automated Federated Learning Experiment Runner},
  author={Med SNP Deconvolution Team},
  year={2026},
  url={https://github.com/your-repo/Med_SNP_Deconvolution}
}
```

## References

1. McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
2. Li et al. "Federated Optimization in Heterogeneous Networks" (2020)
3. Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (2020)
4. Reddi et al. "Adaptive Federated Optimization" (2021)

## Support

For issues or questions:
1. Check workspace logs: `results/fl_experiments/<experiment>/workspace/*/log.txt`
2. Run with `--verbose` for detailed output
3. Review NVFlare documentation: https://nvflare.readthedocs.io/
4. Open an issue on GitHub
