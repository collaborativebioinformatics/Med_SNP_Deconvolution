# Scripts Directory

This directory contains utility scripts for the Med SNP Deconvolution project.

## Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `prepare_federated_data.py` | Prepare federated learning data splits | Data preparation |
| `run_fl_experiments.py` | Automated FL experiment runner | Batch experiments |

## Quick Start

### 1. Prepare Federated Data (Single Configuration)

```bash
# Prepare IID data split
python scripts/prepare_federated_data.py \
    --pipeline_output out_dir/TNFa \
    --num_sites 3 \
    --output_dir data/federated \
    --split_type iid
```

### 2. Run Automated Experiments (Multiple Configurations)

```bash
# Dry run to see what would be executed
python scripts/run_fl_experiments.py --dry_run

# Run experiments comparing strategies
python scripts/run_fl_experiments.py \
    --strategies fedavg,fedprox \
    --splits iid,dirichlet_0.5 \
    --num_rounds 50
```

## Scripts Documentation

### prepare_federated_data.py

**Purpose**: Prepare data splits for federated learning with various non-IID strategies.

**Key Features**:
- Multiple split strategies (IID, Dirichlet, Label Skew, Quantity Skew)
- Support for cluster and SNP features
- Verification and validation
- Comprehensive statistics

**Common Use Cases**:

```bash
# 1. IID split (baseline)
python scripts/prepare_federated_data.py \
    --pipeline_output out_dir/TNFa \
    --num_sites 3 \
    --split_type iid \
    --output_dir data/federated/iid

# 2. Non-IID with Dirichlet distribution
python scripts/prepare_federated_data.py \
    --pipeline_output out_dir/TNFa \
    --num_sites 3 \
    --split_type dirichlet \
    --alpha 0.5 \
    --output_dir data/federated/dirichlet_0.5

# 3. Label skew (extreme heterogeneity)
python scripts/prepare_federated_data.py \
    --pipeline_output out_dir/TNFa \
    --num_sites 3 \
    --split_type label_skew \
    --labels_per_site 2 \
    --output_dir data/federated/label_skew
```

**Output Structure**:
```
data/federated/
├── site1/
│   ├── train_cluster.pt
│   ├── train_cluster.npz
│   ├── val_cluster.pt
│   └── val_cluster.npz
├── site2/
│   └── ...
├── site3/
│   └── ...
└── dataset_metadata.json
```

**Documentation**: See inline help with `--help`

---

### run_fl_experiments.py

**Purpose**: Automate running multiple federated learning experiments with different configurations.

**Key Features**:
- Automated data preparation
- NVFlare POC simulation
- Metrics collection and analysis
- Summary report generation
- Error handling and recovery
- Progress tracking

**Common Use Cases**:

```bash
# 1. Quick test
python scripts/run_fl_experiments.py \
    --strategies fedavg \
    --splits iid \
    --num_rounds 5 \
    --dry_run

# 2. Strategy comparison
python scripts/run_fl_experiments.py \
    --strategies fedavg,fedprox,scaffold \
    --splits dirichlet_0.5 \
    --num_rounds 50

# 3. Heterogeneity study
python scripts/run_fl_experiments.py \
    --strategies fedprox \
    --splits iid,dirichlet_0.7,dirichlet_0.5,dirichlet_0.1 \
    --num_rounds 50

# 4. Full benchmark
python scripts/run_fl_experiments.py \
    --strategies all \
    --splits all \
    --num_rounds 100
```

**Output Structure**:
```
results/fl_experiments/
├── experiment_report.json          # Full report
├── experiment_summary.csv          # Summary table
├── experiment_results.json         # Successful experiments
├── failed_experiments.json         # Failed experiments
│
├── lightning_iid_fedavg/           # Individual experiment
│   ├── workspace/                  # NVFlare workspace
│   └── results/                    # Metrics
│
└── ...
```

**Documentation**:
- Full guide: `README_EXPERIMENTS.md`
- Quick start: `EXPERIMENTS_QUICKSTART.md`

## Workflow

### Complete Workflow: Data Preparation → Training → Analysis

```bash
# Step 1: Prepare data (if not using run_fl_experiments.py)
python scripts/prepare_federated_data.py \
    --pipeline_output out_dir/TNFa \
    --num_sites 3 \
    --split_type dirichlet \
    --alpha 0.5 \
    --output_dir data/federated/dirichlet_0.5

# Step 2: Run experiments
python scripts/run_fl_experiments.py \
    --strategies fedavg,fedprox \
    --splits dirichlet_0.5 \
    --num_rounds 50

# Step 3: Analyze results
python scripts/analyze_experiments.py  # (if available)
# Or manually:
cat results/fl_experiments/experiment_summary.csv
```

### Manual vs Automated

**Use `prepare_federated_data.py` when**:
- You need a specific data split configuration
- You want to prepare data once and reuse it
- You're testing data preparation parameters

**Use `run_fl_experiments.py` when**:
- You want to run multiple experiments
- You need to compare strategies or splits
- You want automated metrics collection

## Supported Configurations

### Data Split Types

| Type | Alpha | Description | Use Case |
|------|-------|-------------|----------|
| `iid` | N/A | Independent and Identically Distributed | Baseline |
| `dirichlet_0.7` | 0.7 | Mild heterogeneity | Realistic scenario |
| `dirichlet_0.5` | 0.5 | Moderate heterogeneity | Common non-IID |
| `dirichlet_0.3` | 0.3 | Strong heterogeneity | Challenging |
| `dirichlet_0.1` | 0.1 | Extreme heterogeneity | Stress test |
| `label_skew` | N/A | Label partitioning | Extreme case |
| `quantity_skew` | N/A | Unbalanced sample counts | Real-world |

### FL Strategies

| Strategy | Description | Best For | Reference |
|----------|-------------|----------|-----------|
| `fedavg` | Federated Averaging | Baseline, IID data | McMahan et al., 2017 |
| `fedprox` | Federated Proximal | Moderate non-IID | Li et al., 2020 |
| `scaffold` | Controlled Averaging | Strong non-IID | Karimireddy et al., 2020 |
| `fedopt_adam` | FedOpt with Adam | Fast convergence | Reddi et al., 2021 |
| `fedopt_yogi` | FedOpt with Yogi | Stable training | Reddi et al., 2021 |

### Models

- **lightning**: PyTorch Lightning with CNN+Transformer (current)
- **xgboost**: XGBoost with federated learning (planned)

## Requirements

### Python Packages

```bash
# Core dependencies
pip install numpy pandas tqdm

# NVFlare
pip install nvflare

# PyTorch Lightning
pip install pytorch-lightning torch

# For visualization (optional)
pip install matplotlib seaborn
```

### System Requirements

- **Disk Space**: 5-10 GB for full experiments
- **Memory**: 8+ GB RAM recommended
- **GPU**: Optional but recommended (2-5x speedup)

## Troubleshooting

### Common Issues

1. **"Pipeline output not found"**
   ```bash
   # Check if pipeline output exists
   ls out_dir/TNFa/
   ```

2. **"Data preparation failed"**
   ```bash
   # Run with verbose logging
   python scripts/prepare_federated_data.py --verbose ...
   ```

3. **"NVFlare simulation failed"**
   ```bash
   # Check workspace logs
   find results -name "log.txt" -exec tail -n 20 {} +
   ```

4. **"Out of memory"**
   ```bash
   # Reduce batch size or number of sites
   python scripts/run_fl_experiments.py --num_sites 2 ...
   ```

### Debug Mode

```bash
# Enable verbose logging for any script
python scripts/run_fl_experiments.py --verbose ...
python scripts/prepare_federated_data.py --verbose ...
```

## Performance Tips

1. **Start small**: Test with `--num_rounds 5` first
2. **Use dry run**: Verify with `--dry_run` before full runs
3. **Monitor resources**: Watch CPU/GPU/disk during experiments
4. **Clean up**: Remove old workspaces to save space
5. **Parallel execution**: Coming soon for faster throughput

## Examples

### Example 1: Quick Test

```bash
# Verify everything works
python scripts/run_fl_experiments.py \
    --strategies fedavg \
    --splits iid \
    --num_rounds 5 \
    --dry_run
```

### Example 2: Algorithm Comparison

```bash
# Compare 3 algorithms on non-IID data
python scripts/run_fl_experiments.py \
    --strategies fedavg,fedprox,scaffold \
    --splits dirichlet_0.5 \
    --num_rounds 50
```

### Example 3: Publication-Ready Benchmark

```bash
# Full benchmark for paper
python scripts/run_fl_experiments.py \
    --strategies fedavg,fedprox,scaffold \
    --splits iid,dirichlet_0.5,dirichlet_0.1 \
    --num_rounds 100 \
    --output_dir results/paper_experiments
```

## Additional Resources

- **Full Experiment Guide**: `README_EXPERIMENTS.md`
- **Quick Start Guide**: `EXPERIMENTS_QUICKSTART.md`
- **NVFlare Docs**: https://nvflare.readthedocs.io/
- **Project README**: `../README.md`

## Contributing

When adding new scripts:

1. Add entry to this README
2. Include inline documentation
3. Add `--help` argument
4. Provide usage examples
5. Update workflow documentation

## Support

For issues or questions:
1. Check script help: `python scripts/<script>.py --help`
2. Review documentation files
3. Check workspace logs for NVFlare issues
4. Open GitHub issue with error details
