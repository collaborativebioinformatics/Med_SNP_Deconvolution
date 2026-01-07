# NVFlare Lightning - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install nvflare pytorch-lightning torch numpy
```

### 2. Test Setup

```bash
cd /Users/saltfish/Files/Coding/Med_SNP_Deconvolution/snp_deconvolution/nvflare_real/lightning

# Create test data and verify installation
python test_setup.py
```

### 3. Run POC Simulation

```bash
# Quick 2-round simulation with test data
python job.py \
  --mode poc \
  --num_rounds 2 \
  --num_clients 3 \
  --run_now \
  --data_dir ./test_data
```

Done! You should see federated learning training progress.

## Common Use Cases

### Production Training (Cluster Features)

```bash
# 1. Prepare your data (ensure site-specific .npz files exist)
ls /path/to/data/
# Should show: site1_cluster.npz, site2_cluster.npz, etc.

# 2. Run POC simulation
python job.py \
  --mode poc \
  --num_rounds 10 \
  --num_clients 3 \
  --feature_type cluster \
  --architecture cnn_transformer \
  --batch_size 128 \
  --run_now \
  --data_dir /path/to/data
```

### Production Training (SNP Features)

```bash
python job.py \
  --mode poc \
  --num_rounds 10 \
  --num_clients 3 \
  --feature_type snp \
  --architecture cnn_transformer \
  --batch_size 128 \
  --run_now \
  --data_dir /path/to/data
```

### Export for Deployment

```bash
# Create job package
python job.py \
  --mode export \
  --export_dir ./production_jobs \
  --num_rounds 100 \
  --clients hospital_A,hospital_B,hospital_C \
  --feature_type cluster

# Submit to deployed NVFlare
nvflare job submit -j ./production_jobs/snp_fedavg
```

## File Locations

```
Project Root: /Users/saltfish/Files/Coding/Med_SNP_Deconvolution

Main Scripts:
  - client.py:     snp_deconvolution/nvflare_real/lightning/client.py
  - job.py:        snp_deconvolution/nvflare_real/lightning/job.py
  - test_setup.py: snp_deconvolution/nvflare_real/lightning/test_setup.py

Data Module:
  - federated_data_module.py: snp_deconvolution/nvflare_real/data/federated_data_module.py

Documentation:
  - README.md:     snp_deconvolution/nvflare_real/lightning/README.md
  - SUMMARY.md:    snp_deconvolution/nvflare_real/lightning/SUMMARY.md
  - QUICKSTART.md: snp_deconvolution/nvflare_real/lightning/QUICKSTART.md (this file)
```

## Data Format Quick Reference

### Required Files

```
data/
├── site1_cluster.npz   # or site1_snp.npz
├── site2_cluster.npz
└── site3_cluster.npz
```

### .npz Structure

```python
{
    'X_train': np.array([n_samples, n_features, encoding_dim], dtype=np.float32),
    'y_train': np.array([n_samples], dtype=np.int64),
    'X_val': np.array([n_val, n_features, encoding_dim], dtype=np.float32),
    'y_val': np.array([n_val], dtype=np.int64),
    'X_test': np.array([n_test, n_features, encoding_dim], dtype=np.float32),
    'y_test': np.array([n_test], dtype=np.int64),
}
```

## Key Parameters

### Most Important

- `--mode`: `poc` (simulation) or `export` (deployment)
- `--num_rounds`: Number of FL rounds (10-100)
- `--feature_type`: `cluster` or `snp`
- `--data_dir`: Path to data directory

### Performance Tuning

- `--batch_size`: 64, 128, 256, 512
- `--learning_rate`: 1e-5 to 1e-3
- `--local_epochs`: 1-5 epochs per round
- `--architecture`: `cnn`, `cnn_transformer`, `gnn`

### Hardware

- `--precision`: `bf16-mixed` (A100/H100), `16-mixed` (V100), `32` (CPU)
- `--num_workers`: 0 (debug), 4 (normal), 8 (fast I/O)

## Troubleshooting

### No GPU
```bash
# Will automatically use CPU
python job.py --mode poc --num_rounds 2 --run_now
```

### Data Not Found
```bash
# Check files exist
ls data/
# Verify format
python -c "import numpy as np; print(list(np.load('data/site1_cluster.npz').keys()))"
```

### Import Error
```bash
pip install nvflare pytorch-lightning torch numpy
```

## Next Steps

1. Read full documentation: `cat README.md`
2. Review implementation: `cat SUMMARY.md`
3. Customize for your data
4. Run POC simulation
5. Deploy to production

## Quick Help

```bash
# Get help on job.py
python job.py --help

# Get help on client.py
python client.py --help
```

## Support

See README.md for detailed documentation and troubleshooting.
