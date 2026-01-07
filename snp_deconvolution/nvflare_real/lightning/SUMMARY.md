# NVFlare Lightning Implementation Summary

## Files Created

### 1. Core Files

```
snp_deconvolution/nvflare_real/
├── lightning/
│   ├── __init__.py                    # Module initialization
│   ├── client.py                      # NVFlare client script (12KB)
│   ├── job.py                         # Job configuration script (12KB)
│   ├── README.md                      # Complete documentation (10KB)
│   ├── test_setup.py                  # Test script (6KB)
│   └── SUMMARY.md                     # This file
└── data/
    ├── __init__.py                    # Module initialization
    └── federated_data_module.py       # Lightning DataModule (14KB)
```

## Quick Reference

### 1. Test Installation

```bash
cd /Users/saltfish/Files/Coding/Med_SNP_Deconvolution/snp_deconvolution/nvflare_real/lightning

# Run setup test
python test_setup.py
```

### 2. POC Simulation (Recommended First Step)

```bash
# Create job and run simulation
python job.py \
  --mode poc \
  --num_rounds 5 \
  --num_clients 3 \
  --run_now \
  --data_dir /path/to/your/data
```

### 3. Production Deployment

```bash
# Export job package
python job.py \
  --mode export \
  --export_dir ./jobs \
  --num_rounds 50 \
  --clients hospital_A,hospital_B,hospital_C

# Submit to NVFlare
nvflare job submit -j ./jobs/snp_fedavg
```

## Key Implementation Details

### Client API Pattern (client.py)

```python
import nvflare.client as flare

# 1. Initialize
flare.init()

# 2. Get site name
site_name = flare.get_site_name()

# 3. Create trainer and model
trainer = pl.Trainer(...)
model = SNPLightningModule(...)

# 4. CRITICAL: Patch trainer
flare.patch(trainer)

# 5. FL loop
while flare.is_running():
    input_model = flare.receive()
    trainer.fit(model, datamodule=data_module)
    # Updates sent automatically via patch()
```

### Job Configuration (job.py)

Uses `FedAvgRecipe` for simplified configuration:

```python
from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob

job = FedAvgJob(
    name="snp_fedavg",
    num_rounds=10,
    n_clients=3,
    min_clients=3,
)

# Add clients
for client in clients:
    job.to(client, "client.py", script_args=args)
```

### Data Module (federated_data_module.py)

Site-specific data loading:

```python
from snp_deconvolution.nvflare_real.data import SNPFederatedDataModule

dm = SNPFederatedDataModule(
    data_dir='./data',
    site_name='site1',      # Loads site1_cluster.npz
    feature_type='cluster',
    batch_size=128,
)
```

## Data Format Required

Each site needs: `{site_name}_{feature_type}.npz`

```python
{
    'X_train': [n_samples, n_features, encoding_dim],  # float32
    'y_train': [n_samples],                             # int64
    'X_val': [n_samples, n_features, encoding_dim],
    'y_val': [n_samples],
    'X_test': [n_samples, n_features, encoding_dim],
    'y_test': [n_samples],
}
```

## Architecture Comparison

| Feature | client.py | job.py | federated_data_module.py |
|---------|-----------|--------|--------------------------|
| Lines of Code | ~350 | ~350 | ~400 |
| Main Function | FL training loop | Job creation | Data loading |
| NVFlare APIs | `flare.patch()`, `flare.receive()` | `FedAvgJob` | N/A |
| Dependencies | nvflare, pytorch_lightning | nvflare | pytorch_lightning |
| Execution | Auto by NVFlare | Manual | Imported |

## Command Line Arguments

### job.py (Most Common)

```bash
--mode poc|export          # Execution mode
--num_rounds 10            # FL rounds
--num_clients 3            # Number of sites
--local_epochs 1           # Epochs per round
--batch_size 128           # Batch size
--learning_rate 1e-4       # Learning rate
--architecture cnn_transformer  # Model type
--feature_type cluster     # cluster or snp
--run_now                  # Immediate execution (POC only)
```

### client.py (Auto-invoked by NVFlare)

```bash
--data_dir ./data          # Data directory
--feature_type cluster     # Feature type
--local_epochs 1           # Epochs per round
--batch_size 128           # Batch size
--architecture cnn_transformer
--use_focal_loss           # Enable focal loss
--precision bf16-mixed     # Training precision
```

## Testing Checklist

- [ ] Test data module: `python federated_data_module.py` (from data/ dir)
- [ ] Test setup: `python test_setup.py`
- [ ] POC simulation: `python job.py --mode poc --num_rounds 2 --run_now`
- [ ] Export job: `python job.py --mode export --export_dir ./jobs`

## Common Commands

```bash
# Test with dummy data
python test_setup.py

# Quick POC (2 rounds, 3 clients)
python job.py --mode poc --num_rounds 2 --run_now --data_dir ./test_data

# Full POC (10 rounds, custom clients)
python job.py --mode poc --num_rounds 10 --clients site1,site2,site3 --run_now

# Export for production
python job.py --mode export --export_dir ./production_jobs --num_rounds 100

# Submit to deployed system
nvflare job submit -j ./production_jobs/snp_fedavg

# Monitor job
nvflare job list
nvflare job show <JOB_ID>

# Download results
nvflare job download <JOB_ID> -d ./results
```

## Integration Points

### With Existing Code

The implementation integrates with:

1. `snp_deconvolution/attention_dl/lightning_trainer.py`
   - Uses `SNPLightningModule` class
   - Compatible with existing model architectures

2. `dl_models/snp_interpretable_models.py`
   - Model definitions (CNN, Transformer, GNN)
   - Architecture selection via `--architecture` flag

### With NVFlare

- Compatible with NVFlare 2.7.0+
- Uses official Client API pattern
- Follows hello-lightning tutorial architecture

## Performance Considerations

### Hardware

- **GPU (A100/H100)**: Use `--precision bf16-mixed`
- **GPU (V100/older)**: Use `--precision 16-mixed`
- **CPU**: Use `--precision 32`

### Batch Size

- Small datasets: 64-128
- Medium datasets: 128-256
- Large datasets: 256-512

### Workers

- SSD storage: 4-8 workers
- HDD storage: 2-4 workers
- Debugging: 0 workers

## Troubleshooting

### Import Errors

```bash
# Install dependencies
pip install nvflare pytorch-lightning torch
```

### Data Not Found

```bash
# Check data exists
ls data/
# Should show: site1_cluster.npz, site2_cluster.npz, etc.

# Verify data format
python -c "import numpy as np; data=np.load('data/site1_cluster.npz'); print(list(data.keys()))"
```

### GPU Issues

```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Use CPU if needed
python job.py --precision 32  # Automatically uses CPU if no GPU
```

## References

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [Hello Lightning Tutorial](https://nvflare.readthedocs.io/en/2.7.0/hello-world/hello-lightning/)
- [Client API Guide](https://nvflare.readthedocs.io/en/2.7.0/programming_guide/fl_client_api.html)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## Next Steps

1. Test the setup: `python test_setup.py`
2. Prepare your real data (see README.md)
3. Run POC simulation
4. Adjust hyperparameters
5. Deploy to production

## License

Part of Med_SNP_Deconvolution project.
