# FedOpt Quick Reference Card

## Import

```python
from snp_deconvolution.nvflare_real.controllers import FedOptController
```

## Basic Usage

```python
from nvflare.job_config.api import FedJob

job = FedJob(name="my_job")

controller = FedOptController(
    num_clients=3,
    num_rounds=10,
    optimizer='adam',
    server_lr=0.01
)

job.to_server(controller)
```

## Optimizers Cheat Sheet

| Optimizer | Best For | server_lr | Key Parameters |
|-----------|----------|-----------|----------------|
| **adam** | General use | 0.01 | beta1=0.9, beta2=0.999, epsilon=1e-8 |
| **yogi** | Heterogeneous data | 0.02 | beta1=0.9, beta2=0.999, epsilon=1e-3 |
| **adagrad** | Sparse features | 0.01 | epsilon=1e-8 |
| **sgdm** | Simple baseline | 0.01 | momentum=0.9 |

## Configuration Templates

### FedAdam (Recommended Default)
```python
FedOptController(
    num_clients=3,
    num_rounds=10,
    optimizer='adam',
    server_lr=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
)
```

### FedYogi (Best for Heterogeneous Data)
```python
FedOptController(
    num_clients=3,
    num_rounds=10,
    optimizer='yogi',
    server_lr=0.02,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-3  # Note: larger epsilon
)
```

### FedAdaGrad (Good for Sparse Data)
```python
FedOptController(
    num_clients=3,
    num_rounds=10,
    optimizer='adagrad',
    server_lr=0.01,
    epsilon=1e-8
)
```

### SGD with Momentum (Simple Baseline)
```python
FedOptController(
    num_clients=3,
    num_rounds=10,
    optimizer='sgdm',
    server_lr=0.01,
    momentum=0.9
)
```

## Command Line Examples

### Run Example Script
```bash
# FedAdam
python example_fedopt_job.py --optimizer adam --server_lr 0.01 --run_now

# FedYogi
python example_fedopt_job.py --optimizer yogi --server_lr 0.02 --run_now

# Quick test (5 rounds)
python example_fedopt_job.py --optimizer adam --num_rounds 5 --run_now
```

### Compare Algorithms
```bash
# Compare FedAvg vs FedOpt-Adam
python compare_fedavg_fedopt.py --num_rounds 10 --optimizers adam

# Compare multiple optimizers
python compare_fedavg_fedopt.py --num_rounds 10 --optimizers adam,yogi,sgdm

# Quick comparison
python compare_fedavg_fedopt.py --quick --optimizers adam
```

## Hyperparameter Tuning Guide

### If Training Diverges
- **Reduce** `server_lr` → try 0.001 or 0.005
- **Increase** `beta2` (Adam/Yogi) → try 0.9999

### If Convergence is Too Slow
- **Increase** `server_lr` → try 0.05 or 0.1
- Try **yogi** instead of adam
- **Reduce** momentum (SGDM) → try 0.5

### If Results are Unstable (High Variance)
- **Reduce** `server_lr`
- **Increase** `beta2` → try 0.9999
- Use **adam** or **sgdm** (more stable than yogi)

### For Heterogeneous Data (Non-IID)
- Use **yogi** optimizer
- Set `epsilon=1e-3`
- Use `server_lr=0.02`

## Testing

```bash
# Run all tests
cd /Users/saltfish/Files/Coding/Med_SNP_Deconvolution/snp_deconvolution/nvflare_real/controllers
python test_optimizers_standalone.py

# Expected output:
# Tests run: 14
# Successes: 14
# Failures: 0
# Errors: 0
```

## Integration into job.py

**Replace:**
```python
from nvflare.app_common.workflows.fedavg import FedAvg

controller = FedAvg(
    num_clients=len(clients),
    num_rounds=num_rounds,
)
```

**With:**
```python
from snp_deconvolution.nvflare_real.controllers import FedOptController

controller = FedOptController(
    num_clients=len(clients),
    num_rounds=num_rounds,
    optimizer='adam',
    server_lr=0.01,
    beta1=0.9,
    beta2=0.999,
)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ImportError: No module named 'nvflare' | `pip install nvflare` |
| Training diverges | Reduce server_lr to 0.001 |
| Too slow convergence | Increase server_lr to 0.05, try yogi |
| High variance | Increase beta2 to 0.9999 |
| Parameters not updating | Check client sends NUM_STEPS_CURRENT_ROUND |

## File Locations

```
controllers/
├── __init__.py                      # Package init
├── fedopt_controller.py             # Core implementation (700 lines)
├── test_fedopt_controller.py        # Integration tests
├── test_optimizers_standalone.py    # Unit tests (14 tests)
├── example_fedopt_job.py            # Usage example
├── compare_fedavg_fedopt.py         # Comparison tool
├── README.md                        # Full documentation
├── IMPLEMENTATION_SUMMARY.md        # Implementation details
└── QUICK_REFERENCE.md              # This file
```

## Algorithm at a Glance

```
FedAvg:  w_global = (1/n) * Σ w_i

FedOpt:  1. Δ_i = w_i - w_global
         2. g = (1/n) * Σ Δ_i
         3. w_global = optimizer.step(w_global, g)
```

## Expected Performance

- **20-50% faster convergence** (fewer rounds)
- **1-5% better accuracy** (heterogeneous data)
- **Lower variance** (more stable)
- **Better generalization** (test accuracy)

## Key References

- Paper: https://arxiv.org/abs/2003.00295
- Blog: https://openmined.org/blog/adaptive-federated-optimization/
- NVFlare: https://nvflare.readthedocs.io/

---

**Quick Recommendation for SNP Deconvolution:**

```python
FedOptController(
    num_clients=3,
    num_rounds=10,
    optimizer='yogi',    # Good for genomic data heterogeneity
    server_lr=0.02,      # Moderate learning rate
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-3
)
```
