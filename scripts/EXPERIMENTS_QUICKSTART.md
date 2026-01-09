# Quick Start Guide: Automated FL Experiments

## 1-Minute Quick Start

```bash
# Navigate to project root
cd /Users/saltfish/Files/Coding/Med_SNP_Deconvolution

# Dry run to see what would happen
python scripts/run_fl_experiments.py --dry_run

# Run 2 quick experiments (IID + non-IID with FedAvg)
python scripts/run_fl_experiments.py \
    --strategies fedavg \
    --splits iid,dirichlet_0.5 \
    --num_rounds 10

# View results
cat results/fl_experiments/experiment_summary.csv
```

## Common Use Cases

### Use Case 1: Algorithm Comparison

**Goal**: Compare FedAvg, FedProx, and Scaffold on non-IID data

```bash
python scripts/run_fl_experiments.py \
    --strategies fedavg,fedprox,scaffold \
    --splits dirichlet_0.5 \
    --num_rounds 50
```

**Expected Output**: 3 experiments comparing different algorithms

### Use Case 2: Data Heterogeneity Study

**Goal**: Test how FedProx handles different levels of heterogeneity

```bash
python scripts/run_fl_experiments.py \
    --strategies fedprox \
    --splits iid,dirichlet_0.7,dirichlet_0.5,dirichlet_0.3,dirichlet_0.1 \
    --num_rounds 50
```

**Expected Output**: 5 experiments showing degradation with increasing heterogeneity

### Use Case 3: Quick Validation

**Goal**: Quickly test if everything works

```bash
python scripts/run_fl_experiments.py \
    --strategies fedavg \
    --splits iid \
    --num_rounds 5
```

**Expected Output**: 1 experiment completing in ~5-10 minutes

### Use Case 4: Full Benchmark

**Goal**: Run comprehensive benchmark of all configurations

```bash
python scripts/run_fl_experiments.py \
    --strategies all \
    --splits all \
    --num_rounds 100
```

**Expected Output**: 20+ experiments (may take hours)

## Understanding Results

### Check Experiment Status

```bash
# View summary table
column -t -s, results/fl_experiments/experiment_summary.csv | less -S

# Count successful experiments
jq '.successful_experiments' results/fl_experiments/experiment_report.json

# List failed experiments
jq '.experiments.failed[].config.name' results/fl_experiments/failed_experiments.json
```

### Find Best Configuration

```bash
# Best accuracy per split type
python -c "
import pandas as pd
df = pd.read_csv('results/fl_experiments/experiment_summary.csv')
if 'val_accuracy' in df.columns:
    best = df.groupby('split_type')['val_accuracy'].max()
    print('Best accuracy per split:')
    print(best)
"
```

### Check Execution Time

```bash
# Total time for all experiments
python -c "
import json
with open('results/fl_experiments/experiment_results.json') as f:
    results = json.load(f)
total = sum(r['duration_seconds'] for r in results)
print(f'Total execution time: {total/3600:.2f} hours')
"
```

## Experiment Matrix Reference

| Split Type | Description | Alpha | Non-IID Level |
|------------|-------------|-------|---------------|
| `iid` | Balanced distribution | N/A | None |
| `dirichlet_0.7` | Mild heterogeneity | 0.7 | Low |
| `dirichlet_0.5` | Moderate heterogeneity | 0.5 | Medium |
| `dirichlet_0.3` | Strong heterogeneity | 0.3 | High |
| `dirichlet_0.1` | Extreme heterogeneity | 0.1 | Very High |
| `label_skew` | Label partitioning | N/A | Extreme |

| Strategy | Best For | Overhead |
|----------|----------|----------|
| `fedavg` | Baseline, IID data | Low |
| `fedprox` | Moderate non-IID | Low |
| `scaffold` | Strong non-IID | Medium |
| `fedopt_adam` | Fast convergence | Low |
| `fedopt_yogi` | Stable convergence | Low |

## Typical Experiment Runtimes

Assuming 3 sites, 50 rounds, ~1000 samples per site:

- **Single experiment**: 10-20 minutes
- **3 experiments**: 30-60 minutes
- **10 experiments**: 2-4 hours
- **Full matrix (20+)**: 4-8 hours

GPU significantly reduces runtime (2-5x speedup).

## Disk Space Requirements

Per experiment:
- **Data**: ~50-100 MB (federated splits)
- **Workspace**: ~200-500 MB (models + logs)
- **Results**: ~1-5 MB (metrics)

For 20 experiments: ~5-10 GB total

## Troubleshooting Quick Fixes

### "Data preparation failed"

```bash
# Check pipeline output exists
ls out_dir/TNFa/

# Check population files exist
ls data/igsr-*.tsv
```

### "NVFlare simulation failed"

```bash
# Check workspace logs
find results/fl_experiments -name "log.txt" -exec tail -n 20 {} +

# Try running one experiment manually
python scripts/run_fl_experiments.py \
    --strategies fedavg \
    --splits iid \
    --num_rounds 5 \
    --verbose
```

### "Out of memory"

```bash
# Reduce batch size or sites
python scripts/run_fl_experiments.py \
    --strategies fedavg \
    --splits iid \
    --num_sites 2  # Reduce from 3 to 2
```

### "Metrics not collected"

```bash
# Check if workspace was created
ls results/fl_experiments/*/workspace/

# Manually inspect logs for metrics
grep -r "accuracy\|loss" results/fl_experiments/*/workspace/
```

## Next Steps

1. **Run dry run** to verify configuration
2. **Start with 1-2 quick experiments** (5-10 rounds)
3. **Check results format** and troubleshoot
4. **Scale up to full experiments** (50-100 rounds)
5. **Analyze and visualize** results

## Pro Tips

1. Always start with `--dry_run`
2. Test with `--num_rounds 5` first
3. Use `--verbose` when debugging
4. Monitor disk space during long runs
5. Save experiment configurations in scripts
6. Clean up old workspaces regularly

## Example Analysis Script

```python
#!/usr/bin/env python3
"""Analyze experiment results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('results/fl_experiments/experiment_summary.csv')

# Filter successful experiments
df = df[df['status'] == 'success']

print("Summary Statistics:")
print(df.groupby(['split_type', 'strategy'])['val_accuracy'].describe())

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy by strategy
df.groupby('strategy')['val_accuracy'].mean().plot(kind='bar', ax=axes[0])
axes[0].set_title('Average Accuracy by Strategy')
axes[0].set_ylabel('Validation Accuracy')

# Accuracy by split type
df.groupby('split_type')['val_accuracy'].mean().plot(kind='bar', ax=axes[1])
axes[1].set_title('Average Accuracy by Data Split')
axes[1].set_ylabel('Validation Accuracy')

plt.tight_layout()
plt.savefig('results/fl_experiments/comparison.png')
print("Saved plot to results/fl_experiments/comparison.png")
```

Save this as `scripts/analyze_experiments.py` and run after experiments complete.
