# Quick Start: FL Experiment Visualizations

## Quick Commands

### 1. Create Example Data
```bash
python scripts/example_create_visualization_data.py --output_dir my_results
```

### 2. Generate Visualizations
```bash
python scripts/visualize_fl_experiments.py --results_dir my_results --output_dir figures
```

### 3. Generate Paper-Quality PDFs
```bash
python scripts/visualize_fl_experiments.py \
    --results_dir my_results \
    --output_dir paper/figures \
    --format pdf \
    --style paper
```

## File Formats

### JSON Format (`strategy_split.json`)
```json
{
  "rounds": [1, 2, 3, ...],
  "global_accuracy": [0.5, 0.6, 0.7, ...],
  "site_accuracies": {
    "site_0": [0.48, 0.58, ...],
    "site_1": [0.52, 0.62, ...]
  }
}
```

### CSV Format (`strategy_split.csv`)
```csv
round,accuracy,site_0,site_1,site_2
1,0.50,0.48,0.52,0.49
2,0.60,0.58,0.62,0.59
```

## Naming Convention

Files should follow this pattern:
- `{strategy}_{split_type}.{json|csv}`

Examples:
- `fedavg_iid.json`
- `fedprox_dirichlet_0.5.csv`
- `scaffold_label_skew.json`

## Supported Strategies

- FedAvg
- FedProx
- Scaffold
- FedOpt (FedAdam, FedYogi)

## Supported Split Types

- `iid` - Independent and identically distributed
- `dirichlet_0.5` - Dirichlet with α=0.5
- `dirichlet_0.1` - Dirichlet with α=0.1
- `label_skew` - Label distribution skew
- `quantity_skew` - Data quantity skew

## Generated Visualizations

1. **convergence_curves** - Accuracy over rounds
2. **strategy_comparison** - Final accuracy bar chart
3. **accuracy_heatmap** - Strategy vs split performance
4. **site_performance_radar** - Per-site comparison
5. **convergence_speed** - Rounds to target accuracy

## Common Options

| Option | Values | Description |
|--------|--------|-------------|
| `--format` | png, pdf, svg | Output file format |
| `--style` | default, paper, presentation | Visual style |
| `--target_accuracy` | 0.0-1.0 | Target for convergence speed |
| `--verbose` | flag | Detailed logging |

## Examples

### Basic workflow
```bash
# 1. Create test data
python scripts/example_create_visualization_data.py --output_dir test_results

# 2. Visualize
python scripts/visualize_fl_experiments.py --results_dir test_results
```

### Publication workflow
```bash
# High-quality PDFs for paper
python scripts/visualize_fl_experiments.py \
    --results_dir experiments/final_run \
    --output_dir paper/figures \
    --format pdf \
    --style paper \
    --verbose
```

### Presentation workflow
```bash
# Large fonts for slides
python scripts/visualize_fl_experiments.py \
    --results_dir results/ \
    --output_dir slides/images \
    --format png \
    --style presentation
```

## Troubleshooting

### No data loaded
- Check directory exists: `ls -la results_dir`
- Verify file naming matches pattern
- Use `--verbose` for detailed logs

### Missing plots
- Check log messages for errors
- Verify data format is correct
- Ensure all required fields present

### Poor quality
- Use `--style paper` for publications
- Use `--format pdf` for vector graphics
- Increase DPI in style config if needed

## Tips

1. **Consistent naming** - Use standard patterns for automatic detection
2. **Complete data** - Include all rounds and sites for best results
3. **Multiple formats** - Generate both PNG (preview) and PDF (paper)
4. **Batch processing** - Organize experiments in subdirectories
5. **Version control** - Save raw results and generated figures separately

## Quick Test

```bash
# Full test in 3 commands
mkdir -p test_exp
python scripts/example_create_visualization_data.py --output_dir test_exp
python scripts/visualize_fl_experiments.py --results_dir test_exp --verbose
# Check: ls figures/
```

## Next Steps

- See `README_visualize_fl_experiments.md` for detailed documentation
- Adapt `example_create_visualization_data.py` for your experiment format
- Customize colors and styles in `visualize_fl_experiments.py`
