#!/usr/bin/env python3
"""
Federated Learning Experiment Results Visualization Script

This script generates comprehensive visualizations for federated learning experiments,
including convergence curves, strategy comparisons, heatmaps, and radar charts.

Usage:
    python visualize_fl_experiments.py --results_dir experiments/results --output_dir figures/
    python visualize_fl_experiments.py --results_dir results/ --format pdf --style paper
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.patches import Circle
from matplotlib.projections import polar

# Try to import seaborn for better styling
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Plotting style configurations
STYLE_CONFIGS = {
    'default': {
        'figure.figsize': (10, 6),
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 100,
    },
    'paper': {
        'figure.figsize': (8, 5),
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'lines.linewidth': 2,
    },
    'presentation': {
        'figure.figsize': (12, 7),
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 150,
        'lines.linewidth': 2.5,
    }
}

# Color palettes for different strategies
STRATEGY_COLORS = {
    'FedAvg': '#1f77b4',
    'FedProx': '#ff7f0e',
    'Scaffold': '#2ca02c',
    'FedOpt': '#d62728',
    'FedAdam': '#9467bd',
    'FedYogi': '#8c564b',
}

# Data split types
SPLIT_TYPES = ['IID', 'Dirichlet_0.5', 'Dirichlet_0.1', 'Label_Skew']


class ExperimentDataLoader:
    """Load and parse experiment results from various formats."""

    def __init__(self, results_dir: Path):
        """
        Initialize the data loader.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.data: Dict[str, Any] = {}

    def load_results(self) -> Dict[str, Any]:
        """
        Load all experiment results from the directory.

        Returns:
            Dictionary containing parsed experiment data

        Expected structure:
        {
            'strategy_name': {
                'split_type': {
                    'rounds': [1, 2, 3, ...],
                    'global_accuracy': [0.5, 0.6, 0.7, ...],
                    'site_accuracies': {
                        'site_1': [0.48, 0.58, 0.68, ...],
                        'site_2': [0.52, 0.62, 0.72, ...],
                    },
                    'final_accuracy': 0.85,
                    'std': 0.02,
                }
            }
        }
        """
        # Try loading JSON files
        json_files = list(self.results_dir.glob('**/*.json'))
        if json_files:
            logger.info(f"Found {len(json_files)} JSON files")
            self.data = self._load_json_files(json_files)

        # Try loading CSV files
        csv_files = list(self.results_dir.glob('**/*.csv'))
        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files")
            csv_data = self._load_csv_files(csv_files)
            # Merge with existing data
            self.data = self._merge_data(self.data, csv_data)

        if not self.data:
            logger.warning("No data loaded. Creating synthetic data for demonstration.")
            self.data = self._generate_synthetic_data()

        return self.data

    def _load_json_files(self, json_files: List[Path]) -> Dict[str, Any]:
        """Load data from JSON files."""
        data = {}
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    file_data = json.load(f)

                # Parse filename to extract strategy and split type
                strategy, split_type = self._parse_filename(json_file.stem)

                if strategy not in data:
                    data[strategy] = {}

                data[strategy][split_type] = file_data
                logger.info(f"Loaded {json_file.name}: {strategy} - {split_type}")

            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        return data

    def _load_csv_files(self, csv_files: List[Path]) -> Dict[str, Any]:
        """Load data from CSV files."""
        data = {}
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)

                # Parse filename to extract strategy and split type
                strategy, split_type = self._parse_filename(csv_file.stem)

                if strategy not in data:
                    data[strategy] = {}

                # Convert DataFrame to dictionary format
                data[strategy][split_type] = self._dataframe_to_dict(df)
                logger.info(f"Loaded {csv_file.name}: {strategy} - {split_type}")

            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")

        return data

    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Parse filename to extract strategy and split type.

        Expected formats:
        - fedavg_iid.json
        - fedprox_dirichlet_0.5.json
        - scaffold_label_skew.csv
        """
        parts = filename.lower().replace('.', '_').split('_')

        # Extract strategy
        strategy = 'FedAvg'  # default
        if 'fedavg' in filename.lower():
            strategy = 'FedAvg'
        elif 'fedprox' in filename.lower():
            strategy = 'FedProx'
        elif 'scaffold' in filename.lower():
            strategy = 'Scaffold'
        elif 'fedopt' in filename.lower() or 'fedadam' in filename.lower():
            strategy = 'FedOpt'
        elif 'fedyogi' in filename.lower():
            strategy = 'FedYogi'

        # Extract split type
        split_type = 'IID'  # default
        if 'dirichlet' in filename.lower():
            if '0_1' in filename or '01' in filename:
                split_type = 'Dirichlet_0.1'
            elif '0_5' in filename or '05' in filename:
                split_type = 'Dirichlet_0.5'
            else:
                split_type = 'Dirichlet_0.5'
        elif 'label' in filename.lower() and 'skew' in filename.lower():
            split_type = 'Label_Skew'
        elif 'quantity' in filename.lower() and 'skew' in filename.lower():
            split_type = 'Quantity_Skew'
        elif 'iid' in filename.lower():
            split_type = 'IID'

        return strategy, split_type

    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Convert DataFrame to dictionary format."""
        result = {
            'rounds': df['round'].tolist() if 'round' in df.columns else list(range(len(df))),
            'global_accuracy': df['accuracy'].tolist() if 'accuracy' in df.columns else df['global_accuracy'].tolist(),
        }

        # Extract site-specific accuracies
        site_cols = [col for col in df.columns if col.startswith('site_')]
        if site_cols:
            result['site_accuracies'] = {}
            for col in site_cols:
                result['site_accuracies'][col] = df[col].tolist()

        # Calculate final accuracy and std
        result['final_accuracy'] = result['global_accuracy'][-1]
        result['std'] = np.std(result['global_accuracy'][-5:]) if len(result['global_accuracy']) >= 5 else 0.01

        return result

    def _merge_data(self, data1: Dict, data2: Dict) -> Dict:
        """Merge two data dictionaries."""
        result = data1.copy()
        for strategy, splits in data2.items():
            if strategy not in result:
                result[strategy] = {}
            result[strategy].update(splits)
        return result

    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """Generate synthetic data for demonstration purposes."""
        logger.info("Generating synthetic experiment data")

        strategies = ['FedAvg', 'FedProx', 'Scaffold', 'FedOpt']
        split_types = ['IID', 'Dirichlet_0.5', 'Dirichlet_0.1', 'Label_Skew']
        num_rounds = 50
        num_sites = 4

        data = {}

        for strategy in strategies:
            data[strategy] = {}

            # Strategy-specific characteristics
            base_convergence_rate = {
                'FedAvg': 0.015,
                'FedProx': 0.018,
                'Scaffold': 0.020,
                'FedOpt': 0.022,
            }[strategy]

            for split_type in split_types:
                # Split-specific characteristics
                if split_type == 'IID':
                    final_acc = 0.85 + np.random.normal(0, 0.02)
                    noise_level = 0.01
                elif 'Dirichlet_0.5' in split_type:
                    final_acc = 0.82 + np.random.normal(0, 0.02)
                    noise_level = 0.015
                elif 'Dirichlet_0.1' in split_type:
                    final_acc = 0.78 + np.random.normal(0, 0.02)
                    noise_level = 0.02
                else:  # Label_Skew
                    final_acc = 0.75 + np.random.normal(0, 0.02)
                    noise_level = 0.025

                # Generate convergence curve
                rounds = list(range(1, num_rounds + 1))
                global_accuracy = []

                for r in rounds:
                    # Exponential convergence with noise
                    acc = final_acc * (1 - np.exp(-base_convergence_rate * r))
                    acc += np.random.normal(0, noise_level)
                    acc = np.clip(acc, 0.4, 1.0)
                    global_accuracy.append(acc)

                # Generate site-specific accuracies
                site_accuracies = {}
                for site_id in range(num_sites):
                    site_offset = np.random.normal(0, 0.03)
                    site_accuracies[f'site_{site_id}'] = [
                        np.clip(acc + site_offset + np.random.normal(0, noise_level * 0.5), 0.3, 1.0)
                        for acc in global_accuracy
                    ]

                data[strategy][split_type] = {
                    'rounds': rounds,
                    'global_accuracy': global_accuracy,
                    'site_accuracies': site_accuracies,
                    'final_accuracy': global_accuracy[-1],
                    'std': np.std([site_accuracies[s][-1] for s in site_accuracies]),
                }

        return data


class FLVisualizer:
    """Generate visualizations for federated learning experiments."""

    def __init__(
        self,
        data: Dict[str, Any],
        output_dir: Path,
        output_format: str = 'png',
        style: str = 'default'
    ):
        """
        Initialize the visualizer.

        Args:
            data: Experiment data dictionary
            output_dir: Directory to save figures
            output_format: Output format (png, pdf, svg)
            style: Plot style (default, paper, presentation)
        """
        self.data = data
        self.output_dir = output_dir
        self.output_format = output_format
        self.style = style

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Apply style
        self._apply_style()

    def _apply_style(self):
        """Apply plotting style configuration."""
        if self.style in STYLE_CONFIGS:
            rcParams.update(STYLE_CONFIGS[self.style])

        # Set seaborn style if available
        if HAS_SEABORN:
            if self.style == 'paper':
                sns.set_style("whitegrid")
            elif self.style == 'presentation':
                sns.set_style("darkgrid")
            else:
                sns.set_style("whitegrid")

    def generate_all_visualizations(self):
        """Generate all visualization types."""
        logger.info("Generating all visualizations...")

        try:
            self.plot_convergence_curves()
        except Exception as e:
            logger.error(f"Error generating convergence curves: {e}")

        try:
            self.plot_strategy_comparison()
        except Exception as e:
            logger.error(f"Error generating strategy comparison: {e}")

        try:
            self.plot_accuracy_heatmap()
        except Exception as e:
            logger.error(f"Error generating accuracy heatmap: {e}")

        try:
            self.plot_site_performance_radar()
        except Exception as e:
            logger.error(f"Error generating site performance radar: {e}")

        try:
            self.plot_convergence_speed()
        except Exception as e:
            logger.error(f"Error generating convergence speed: {e}")

        logger.info(f"All visualizations saved to {self.output_dir}")

    def plot_convergence_curves(self):
        """
        Plot convergence curves showing global accuracy over communication rounds.
        Creates subplots for different data splits.
        """
        logger.info("Generating convergence curves...")

        # Get all split types from data
        split_types = set()
        for strategy_data in self.data.values():
            split_types.update(strategy_data.keys())
        split_types = sorted(list(split_types))

        # Create subplots
        n_splits = len(split_types)
        n_cols = 2
        n_rows = (n_splits + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, split_type in enumerate(split_types):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Plot each strategy
            for strategy in sorted(self.data.keys()):
                if split_type not in self.data[strategy]:
                    continue

                split_data = self.data[strategy][split_type]
                rounds = split_data['rounds']
                accuracy = split_data['global_accuracy']

                color = STRATEGY_COLORS.get(strategy, None)
                ax.plot(rounds, accuracy, label=strategy, linewidth=2, color=color, alpha=0.8)

            ax.set_xlabel('Communication Rounds')
            ax.set_ylabel('Global Model Accuracy')
            ax.set_title(f'Convergence: {split_type.replace("_", " ")}')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.4, 1.0])

        # Hide empty subplots
        for idx in range(n_splits, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        output_path = self.output_dir / f'convergence_curves.{self.output_format}'
        plt.savefig(output_path, dpi=rcParams['figure.dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"Saved convergence curves to {output_path}")

    def plot_strategy_comparison(self):
        """
        Plot bar chart comparing final accuracies of different strategies.
        Groups by data split type with error bars.
        """
        logger.info("Generating strategy comparison bar chart...")

        # Prepare data for plotting
        strategies = sorted(self.data.keys())
        split_types = set()
        for strategy_data in self.data.values():
            split_types.update(strategy_data.keys())
        split_types = sorted(list(split_types))

        # Create grouped bar chart
        x = np.arange(len(split_types))
        width = 0.8 / len(strategies)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, strategy in enumerate(strategies):
            accuracies = []
            errors = []

            for split_type in split_types:
                if split_type in self.data[strategy]:
                    split_data = self.data[strategy][split_type]
                    accuracies.append(split_data['final_accuracy'])
                    errors.append(split_data.get('std', 0.01))
                else:
                    accuracies.append(0)
                    errors.append(0)

            offset = (i - len(strategies) / 2) * width + width / 2
            color = STRATEGY_COLORS.get(strategy, None)
            ax.bar(x + offset, accuracies, width, label=strategy,
                   yerr=errors, capsize=5, color=color, alpha=0.8)

        ax.set_xlabel('Data Split Type')
        ax.set_ylabel('Final Accuracy')
        ax.set_title('Strategy Comparison: Final Accuracy by Data Split')
        ax.set_xticks(x)
        ax.set_xticklabels([st.replace('_', ' ') for st in split_types], rotation=15, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0.6, 1.0])

        plt.tight_layout()
        output_path = self.output_dir / f'strategy_comparison.{self.output_format}'
        plt.savefig(output_path, dpi=rcParams['figure.dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"Saved strategy comparison to {output_path}")

    def plot_accuracy_heatmap(self):
        """
        Plot heatmap showing final accuracy for each strategy-split combination.
        """
        logger.info("Generating accuracy heatmap...")

        # Prepare data matrix
        strategies = sorted(self.data.keys())
        split_types = set()
        for strategy_data in self.data.values():
            split_types.update(strategy_data.keys())
        split_types = sorted(list(split_types))

        # Create accuracy matrix
        accuracy_matrix = np.zeros((len(strategies), len(split_types)))

        for i, strategy in enumerate(strategies):
            for j, split_type in enumerate(split_types):
                if split_type in self.data[strategy]:
                    accuracy_matrix[i, j] = self.data[strategy][split_type]['final_accuracy']
                else:
                    accuracy_matrix[i, j] = np.nan

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0.65, vmax=0.90)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(split_types)))
        ax.set_yticks(np.arange(len(strategies)))
        ax.set_xticklabels([st.replace('_', '\n') for st in split_types])
        ax.set_yticklabels(strategies)

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Final Accuracy', rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(split_types)):
                if not np.isnan(accuracy_matrix[i, j]):
                    text = ax.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontsize=9)

        ax.set_title('Final Accuracy Heatmap: Strategies vs Data Splits')
        ax.set_xlabel('Data Split Type')
        ax.set_ylabel('Strategy')

        plt.tight_layout()
        output_path = self.output_dir / f'accuracy_heatmap.{self.output_format}'
        plt.savefig(output_path, dpi=rcParams['figure.dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"Saved accuracy heatmap to {output_path}")

    def plot_site_performance_radar(self):
        """
        Plot radar charts showing per-site performance under different strategies.
        """
        logger.info("Generating site performance radar charts...")

        # Select one split type for radar chart (preferably IID or most complete)
        split_type = 'IID'
        available_splits = set()
        for strategy_data in self.data.values():
            available_splits.update(strategy_data.keys())

        if split_type not in available_splits:
            split_type = sorted(list(available_splits))[0]

        logger.info(f"Using split type: {split_type} for radar chart")

        # Get site names from first strategy that has this split
        site_names = []
        for strategy in self.data.keys():
            if split_type in self.data[strategy]:
                site_accuracies = self.data[strategy][split_type].get('site_accuracies', {})
                if site_accuracies:
                    site_names = sorted(list(site_accuracies.keys()))
                    break

        if not site_names:
            logger.warning("No site-specific data found for radar chart")
            return

        # Create radar chart for each site
        n_sites = len(site_names)
        n_cols = 2
        n_rows = (n_sites + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(12, 6 * n_rows))

        for idx, site_name in enumerate(site_names):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='polar')

            # Collect final accuracies for this site across strategies
            strategies = []
            accuracies = []

            for strategy in sorted(self.data.keys()):
                if split_type in self.data[strategy]:
                    site_accs = self.data[strategy][split_type].get('site_accuracies', {})
                    if site_name in site_accs:
                        strategies.append(strategy)
                        accuracies.append(site_accs[site_name][-1])

            if not strategies:
                continue

            # Compute angles for radar chart
            angles = np.linspace(0, 2 * np.pi, len(strategies), endpoint=False).tolist()
            accuracies = accuracies + [accuracies[0]]  # Close the plot
            angles = angles + [angles[0]]

            # Plot
            ax.plot(angles, accuracies, 'o-', linewidth=2, label=site_name)
            ax.fill(angles, accuracies, alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(strategies)
            ax.set_ylim(0.5, 1.0)
            ax.set_title(f'{site_name.replace("_", " ").title()}', pad=20)
            ax.grid(True)

        plt.suptitle(f'Site Performance Radar Chart ({split_type.replace("_", " ")})',
                     fontsize=14, y=1.02)
        plt.tight_layout()
        output_path = self.output_dir / f'site_performance_radar.{self.output_format}'
        plt.savefig(output_path, dpi=rcParams['figure.dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"Saved site performance radar to {output_path}")

    def plot_convergence_speed(self, target_accuracy: float = 0.75):
        """
        Plot convergence speed comparison showing rounds needed to reach target accuracy.

        Args:
            target_accuracy: Target accuracy threshold
        """
        logger.info(f"Generating convergence speed comparison (target: {target_accuracy})...")

        # Prepare data
        strategies = sorted(self.data.keys())
        split_types = set()
        for strategy_data in self.data.values():
            split_types.update(strategy_data.keys())
        split_types = sorted(list(split_types))

        # Calculate rounds to reach target accuracy
        rounds_to_target = {strategy: {} for strategy in strategies}

        for strategy in strategies:
            for split_type in split_types:
                if split_type not in self.data[strategy]:
                    rounds_to_target[strategy][split_type] = None
                    continue

                split_data = self.data[strategy][split_type]
                rounds = split_data['rounds']
                accuracies = split_data['global_accuracy']

                # Find first round where accuracy exceeds target
                target_round = None
                for r, acc in zip(rounds, accuracies):
                    if acc >= target_accuracy:
                        target_round = r
                        break

                rounds_to_target[strategy][split_type] = target_round

        # Create grouped bar chart
        x = np.arange(len(split_types))
        width = 0.8 / len(strategies)

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, strategy in enumerate(strategies):
            rounds_list = []

            for split_type in split_types:
                rounds = rounds_to_target[strategy].get(split_type)
                if rounds is None:
                    rounds_list.append(0)
                else:
                    rounds_list.append(rounds)

            offset = (i - len(strategies) / 2) * width + width / 2
            color = STRATEGY_COLORS.get(strategy, None)
            bars = ax.bar(x + offset, rounds_list, width, label=strategy,
                          color=color, alpha=0.8)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Data Split Type')
        ax.set_ylabel('Rounds to Reach Target Accuracy')
        ax.set_title(f'Convergence Speed Comparison (Target: {target_accuracy:.1%})')
        ax.set_xticks(x)
        ax.set_xticklabels([st.replace('_', ' ') for st in split_types], rotation=15, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / f'convergence_speed.{self.output_format}'
        plt.savefig(output_path, dpi=rcParams['figure.dpi'], bbox_inches='tight')
        plt.close()
        logger.info(f"Saved convergence speed comparison to {output_path}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize federated learning experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python visualize_fl_experiments.py --results_dir experiments/results

  # Generate paper-quality figures in PDF format
  python visualize_fl_experiments.py --results_dir results/ --format pdf --style paper

  # Generate presentation slides with larger fonts
  python visualize_fl_experiments.py --results_dir results/ --format png --style presentation

  # Specify custom output directory
  python visualize_fl_experiments.py --results_dir results/ --output_dir figures/my_experiment
"""
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing experiment results (JSON/CSV files)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures',
        help='Directory to save generated figures (default: figures/)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for figures (default: png)'
    )

    parser.add_argument(
        '--style',
        type=str,
        choices=['default', 'paper', 'presentation'],
        default='default',
        help='Plot style preset (default: default)'
    )

    parser.add_argument(
        '--target_accuracy',
        type=float,
        default=0.75,
        help='Target accuracy for convergence speed analysis (default: 0.75)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed arguments

    Raises:
        ValueError: If validation fails
    """
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    if not results_dir.is_dir():
        raise ValueError(f"Results path is not a directory: {results_dir}")

    if not 0 < args.target_accuracy < 1:
        raise ValueError(f"Target accuracy must be between 0 and 1: {args.target_accuracy}")


def main():
    """Main execution function."""
    # Parse and validate arguments
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        validate_arguments(args)
    except ValueError as e:
        logger.error(f"Argument validation failed: {e}")
        sys.exit(1)

    # Convert paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 80)
    logger.info("Federated Learning Experiment Visualization")
    logger.info("=" * 80)
    logger.info(f"Results directory: {results_dir.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info(f"Output format: {args.format}")
    logger.info(f"Plot style: {args.style}")
    logger.info(f"Target accuracy: {args.target_accuracy:.2%}")
    logger.info("=" * 80)

    # Load experiment results
    loader = ExperimentDataLoader(results_dir)
    data = loader.load_results()

    if not data:
        logger.error("No experiment data loaded. Exiting.")
        sys.exit(1)

    # Log data summary
    logger.info(f"Loaded data for {len(data)} strategies:")
    for strategy, splits in data.items():
        logger.info(f"  - {strategy}: {len(splits)} split types")

    # Create visualizer and generate plots
    visualizer = FLVisualizer(
        data=data,
        output_dir=output_dir,
        output_format=args.format,
        style=args.style
    )

    visualizer.generate_all_visualizations()

    logger.info("=" * 80)
    logger.info("Visualization complete!")
    logger.info(f"Figures saved to: {output_dir.absolute()}")
    logger.info("=" * 80)

    # List generated files
    generated_files = sorted(output_dir.glob(f'*.{args.format}'))
    if generated_files:
        logger.info("Generated files:")
        for file in generated_files:
            logger.info(f"  - {file.name}")


if __name__ == '__main__':
    main()
