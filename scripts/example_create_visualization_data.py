#!/usr/bin/env python3
"""
Example script showing how to create experiment result files for visualization.

This demonstrates how to format your federated learning experiment results
so they can be visualized using visualize_fl_experiments.py.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def create_json_results(output_dir: Path, strategy: str, split_type: str, num_rounds: int = 50, num_sites: int = 4):
    """
    Create a JSON results file for a single experiment.

    Args:
        output_dir: Directory to save results
        strategy: Strategy name (e.g., 'fedavg', 'fedprox')
        split_type: Data split type (e.g., 'iid', 'dirichlet_0.5')
        num_rounds: Number of communication rounds
        num_sites: Number of federated sites
    """
    # Generate synthetic data (replace with actual experiment results)
    rounds = list(range(1, num_rounds + 1))

    # Simulate accuracy improvement
    base_acc = 0.5
    final_acc = 0.85 if split_type == 'iid' else 0.75
    noise_level = 0.02

    global_accuracy = []
    for r in rounds:
        acc = final_acc * (1 - np.exp(-0.02 * r))
        acc += np.random.normal(0, noise_level)
        acc = np.clip(acc, 0.4, 1.0)
        global_accuracy.append(float(acc))

    # Generate site-specific accuracies
    site_accuracies = {}
    for site_id in range(num_sites):
        site_offset = np.random.normal(0, 0.03)
        site_accuracies[f'site_{site_id}'] = [
            float(np.clip(acc + site_offset + np.random.normal(0, noise_level * 0.5), 0.3, 1.0))
            for acc in global_accuracy
        ]

    # Calculate statistics
    final_site_accs = [site_accuracies[f'site_{i}'][-1] for i in range(num_sites)]

    result = {
        'rounds': rounds,
        'global_accuracy': global_accuracy,
        'site_accuracies': site_accuracies,
        'final_accuracy': float(global_accuracy[-1]),
        'std': float(np.std(final_site_accs)),
        'metadata': {
            'strategy': strategy,
            'split_type': split_type,
            'num_rounds': num_rounds,
            'num_sites': num_sites,
        }
    }

    # Save to JSON file
    filename = f'{strategy}_{split_type}.json'
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Created: {filepath}")
    return filepath


def create_csv_results(output_dir: Path, strategy: str, split_type: str, num_rounds: int = 50, num_sites: int = 4):
    """
    Create a CSV results file for a single experiment.

    Args:
        output_dir: Directory to save results
        strategy: Strategy name (e.g., 'fedavg', 'fedprox')
        split_type: Data split type (e.g., 'iid', 'dirichlet_0.5')
        num_rounds: Number of communication rounds
        num_sites: Number of federated sites
    """
    # Generate synthetic data
    rounds = list(range(1, num_rounds + 1))

    # Simulate accuracy improvement
    final_acc = 0.85 if split_type == 'iid' else 0.75
    noise_level = 0.02

    data = {'round': rounds}

    # Global accuracy
    global_accuracy = []
    for r in rounds:
        acc = final_acc * (1 - np.exp(-0.02 * r))
        acc += np.random.normal(0, noise_level)
        acc = np.clip(acc, 0.4, 1.0)
        global_accuracy.append(float(acc))

    data['accuracy'] = global_accuracy

    # Site-specific accuracies
    for site_id in range(num_sites):
        site_offset = np.random.normal(0, 0.03)
        data[f'site_{site_id}'] = [
            float(np.clip(acc + site_offset + np.random.normal(0, noise_level * 0.5), 0.3, 1.0))
            for acc in global_accuracy
        ]

    # Create DataFrame and save
    df = pd.DataFrame(data)

    filename = f'{strategy}_{split_type}.csv'
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)

    print(f"Created: {filepath}")
    return filepath


def create_example_results(output_dir: str = 'example_results'):
    """
    Create a complete set of example results for multiple strategies and splits.

    Args:
        output_dir: Directory to save example results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    strategies = ['fedavg', 'fedprox', 'scaffold', 'fedopt']
    split_types = ['iid', 'dirichlet_0.5', 'dirichlet_0.1', 'label_skew']

    print("Creating example experiment results...")
    print("=" * 80)

    # Create JSON files for half, CSV for the other half
    for i, strategy in enumerate(strategies):
        for j, split_type in enumerate(split_types):
            # Alternate between JSON and CSV
            if (i + j) % 2 == 0:
                create_json_results(output_path, strategy, split_type)
            else:
                create_csv_results(output_path, strategy, split_type)

    print("=" * 80)
    print(f"Created example results in: {output_path.absolute()}")
    print("\nTo visualize these results, run:")
    print(f"  python scripts/visualize_fl_experiments.py --results_dir {output_dir}")


def load_actual_experiment_results(experiment_log_file: str, output_dir: str = 'experiment_results'):
    """
    Example function showing how to convert actual NVFlare experiment logs to visualization format.

    Args:
        experiment_log_file: Path to experiment log file
        output_dir: Directory to save formatted results

    Note: This is a template - adapt based on your actual log format
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Example: Parse your experiment logs
    # This is pseudocode - adjust based on your actual log format

    """
    # If your logs are in JSON format:
    with open(experiment_log_file, 'r') as f:
        logs = json.load(f)

    # Extract relevant information
    strategy = logs.get('strategy', 'fedavg')
    split_type = logs.get('split_type', 'iid')

    # Parse round-by-round results
    rounds = []
    global_accuracy = []
    site_accuracies = {f'site_{i}': [] for i in range(logs['num_sites'])}

    for round_data in logs['rounds']:
        rounds.append(round_data['round_num'])
        global_accuracy.append(round_data['global_accuracy'])

        for site_id, site_acc in round_data['site_accuracies'].items():
            site_accuracies[site_id].append(site_acc)

    # Create result dictionary
    result = {
        'rounds': rounds,
        'global_accuracy': global_accuracy,
        'site_accuracies': site_accuracies,
        'final_accuracy': global_accuracy[-1],
        'std': np.std([site_accuracies[s][-1] for s in site_accuracies]),
    }

    # Save to JSON
    filename = f'{strategy}_{split_type}.json'
    filepath = output_path / filename
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Converted experiment results to: {filepath}")
    """

    print("This is a template function - adapt to your actual log format")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Create example visualization data')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='example_results',
        help='Directory to save example results'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'both'],
        default='both',
        help='Output format'
    )

    args = parser.parse_args()

    # Create example results
    create_example_results(args.output_dir)


if __name__ == '__main__':
    main()
