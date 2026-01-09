#!/usr/bin/env python3
"""
Comparison Script: FedAvg vs FedOpt

This script runs both FedAvg and FedOpt experiments side by side to compare
their convergence speed and final performance on the SNP deconvolution task.

The script:
1. Creates two separate jobs (FedAvg and FedOpt)
2. Runs them in POC mode sequentially or in parallel
3. Compares results and generates performance plots

Usage:
    # Quick comparison (5 rounds)
    python compare_fedavg_fedopt.py --num_rounds 5 --quick

    # Full comparison (20 rounds)
    python compare_fedavg_fedopt.py --num_rounds 20

    # Compare multiple FedOpt variants
    python compare_fedavg_fedopt.py --num_rounds 10 --optimizers adam,yogi,sgdm

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare FedAvg vs FedOpt Performance',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Experiment configuration
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='Number of FL rounds')
    parser.add_argument('--num_clients', type=int, default=3,
                        help='Number of clients')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Local epochs per round')

    # FedOpt optimizers to compare
    parser.add_argument('--optimizers', type=str, default='adam',
                        help='Comma-separated list of optimizers (adam,yogi,sgdm,adagrad)')
    parser.add_argument('--server_lr', type=float, default=0.01,
                        help='Server learning rate for FedOpt')

    # Model configuration
    parser.add_argument('--architecture', type=str, default='cnn_transformer',
                        choices=['cnn', 'cnn_transformer', 'gnn'],
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--feature_type', type=str, default='cluster',
                        choices=['cluster', 'snp'],
                        help='Feature type')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Client learning rate')

    # Execution configuration
    parser.add_argument('--workspace_base', type=str, default='./comparison_workspace',
                        help='Base workspace directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: reduced rounds for testing')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots after experiments')

    # Data configuration
    default_data_dir = str(project_root / 'data' / 'federated')
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='Data directory')

    return parser.parse_args()


def create_fedavg_job(args, workspace: Path):
    """Create FedAvg baseline job."""
    try:
        from nvflare.job_config.api import FedJob
        from nvflare.app_common.workflows.fedavg import FedAvg
        from nvflare.job_config.script_runner import ScriptRunner
    except ImportError as e:
        logger.error(f"NVFlare not installed: {e}")
        raise

    logger.info("="*70)
    logger.info("Creating FedAvg Baseline Job")
    logger.info("="*70)

    job = FedJob(name='fedavg_baseline', min_clients=args.num_clients)

    # Add FedAvg controller
    controller = FedAvg(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
    )
    job.to_server(controller)
    logger.info("Added FedAvg controller")

    # Configure client
    client_script = str(Path(__file__).parent.parent / "lightning" / "client.py")
    script_args = (
        f"--data_dir {args.data_dir} "
        f"--feature_type {args.feature_type} "
        f"--architecture {args.architecture} "
        f"--num_classes {args.num_classes} "
        f"--local_epochs {args.local_epochs} "
        f"--batch_size {args.batch_size} "
        f"--learning_rate {args.learning_rate} "
        f"--use_focal_loss "
        f"--precision bf16-mixed"
    )

    runner = ScriptRunner(script=client_script, script_args=script_args)
    job.to_clients(runner)

    return job


def create_fedopt_job(args, optimizer: str, workspace: Path):
    """Create FedOpt job with specified optimizer."""
    try:
        from nvflare.job_config.api import FedJob
        from nvflare.job_config.script_runner import ScriptRunner
        from snp_deconvolution.nvflare_real.controllers import FedOptController
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise

    logger.info("="*70)
    logger.info(f"Creating FedOpt Job with {optimizer.upper()}")
    logger.info("="*70)

    job = FedJob(name=f'fedopt_{optimizer}', min_clients=args.num_clients)

    # Add FedOpt controller
    controller_kwargs = {
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'optimizer': optimizer,
        'server_lr': args.server_lr,
    }

    if optimizer in ['adam', 'yogi']:
        controller_kwargs.update({
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8 if optimizer == 'adam' else 1e-3,
        })
    elif optimizer == 'sgdm':
        controller_kwargs['momentum'] = 0.9
    elif optimizer == 'adagrad':
        controller_kwargs['epsilon'] = 1e-8

    controller = FedOptController(**controller_kwargs)
    job.to_server(controller)
    logger.info(f"Added FedOpt-{optimizer.upper()} controller")

    # Configure client
    client_script = str(Path(__file__).parent.parent / "lightning" / "client.py")
    script_args = (
        f"--data_dir {args.data_dir} "
        f"--feature_type {args.feature_type} "
        f"--architecture {args.architecture} "
        f"--num_classes {args.num_classes} "
        f"--local_epochs {args.local_epochs} "
        f"--batch_size {args.batch_size} "
        f"--learning_rate {args.learning_rate} "
        f"--use_focal_loss "
        f"--precision bf16-mixed"
    )

    runner = ScriptRunner(script=client_script, script_args=script_args)
    job.to_clients(runner)

    return job


def run_experiment(job, workspace: Path, name: str, args):
    """Run a single experiment."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Running Experiment: {name}")
    logger.info(f"{'='*70}")

    workspace.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting job to: {workspace}")
    job.export_job(str(workspace))

    logger.info(f"Starting simulation...")
    try:
        job.simulator_run(
            workspace=str(workspace),
            n_clients=args.num_clients,
            threads=args.num_clients,
        )
        logger.info(f"Experiment {name} completed successfully")
        return True
    except Exception as e:
        logger.error(f"Experiment {name} failed: {e}")
        return False


def extract_results(workspace: Path) -> Dict:
    """Extract results from workspace."""
    results = {
        'rounds': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
    }

    # Try to find and parse results
    # This is a simplified version - actual implementation would parse
    # NVFlare logs and extract metrics
    result_file = workspace / "results.json"
    if result_file.exists():
        with open(result_file) as f:
            results = json.load(f)

    return results


def compare_results(results: Dict[str, Dict]):
    """Compare results from different algorithms."""
    logger.info("\n" + "="*70)
    logger.info("COMPARISON RESULTS")
    logger.info("="*70)

    for name, result in results.items():
        if result.get('val_accuracy'):
            final_acc = result['val_accuracy'][-1] if result['val_accuracy'] else 'N/A'
            logger.info(f"{name:20s}: Final Accuracy = {final_acc}")

    logger.info("="*70)


def generate_plots(results: Dict[str, Dict], output_dir: Path):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed, skipping plots")
        return

    logger.info(f"\nGenerating comparison plots in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot convergence curves
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for name, result in results.items():
        if result.get('rounds') and result.get('val_accuracy'):
            axes[0].plot(result['rounds'], result['val_accuracy'], label=name, marker='o')
            axes[1].plot(result['rounds'], result['val_loss'], label=name, marker='o')

    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Convergence: Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Convergence: Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot: {output_dir / 'comparison.png'}")
    plt.close()


def main():
    """Main entry point."""
    args = parse_args()

    if args.quick:
        args.num_rounds = min(args.num_rounds, 5)
        logger.info("Quick mode: Running with reduced rounds")

    # Parse optimizers
    optimizers = [opt.strip() for opt in args.optimizers.split(',')]

    logger.info("\n" + "="*70)
    logger.info("FEDAVG VS FEDOPT COMPARISON")
    logger.info("="*70)
    logger.info(f"Rounds: {args.num_rounds}")
    logger.info(f"Clients: {args.num_clients}")
    logger.info(f"FedOpt Optimizers: {', '.join(optimizers)}")
    logger.info("="*70)

    # Setup workspace
    workspace_base = Path(args.workspace_base)
    workspace_base.mkdir(parents=True, exist_ok=True)

    results = {}

    # Run FedAvg baseline
    logger.info("\n### RUNNING FEDAVG BASELINE ###")
    fedavg_workspace = workspace_base / "fedavg_baseline"
    fedavg_job = create_fedavg_job(args, fedavg_workspace)
    if run_experiment(fedavg_job, fedavg_workspace, "FedAvg", args):
        results['FedAvg'] = extract_results(fedavg_workspace)

    # Run FedOpt variants
    for optimizer in optimizers:
        logger.info(f"\n### RUNNING FEDOPT-{optimizer.upper()} ###")
        fedopt_workspace = workspace_base / f"fedopt_{optimizer}"
        fedopt_job = create_fedopt_job(args, optimizer, fedopt_workspace)
        if run_experiment(fedopt_job, fedopt_workspace, f"FedOpt-{optimizer}", args):
            results[f'FedOpt-{optimizer}'] = extract_results(fedopt_workspace)

    # Compare results
    if results:
        compare_results(results)

        if args.plot:
            generate_plots(results, workspace_base / "plots")

    logger.info("\n" + "="*70)
    logger.info("COMPARISON COMPLETED")
    logger.info(f"Results saved in: {workspace_base}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
