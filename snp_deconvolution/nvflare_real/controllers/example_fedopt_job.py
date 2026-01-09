#!/usr/bin/env python3
"""
Example: Using FedOpt Controller in NVFlare Job

This script demonstrates how to create a federated learning job using
the FedOpt (Federated Optimization) controller instead of FedAvg.

FedOpt applies server-side adaptive optimization (Adam, AdaGrad, Yogi, SGD)
to aggregate client updates, often leading to faster convergence and better
performance compared to simple averaging (FedAvg).

Usage:
    # Run with FedAdam
    python example_fedopt_job.py --optimizer adam --server_lr 0.01

    # Run with FedYogi
    python example_fedopt_job.py --optimizer yogi --server_lr 0.02

    # Run with SGD + Momentum
    python example_fedopt_job.py --optimizer sgdm --server_lr 0.01 --momentum 0.9

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

import argparse
import logging
import sys
from pathlib import Path

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
        description='NVFlare Job with FedOpt Controller',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basic job configuration
    parser.add_argument('--job_name', type=str, default='snp_fedopt',
                        help='Job name')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='Number of FL rounds')
    parser.add_argument('--num_clients', type=int, default=3,
                        help='Number of clients')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Local epochs per round')

    # FedOpt configuration
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adagrad', 'yogi', 'sgdm'],
                        help='Server optimizer type')
    parser.add_argument('--server_lr', type=float, default=0.01,
                        help='Server learning rate')

    # Adam/Yogi parameters
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='First moment decay (Adam/Yogi)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Second moment decay (Adam/Yogi)')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Numerical stability constant')

    # SGD with momentum parameters
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum coefficient (SGDM)')

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Client learning rate')
    parser.add_argument('--architecture', type=str, default='cnn_transformer',
                        choices=['cnn', 'cnn_transformer', 'gnn'],
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of population classes')
    parser.add_argument('--feature_type', type=str, default='cluster',
                        choices=['cluster', 'snp'],
                        help='Feature type')

    # Execution configuration
    parser.add_argument('--mode', type=str, default='poc',
                        choices=['poc', 'export'],
                        help='Job mode')
    parser.add_argument('--workspace', type=str, default='./nvflare_workspace_fedopt',
                        help='Workspace directory')
    parser.add_argument('--run_now', action='store_true',
                        help='Run job immediately in POC mode')

    # Data configuration
    default_data_dir = str(project_root / 'data' / 'federated')
    parser.add_argument('--data_dir', type=str, default=default_data_dir,
                        help='Data directory')

    return parser.parse_args()


def create_fedopt_job(args):
    """
    Create NVFlare job with FedOpt controller.

    Args:
        args: Command line arguments

    Returns:
        FedJob instance
    """
    try:
        from nvflare.job_config.api import FedJob
        from nvflare.job_config.script_runner import ScriptRunner
        from snp_deconvolution.nvflare_real.controllers import FedOptController
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure NVFlare is installed: pip install nvflare")
        raise

    logger.info("="*70)
    logger.info(f"Creating NVFlare job with FedOpt Controller")
    logger.info("="*70)
    logger.info(f"Job Name: {args.job_name}")
    logger.info(f"Algorithm: FedOpt-{args.optimizer.upper()}")
    logger.info(f"Clients: {args.num_clients}")
    logger.info(f"FL Rounds: {args.num_rounds}")
    logger.info(f"Server LR: {args.server_lr}")

    if args.optimizer in ['adam', 'yogi']:
        logger.info(f"Beta1: {args.beta1}, Beta2: {args.beta2}, Epsilon: {args.epsilon}")
    elif args.optimizer == 'sgdm':
        logger.info(f"Momentum: {args.momentum}")

    logger.info("="*70)

    # Create FedJob
    job = FedJob(
        name=args.job_name,
        min_clients=args.num_clients,
    )

    # Add FedOpt controller to server
    controller_kwargs = {
        'num_clients': args.num_clients,
        'num_rounds': args.num_rounds,
        'optimizer': args.optimizer,
        'server_lr': args.server_lr,
    }

    # Add optimizer-specific parameters
    if args.optimizer in ['adam', 'yogi']:
        controller_kwargs.update({
            'beta1': args.beta1,
            'beta2': args.beta2,
            'epsilon': args.epsilon,
        })
    elif args.optimizer == 'sgdm':
        controller_kwargs['momentum'] = args.momentum
    elif args.optimizer == 'adagrad':
        controller_kwargs['epsilon'] = args.epsilon

    controller = FedOptController(**controller_kwargs)
    job.to_server(controller)

    logger.info(f"Added FedOpt-{args.optimizer.upper()} controller to server")

    # Configure client script
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

    # Add ScriptRunner to all clients
    runner = ScriptRunner(
        script=client_script,
        script_args=script_args,
    )
    job.to_clients(runner)

    logger.info(f"Configured ScriptRunner for {args.num_clients} clients")
    logger.info(f"Job created successfully: {args.job_name}")

    return job


def run_poc_mode(args, job):
    """Run job in POC mode."""
    logger.info("\n" + "="*70)
    logger.info("POC MODE - Local Simulation with FedOpt")
    logger.info("="*70)
    logger.info(f"Workspace: {args.workspace}")
    logger.info(f"Optimizer: {args.optimizer.upper()}")
    logger.info("="*70)

    # Export job
    workspace_path = Path(args.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nExporting job to: {workspace_path}")
    job.export_job(str(workspace_path))
    logger.info("Job exported successfully")

    # Run simulation if requested
    if args.run_now:
        logger.info("\n" + "="*70)
        logger.info("STARTING POC SIMULATION")
        logger.info("="*70)

        try:
            job.simulator_run(
                workspace=str(workspace_path),
                n_clients=args.num_clients,
                threads=args.num_clients,
            )
            logger.info("\n" + "="*70)
            logger.info("POC SIMULATION COMPLETED SUCCESSFULLY")
            logger.info("="*70)
            logger.info(f"Results saved in: {workspace_path}")
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    else:
        logger.info("\nJob ready for POC simulation")
        logger.info(f"To run manually:")
        logger.info(f"  cd {workspace_path}")
        logger.info(f"  nvflare simulator -w {workspace_path} -n {args.num_clients} -t {args.num_rounds}")


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("NVFlare FedOpt Job Configuration Tool")
    logger.info(f"Mode: {args.mode.upper()}")

    try:
        # Create job with FedOpt controller
        job = create_fedopt_job(args)

        # Execute based on mode
        if args.mode == 'poc':
            run_poc_mode(args, job)
        elif args.mode == 'export':
            export_dir = Path('./nvflare_jobs_fedopt')
            export_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"\nExporting job to: {export_dir}")
            job.export_job(str(export_dir))
            logger.info("Job export completed")
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Job creation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
