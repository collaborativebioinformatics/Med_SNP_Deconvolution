#!/usr/bin/env python3
"""
NVFlare Job Configuration and Submission Script

Creates and manages NVFlare federated learning jobs using FedAvgRecipe.
Supports both POC (Proof of Concept) simulation and production deployment.

Reference: https://nvflare.readthedocs.io/en/2.7.0/hello-world/hello-lightning/

Usage:
    # POC Mode - Simulate FL with multiple clients locally
    python job.py --mode poc --num_rounds 10 --num_clients 3

    # Export Mode - Create job package for deployment
    python job.py --mode export --export_dir ./jobs --num_rounds 50

    # Run POC simulation directly
    python job.py --mode poc --num_rounds 5 --run_now

    # Customize clients
    python job.py --mode poc --clients site1,site2,site3 --num_rounds 10

Architecture:
    - Uses FedAvgRecipe for simplified job configuration
    - Automatically configures server and client components
    - Supports ScatterAndGather workflow (FedAvg algorithm)
    - Compatible with both POC and production environments

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-07
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

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
        description='NVFlare Job Configuration for SNP Deconvolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        choices=['poc', 'export'],
        default='poc',
        help='Job mode: poc (simulate locally) or export (create deployment package)'
    )

    # Job configuration
    parser.add_argument(
        '--job_name',
        type=str,
        default='snp_fedavg',
        help='Job name'
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=10,
        help='Number of federated learning rounds'
    )
    parser.add_argument(
        '--num_clients',
        type=int,
        default=3,
        help='Number of clients (POC mode only)'
    )
    parser.add_argument(
        '--clients',
        type=str,
        default=None,
        help='Comma-separated client names (e.g., "site1,site2,site3"). If not provided, auto-generated.'
    )

    # Client configuration
    parser.add_argument(
        '--local_epochs',
        type=int,
        default=1,
        help='Number of local epochs per FL round'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    # Model configuration
    parser.add_argument(
        '--architecture',
        type=str,
        default='cnn_transformer',
        choices=['cnn', 'cnn_transformer', 'gnn'],
        help='Model architecture'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=3,
        help='Number of population classes'
    )
    parser.add_argument(
        '--feature_type',
        type=str,
        choices=['cluster', 'snp'],
        default='cluster',
        help='Feature type'
    )

    # Server aggregation configuration
    parser.add_argument(
        '--min_clients',
        type=int,
        default=None,
        help='Minimum number of clients required per round (default: all clients)'
    )

    # Export configuration
    parser.add_argument(
        '--export_dir',
        type=str,
        default='./nvflare_jobs',
        help='Directory to export job package (export mode only)'
    )

    # POC configuration
    parser.add_argument(
        '--workspace',
        type=str,
        default='./nvflare_workspace',
        help='Workspace directory for POC mode'
    )
    parser.add_argument(
        '--run_now',
        action='store_true',
        help='Immediately run the job in POC mode after creation'
    )

    # Data configuration
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/Users/saltfish/Files/Coding/Med_SNP_Deconvolution/snp_deconvolution/nvflare_real/data',
        help='Data directory (must contain site-specific data splits)'
    )

    return parser.parse_args()


def create_job_with_recipe(
    job_name: str,
    num_rounds: int,
    clients: List[str],
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    architecture: str,
    num_classes: int,
    feature_type: str,
    data_dir: str,
    min_clients: Optional[int] = None,
):
    """
    Create NVFlare job using FedAvgRecipe.

    Args:
        job_name: Name of the job
        num_rounds: Number of FL rounds
        clients: List of client site names
        local_epochs: Local training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        architecture: Model architecture
        num_classes: Number of classes
        feature_type: Feature type
        data_dir: Data directory
        min_clients: Minimum clients per round

    Returns:
        FedJob instance
    """
    try:
        from nvflare import FedJob
        from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
    except ImportError:
        logger.error("NVFlare not installed. Install with: pip install nvflare")
        raise

    logger.info("="*60)
    logger.info(f"Creating NVFlare job: {job_name}")
    logger.info("="*60)
    logger.info(f"Algorithm: FedAvg (Federated Averaging)")
    logger.info(f"Clients: {', '.join(clients)}")
    logger.info(f"FL Rounds: {num_rounds}")
    logger.info(f"Local Epochs: {local_epochs}")
    logger.info(f"Min Clients: {min_clients if min_clients else 'all'}")
    logger.info("="*60)

    # Create FedAvg job
    job = FedAvgJob(
        name=job_name,
        num_rounds=num_rounds,
        n_clients=len(clients),
        min_clients=min_clients if min_clients else len(clients),
    )

    # Configure client script and arguments
    client_script = str(Path(__file__).parent / "client.py")

    # Build client script arguments
    script_args = (
        f"--data_dir {data_dir} "
        f"--feature_type {feature_type} "
        f"--architecture {architecture} "
        f"--num_classes {num_classes} "
        f"--local_epochs {local_epochs} "
        f"--batch_size {batch_size} "
        f"--learning_rate {learning_rate} "
        f"--use_focal_loss "
        f"--precision bf16-mixed"
    )

    # Add client configuration to job
    for client in clients:
        job.to(
            client,
            client_script,
            script_args=script_args,
        )
        logger.info(f"Configured client: {client}")

    logger.info(f"Job created successfully: {job_name}")
    return job


def run_poc_mode(args):
    """
    Run job in POC (Proof of Concept) mode.

    POC mode simulates federated learning locally with multiple clients.
    Useful for development and testing.

    Args:
        args: Command line arguments
    """
    try:
        from nvflare.job import SimEnv
    except ImportError:
        logger.error("NVFlare not installed. Install with: pip install nvflare")
        raise

    # Generate client names if not provided
    if args.clients:
        clients = [c.strip() for c in args.clients.split(',')]
    else:
        clients = [f"site{i+1}" for i in range(args.num_clients)]

    logger.info("="*60)
    logger.info("POC MODE - Local Simulation")
    logger.info("="*60)
    logger.info(f"Workspace: {args.workspace}")
    logger.info(f"Clients: {clients}")
    logger.info("="*60)

    # Create job
    job = create_job_with_recipe(
        job_name=args.job_name,
        num_rounds=args.num_rounds,
        clients=clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        architecture=args.architecture,
        num_classes=args.num_classes,
        feature_type=args.feature_type,
        data_dir=args.data_dir,
        min_clients=args.min_clients,
    )

    # Verify data exists for each client
    data_path = Path(args.data_dir)
    logger.info("\nVerifying client data...")
    for client in clients:
        client_data_file = data_path / f"{client}_{args.feature_type}.npz"
        if client_data_file.exists():
            logger.info(f"  {client}: Data found at {client_data_file}")
        else:
            logger.warning(f"  {client}: Data NOT found at {client_data_file}")
            logger.warning(f"    Please ensure data is prepared before running!")

    # Export job to workspace
    workspace_path = Path(args.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nExporting job to workspace: {workspace_path}")
    job.export_job(str(workspace_path))
    logger.info("Job exported successfully")

    # Run simulation if requested
    if args.run_now:
        logger.info("\n" + "="*60)
        logger.info("STARTING POC SIMULATION")
        logger.info("="*60)

        # Create simulation environment
        sim_env = SimEnv(workspace=str(workspace_path))

        # Run the job
        logger.info(f"Running job: {args.job_name}")
        logger.info(f"Simulating {len(clients)} clients for {args.num_rounds} rounds...")

        try:
            sim_env.run(job_name=args.job_name, clients=clients)
            logger.info("\n" + "="*60)
            logger.info("POC SIMULATION COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            logger.info(f"Results saved in: {workspace_path}")
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
    else:
        logger.info("\nJob ready for POC simulation")
        logger.info(f"To run manually:")
        logger.info(f"  cd {workspace_path}")
        logger.info(f"  nvflare simulator -w {workspace_path} -n {len(clients)} -t {args.num_rounds}")


def run_export_mode(args):
    """
    Export job package for production deployment.

    Creates a job package that can be submitted to a deployed NVFlare system.

    Args:
        args: Command line arguments
    """
    # Generate client names if not provided
    if args.clients:
        clients = [c.strip() for c in args.clients.split(',')]
    else:
        clients = [f"site{i+1}" for i in range(args.num_clients)]

    logger.info("="*60)
    logger.info("EXPORT MODE - Production Deployment")
    logger.info("="*60)
    logger.info(f"Export directory: {args.export_dir}")
    logger.info(f"Clients: {clients}")
    logger.info("="*60)

    # Create job
    job = create_job_with_recipe(
        job_name=args.job_name,
        num_rounds=args.num_rounds,
        clients=clients,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        architecture=args.architecture,
        num_classes=args.num_classes,
        feature_type=args.feature_type,
        data_dir=args.data_dir,
        min_clients=args.min_clients,
    )

    # Export job
    export_path = Path(args.export_dir)
    export_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nExporting job package to: {export_path}")
    job.export_job(str(export_path))

    logger.info("\n" + "="*60)
    logger.info("JOB EXPORT COMPLETED")
    logger.info("="*60)
    logger.info(f"Job package: {export_path / args.job_name}")
    logger.info("\nTo submit to deployed NVFlare system:")
    logger.info(f"  nvflare job submit -j {export_path / args.job_name}")
    logger.info("\nTo check job status:")
    logger.info("  nvflare job list")
    logger.info("="*60)


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("NVFlare Job Configuration Tool")
    logger.info(f"Mode: {args.mode.upper()}")

    try:
        if args.mode == 'poc':
            run_poc_mode(args)
        elif args.mode == 'export':
            run_export_mode(args)
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
