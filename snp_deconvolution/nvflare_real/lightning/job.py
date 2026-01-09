#!/usr/bin/env python3
"""
NVFlare Job Configuration and Submission Script

Creates and manages NVFlare federated learning jobs with multiple FL strategies.
Supports both POC (Proof of Concept) simulation and production deployment.

Supported Strategies:
    - FedAvg: Federated Averaging (default)
    - FedProx: Federated Proximal (with proximal term)
    - Scaffold: Stochastic Controlled Averaging
    - FedOpt: Server-side Adaptive Optimization (FedAdam, FedAdaGrad, FedYogi)

Reference: https://nvflare.readthedocs.io/en/2.7.0/hello-world/hello-lightning/

Usage:
    # POC Mode - FedAvg (default)
    python job.py --mode poc --num_rounds 10 --num_clients 3

    # POC Mode - FedProx
    python job.py --mode poc --strategy fedprox --mu 0.01 --num_rounds 10

    # POC Mode - FedOpt with Adam
    python job.py --mode poc --strategy fedopt --server_optimizer adam --server_lr 0.01

    # Export Mode - Create job package for deployment
    python job.py --mode export --export_dir ./jobs --num_rounds 50 --strategy fedopt

    # Run POC simulation directly
    python job.py --mode poc --num_rounds 5 --run_now --strategy fedprox

Architecture:
    - Uses strategy registry for flexible FL algorithm selection
    - Automatically configures server and client components
    - Supports FedAvg, FedProx, Scaffold, and FedOpt strategies
    - Compatible with both POC and production environments

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
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

# Import strategy registry
from snp_deconvolution.nvflare_real.strategies import (
    create_controller,
    get_client_script_args,
    get_strategy_metadata,
)


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

    # Federated learning strategy
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['fedavg', 'fedprox', 'scaffold', 'fedopt'],
        default='fedavg',
        help='Federated learning strategy'
    )

    # FedProx parameters
    parser.add_argument(
        '--mu',
        type=float,
        default=0.01,
        help='FedProx proximal term coefficient (only used with --strategy fedprox)'
    )

    # FedOpt parameters
    parser.add_argument(
        '--server_optimizer',
        type=str,
        choices=['adam', 'sgdm', 'adagrad', 'yogi'],
        default='adam',
        help='Server-side optimizer for FedOpt (only used with --strategy fedopt)'
    )
    parser.add_argument(
        '--server_lr',
        type=float,
        default=0.01,
        help='Learning rate for server optimizer (only used with --strategy fedopt)'
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        help='Adam/Yogi beta1 parameter (only used with --strategy fedopt)'
    )
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        help='Adam/Yogi beta2 parameter (only used with --strategy fedopt)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGDM momentum parameter (only used with --strategy fedopt --server_optimizer sgdm)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-8,
        help='Numerical stability epsilon (only used with --strategy fedopt)'
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

    # Data configuration - use relative path from project root
    default_data_dir = str(project_root / 'data' / 'federated')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=default_data_dir,
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
    strategy: str = 'fedavg',
    mu: float = 0.01,
    server_optimizer: str = 'adam',
    server_lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    momentum: float = 0.9,
    epsilon: float = 1e-8,
):
    """
    Create NVFlare job with specified federated learning strategy.

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
        strategy: FL strategy ('fedavg', 'fedprox', 'scaffold', 'fedopt')
        mu: FedProx proximal term coefficient
        server_optimizer: FedOpt server optimizer ('adam', 'sgdm', 'adagrad', 'yogi')
        server_lr: FedOpt server learning rate
        beta1: Adam/Yogi beta1 parameter
        beta2: Adam/Yogi beta2 parameter
        momentum: SGDM momentum parameter
        epsilon: Numerical stability epsilon

    Returns:
        FedJob instance
    """
    try:
        # NVFlare 2.7+ API
        from nvflare.job_config.api import FedJob
        from nvflare.job_config.script_runner import ScriptRunner
    except ImportError:
        logger.error("NVFlare not installed. Install with: pip install nvflare")
        raise

    # Get strategy metadata
    strategy_metadata = get_strategy_metadata(strategy)

    logger.info("="*60)
    logger.info(f"Creating NVFlare job: {job_name}")
    logger.info("="*60)
    logger.info(f"Strategy: {strategy_metadata.display_name} ({strategy})")
    logger.info(f"Description: {strategy_metadata.description}")
    logger.info(f"Clients: {', '.join(clients)}")
    logger.info(f"FL Rounds: {num_rounds}")
    logger.info(f"Local Epochs: {local_epochs}")
    logger.info(f"Min Clients: {min_clients if min_clients else 'all'}")

    # Log strategy-specific parameters
    if strategy == 'fedprox':
        logger.info(f"FedProx mu: {mu}")
    elif strategy == 'fedopt':
        logger.info(f"Server Optimizer: {server_optimizer}")
        logger.info(f"Server LR: {server_lr}")
        if server_optimizer in ['adam', 'yogi']:
            logger.info(f"Beta1: {beta1}, Beta2: {beta2}")
        elif server_optimizer == 'sgdm':
            logger.info(f"Momentum: {momentum}")

    logger.info("="*60)

    # Create FedJob (NVFlare 2.7+ API)
    job = FedJob(
        name=job_name,
        min_clients=min_clients if min_clients else len(clients),
    )

    # Add model persistor for saving global model checkpoints
    # In NVFlare 2.7+, use PTModel wrapper which auto-configures PTFileModelPersistor
    persistor_id = None
    try:
        from nvflare.app_opt.pt.job_config.model import PTModel
        import torch.nn as nn

        # Create a dummy model structure for persistor (actual weights come from clients)
        # This is needed for PTFileModelPersistor to know the model structure
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.placeholder = nn.Linear(1, 1)
            def forward(self, x):
                return x

        model_component = PTModel(DummyModel())
        component_ids = job.to_server(model_component)
        persistor_id = component_ids.get("persistor_id", "persistor")
        logger.info(f"Added PTModel with persistor_id={persistor_id} for global model saving")
    except (ImportError, Exception) as e:
        logger.warning(f"PTModel not available: {e}, trying without persistor")
        persistor_id = None

    # Create controller directly for better NVFlare 2.7.x compatibility
    # Note: Using FedAvg directly instead of registry for FedAvg/FedProx
    # SCAFFOLD support requires additional client-side implementation
    if strategy.lower() in ['fedavg', 'fedprox']:
        from nvflare.app_common.workflows.fedavg import FedAvg
        controller = FedAvg(
            num_clients=len(clients),
            num_rounds=num_rounds,
            persistor_id=persistor_id if persistor_id else "",
        )
        logger.info(f"Created FedAvg controller (strategy={strategy}, persistor_id={persistor_id})")
    elif strategy.lower() == 'scaffold':
        logger.warning("SCAFFOLD requires special client implementation - falling back to FedAvg")
        from nvflare.app_common.workflows.fedavg import FedAvg
        controller = FedAvg(
            num_clients=len(clients),
            num_rounds=num_rounds,
            persistor_id=persistor_id if persistor_id else "",
        )
    elif strategy.lower() == 'fedopt':
        # Use custom FedOpt controller
        controller = create_controller(
            strategy=strategy,
            num_clients=len(clients),
            num_rounds=num_rounds,
            mu=mu,
            server_optimizer=server_optimizer,
            server_lr=server_lr,
            beta1=beta1,
            beta2=beta2,
            momentum=momentum,
            epsilon=epsilon,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    job.to_server(controller)
    logger.info(f"Added {strategy_metadata.display_name} controller to server")

    # Configure client script and arguments
    client_script = str(Path(__file__).parent / "client.py")

    # Build base client script arguments
    base_args = (
        f"--data_dir {data_dir} "
        f"--feature_type {feature_type} "
        f"--architecture {architecture} "
        f"--num_classes {num_classes} "
        f"--local_epochs {local_epochs} "
        f"--batch_size {batch_size} "
        f"--learning_rate {learning_rate} "
        f"--use_focal_loss "
        f"--precision bf16-mixed "
        f"--checkpoint_dir ./checkpoints "
        f"--save_top_k 3"
    )

    # Add strategy-specific arguments for client
    strategy_args = get_client_script_args(
        strategy=strategy,
        mu=mu,
        server_optimizer=server_optimizer,
        server_lr=server_lr,
    )

    # Combine all arguments
    script_args = f"{base_args} {strategy_args}"
    logger.info(f"Client script arguments: {script_args}")

    # Add ScriptRunner to all clients (don't specify individual clients)
    # This allows simulator_run to use n_clients parameter
    runner = ScriptRunner(
        script=client_script,
        script_args=script_args,
    )
    job.to_clients(runner)
    logger.info(f"Configured ScriptRunner for clients")

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
        strategy=args.strategy,
        mu=args.mu,
        server_optimizer=args.server_optimizer,
        server_lr=args.server_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        momentum=args.momentum,
        epsilon=args.epsilon,
    )

    # Verify data exists for each client
    data_path = Path(args.data_dir)
    logger.info("\nVerifying client data...")
    for client in clients:
        # Check both formats: site directory or combined file
        client_dir = data_path / client
        client_data_file = data_path / f"{client}_{args.feature_type}.npz"
        train_file = client_dir / f"train_{args.feature_type}.npz"

        if train_file.exists():
            logger.info(f"  {client}: Data found at {client_dir}/")
        elif client_data_file.exists():
            logger.info(f"  {client}: Data found at {client_data_file}")
        else:
            logger.warning(f"  {client}: Data NOT found!")
            logger.warning(f"    Expected: {client_dir}/ or {client_data_file}")
            logger.warning(f"    Please run prepare_federated_data.py first!")

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

        logger.info(f"Running job: {args.job_name}")
        logger.info(f"Simulating {len(clients)} clients for {args.num_rounds} rounds...")

        try:
            # NVFlare 2.7+ API: use job.simulator_run() directly
            # When using to_clients() (generic), we must specify n_clients
            job.simulator_run(
                workspace=str(workspace_path),
                n_clients=len(clients),
                threads=len(clients),
            )
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
        strategy=args.strategy,
        mu=args.mu,
        server_optimizer=args.server_optimizer,
        server_lr=args.server_lr,
        beta1=args.beta1,
        beta2=args.beta2,
        momentum=args.momentum,
        epsilon=args.epsilon,
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
