#!/usr/bin/env python3
"""
Strategy Registry Usage Examples

This script demonstrates how to use the strategy registry to create
different federated learning controllers and configurations.

Run this script to see examples of:
- Listing available strategies
- Getting strategy metadata
- Creating controllers for different strategies
- Generating client script arguments

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from snp_deconvolution.nvflare_real.strategies import (
    list_strategies,
    get_strategy_metadata,
    create_controller,
    get_client_script_args,
    print_strategy_info,
)


def example_list_strategies():
    """Example: List all available strategies."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: List Available Strategies")
    print("=" * 70)

    strategies = list_strategies()
    print(f"Available strategies: {strategies}")
    print(f"Total strategies: {len(strategies)}")


def example_get_metadata():
    """Example: Get metadata for a specific strategy."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Get Strategy Metadata")
    print("=" * 70)

    strategy = 'fedopt'
    metadata = get_strategy_metadata(strategy)

    print(f"Strategy: {metadata.display_name}")
    print(f"Description: {metadata.description}")
    print(f"Controller: {metadata.controller_class}")
    print(f"Parameters: {list(metadata.parameters.keys())}")
    print(f"Reference: {metadata.reference}")


def example_create_fedavg():
    """Example: Create FedAvg controller."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Create FedAvg Controller")
    print("=" * 70)

    try:
        controller = create_controller(
            strategy='fedavg',
            num_clients=3,
            num_rounds=10,
        )
        print(f"Controller created: {type(controller).__name__}")
        print("Configuration: Standard FedAvg with weighted averaging")
    except ImportError as e:
        print(f"Cannot create controller (NVFlare not installed): {e}")


def example_create_fedprox():
    """Example: Create FedProx controller."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Create FedProx Controller")
    print("=" * 70)

    try:
        controller = create_controller(
            strategy='fedprox',
            num_clients=3,
            num_rounds=10,
            mu=0.01,  # Proximal term coefficient
        )
        print(f"Controller created: {type(controller).__name__}")
        print("Configuration: FedProx with mu=0.01")
        print("Note: Proximal term applied client-side during training")
    except ImportError as e:
        print(f"Cannot create controller (NVFlare not installed): {e}")


def example_create_fedopt():
    """Example: Create FedOpt controller with different optimizers."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Create FedOpt Controllers")
    print("=" * 70)

    optimizers = ['adam', 'sgdm', 'adagrad', 'yogi']

    for opt in optimizers:
        print(f"\n--- FedOpt with {opt.upper()} ---")
        try:
            controller = create_controller(
                strategy='fedopt',
                num_clients=3,
                num_rounds=10,
                server_optimizer=opt,
                server_lr=0.01,
                beta1=0.9,      # For Adam/Yogi
                beta2=0.999,    # For Adam/Yogi
                momentum=0.9,   # For SGDM
                epsilon=1e-8,
            )
            print(f"Controller created: {type(controller).__name__}")
            print(f"Server optimizer: {opt}")
            print(f"Server learning rate: 0.01")
        except ImportError as e:
            print(f"Cannot create controller: {e}")


def example_client_script_args():
    """Example: Generate client script arguments."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Generate Client Script Arguments")
    print("=" * 70)

    # FedAvg
    args = get_client_script_args('fedavg')
    print(f"FedAvg args: {args}")

    # FedProx
    args = get_client_script_args('fedprox', mu=0.05)
    print(f"FedProx args: {args}")

    # FedOpt (no client-side args needed)
    args = get_client_script_args('fedopt', server_optimizer='adam')
    print(f"FedOpt args: {args}")

    # Scaffold
    args = get_client_script_args('scaffold')
    print(f"Scaffold args: {args}")


def example_strategy_info():
    """Example: Print detailed strategy information."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Print Detailed Strategy Information")
    print("=" * 70)

    for strategy in ['fedavg', 'fedprox', 'scaffold', 'fedopt']:
        print_strategy_info(strategy)


def example_validation():
    """Example: Parameter validation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Parameter Validation")
    print("=" * 70)

    # Valid FedOpt optimizer
    print("\n--- Valid optimizer ---")
    try:
        controller = create_controller(
            strategy='fedopt',
            num_clients=3,
            num_rounds=10,
            server_optimizer='adam',
            server_lr=0.01,
        )
        print("Success: Created FedOpt with Adam optimizer")
    except (ValueError, ImportError) as e:
        print(f"Error: {e}")

    # Invalid FedOpt optimizer
    print("\n--- Invalid optimizer ---")
    try:
        controller = create_controller(
            strategy='fedopt',
            num_clients=3,
            num_rounds=10,
            server_optimizer='invalid_optimizer',  # Invalid!
            server_lr=0.01,
        )
        print("Should not reach here!")
    except ValueError as e:
        print(f"Expected error caught: {e}")
    except ImportError:
        print("NVFlare not installed, skipping validation test")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("FEDERATED LEARNING STRATEGY REGISTRY - USAGE EXAMPLES")
    print("=" * 70)

    # Run all examples
    example_list_strategies()
    example_get_metadata()
    example_create_fedavg()
    example_create_fedprox()
    example_create_fedopt()
    example_client_script_args()
    example_validation()

    # Uncomment to see full strategy info
    # example_strategy_info()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETED")
    print("=" * 70)
    print("\nFor more information, run:")
    print("  python -m snp_deconvolution.nvflare_real.strategies.registry")
    print("\n")


if __name__ == "__main__":
    main()
