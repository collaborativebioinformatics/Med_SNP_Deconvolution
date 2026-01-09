"""
Strategies Module - Federated Learning Strategy Registry

This module provides a centralized registry and factory functions for creating
NVFlare federated learning controllers with different strategies.

Available Strategies:
    - FedAvg: Federated Averaging
    - FedProx: Federated Proximal (with proximal term)
    - Scaffold: Stochastic Controlled Averaging
    - FedOpt: Server-side Adaptive Optimization (FedAdam, FedAdaGrad, FedYogi)

Main Functions:
    - list_strategies(): Get list of available strategies
    - get_strategy_metadata(): Get metadata for a specific strategy
    - create_controller(): Factory function to create strategy controller
    - get_client_script_args(): Generate client script arguments for a strategy
    - validate_strategy_parameters(): Validate strategy-specific parameters
    - print_strategy_info(): Print detailed strategy information

Usage:
    ```python
    from snp_deconvolution.nvflare_real.strategies import (
        list_strategies,
        create_controller,
        get_strategy_metadata
    )

    # List available strategies
    strategies = list_strategies()

    # Create a controller
    controller = create_controller(
        strategy='fedopt',
        num_clients=3,
        num_rounds=10,
        server_optimizer='adam',
        server_lr=0.01
    )
    job.to_server(controller)
    ```

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

from .registry import (
    # Core functions
    list_strategies,
    get_strategy_metadata,
    create_controller,
    get_client_script_args,
    validate_strategy_parameters,
    print_strategy_info,
    # Data classes
    StrategyMetadata,
    StrategyParameter,
    # Registry
    STRATEGY_REGISTRY,
    STRATEGY_PARAMETERS,
)

__all__ = [
    # Core functions
    'list_strategies',
    'get_strategy_metadata',
    'create_controller',
    'get_client_script_args',
    'validate_strategy_parameters',
    'print_strategy_info',
    # Data classes
    'StrategyMetadata',
    'StrategyParameter',
    # Registry
    'STRATEGY_REGISTRY',
    'STRATEGY_PARAMETERS',
]
