#!/usr/bin/env python3
"""
Strategy Registry for NVFlare Federated Learning

This module provides a centralized registry of available federated learning strategies
and factory functions to create the appropriate controllers with their configurations.

Supported Strategies:
    - FedAvg: Federated Averaging (default)
    - FedProx: Federated Proximal (adds proximal term to client optimization)
    - Scaffold: Variance reduction with control variates
    - FedOpt: Server-side adaptive optimization (FedAdam, FedAdaGrad, FedYogi)

Usage:
    ```python
    from snp_deconvolution.nvflare_real.strategies import (
        get_strategy_metadata,
        create_controller,
        list_strategies
    )

    # List available strategies
    strategies = list_strategies()

    # Get strategy information
    metadata = get_strategy_metadata('fedopt')

    # Create controller
    controller = create_controller(
        strategy='fedopt',
        num_clients=3,
        num_rounds=10,
        server_optimizer='adam',
        server_lr=0.01
    )
    ```

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyParameter:
    """
    Metadata for a strategy parameter.

    Attributes:
        name: Parameter name
        type: Parameter type (str, float, int, bool)
        default: Default value
        choices: Valid choices for the parameter (optional)
        description: Human-readable description
        required_for: List of strategies that require this parameter
    """
    name: str
    type: type
    default: Any
    choices: Optional[List[Any]] = None
    description: str = ""
    required_for: List[str] = field(default_factory=list)


@dataclass
class StrategyMetadata:
    """
    Metadata for a federated learning strategy.

    Attributes:
        name: Strategy name
        display_name: Human-readable name
        description: Strategy description
        controller_class: NVFlare controller class name
        parameters: Strategy-specific parameters
        reference: Paper/documentation reference
    """
    name: str
    display_name: str
    description: str
    controller_class: str
    parameters: Dict[str, StrategyParameter]
    reference: str = ""


# Define strategy parameters
STRATEGY_PARAMETERS = {
    'mu': StrategyParameter(
        name='mu',
        type=float,
        default=0.01,
        description='Proximal term coefficient for FedProx (regularizes deviation from global model)',
        required_for=['fedprox']
    ),
    'server_optimizer': StrategyParameter(
        name='server_optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgdm', 'adagrad', 'yogi'],
        description='Server-side optimizer for FedOpt',
        required_for=['fedopt']
    ),
    'server_lr': StrategyParameter(
        name='server_lr',
        type=float,
        default=0.01,
        description='Learning rate for server optimizer (FedOpt)',
        required_for=['fedopt']
    ),
    'beta1': StrategyParameter(
        name='beta1',
        type=float,
        default=0.9,
        description='First moment decay rate for Adam/Yogi (FedOpt)',
        required_for=['fedopt']
    ),
    'beta2': StrategyParameter(
        name='beta2',
        type=float,
        default=0.999,
        description='Second moment decay rate for Adam/Yogi (FedOpt)',
        required_for=['fedopt']
    ),
    'momentum': StrategyParameter(
        name='momentum',
        type=float,
        default=0.9,
        description='Momentum coefficient for SGDM (FedOpt)',
        required_for=['fedopt']
    ),
    'epsilon': StrategyParameter(
        name='epsilon',
        type=float,
        default=1e-8,
        description='Small constant for numerical stability (FedOpt)',
        required_for=['fedopt']
    ),
}


# Define strategy registry
STRATEGY_REGISTRY: Dict[str, StrategyMetadata] = {
    'fedavg': StrategyMetadata(
        name='fedavg',
        display_name='FedAvg',
        description='Federated Averaging - Simple weighted average of client models',
        controller_class='nvflare.app_common.workflows.fedavg.FedAvg',
        parameters={},
        reference='McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (AISTATS 2017)'
    ),
    'fedprox': StrategyMetadata(
        name='fedprox',
        display_name='FedProx',
        description='Federated Proximal - FedAvg with proximal term to handle data heterogeneity',
        controller_class='nvflare.app_common.workflows.fedavg.FedAvg',
        parameters={
            'mu': STRATEGY_PARAMETERS['mu']
        },
        reference='Li et al. "Federated Optimization in Heterogeneous Networks" (MLSys 2020)'
    ),
    'scaffold': StrategyMetadata(
        name='scaffold',
        display_name='SCAFFOLD',
        description='Stochastic Controlled Averaging for Federated Learning - Variance reduction with control variates',
        controller_class='nvflare.app_common.workflows.scaffold.Scaffold',
        parameters={},
        reference='Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning" (ICML 2020)'
    ),
    'fedopt': StrategyMetadata(
        name='fedopt',
        display_name='FedOpt',
        description='Federated Optimization - Server-side adaptive optimizers (FedAdam, FedAdaGrad, FedYogi)',
        controller_class='snp_deconvolution.nvflare_real.controllers.fedopt_controller.FedOptController',
        parameters={
            'server_optimizer': STRATEGY_PARAMETERS['server_optimizer'],
            'server_lr': STRATEGY_PARAMETERS['server_lr'],
            'beta1': STRATEGY_PARAMETERS['beta1'],
            'beta2': STRATEGY_PARAMETERS['beta2'],
            'momentum': STRATEGY_PARAMETERS['momentum'],
            'epsilon': STRATEGY_PARAMETERS['epsilon'],
        },
        reference='Reddi et al. "Adaptive Federated Optimization" (ICLR 2021)'
    ),
}


def list_strategies() -> List[str]:
    """
    Get list of available strategy names.

    Returns:
        List of strategy names
    """
    return list(STRATEGY_REGISTRY.keys())


def get_strategy_metadata(strategy: str) -> StrategyMetadata:
    """
    Get metadata for a specific strategy.

    Args:
        strategy: Strategy name (e.g., 'fedavg', 'fedprox', 'fedopt')

    Returns:
        StrategyMetadata object

    Raises:
        ValueError: If strategy is not found in registry
    """
    strategy_lower = strategy.lower()
    if strategy_lower not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy: {strategy}. Available strategies: {available}"
        )
    return STRATEGY_REGISTRY[strategy_lower]


def validate_strategy_parameters(strategy: str, **kwargs) -> Dict[str, Any]:
    """
    Validate and extract strategy-specific parameters.

    Args:
        strategy: Strategy name
        **kwargs: Parameters to validate

    Returns:
        Dictionary of validated parameters for the strategy

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    metadata = get_strategy_metadata(strategy)
    validated = {}

    for param_name, param_spec in metadata.parameters.items():
        value = kwargs.get(param_name, param_spec.default)

        # Validate choices if specified
        if param_spec.choices is not None and value not in param_spec.choices:
            raise ValueError(
                f"Invalid value for {param_name}: {value}. "
                f"Valid choices: {param_spec.choices}"
            )

        # Type check
        if not isinstance(value, param_spec.type):
            try:
                value = param_spec.type(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Invalid type for {param_name}: expected {param_spec.type.__name__}, "
                    f"got {type(value).__name__}"
                )

        validated[param_name] = value

    return validated


def create_controller(
    strategy: str,
    num_clients: int,
    num_rounds: int,
    **kwargs
) -> Any:
    """
    Factory function to create the appropriate controller for a strategy.

    Args:
        strategy: Strategy name ('fedavg', 'fedprox', 'scaffold', 'fedopt')
        num_clients: Number of clients to aggregate per round
        num_rounds: Total number of federated learning rounds
        **kwargs: Strategy-specific parameters

    Returns:
        Controller instance (FedAvg, Scaffold, or FedOptController)

    Raises:
        ValueError: If strategy is unknown or parameters are invalid
        ImportError: If required controller class cannot be imported

    Examples:
        >>> # Create FedAvg controller
        >>> controller = create_controller('fedavg', num_clients=3, num_rounds=10)

        >>> # Create FedProx controller
        >>> controller = create_controller('fedprox', num_clients=3, num_rounds=10, mu=0.01)

        >>> # Create FedOpt controller
        >>> controller = create_controller(
        ...     'fedopt', num_clients=3, num_rounds=10,
        ...     server_optimizer='adam', server_lr=0.01
        ... )
    """
    metadata = get_strategy_metadata(strategy)
    strategy_lower = strategy.lower()

    logger.info(f"Creating controller for strategy: {metadata.display_name}")
    logger.info(f"  Description: {metadata.description}")
    logger.info(f"  Clients: {num_clients}, Rounds: {num_rounds}")

    # Validate strategy-specific parameters
    validated_params = validate_strategy_parameters(strategy_lower, **kwargs)

    # Import and create controller based on strategy
    if strategy_lower == 'fedavg':
        # Standard FedAvg - simple weighted averaging
        try:
            from nvflare.app_common.workflows.fedavg import FedAvg
        except ImportError:
            raise ImportError(
                "NVFlare not installed. Install with: pip install nvflare"
            )

        controller = FedAvg(
            num_clients=num_clients,
            num_rounds=num_rounds,
        )
        logger.info("Created FedAvg controller")

    elif strategy_lower == 'fedprox':
        # FedProx - FedAvg with proximal term (implemented client-side)
        # Server uses FedAvg aggregation, but clients apply proximal regularization
        try:
            from nvflare.app_common.workflows.fedavg import FedAvg
        except ImportError:
            raise ImportError(
                "NVFlare not installed. Install with: pip install nvflare"
            )

        controller = FedAvg(
            num_clients=num_clients,
            num_rounds=num_rounds,
        )
        logger.info(f"Created FedProx controller (server-side FedAvg with mu={validated_params['mu']})")
        logger.info("Note: Proximal term (mu) is applied client-side during training")

    elif strategy_lower == 'scaffold':
        # SCAFFOLD - Variance reduction with control variates
        try:
            from nvflare.app_common.workflows.scaffold import Scaffold
        except ImportError:
            raise ImportError(
                "NVFlare SCAFFOLD not available. "
                "Ensure you have the latest NVFlare version or implement custom SCAFFOLD controller."
            )

        controller = Scaffold(
            num_clients=num_clients,
            num_rounds=num_rounds,
        )
        logger.info("Created SCAFFOLD controller")

    elif strategy_lower == 'fedopt':
        # FedOpt - Server-side adaptive optimization
        try:
            from snp_deconvolution.nvflare_real.controllers.fedopt_controller import FedOptController
        except ImportError:
            raise ImportError(
                "FedOptController not found. "
                "Ensure snp_deconvolution.nvflare_real.controllers.fedopt_controller is available."
            )

        server_optimizer = validated_params['server_optimizer']

        controller = FedOptController(
            num_clients=num_clients,
            num_rounds=num_rounds,
            optimizer=server_optimizer,
            server_lr=validated_params['server_lr'],
            beta1=validated_params.get('beta1', 0.9),
            beta2=validated_params.get('beta2', 0.999),
            momentum=validated_params.get('momentum', 0.9),
            epsilon=validated_params.get('epsilon', 1e-8),
        )
        logger.info(f"Created FedOpt controller with {server_optimizer.upper()} optimizer")
        logger.info(f"  Server LR: {validated_params['server_lr']}")

    else:
        raise ValueError(f"Strategy implementation not found: {strategy_lower}")

    return controller


def get_client_script_args(strategy: str, **kwargs) -> str:
    """
    Generate client script arguments for a specific strategy.

    This function creates the command-line arguments that should be passed
    to the client script to enable strategy-specific features.

    Args:
        strategy: Strategy name
        **kwargs: Strategy-specific parameters

    Returns:
        String of command-line arguments to append to client script

    Examples:
        >>> get_client_script_args('fedavg')
        '--strategy fedavg'

        >>> get_client_script_args('fedprox', mu=0.01)
        '--strategy fedprox --mu 0.01'

        >>> get_client_script_args('fedopt', server_optimizer='adam')
        '--strategy fedopt'
    """
    metadata = get_strategy_metadata(strategy)
    validated_params = validate_strategy_parameters(strategy, **kwargs)

    args = [f"--strategy {strategy}"]

    # Add strategy-specific client-side parameters
    if strategy.lower() == 'fedprox':
        # FedProx needs mu parameter on client side
        mu = validated_params.get('mu', 0.01)
        args.append(f"--mu {mu}")

    # FedOpt and SCAFFOLD don't need additional client args
    # (server-side aggregation handles the logic)

    return " ".join(args)


def print_strategy_info(strategy: str) -> None:
    """
    Print detailed information about a strategy.

    Args:
        strategy: Strategy name
    """
    metadata = get_strategy_metadata(strategy)

    print(f"\n{'='*70}")
    print(f"Strategy: {metadata.display_name} ({metadata.name})")
    print(f"{'='*70}")
    print(f"\nDescription:")
    print(f"  {metadata.description}")
    print(f"\nController: {metadata.controller_class}")

    if metadata.parameters:
        print(f"\nParameters:")
        for param_name, param_spec in metadata.parameters.items():
            print(f"  --{param_name}")
            print(f"    Type: {param_spec.type.__name__}")
            print(f"    Default: {param_spec.default}")
            if param_spec.choices:
                print(f"    Choices: {param_spec.choices}")
            print(f"    Description: {param_spec.description}")
    else:
        print(f"\nParameters: None (uses default FedAvg configuration)")

    if metadata.reference:
        print(f"\nReference:")
        print(f"  {metadata.reference}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Demo usage
    print("Available Federated Learning Strategies:")
    print("=" * 70)

    for strategy_name in list_strategies():
        print_strategy_info(strategy_name)
