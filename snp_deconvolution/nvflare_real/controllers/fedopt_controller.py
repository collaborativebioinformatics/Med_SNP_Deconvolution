#!/usr/bin/env python3
"""
FedOpt Controller - Server-Side Adaptive Optimization

This module implements the FedOpt (Federated Optimization) strategy, which applies
server-side adaptive optimizers (Adam, AdaGrad, Yogi, SGD with momentum) instead
of simple averaging for aggregating client model updates.

Algorithm Overview:
    1. Clients compute local updates: delta_i = w_i - w_global
    2. Server aggregates deltas as pseudo-gradient: g = (1/n) * sum(delta_i)
    3. Server applies optimizer: w_global = optimizer.step(w_global, g)
       Note: delta represents the direction of gradient DESCENT (clients moved towards lower loss),
       so we ADD the delta (not subtract) when updating the global model.

Key Features:
    - Supports multiple optimizers: Adam, AdaGrad, Yogi, SGD with momentum
    - Configurable learning rate and optimizer hyperparameters
    - Compatible with NVFlare's workflow architecture
    - Extends FedAvg for seamless integration

Reference:
    Reddi et al. "Adaptive Federated Optimization" (ICLR 2021)
    https://arxiv.org/abs/2003.00295

Usage:
    ```python
    from snp_deconvolution.nvflare_real.controllers import FedOptController

    # FedAdam configuration
    controller = FedOptController(
        num_clients=3,
        num_rounds=10,
        optimizer='adam',
        server_lr=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    )
    job.to_server(controller)
    ```

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

import copy
import logging
from typing import Dict, Any, Optional, List

import numpy as np
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.workflows.fedavg import FedAvg


logger = logging.getLogger(__name__)


class ServerOptimizer:
    """
    Base class for server-side optimizers used in FedOpt.

    These optimizers apply updates to the global model using aggregated
    client deltas as pseudo-gradients.
    """

    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        """
        Initialize server optimizer.

        Args:
            lr: Learning rate for server optimizer
            epsilon: Small constant for numerical stability
        """
        self.lr = lr
        self.epsilon = epsilon
        self.state: Dict[str, Any] = {}

    def step(
        self,
        params: Dict[str, np.ndarray],
        pseudo_gradient: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply optimizer step to global model parameters.

        Args:
            params: Current global model parameters
            pseudo_gradient: Aggregated client deltas (w_client - w_global),
                           representing the direction clients moved during local training.
                           This is effectively the negative gradient direction since
                           clients performed gradient descent.

        Returns:
            Updated global model parameters
        """
        raise NotImplementedError("Subclasses must implement step()")

    def reset_state(self):
        """Reset optimizer state (useful when starting new experiments)."""
        self.state = {}


class ServerSGDM(ServerOptimizer):
    """
    Server-side SGD with momentum.

    This optimizer applies momentum to the aggregated updates, which can
    help accelerate convergence and reduce oscillations.

    Update rule:
        m_t = beta * m_{t-1} + g_t
        w_t = w_{t-1} - lr * m_t
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, epsilon: float = 1e-8):
        """
        Initialize SGD with momentum optimizer.

        Args:
            lr: Learning rate
            momentum: Momentum coefficient (beta)
            epsilon: Small constant for numerical stability
        """
        super().__init__(lr=lr, epsilon=epsilon)
        self.momentum = momentum
        self.state = {'velocity': {}}

    def step(
        self,
        params: Dict[str, np.ndarray],
        pseudo_gradient: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply SGD with momentum step."""
        updated_params = {}

        for name, param in params.items():
            if name not in pseudo_gradient:
                # Keep parameter unchanged if no gradient
                updated_params[name] = param
                continue

            grad = pseudo_gradient[name]

            # Initialize velocity if needed
            if name not in self.state['velocity']:
                self.state['velocity'][name] = np.zeros_like(param)

            # Update velocity: v = beta * v + g
            velocity = self.momentum * self.state['velocity'][name] + grad
            self.state['velocity'][name] = velocity

            # Update parameters: w = w + lr * v
            # Note: grad is already the descent direction (delta = w_client - w_global),
            # so we ADD the update instead of subtracting
            updated_params[name] = param + self.lr * velocity

        return updated_params


class ServerAdam(ServerOptimizer):
    """
    Server-side Adam optimizer.

    Adam combines momentum and adaptive learning rates for each parameter,
    often leading to faster and more stable convergence.

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        w_t = w_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize Adam optimizer.

        Args:
            lr: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(lr=lr, epsilon=epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.state = {
            'step': 0,
            'first_moment': {},
            'second_moment': {}
        }

    def step(
        self,
        params: Dict[str, np.ndarray],
        pseudo_gradient: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply Adam step."""
        self.state['step'] += 1
        t = self.state['step']
        updated_params = {}

        for name, param in params.items():
            if name not in pseudo_gradient:
                updated_params[name] = param
                continue

            grad = pseudo_gradient[name]

            # Initialize moments if needed
            if name not in self.state['first_moment']:
                self.state['first_moment'][name] = np.zeros_like(param)
                self.state['second_moment'][name] = np.zeros_like(param)

            # Update biased first moment: m = beta1 * m + (1 - beta1) * g
            m = self.beta1 * self.state['first_moment'][name] + (1 - self.beta1) * grad
            self.state['first_moment'][name] = m

            # Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
            v = self.beta2 * self.state['second_moment'][name] + (1 - self.beta2) * (grad ** 2)
            self.state['second_moment'][name] = v

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Update parameters: w = w + lr * m_hat / (sqrt(v_hat) + epsilon)
            # Note: grad is already the descent direction (delta = w_client - w_global),
            # so we ADD the update instead of subtracting
            updated_params[name] = param + self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return updated_params


class ServerAdaGrad(ServerOptimizer):
    """
    Server-side AdaGrad optimizer.

    AdaGrad adapts the learning rate for each parameter based on historical
    gradient information, giving frequently updated parameters smaller learning rates.

    Update rule:
        G_t = G_{t-1} + g_t^2
        w_t = w_{t-1} - lr * g_t / (sqrt(G_t) + epsilon)
    """

    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        """
        Initialize AdaGrad optimizer.

        Args:
            lr: Learning rate
            epsilon: Small constant for numerical stability
        """
        super().__init__(lr=lr, epsilon=epsilon)
        self.state = {'sum_squared_gradients': {}}

    def step(
        self,
        params: Dict[str, np.ndarray],
        pseudo_gradient: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply AdaGrad step."""
        updated_params = {}

        for name, param in params.items():
            if name not in pseudo_gradient:
                updated_params[name] = param
                continue

            grad = pseudo_gradient[name]

            # Initialize accumulated squared gradients if needed
            if name not in self.state['sum_squared_gradients']:
                self.state['sum_squared_gradients'][name] = np.zeros_like(param)

            # Accumulate squared gradients: G = G + g^2
            G = self.state['sum_squared_gradients'][name] + grad ** 2
            self.state['sum_squared_gradients'][name] = G

            # Update parameters: w = w + lr * g / (sqrt(G) + epsilon)
            # Note: grad is already the descent direction (delta = w_client - w_global),
            # so we ADD the update instead of subtracting
            updated_params[name] = param + self.lr * grad / (np.sqrt(G) + self.epsilon)

        return updated_params


class ServerYogi(ServerOptimizer):
    """
    Server-side Yogi optimizer.

    Yogi is a variant of Adam that uses a different second moment update rule,
    often providing better convergence properties in federated settings.

    Update rule:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - g_t^2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        w_t = w_{t-1} - lr * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(
        self,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize Yogi optimizer.

        Args:
            lr: Learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(lr=lr, epsilon=epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.state = {
            'step': 0,
            'first_moment': {},
            'second_moment': {}
        }

    def step(
        self,
        params: Dict[str, np.ndarray],
        pseudo_gradient: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply Yogi step."""
        self.state['step'] += 1
        t = self.state['step']
        updated_params = {}

        for name, param in params.items():
            if name not in pseudo_gradient:
                updated_params[name] = param
                continue

            grad = pseudo_gradient[name]

            # Initialize moments if needed
            if name not in self.state['first_moment']:
                self.state['first_moment'][name] = np.zeros_like(param)
                self.state['second_moment'][name] = np.zeros_like(param)

            # Update first moment: m = beta1 * m + (1 - beta1) * g
            m = self.beta1 * self.state['first_moment'][name] + (1 - self.beta1) * grad
            self.state['first_moment'][name] = m

            # Yogi's unique second moment update
            # v = v - (1 - beta2) * sign(v - g^2) * g^2
            v = self.state['second_moment'][name]
            grad_squared = grad ** 2
            v = v - (1 - self.beta2) * np.sign(v - grad_squared) * grad_squared
            self.state['second_moment'][name] = v

            # Bias correction
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # Update parameters: w = w + lr * m_hat / (sqrt(|v_hat|) + epsilon)
            # Note: Use absolute value for v_hat since Yogi can produce negative values
            # Note: grad is already the descent direction (delta = w_client - w_global),
            # so we ADD the update instead of subtracting
            updated_params[name] = param + self.lr * m_hat / (np.sqrt(np.abs(v_hat)) + self.epsilon)

        return updated_params


class FedOptController(FedAvg):
    """
    FedOpt Controller - Server-side adaptive optimization for federated learning.

    This controller extends FedAvg to support adaptive server-side optimizers
    (Adam, AdaGrad, Yogi, SGD with momentum) that treat aggregated client updates
    as pseudo-gradients for global model optimization.

    Key Differences from FedAvg:
        - FedAvg: w_global = (1/n) * sum(w_i)
        - FedOpt: w_global = optimizer.step(w_global, pseudo_gradient)

    where pseudo_gradient = (1/n) * sum(w_i - w_global)

    This approach often leads to faster convergence and better performance,
    especially in heterogeneous data settings.
    """

    def __init__(
        self,
        *,
        num_clients: int,
        num_rounds: int,
        optimizer: str = 'adam',
        server_lr: float = 0.01,
        # Adam/Yogi specific
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        # SGD with momentum specific
        momentum: float = 0.9,
        # Gradient clipping
        max_grad_norm: Optional[float] = None,
        # FedAvg inherited args
        persistor_id: str = "persistor",
        ignore_result_error: bool = False,
        allow_empty_global_weights: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
        snapshot_every_n_rounds: int = 1,
        **kwargs
    ):
        """
        Initialize FedOpt controller.

        Args:
            num_clients: Number of clients to aggregate per round
            num_rounds: Total number of federated learning rounds
            optimizer: Server optimizer type ('adam', 'adagrad', 'yogi', 'sgdm')
            server_lr: Learning rate for server optimizer
            beta1: First moment decay rate (for Adam/Yogi)
            beta2: Second moment decay rate (for Adam/Yogi)
            epsilon: Small constant for numerical stability
            momentum: Momentum coefficient (for SGDM)
            persistor_id: ID of model persistor component
            ignore_result_error: Whether to ignore errors in client results
            allow_empty_global_weights: Whether to allow empty global weights
            task_check_period: Period for checking task status (seconds)
            persist_every_n_rounds: Save model every N rounds
            snapshot_every_n_rounds: Snapshot workspace every N rounds
            **kwargs: Additional arguments passed to FedAvg
        """
        # Initialize parent FedAvg class
        super().__init__(
            num_clients=num_clients,
            num_rounds=num_rounds,
            persistor_id=persistor_id,
            ignore_result_error=ignore_result_error,
            allow_empty_global_weights=allow_empty_global_weights,
            task_check_period=task_check_period,
            persist_every_n_rounds=persist_every_n_rounds,
            snapshot_every_n_rounds=snapshot_every_n_rounds,
            **kwargs
        )

        # FedOpt specific configuration
        self.optimizer_name = optimizer.lower()
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = momentum

        # Gradient clipping
        self.max_grad_norm = max_grad_norm

        # Validate parameters
        if self.server_lr <= 0:
            raise ValueError(f"server_lr must be positive, got {self.server_lr}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if not (0 <= self.beta1 < 1):
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not (0 <= self.beta2 < 1):
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if not (0 <= self.momentum < 1):
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")

        # Initialize server optimizer
        self.server_optimizer: Optional[ServerOptimizer] = None

        # Store previous global model for computing deltas
        self.previous_global_model: Optional[Dict[str, np.ndarray]] = None

        logger.info(f"FedOpt Controller initialized:")
        logger.info(f"  Optimizer: {self.optimizer_name}")
        logger.info(f"  Server LR: {self.server_lr}")
        if self.optimizer_name in ['adam', 'yogi']:
            logger.info(f"  Beta1: {self.beta1}, Beta2: {self.beta2}, Epsilon: {self.epsilon}")
        elif self.optimizer_name == 'sgdm':
            logger.info(f"  Momentum: {self.momentum}")
        logger.info(f"  Clients per round: {num_clients}")
        logger.info(f"  Total rounds: {num_rounds}")

    def start_controller(self, fl_ctx: FLContext) -> None:
        """
        Start controller and initialize server optimizer.

        Args:
            fl_ctx: FL context
        """
        # Call parent start_controller
        super().start_controller(fl_ctx)

        # Initialize server optimizer based on configuration
        if self.optimizer_name == 'adam':
            self.server_optimizer = ServerAdam(
                lr=self.server_lr,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon
            )
            logger.info("Initialized ServerAdam optimizer")
        elif self.optimizer_name == 'adagrad':
            self.server_optimizer = ServerAdaGrad(
                lr=self.server_lr,
                epsilon=self.epsilon
            )
            logger.info("Initialized ServerAdaGrad optimizer")
        elif self.optimizer_name == 'yogi':
            self.server_optimizer = ServerYogi(
                lr=self.server_lr,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon
            )
            logger.info("Initialized ServerYogi optimizer")
        elif self.optimizer_name == 'sgdm':
            self.server_optimizer = ServerSGDM(
                lr=self.server_lr,
                momentum=self.momentum,
                epsilon=self.epsilon
            )
            logger.info("Initialized ServerSGDM optimizer")
        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer_name}. "
                f"Supported optimizers: adam, adagrad, yogi, sgdm"
            )

    def aggregate(
        self,
        shareable_list: List[Shareable],
        fl_ctx: FLContext
    ) -> Shareable:
        """
        Aggregate client models using FedOpt algorithm.

        This method overrides FedAvg's aggregation to implement:
        1. Compute client deltas: delta_i = w_i - w_global
        2. Aggregate deltas as pseudo-gradient: g = (1/n) * sum(delta_i)
        3. Apply server optimizer: w_global = optimizer.step(w_global, g)

        Args:
            shareable_list: List of shareables from clients
            fl_ctx: FL context

        Returns:
            Aggregated shareable with updated global model
        """
        logger.info(f"FedOpt aggregation: Received {len(shareable_list)} client updates")

        # Get current global model
        aggregator = self.aggregator
        current_global = aggregator.get_model_params()

        if current_global is None:
            logger.warning("No global model available, falling back to FedAvg aggregation")
            return super().aggregate(shareable_list, fl_ctx)

        # Store current global model as numpy arrays for FedOpt computation
        if self.previous_global_model is None:
            self.previous_global_model = {
                name: np.array(param) for name, param in current_global.items()
            }
            logger.info("Stored initial global model")

        # Collect client models and compute deltas
        client_deltas = []
        total_samples = 0

        for shareable in shareable_list:
            try:
                dxo = from_shareable(shareable)
                client_weights = dxo.data

                # Get sample count for weighted averaging
                n_samples = dxo.get_meta_prop("NUM_STEPS_CURRENT_ROUND", 1)
                total_samples += n_samples

                # Compute delta: delta_i = w_i - w_global
                delta = {}
                for name, client_param in client_weights.items():
                    if name in self.previous_global_model:
                        global_param = self.previous_global_model[name]
                        delta[name] = np.array(client_param) - global_param
                    else:
                        logger.warning(f"Parameter {name} not found in global model, skipping")

                client_deltas.append((delta, n_samples))

            except Exception as e:
                logger.warning(f"Failed to process client shareable: {e}")
                continue

        if not client_deltas:
            logger.error("No valid client deltas, cannot perform FedOpt aggregation")
            return super().aggregate(shareable_list, fl_ctx)

        # Compute weighted average of deltas as pseudo-gradient
        # pseudo_gradient = sum(n_i * delta_i) / sum(n_i)
        pseudo_gradient = {}
        for name in self.previous_global_model.keys():
            weighted_sum = None
            for delta, n_samples in client_deltas:
                if name in delta:
                    contribution = n_samples * delta[name]
                    if weighted_sum is None:
                        weighted_sum = contribution
                    else:
                        weighted_sum += contribution

            if weighted_sum is not None:
                pseudo_gradient[name] = weighted_sum / total_samples
            else:
                # No updates for this parameter, use zero gradient
                pseudo_gradient[name] = np.zeros_like(self.previous_global_model[name])

        logger.info(f"Computed pseudo-gradient from {len(client_deltas)} clients")
        logger.debug(f"Total samples: {total_samples}")

        # Apply gradient clipping if configured
        if self.max_grad_norm is not None:
            total_norm = 0.0
            for name, grad in pseudo_gradient.items():
                total_norm += np.sum(grad ** 2)
            total_norm = np.sqrt(total_norm)

            if total_norm > self.max_grad_norm:
                clip_coef = self.max_grad_norm / (total_norm + 1e-8)
                for name in pseudo_gradient:
                    pseudo_gradient[name] = pseudo_gradient[name] * clip_coef
                logger.info(f"Clipped pseudo-gradient: norm {total_norm:.4f} -> {self.max_grad_norm}")

        # Apply server optimizer to update global model
        updated_global = self.server_optimizer.step(
            params=self.previous_global_model,
            pseudo_gradient=pseudo_gradient
        )

        logger.info("Applied server optimizer step")

        # Store updated global model for next round
        self.previous_global_model = updated_global

        # Create DXO with updated parameters
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=updated_global)

        # Create shareable from DXO
        result_shareable = dxo.to_shareable()

        logger.info("FedOpt aggregation completed successfully")
        return result_shareable
