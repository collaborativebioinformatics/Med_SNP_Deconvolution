"""
Aggregation Strategies for Federated Learning

Implements various aggregation methods for combining model weights from
multiple federated learning sites.

Strategies:
    - FedAvg: Weighted average based on sample counts
    - FedProx: FedAvg with proximal term (client-side)
    - FedOpt: Server-side optimization (Adam, AdaGrad, etc.)
    - Trimmed Mean: Robust aggregation removing outliers
    - Median: Robust aggregation using median
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
from dataclasses import dataclass

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from .model_shareable import deserialize_pytorch_weights, serialize_pytorch_weights

logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of aggregation with metadata"""
    aggregated_weights: Dict[str, Any]
    num_sites: int
    total_samples: int
    strategy: str
    metadata: Optional[Dict[str, Any]] = None


def federated_averaging(
    site_weights: List[Dict[str, Any]],
    site_sample_counts: List[int],
    normalize_by_samples: bool = True,
) -> AggregationResult:
    """
    FedAvg: Weighted average of model weights.

    Each site's contribution is weighted by the number of samples:
        w_global = Σ(n_i / N) * w_i

    where n_i is the number of samples at site i and N is total samples.

    Args:
        site_weights: List of serialized weights from each site
        site_sample_counts: Number of samples at each site
        normalize_by_samples: Weight by sample counts (True) or equal weight (False)

    Returns:
        AggregationResult with aggregated weights

    Raises:
        ValueError: If inputs are inconsistent

    Example:
        >>> weights = [site1_weights, site2_weights, site3_weights]
        >>> counts = [100, 150, 80]
        >>> result = federated_averaging(weights, counts)
        >>> global_weights = result.aggregated_weights
    """
    if len(site_weights) != len(site_sample_counts):
        raise ValueError(
            f"Mismatch: {len(site_weights)} weight sets, "
            f"{len(site_sample_counts)} sample counts"
        )

    if len(site_weights) == 0:
        raise ValueError("No weights provided for aggregation")

    # Determine model type from first site
    model_type = site_weights[0].get('format', 'unknown')

    if 'pytorch' in model_type:
        return _fedavg_pytorch(site_weights, site_sample_counts, normalize_by_samples)
    elif 'xgboost' in model_type:
        return _fedavg_xgboost(site_weights, site_sample_counts)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _fedavg_pytorch(
    site_weights: List[Dict[str, Any]],
    site_sample_counts: List[int],
    normalize_by_samples: bool,
) -> AggregationResult:
    """FedAvg for PyTorch models"""
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch required for PyTorch model aggregation")

    total_samples = sum(site_sample_counts)

    # Compute weights
    if normalize_by_samples:
        weights = [count / total_samples for count in site_sample_counts]
    else:
        weights = [1.0 / len(site_weights)] * len(site_weights)

    # Deserialize all state dicts
    state_dicts = []
    for site_weight in site_weights:
        state_dict = deserialize_pytorch_weights(site_weight, device='cpu')
        state_dicts.append(state_dict)

    # Weighted average
    global_state_dict = {}
    for key in state_dicts[0].keys():
        global_state_dict[key] = sum(
            weights[i] * state_dicts[i][key] for i in range(len(state_dicts))
        )

    # Serialize back
    aggregated = serialize_pytorch_weights(global_state_dict, include_metadata=True)

    # Preserve metadata from first site
    for key in ['num_snps', 'num_populations', 'model_type']:
        if key in site_weights[0]:
            aggregated[key] = site_weights[0][key]

    return AggregationResult(
        aggregated_weights=aggregated,
        num_sites=len(site_weights),
        total_samples=total_samples,
        strategy='fedavg',
        metadata={'weights': weights},
    )


def _fedavg_xgboost(
    site_weights: List[Dict[str, Any]],
    site_sample_counts: List[int],
) -> AggregationResult:
    """
    FedAvg for XGBoost models.

    For XGBoost, we use the model from the site with the most samples.
    True federated XGBoost would use histogram aggregation, which requires
    deeper integration with XGBoost internals.

    Alternative approaches:
        1. Ensemble of trees (requires custom implementation)
        2. Histogram aggregation (requires XGBoost plugin)
        3. Select best performing model
    """
    # Select model from site with most samples
    max_idx = np.argmax(site_sample_counts)
    selected_weights = site_weights[max_idx]

    logger.info(
        f"XGBoost aggregation: selected model from site {max_idx} "
        f"with {site_sample_counts[max_idx]} samples"
    )

    return AggregationResult(
        aggregated_weights=selected_weights,
        num_sites=len(site_weights),
        total_samples=sum(site_sample_counts),
        strategy='fedavg_select_best',
        metadata={
            'selected_site': max_idx,
            'site_samples': site_sample_counts,
        },
    )


def trimmed_mean_aggregation(
    site_weights: List[Dict[str, Any]],
    site_sample_counts: List[int],
    trim_ratio: float = 0.1,
) -> AggregationResult:
    """
    Trimmed mean aggregation for robust federated learning.

    Removes the highest and lowest values before averaging to reduce
    impact of outliers or malicious clients.

    Args:
        site_weights: List of serialized weights from each site
        site_sample_counts: Number of samples at each site
        trim_ratio: Proportion to trim from each end (0-0.5)

    Returns:
        AggregationResult with aggregated weights

    Raises:
        ValueError: If trim_ratio invalid or not enough sites

    Note:
        Only applicable to PyTorch models. For XGBoost, falls back to FedAvg.
    """
    if not (0 <= trim_ratio < 0.5):
        raise ValueError(f"trim_ratio must be in [0, 0.5), got {trim_ratio}")

    num_sites = len(site_weights)
    num_to_trim = int(num_sites * trim_ratio)

    if num_sites - 2 * num_to_trim < 1:
        raise ValueError(
            f"Not enough sites for trimming: {num_sites} sites, "
            f"trim_ratio={trim_ratio}"
        )

    model_type = site_weights[0].get('format', 'unknown')

    if 'pytorch' not in model_type:
        logger.warning(
            "Trimmed mean only supported for PyTorch, falling back to FedAvg"
        )
        return federated_averaging(site_weights, site_sample_counts)

    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch required for trimmed mean aggregation")

    # Deserialize all state dicts
    state_dicts = []
    for site_weight in site_weights:
        state_dict = deserialize_pytorch_weights(site_weight, device='cpu')
        state_dicts.append(state_dict)

    # Compute trimmed mean for each parameter
    global_state_dict = {}
    for key in state_dicts[0].keys():
        # Stack parameters from all sites
        param_stack = torch.stack([sd[key] for sd in state_dicts])

        # Sort along site dimension
        sorted_params, _ = torch.sort(param_stack, dim=0)

        # Trim and average
        if num_to_trim > 0:
            trimmed_params = sorted_params[num_to_trim:-num_to_trim]
        else:
            trimmed_params = sorted_params

        global_state_dict[key] = torch.mean(trimmed_params, dim=0)

    # Serialize
    aggregated = serialize_pytorch_weights(global_state_dict, include_metadata=True)

    # Preserve metadata
    for key in ['num_snps', 'num_populations', 'model_type']:
        if key in site_weights[0]:
            aggregated[key] = site_weights[0][key]

    return AggregationResult(
        aggregated_weights=aggregated,
        num_sites=num_sites,
        total_samples=sum(site_sample_counts),
        strategy='trimmed_mean',
        metadata={
            'trim_ratio': trim_ratio,
            'sites_trimmed': num_to_trim,
            'sites_used': num_sites - 2 * num_to_trim,
        },
    )


def median_aggregation(
    site_weights: List[Dict[str, Any]],
    site_sample_counts: List[int],
) -> AggregationResult:
    """
    Median aggregation for highly robust federated learning.

    Uses element-wise median instead of mean. Very robust to outliers
    but less efficient in benign settings.

    Args:
        site_weights: List of serialized weights from each site
        site_sample_counts: Number of samples at each site

    Returns:
        AggregationResult with aggregated weights

    Note:
        Only applicable to PyTorch models.
    """
    model_type = site_weights[0].get('format', 'unknown')

    if 'pytorch' not in model_type:
        logger.warning("Median only supported for PyTorch, falling back to FedAvg")
        return federated_averaging(site_weights, site_sample_counts)

    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch required for median aggregation")

    # Deserialize all state dicts
    state_dicts = []
    for site_weight in site_weights:
        state_dict = deserialize_pytorch_weights(site_weight, device='cpu')
        state_dicts.append(state_dict)

    # Compute median for each parameter
    global_state_dict = {}
    for key in state_dicts[0].keys():
        param_stack = torch.stack([sd[key] for sd in state_dicts])
        global_state_dict[key] = torch.median(param_stack, dim=0).values

    # Serialize
    aggregated = serialize_pytorch_weights(global_state_dict, include_metadata=True)

    # Preserve metadata
    for key in ['num_snps', 'num_populations', 'model_type']:
        if key in site_weights[0]:
            aggregated[key] = site_weights[0][key]

    return AggregationResult(
        aggregated_weights=aggregated,
        num_sites=len(site_weights),
        total_samples=sum(site_sample_counts),
        strategy='median',
    )


class FedOptAggregator:
    """
    Server-side optimizer for federated learning (FedOpt).

    Instead of simple averaging, applies an optimizer on the server:
        w_{t+1} = w_t - η * Δw

    where Δw is the pseudo-gradient from aggregating client updates.

    Supports:
        - FedAdam: Server-side Adam optimizer
        - FedAdagrad: Server-side AdaGrad
        - FedYogi: Server-side Yogi optimizer
    """

    def __init__(
        self,
        optimizer_type: str = 'adam',
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """
        Initialize FedOpt aggregator.

        Args:
            optimizer_type: 'adam', 'adagrad', or 'yogi'
            learning_rate: Server learning rate
            beta1: First moment decay (Adam/Yogi)
            beta2: Second moment decay (Adam/Yogi)
            epsilon: Numerical stability constant
        """
        self.optimizer_type = optimizer_type.lower()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Server-side optimizer state
        self.m_t: Optional[Dict[str, Any]] = None  # First moment
        self.v_t: Optional[Dict[str, Any]] = None  # Second moment
        self.t: int = 0  # Time step

        logger.info(
            f"Initialized FedOpt: {optimizer_type}, lr={learning_rate}"
        )

    def aggregate(
        self,
        global_weights: Dict[str, Any],
        site_weights: List[Dict[str, Any]],
        site_sample_counts: List[int],
    ) -> AggregationResult:
        """
        Aggregate using server-side optimizer.

        Args:
            global_weights: Current global model weights
            site_weights: Updated weights from each site
            site_sample_counts: Number of samples at each site

        Returns:
            AggregationResult with optimized global weights
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch required for FedOpt")

        self.t += 1

        # Deserialize global weights
        global_state = deserialize_pytorch_weights(global_weights, device='cpu')

        # Compute weighted average of client deltas
        total_samples = sum(site_sample_counts)
        site_states = [
            deserialize_pytorch_weights(w, device='cpu') for w in site_weights
        ]

        # Compute pseudo-gradient: Δw = w_global - avg(w_clients)
        avg_state = {}
        for key in global_state.keys():
            avg_state[key] = sum(
                (site_sample_counts[i] / total_samples) * site_states[i][key]
                for i in range(len(site_states))
            )

        delta = {key: global_state[key] - avg_state[key] for key in global_state.keys()}

        # Initialize optimizer state if needed
        if self.m_t is None:
            self.m_t = {key: torch.zeros_like(val) for key, val in delta.items()}
            self.v_t = {key: torch.zeros_like(val) for key, val in delta.items()}

        # Apply optimizer
        if self.optimizer_type == 'adam':
            new_global_state = self._apply_adam(global_state, delta)
        elif self.optimizer_type == 'adagrad':
            new_global_state = self._apply_adagrad(global_state, delta)
        elif self.optimizer_type == 'yogi':
            new_global_state = self._apply_yogi(global_state, delta)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        # Serialize
        aggregated = serialize_pytorch_weights(new_global_state, include_metadata=True)

        # Preserve metadata
        for key in ['num_snps', 'num_populations', 'model_type']:
            if key in global_weights:
                aggregated[key] = global_weights[key]

        return AggregationResult(
            aggregated_weights=aggregated,
            num_sites=len(site_weights),
            total_samples=total_samples,
            strategy=f'fedopt_{self.optimizer_type}',
            metadata={'timestep': self.t},
        )

    def _apply_adam(
        self,
        weights: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply Adam optimizer"""
        new_weights = {}

        for key in weights.keys():
            # Update biased first moment estimate
            self.m_t[key] = self.beta1 * self.m_t[key] + (1 - self.beta1) * delta[key]

            # Update biased second moment estimate
            self.v_t[key] = self.beta2 * self.v_t[key] + (1 - self.beta2) * (delta[key] ** 2)

            # Bias correction
            m_hat = self.m_t[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v_t[key] / (1 - self.beta2 ** self.t)

            # Update weights
            new_weights[key] = weights[key] - self.learning_rate * m_hat / (
                torch.sqrt(v_hat) + self.epsilon
            )

        return new_weights

    def _apply_adagrad(
        self,
        weights: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply AdaGrad optimizer"""
        new_weights = {}

        for key in weights.keys():
            # Accumulate squared gradients
            self.v_t[key] = self.v_t[key] + delta[key] ** 2

            # Update weights
            new_weights[key] = weights[key] - self.learning_rate * delta[key] / (
                torch.sqrt(self.v_t[key]) + self.epsilon
            )

        return new_weights

    def _apply_yogi(
        self,
        weights: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Apply Yogi optimizer"""
        new_weights = {}

        for key in weights.keys():
            # Update first moment
            self.m_t[key] = self.beta1 * self.m_t[key] + (1 - self.beta1) * delta[key]

            # Update second moment (Yogi-specific)
            self.v_t[key] = self.v_t[key] - (1 - self.beta2) * torch.sign(
                self.v_t[key] - delta[key] ** 2
            ) * (delta[key] ** 2)

            # Update weights
            new_weights[key] = weights[key] - self.learning_rate * self.m_t[key] / (
                torch.sqrt(self.v_t[key]) + self.epsilon
            )

        return new_weights


# Convenience function for common aggregation strategies
def aggregate_weights(
    site_weights: List[Dict[str, Any]],
    site_sample_counts: List[int],
    strategy: str = 'fedavg',
    **kwargs
) -> AggregationResult:
    """
    Aggregate weights using specified strategy.

    Args:
        site_weights: List of serialized weights
        site_sample_counts: Sample counts per site
        strategy: Aggregation strategy name
        **kwargs: Strategy-specific parameters

    Returns:
        AggregationResult

    Strategies:
        - 'fedavg': FedAvg (default)
        - 'trimmed_mean': Trimmed mean (trim_ratio parameter)
        - 'median': Element-wise median

    Example:
        >>> result = aggregate_weights(
        ...     site_weights=[w1, w2, w3],
        ...     site_sample_counts=[100, 150, 80],
        ...     strategy='trimmed_mean',
        ...     trim_ratio=0.1
        ... )
    """
    strategy = strategy.lower()

    if strategy == 'fedavg':
        return federated_averaging(site_weights, site_sample_counts)
    elif strategy == 'trimmed_mean':
        trim_ratio = kwargs.get('trim_ratio', 0.1)
        return trimmed_mean_aggregation(site_weights, site_sample_counts, trim_ratio)
    elif strategy == 'median':
        return median_aggregation(site_weights, site_sample_counts)
    else:
        raise ValueError(
            f"Unknown aggregation strategy: {strategy}. "
            f"Use 'fedavg', 'trimmed_mean', or 'median'"
        )
