"""
Example Usage of NVFlare Base Module

Demonstrates federated learning simulation with XGBoost and PyTorch executors.

This script simulates a federated learning scenario with multiple sites:
    - Site 1: 40% of data
    - Site 2: 35% of data
    - Site 3: 25% of data

Each site trains locally and shares model weights (not data) with the server.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Any
import logging

from base_executor import ExecutorMetrics
from xgb_nvflare_wrapper import XGBoostNVFlareExecutor
from dl_nvflare_wrapper import DLNVFlareExecutor
from model_shareable import (
    serialize_pytorch_weights,
    deserialize_pytorch_weights,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple PyTorch model for demonstration
class SimpleSNPNet(nn.Module):
    """Simple feedforward network for SNP classification"""

    def __init__(self, num_snps: int, num_populations: int):
        super().__init__()
        self.num_snps = num_snps
        self.num_populations = num_populations

        self.network = nn.Sequential(
            nn.Linear(num_snps, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_populations),
        )

    def forward(self, x):
        return self.network(x)


def generate_synthetic_snp_data(
    num_samples: int,
    num_snps: int,
    num_populations: int,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic SNP data for testing.

    Args:
        num_samples: Number of samples
        num_snps: Number of SNP features
        num_populations: Number of population classes
        seed: Random seed

    Returns:
        Tuple of (X, y) where X is genotype matrix and y is population labels
    """
    np.random.seed(seed)

    # Generate genotype matrix (0, 1, 2 for diploid)
    X = np.random.randint(0, 3, size=(num_samples, num_snps), dtype=np.int8)

    # Generate population labels with some structure
    # First 20% of SNPs are informative
    informative_snps = num_snps // 5
    population_signatures = np.random.randint(
        0, 3, size=(num_populations, informative_snps)
    )

    # Assign populations based on similarity to signatures
    y = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        sample_signature = X[i, :informative_snps]
        distances = np.sum(
            (population_signatures - sample_signature) ** 2, axis=1
        )
        y[i] = np.argmin(distances)

    # Add some noise
    noise_idx = np.random.choice(num_samples, size=num_samples // 10)
    y[noise_idx] = np.random.randint(0, num_populations, size=len(noise_idx))

    return X.astype(np.float32), y


def split_data_across_sites(
    X: np.ndarray,
    y: np.ndarray,
    num_sites: int = 3,
    split_ratios: List[float] = None
) -> List[tuple]:
    """
    Split data across federated learning sites.

    Args:
        X: Full dataset features
        y: Full dataset labels
        num_sites: Number of sites
        split_ratios: Proportion of data for each site (default: equal)

    Returns:
        List of (X_site, y_site) tuples
    """
    if split_ratios is None:
        split_ratios = [1.0 / num_sites] * num_sites

    assert len(split_ratios) == num_sites
    assert abs(sum(split_ratios) - 1.0) < 1e-6

    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)

    site_data = []
    start_idx = 0
    for ratio in split_ratios:
        end_idx = start_idx + int(num_samples * ratio)
        site_indices = indices[start_idx:end_idx]
        site_data.append((X[site_indices], y[site_indices]))
        start_idx = end_idx

    return site_data


def federated_averaging_pytorch(
    site_weights: List[Dict[str, Any]],
    site_sample_counts: List[int]
) -> Dict[str, Any]:
    """
    Perform FedAvg aggregation on PyTorch model weights.

    Weights are averaged proportional to the number of samples at each site.

    Args:
        site_weights: List of weight dictionaries from each site
        site_sample_counts: Number of samples at each site

    Returns:
        Aggregated weights dictionary
    """
    total_samples = sum(site_sample_counts)

    # Deserialize all weights
    state_dicts = []
    for weights in site_weights:
        state_dict = deserialize_pytorch_weights(weights, device='cpu')
        state_dicts.append(state_dict)

    # Weighted average
    global_state_dict = {}
    for key in state_dicts[0].keys():
        global_state_dict[key] = sum(
            state_dicts[i][key] * (site_sample_counts[i] / total_samples)
            for i in range(len(state_dicts))
        )

    # Serialize back
    aggregated = serialize_pytorch_weights(global_state_dict)
    return aggregated


def federated_averaging_xgboost(
    site_weights: List[Dict[str, Any]],
    site_sample_counts: List[int]
) -> Dict[str, Any]:
    """
    Aggregate XGBoost models (simple approach: return best performing).

    Note: True XGBoost federated learning would use histogram aggregation,
    but this requires deeper integration with XGBoost internals.

    Args:
        site_weights: List of weight dictionaries from each site
        site_sample_counts: Number of samples at each site

    Returns:
        Selected model weights
    """
    # For simplicity, return the model from the site with most data
    # In practice, would create an ensemble or use histogram aggregation
    max_samples_idx = np.argmax(site_sample_counts)
    logger.info(
        f"XGBoost aggregation: selecting model from site {max_samples_idx} "
        f"({site_sample_counts[max_samples_idx]} samples)"
    )
    return site_weights[max_samples_idx]


def simulate_federated_learning_pytorch(
    num_rounds: int = 5,
    num_sites: int = 3,
    num_snps: int = 1000,
    num_populations: int = 5,
    total_samples: int = 1000,
):
    """Simulate federated learning with PyTorch"""
    logger.info("=" * 60)
    logger.info("PyTorch Federated Learning Simulation")
    logger.info("=" * 60)

    # Generate synthetic data
    X, y = generate_synthetic_snp_data(total_samples, num_snps, num_populations)

    # Split across sites
    site_data = split_data_across_sites(
        X, y, num_sites, split_ratios=[0.4, 0.35, 0.25]
    )

    # Initialize executors for each site
    executors = []
    for i in range(num_sites):
        model = SimpleSNPNet(num_snps, num_populations)
        executor = DLNVFlareExecutor(
            model=model,
            aggregation_strategy='fedavg',
        )

        # Prepare data loaders
        X_site, y_site = site_data[i]
        X_tensor = torch.FloatTensor(X_site)
        y_tensor = torch.LongTensor(y_site)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        executor.set_data_loaders(loader, loader)  # Use same for train/val
        executors.append(executor)

        logger.info(f"Site {i}: {len(X_site)} samples")

    # Federated learning rounds
    for round_num in range(num_rounds):
        logger.info(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        site_weights = []
        site_sample_counts = []

        # Each site trains locally
        for i, executor in enumerate(executors):
            logger.info(f"\nSite {i} training...")
            metrics = executor.local_train(num_epochs=3)
            logger.info(
                f"Site {i}: loss={metrics.loss:.4f}, "
                f"accuracy={metrics.accuracy:.4f}"
            )

            # Export weights
            weights = executor.get_model_weights()
            site_weights.append(weights)
            site_sample_counts.append(metrics.num_samples)

        # Server aggregates weights
        logger.info("\nAggregating weights at server...")
        global_weights = federated_averaging_pytorch(
            site_weights, site_sample_counts
        )

        # Distribute global weights to all sites
        for executor in executors:
            executor.set_model_weights(global_weights)
            executor.increment_round()

    # Final validation
    logger.info("\n" + "=" * 60)
    logger.info("Final Validation Results")
    logger.info("=" * 60)
    for i, executor in enumerate(executors):
        val_metrics = executor.validate()
        logger.info(
            f"Site {i}: loss={val_metrics.loss:.4f}, "
            f"accuracy={val_metrics.accuracy:.4f}"
        )


def simulate_federated_learning_xgboost(
    num_rounds: int = 3,
    num_sites: int = 3,
    num_snps: int = 1000,
    num_populations: int = 5,
    total_samples: int = 1000,
):
    """Simulate federated learning with XGBoost"""
    logger.info("=" * 60)
    logger.info("XGBoost Federated Learning Simulation")
    logger.info("=" * 60)

    # Generate synthetic data
    X, y = generate_synthetic_snp_data(total_samples, num_snps, num_populations)

    # Split across sites
    site_data = split_data_across_sites(
        X, y, num_sites, split_ratios=[0.4, 0.35, 0.25]
    )

    # Initialize executors for each site
    executors = []
    for i in range(num_sites):
        executor = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=num_snps,
            num_populations=num_populations,
        )

        # Prepare data
        X_site, y_site = site_data[i]
        executor.prepare_data(X_site, y_site)
        executor.prepare_data(X_site, y_site, is_validation=True)

        executors.append(executor)
        logger.info(f"Site {i}: {len(X_site)} samples")

    # Federated learning rounds
    for round_num in range(num_rounds):
        logger.info(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        site_weights = []
        site_sample_counts = []

        # Each site trains locally
        for i, executor in enumerate(executors):
            logger.info(f"\nSite {i} training...")
            metrics = executor.local_train(num_epochs=10)
            logger.info(
                f"Site {i}: loss={metrics.loss:.4f}, "
                f"accuracy={metrics.accuracy:.4f}"
            )

            # Export weights
            weights = executor.get_model_weights()
            site_weights.append(weights)
            site_sample_counts.append(metrics.num_samples)

        # Server aggregates weights (simple: select best)
        logger.info("\nAggregating models at server...")
        global_weights = federated_averaging_xgboost(
            site_weights, site_sample_counts
        )

        # Distribute global weights to all sites
        for executor in executors:
            executor.set_model_weights(global_weights)
            executor.increment_round()

    # Final validation and feature importance
    logger.info("\n" + "=" * 60)
    logger.info("Final Validation Results")
    logger.info("=" * 60)
    for i, executor in enumerate(executors):
        val_metrics = executor.validate()
        logger.info(
            f"Site {i}: loss={val_metrics.loss:.4f}, "
            f"accuracy={val_metrics.accuracy:.4f}"
        )

        # Feature importance
        importance = executor.get_feature_importance()
        top_snps = sorted(
            importance.items(), key=lambda x: x[1], reverse=True
        )[:5]
        logger.info(f"Site {i} top 5 SNPs: {top_snps}")


if __name__ == '__main__':
    # Run simulations
    try:
        simulate_federated_learning_pytorch(
            num_rounds=3,
            num_sites=3,
            num_snps=100,
            num_populations=3,
            total_samples=300,
        )
    except Exception as e:
        logger.error(f"PyTorch simulation failed: {e}")

    print("\n" * 2)

    try:
        simulate_federated_learning_xgboost(
            num_rounds=3,
            num_sites=3,
            num_snps=100,
            num_populations=3,
            total_samples=300,
        )
    except Exception as e:
        logger.error(f"XGBoost simulation failed: {e}")
