"""
Custom NVFlare Controllers for Advanced Federated Learning

This module provides custom workflow controllers that extend NVFlare's
built-in aggregation strategies with advanced optimization techniques.

Available Controllers:
    - FedOptController: Server-side adaptive optimization (FedAdam, FedYogi, etc.)
    - More controllers can be added here as needed

Reference:
    Reddi et al. "Adaptive Federated Optimization" (ICLR 2021)
    https://arxiv.org/abs/2003.00295
"""

from .fedopt_controller import FedOptController

__all__ = ['FedOptController']
