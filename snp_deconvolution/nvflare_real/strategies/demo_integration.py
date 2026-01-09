#!/usr/bin/env python3
"""
Strategy Registry Integration Demo

This script demonstrates the complete integration between the strategy
registry and the job.py configuration system. It shows how different
strategies can be configured and what arguments they generate.

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
    get_client_script_args,
    print_strategy_info,
)


def demo_strategy_overview():
    """Demo: Overview of all strategies."""
    print("\n" + "="*80)
    print("STRATEGY REGISTRY OVERVIEW")
    print("="*80)

    strategies = list_strategies()
    print(f"\nAvailable strategies: {len(strategies)}")

    for strategy in strategies:
        metadata = get_strategy_metadata(strategy)
        print(f"\n{metadata.display_name} ({strategy}):")
        print(f"  Description: {metadata.description}")
        if metadata.parameters:
            print(f"  Parameters: {', '.join(metadata.parameters.keys())}")
        else:
            print(f"  Parameters: None (uses default configuration)")


def demo_job_configurations():
    """Demo: Show job.py command-line examples for each strategy."""
    print("\n" + "="*80)
    print("JOB CONFIGURATION EXAMPLES")
    print("="*80)

    base_cmd = "python job.py --mode poc --num_rounds 10 --num_clients 3"

    # FedAvg
    print("\n1. FedAvg (Federated Averaging - Baseline):")
    print(f"   {base_cmd} --strategy fedavg")
    print("   Use case: IID data, baseline performance")

    # FedProx
    print("\n2. FedProx (Federated Proximal):")
    print(f"   {base_cmd} --strategy fedprox --mu 0.01")
    print("   Use case: Non-IID data, data heterogeneity")
    print("   Key parameter: mu (proximal term strength)")

    # Scaffold
    print("\n3. Scaffold (Stochastic Controlled Averaging):")
    print(f"   {base_cmd} --strategy scaffold")
    print("   Use case: Variance reduction, heterogeneous settings")

    # FedOpt variations
    print("\n4. FedOpt (Server-side Adaptive Optimization):")

    print("\n   a) FedAdam (recommended for most cases):")
    print(f"      {base_cmd} --strategy fedopt \\")
    print("          --server_optimizer adam --server_lr 0.01 \\")
    print("          --beta1 0.9 --beta2 0.999")
    print("      Use case: Fast convergence, adaptive learning rates")

    print("\n   b) FedYogi (more stable than Adam):")
    print(f"      {base_cmd} --strategy fedopt \\")
    print("          --server_optimizer yogi --server_lr 0.01 \\")
    print("          --beta1 0.9 --beta2 0.999")
    print("      Use case: Improved stability, federated settings")

    print("\n   c) FedAdaGrad (simple adaptive):")
    print(f"      {base_cmd} --strategy fedopt \\")
    print("          --server_optimizer adagrad --server_lr 0.01")
    print("      Use case: Sparse gradients, simple adaptation")

    print("\n   d) FedSGDM (momentum-based):")
    print(f"      {base_cmd} --strategy fedopt \\")
    print("          --server_optimizer sgdm --server_lr 0.01 \\")
    print("          --momentum 0.9")
    print("      Use case: Accelerated convergence with momentum")


def demo_client_arguments():
    """Demo: Show client script arguments for each strategy."""
    print("\n" + "="*80)
    print("CLIENT SCRIPT ARGUMENTS")
    print("="*80)

    print("\nThese arguments are automatically added to client scripts:")

    # FedAvg
    args = get_client_script_args('fedavg')
    print(f"\nFedAvg: {args}")
    print("  Note: No additional client-side parameters needed")

    # FedProx
    args = get_client_script_args('fedprox', mu=0.01)
    print(f"\nFedProx: {args}")
    print("  Note: mu parameter controls proximal regularization on client")

    # Scaffold
    args = get_client_script_args('scaffold')
    print(f"\nScaffold: {args}")
    print("  Note: Control variates managed by NVFlare controller")

    # FedOpt
    args = get_client_script_args('fedopt', server_optimizer='adam')
    print(f"\nFedOpt: {args}")
    print("  Note: Server-side optimization, no client-side changes needed")


def demo_parameter_recommendations():
    """Demo: Parameter tuning recommendations."""
    print("\n" + "="*80)
    print("PARAMETER TUNING RECOMMENDATIONS")
    print("="*80)

    recommendations = [
        {
            'strategy': 'FedProx',
            'parameter': 'mu',
            'recommendations': [
                ('0.001 - 0.01', 'Mildly heterogeneous data'),
                ('0.01 - 0.1', 'Moderately heterogeneous data'),
                ('0.1 - 1.0', 'Highly heterogeneous data'),
            ]
        },
        {
            'strategy': 'FedOpt (All)',
            'parameter': 'server_lr',
            'recommendations': [
                ('0.001 - 0.01', 'Conservative, stable convergence'),
                ('0.01 - 0.1', 'Standard, balanced performance'),
                ('0.1 - 1.0', 'Aggressive, faster but risky'),
            ]
        },
        {
            'strategy': 'FedAdam/FedYogi',
            'parameter': 'beta1 (momentum)',
            'recommendations': [
                ('0.9', 'Standard momentum (default)'),
                ('0.8', 'Less momentum, more responsive'),
                ('0.95', 'More momentum, smoother updates'),
            ]
        },
        {
            'strategy': 'FedAdam/FedYogi',
            'parameter': 'beta2 (variance)',
            'recommendations': [
                ('0.999', 'Standard variance tracking (default)'),
                ('0.99', 'More responsive to recent gradients'),
                ('0.9999', 'More stable, long-term averaging'),
            ]
        },
    ]

    for rec in recommendations:
        print(f"\n{rec['strategy']} - {rec['parameter']}:")
        for value_range, description in rec['recommendations']:
            print(f"  {value_range:15s} : {description}")


def demo_strategy_comparison():
    """Demo: When to use each strategy."""
    print("\n" + "="*80)
    print("STRATEGY SELECTION GUIDE")
    print("="*80)

    comparison = [
        {
            'scenario': 'IID data (balanced distribution)',
            'recommendation': 'FedAvg',
            'reason': 'Simple averaging works well with IID data'
        },
        {
            'scenario': 'Non-IID data (heterogeneous)',
            'recommendation': 'FedProx or Scaffold',
            'reason': 'Better handles client drift and data heterogeneity'
        },
        {
            'scenario': 'Need fast convergence',
            'recommendation': 'FedOpt (FedAdam or FedYogi)',
            'reason': 'Adaptive server optimization accelerates training'
        },
        {
            'scenario': 'Limited communication rounds',
            'recommendation': 'FedOpt (FedAdam)',
            'reason': 'More efficient use of each communication round'
        },
        {
            'scenario': 'Unstable training',
            'recommendation': 'FedProx with higher mu',
            'reason': 'Proximal term prevents excessive client drift'
        },
        {
            'scenario': 'High client variance',
            'recommendation': 'Scaffold',
            'reason': 'Control variates reduce variance in updates'
        },
    ]

    for item in comparison:
        print(f"\nScenario: {item['scenario']}")
        print(f"  Recommendation: {item['recommendation']}")
        print(f"  Reason: {item['reason']}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("FEDERATED LEARNING STRATEGY REGISTRY - INTEGRATION DEMO")
    print("="*80)

    demo_strategy_overview()
    demo_job_configurations()
    demo_client_arguments()
    demo_parameter_recommendations()
    demo_strategy_comparison()

    print("\n" + "="*80)
    print("ADDITIONAL RESOURCES")
    print("="*80)

    print("\nFor detailed strategy information:")
    print("  python -c \"from snp_deconvolution.nvflare_real.strategies import print_strategy_info; print_strategy_info('fedopt')\"")

    print("\nFor usage examples:")
    print("  python snp_deconvolution/nvflare_real/strategies/example_usage.py")

    print("\nFor unit tests:")
    print("  python snp_deconvolution/nvflare_real/strategies/test_registry.py")

    print("\nFor job.py help:")
    print("  python snp_deconvolution/nvflare_real/lightning/job.py --help")

    print("\n" + "="*80)
    print()


if __name__ == "__main__":
    main()
