#!/usr/bin/env python3
"""
Unit tests for Strategy Registry

Tests the core functionality of the strategy registry including:
- Strategy listing
- Metadata retrieval
- Parameter validation
- Client script argument generation

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
    validate_strategy_parameters,
    get_client_script_args,
)


def test_list_strategies():
    """Test listing available strategies."""
    strategies = list_strategies()
    assert len(strategies) == 4, f"Expected 4 strategies, got {len(strategies)}"
    assert 'fedavg' in strategies
    assert 'fedprox' in strategies
    assert 'scaffold' in strategies
    assert 'fedopt' in strategies
    print("✓ test_list_strategies passed")


def test_get_strategy_metadata():
    """Test getting strategy metadata."""
    # Test FedAvg
    metadata = get_strategy_metadata('fedavg')
    assert metadata.name == 'fedavg'
    assert metadata.display_name == 'FedAvg'
    assert len(metadata.parameters) == 0
    print("✓ test_get_strategy_metadata (FedAvg) passed")

    # Test FedProx
    metadata = get_strategy_metadata('fedprox')
    assert metadata.name == 'fedprox'
    assert 'mu' in metadata.parameters
    print("✓ test_get_strategy_metadata (FedProx) passed")

    # Test FedOpt
    metadata = get_strategy_metadata('fedopt')
    assert metadata.name == 'fedopt'
    assert 'server_optimizer' in metadata.parameters
    assert 'server_lr' in metadata.parameters
    print("✓ test_get_strategy_metadata (FedOpt) passed")


def test_get_metadata_invalid_strategy():
    """Test getting metadata for invalid strategy."""
    try:
        get_strategy_metadata('invalid_strategy')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Unknown strategy' in str(e)
        print("✓ test_get_metadata_invalid_strategy passed")


def test_validate_parameters_fedavg():
    """Test parameter validation for FedAvg."""
    params = validate_strategy_parameters('fedavg')
    assert len(params) == 0, f"FedAvg should have no parameters, got {params}"
    print("✓ test_validate_parameters_fedavg passed")


def test_validate_parameters_fedprox():
    """Test parameter validation for FedProx."""
    # Test with default mu
    params = validate_strategy_parameters('fedprox')
    assert 'mu' in params
    assert params['mu'] == 0.01
    print("✓ test_validate_parameters_fedprox (default) passed")

    # Test with custom mu
    params = validate_strategy_parameters('fedprox', mu=0.05)
    assert params['mu'] == 0.05
    print("✓ test_validate_parameters_fedprox (custom) passed")


def test_validate_parameters_fedopt():
    """Test parameter validation for FedOpt."""
    # Test with defaults
    params = validate_strategy_parameters('fedopt')
    assert params['server_optimizer'] == 'adam'
    assert params['server_lr'] == 0.01
    assert params['beta1'] == 0.9
    assert params['beta2'] == 0.999
    print("✓ test_validate_parameters_fedopt (defaults) passed")

    # Test with custom values
    params = validate_strategy_parameters(
        'fedopt',
        server_optimizer='sgdm',
        server_lr=0.02,
        momentum=0.95
    )
    assert params['server_optimizer'] == 'sgdm'
    assert params['server_lr'] == 0.02
    assert params['momentum'] == 0.95
    print("✓ test_validate_parameters_fedopt (custom) passed")


def test_validate_parameters_invalid_optimizer():
    """Test parameter validation with invalid optimizer."""
    try:
        validate_strategy_parameters('fedopt', server_optimizer='invalid')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Invalid value for server_optimizer' in str(e)
        print("✓ test_validate_parameters_invalid_optimizer passed")


def test_get_client_script_args_fedavg():
    """Test client script args for FedAvg."""
    args = get_client_script_args('fedavg')
    assert args == '--strategy fedavg'
    print("✓ test_get_client_script_args_fedavg passed")


def test_get_client_script_args_fedprox():
    """Test client script args for FedProx."""
    args = get_client_script_args('fedprox', mu=0.05)
    assert '--strategy fedprox' in args
    assert '--mu 0.05' in args
    print("✓ test_get_client_script_args_fedprox passed")


def test_get_client_script_args_fedopt():
    """Test client script args for FedOpt."""
    args = get_client_script_args('fedopt', server_optimizer='adam')
    assert args == '--strategy fedopt'
    print("✓ test_get_client_script_args_fedopt passed")


def test_get_client_script_args_scaffold():
    """Test client script args for Scaffold."""
    args = get_client_script_args('scaffold')
    assert args == '--strategy scaffold'
    print("✓ test_get_client_script_args_scaffold passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RUNNING STRATEGY REGISTRY TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_list_strategies,
        test_get_strategy_metadata,
        test_get_metadata_invalid_strategy,
        test_validate_parameters_fedavg,
        test_validate_parameters_fedprox,
        test_validate_parameters_fedopt,
        test_validate_parameters_invalid_optimizer,
        test_get_client_script_args_fedavg,
        test_get_client_script_args_fedprox,
        test_get_client_script_args_fedopt,
        test_get_client_script_args_scaffold,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
