#!/usr/bin/env python3
"""
Unit Tests for FedOpt Controller

Tests the FedOpt controller and server-side optimizers to ensure correct
implementation of adaptive federated optimization algorithms.

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

import unittest
import numpy as np
from typing import Dict

from fedopt_controller import (
    ServerOptimizer,
    ServerSGDM,
    ServerAdam,
    ServerAdaGrad,
    ServerYogi,
    FedOptController
)


class TestServerOptimizers(unittest.TestCase):
    """Test suite for server-side optimizers."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple test parameters
        self.params = {
            'w1': np.array([1.0, 2.0, 3.0]),
            'w2': np.array([[1.0, 2.0], [3.0, 4.0]])
        }

        # Simple pseudo-gradient
        self.gradient = {
            'w1': np.array([0.1, 0.1, 0.1]),
            'w2': np.array([[0.1, 0.1], [0.1, 0.1]])
        }

    def test_sgdm_basic_step(self):
        """Test SGD with momentum basic update."""
        optimizer = ServerSGDM(lr=0.1, momentum=0.9)

        # First step - no momentum yet
        updated = optimizer.step(self.params, self.gradient)

        # Expected: w = w - lr * g (since velocity starts at 0)
        expected_w1 = self.params['w1'] - 0.1 * self.gradient['w1']
        np.testing.assert_allclose(updated['w1'], expected_w1, rtol=1e-5)

        # Second step - with momentum
        updated2 = optimizer.step(updated, self.gradient)

        # Velocity should have accumulated
        # v = 0.9 * g + g = 1.9 * g
        expected_velocity = 1.9 * self.gradient['w1']
        expected_w1_2 = updated['w1'] - 0.1 * expected_velocity
        np.testing.assert_allclose(updated2['w1'], expected_w1_2, rtol=1e-5)

    def test_sgdm_zero_momentum(self):
        """Test SGD with zero momentum (equivalent to vanilla SGD)."""
        optimizer = ServerSGDM(lr=0.1, momentum=0.0)

        updated = optimizer.step(self.params, self.gradient)

        # Should be equivalent to vanilla SGD
        expected_w1 = self.params['w1'] - 0.1 * self.gradient['w1']
        np.testing.assert_allclose(updated['w1'], expected_w1, rtol=1e-5)

    def test_adam_basic_step(self):
        """Test Adam optimizer basic update."""
        optimizer = ServerAdam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        updated = optimizer.step(self.params, self.gradient)

        # First step should have bias correction
        # m = (1 - beta1) * g = 0.1 * g
        # v = (1 - beta2) * g^2 = 0.001 * g^2
        # m_hat = m / (1 - 0.9^1) = m / 0.1 = g
        # v_hat = v / (1 - 0.999^1) = v / 0.001 = g^2
        # w = w - lr * m_hat / (sqrt(v_hat) + eps)

        # Verify parameters have been updated
        self.assertFalse(np.allclose(updated['w1'], self.params['w1']))
        self.assertFalse(np.allclose(updated['w2'], self.params['w2']))

        # Verify optimizer state
        self.assertEqual(optimizer.state['step'], 1)
        self.assertIn('w1', optimizer.state['first_moment'])
        self.assertIn('w1', optimizer.state['second_moment'])

    def test_adam_convergence(self):
        """Test Adam convergence on simple quadratic."""
        # Minimize f(w) = w^2, optimal w = 0
        optimizer = ServerAdam(lr=0.1, beta1=0.9, beta2=0.999)

        w = {'x': np.array([10.0])}

        for _ in range(100):
            # Gradient of w^2 is 2*w
            grad = {'x': 2 * w['x']}
            w = optimizer.step(w, grad)

        # Should converge close to 0
        self.assertLess(np.abs(w['x'][0]), 0.5)

    def test_adagrad_basic_step(self):
        """Test AdaGrad optimizer basic update."""
        optimizer = ServerAdaGrad(lr=0.1, epsilon=1e-8)

        updated = optimizer.step(self.params, self.gradient)

        # First step: G = g^2, w = w - lr * g / (sqrt(g^2) + eps)
        # Should be approximately: w = w - lr * g / |g|

        # Verify parameters have been updated
        self.assertFalse(np.allclose(updated['w1'], self.params['w1']))

        # Verify accumulated gradients
        self.assertIn('w1', optimizer.state['sum_squared_gradients'])

    def test_adagrad_decreasing_lr(self):
        """Test that AdaGrad effectively decreases learning rate over time."""
        optimizer = ServerAdaGrad(lr=1.0, epsilon=1e-8)

        w = {'x': np.array([10.0])}
        grad = {'x': np.array([1.0])}

        # First step
        w1 = optimizer.step(w, grad)
        step1_change = np.abs(w1['x'] - w['x'])

        # Second step with same gradient
        w2 = optimizer.step(w1, grad)
        step2_change = np.abs(w2['x'] - w1['x'])

        # Second step should have smaller change (adaptive LR)
        self.assertLess(step2_change[0], step1_change[0])

    def test_yogi_basic_step(self):
        """Test Yogi optimizer basic update."""
        optimizer = ServerYogi(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        updated = optimizer.step(self.params, self.gradient)

        # Verify parameters have been updated
        self.assertFalse(np.allclose(updated['w1'], self.params['w1']))

        # Verify optimizer state
        self.assertEqual(optimizer.state['step'], 1)
        self.assertIn('w1', optimizer.state['first_moment'])
        self.assertIn('w1', optimizer.state['second_moment'])

    def test_yogi_vs_adam(self):
        """Test that Yogi differs from Adam in second moment update."""
        adam = ServerAdam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
        yogi = ServerYogi(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

        params = {'x': np.array([1.0])}
        grad = {'x': np.array([0.5])}

        # Take several steps
        params_adam = params.copy()
        params_yogi = params.copy()

        for _ in range(10):
            params_adam = adam.step(params_adam, grad)
            params_yogi = yogi.step(params_yogi, grad)

        # Results should differ due to different second moment updates
        self.assertFalse(np.allclose(params_adam['x'], params_yogi['x']))

    def test_optimizer_state_reset(self):
        """Test optimizer state reset."""
        optimizer = ServerAdam(lr=0.01)

        # Take a step
        optimizer.step(self.params, self.gradient)
        self.assertEqual(optimizer.state['step'], 1)

        # Reset state
        optimizer.reset_state()
        self.assertEqual(len(optimizer.state), 0)

    def test_missing_gradient_params(self):
        """Test handling of parameters without gradients."""
        optimizer = ServerAdam(lr=0.01)

        # Gradient missing 'w2'
        partial_gradient = {'w1': self.gradient['w1']}

        updated = optimizer.step(self.params, partial_gradient)

        # w1 should be updated
        self.assertFalse(np.allclose(updated['w1'], self.params['w1']))

        # w2 should remain unchanged
        np.testing.assert_array_equal(updated['w2'], self.params['w2'])

    def test_zero_gradient(self):
        """Test optimizer behavior with zero gradient."""
        optimizer = ServerAdam(lr=0.01)

        zero_gradient = {
            'w1': np.zeros_like(self.params['w1']),
            'w2': np.zeros_like(self.params['w2'])
        }

        updated = optimizer.step(self.params, zero_gradient)

        # Parameters should be approximately unchanged (small numerical changes ok)
        np.testing.assert_allclose(updated['w1'], self.params['w1'], rtol=1e-5)
        np.testing.assert_allclose(updated['w2'], self.params['w2'], rtol=1e-5)

    def test_large_gradient(self):
        """Test optimizer stability with large gradients."""
        optimizer = ServerAdam(lr=0.01, epsilon=1e-8)

        large_gradient = {
            'w1': np.array([100.0, 100.0, 100.0]),
            'w2': np.array([[100.0, 100.0], [100.0, 100.0]])
        }

        # Should not crash or produce inf/nan
        updated = optimizer.step(self.params, large_gradient)

        self.assertTrue(np.all(np.isfinite(updated['w1'])))
        self.assertTrue(np.all(np.isfinite(updated['w2'])))


class TestFedOptController(unittest.TestCase):
    """Test suite for FedOpt controller integration."""

    def test_controller_initialization(self):
        """Test FedOpt controller initialization with different optimizers."""
        # Test Adam
        controller = FedOptController(
            num_clients=3,
            num_rounds=10,
            optimizer='adam',
            server_lr=0.01
        )
        self.assertEqual(controller.optimizer_name, 'adam')
        self.assertEqual(controller.server_lr, 0.01)

        # Test SGD with momentum
        controller = FedOptController(
            num_clients=3,
            num_rounds=10,
            optimizer='sgdm',
            server_lr=0.01,
            momentum=0.9
        )
        self.assertEqual(controller.optimizer_name, 'sgdm')
        self.assertEqual(controller.momentum, 0.9)

        # Test AdaGrad
        controller = FedOptController(
            num_clients=3,
            num_rounds=10,
            optimizer='adagrad',
            server_lr=0.01
        )
        self.assertEqual(controller.optimizer_name, 'adagrad')

        # Test Yogi
        controller = FedOptController(
            num_clients=3,
            num_rounds=10,
            optimizer='yogi',
            server_lr=0.01
        )
        self.assertEqual(controller.optimizer_name, 'yogi')

    def test_invalid_optimizer(self):
        """Test that invalid optimizer raises error."""
        controller = FedOptController(
            num_clients=3,
            num_rounds=10,
            optimizer='invalid_opt',
            server_lr=0.01
        )

        # Mock FL context
        from unittest.mock import MagicMock
        fl_ctx = MagicMock()

        # Should raise ValueError on start
        with self.assertRaises(ValueError):
            controller.start_controller(fl_ctx)

    def test_optimizer_parameters(self):
        """Test optimizer parameter configuration."""
        controller = FedOptController(
            num_clients=3,
            num_rounds=10,
            optimizer='adam',
            server_lr=0.02,
            beta1=0.95,
            beta2=0.998,
            epsilon=1e-7
        )

        self.assertEqual(controller.server_lr, 0.02)
        self.assertEqual(controller.beta1, 0.95)
        self.assertEqual(controller.beta2, 0.998)
        self.assertEqual(controller.epsilon, 1e-7)


class TestFedOptAlgorithm(unittest.TestCase):
    """Test FedOpt algorithm correctness."""

    def test_pseudo_gradient_computation(self):
        """Test pseudo-gradient computation from client deltas."""
        # Simulate global model and client models
        global_model = {
            'w': np.array([1.0, 2.0, 3.0])
        }

        # Two clients with different updates
        client1_model = {
            'w': np.array([1.1, 2.1, 3.1])  # delta = [0.1, 0.1, 0.1]
        }

        client2_model = {
            'w': np.array([0.9, 1.9, 2.9])  # delta = [-0.1, -0.1, -0.1]
        }

        # Compute deltas
        delta1 = client1_model['w'] - global_model['w']
        delta2 = client2_model['w'] - global_model['w']

        # Pseudo-gradient should be average of deltas
        expected_pseudo_grad = (delta1 + delta2) / 2.0

        # Should be [0, 0, 0] since deltas cancel out
        np.testing.assert_allclose(expected_pseudo_grad, np.zeros(3), atol=1e-10)

    def test_fedopt_vs_fedavg_difference(self):
        """Test that FedOpt differs from FedAvg in non-trivial cases."""
        # Global model
        global_model = {
            'w': np.array([1.0, 2.0, 3.0])
        }

        # Client models
        client1 = {'w': np.array([1.2, 2.1, 3.0])}
        client2 = {'w': np.array([1.1, 1.9, 3.2])}

        # FedAvg result: simple average
        fedavg_result = (client1['w'] + client2['w']) / 2.0

        # FedOpt result: optimizer step
        optimizer = ServerAdam(lr=0.01)

        # Compute pseudo-gradient
        delta1 = client1['w'] - global_model['w']
        delta2 = client2['w'] - global_model['w']
        pseudo_grad = (delta1 + delta2) / 2.0

        fedopt_result = optimizer.step(global_model, {'w': pseudo_grad})['w']

        # Results should differ
        self.assertFalse(np.allclose(fedavg_result, fedopt_result))


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestServerOptimizers))
    suite.addTests(loader.loadTestsFromTestCase(TestFedOptController))
    suite.addTests(loader.loadTestsFromTestCase(TestFedOptAlgorithm))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
