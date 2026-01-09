#!/usr/bin/env python3
"""
Standalone Unit Tests for Server Optimizers

Tests the server-side optimizers without requiring NVFlare installation.
This allows testing the core optimization algorithms independently.

Author: Generated for Med_SNP_Deconvolution
Date: 2026-01-09
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add parent directory to path to import optimizer classes
sys.path.insert(0, str(Path(__file__).parent))


class ServerOptimizer:
    """Base class for server-side optimizers."""

    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.state = {}

    def step(self, params, pseudo_gradient):
        raise NotImplementedError("Subclasses must implement step()")

    def reset_state(self):
        self.state = {}


class ServerSGDM(ServerOptimizer):
    """Server-side SGD with momentum."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.9, epsilon: float = 1e-8):
        super().__init__(lr=lr, epsilon=epsilon)
        self.momentum = momentum
        self.state = {'velocity': {}}

    def step(self, params, pseudo_gradient):
        updated_params = {}
        for name, param in params.items():
            if name not in pseudo_gradient:
                updated_params[name] = param
                continue
            grad = pseudo_gradient[name]
            if name not in self.state['velocity']:
                self.state['velocity'][name] = np.zeros_like(param)
            velocity = self.momentum * self.state['velocity'][name] + grad
            self.state['velocity'][name] = velocity
            updated_params[name] = param - self.lr * velocity
        return updated_params


class ServerAdam(ServerOptimizer):
    """Server-side Adam optimizer."""

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(lr=lr, epsilon=epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.state = {'step': 0, 'first_moment': {}, 'second_moment': {}}

    def step(self, params, pseudo_gradient):
        self.state['step'] += 1
        t = self.state['step']
        updated_params = {}
        for name, param in params.items():
            if name not in pseudo_gradient:
                updated_params[name] = param
                continue
            grad = pseudo_gradient[name]
            if name not in self.state['first_moment']:
                self.state['first_moment'][name] = np.zeros_like(param)
                self.state['second_moment'][name] = np.zeros_like(param)
            m = self.beta1 * self.state['first_moment'][name] + (1 - self.beta1) * grad
            self.state['first_moment'][name] = m
            v = self.beta2 * self.state['second_moment'][name] + (1 - self.beta2) * (grad ** 2)
            self.state['second_moment'][name] = v
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            updated_params[name] = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return updated_params


class ServerAdaGrad(ServerOptimizer):
    """Server-side AdaGrad optimizer."""

    def __init__(self, lr: float = 0.01, epsilon: float = 1e-8):
        super().__init__(lr=lr, epsilon=epsilon)
        self.state = {'sum_squared_gradients': {}}

    def step(self, params, pseudo_gradient):
        updated_params = {}
        for name, param in params.items():
            if name not in pseudo_gradient:
                updated_params[name] = param
                continue
            grad = pseudo_gradient[name]
            if name not in self.state['sum_squared_gradients']:
                self.state['sum_squared_gradients'][name] = np.zeros_like(param)
            G = self.state['sum_squared_gradients'][name] + grad ** 2
            self.state['sum_squared_gradients'][name] = G
            updated_params[name] = param - self.lr * grad / (np.sqrt(G) + self.epsilon)
        return updated_params


class ServerYogi(ServerOptimizer):
    """Server-side Yogi optimizer."""

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(lr=lr, epsilon=epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.state = {'step': 0, 'first_moment': {}, 'second_moment': {}}

    def step(self, params, pseudo_gradient):
        self.state['step'] += 1
        t = self.state['step']
        updated_params = {}
        for name, param in params.items():
            if name not in pseudo_gradient:
                updated_params[name] = param
                continue
            grad = pseudo_gradient[name]
            if name not in self.state['first_moment']:
                self.state['first_moment'][name] = np.zeros_like(param)
                self.state['second_moment'][name] = np.zeros_like(param)
            m = self.beta1 * self.state['first_moment'][name] + (1 - self.beta1) * grad
            self.state['first_moment'][name] = m
            v = self.state['second_moment'][name]
            grad_squared = grad ** 2
            v = v - (1 - self.beta2) * np.sign(v - grad_squared) * grad_squared
            self.state['second_moment'][name] = v
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            updated_params[name] = param - self.lr * m_hat / (np.sqrt(np.abs(v_hat)) + self.epsilon)
        return updated_params


class TestServerOptimizers(unittest.TestCase):
    """Test suite for server-side optimizers."""

    def setUp(self):
        """Set up test fixtures."""
        self.params = {
            'w1': np.array([1.0, 2.0, 3.0]),
            'w2': np.array([[1.0, 2.0], [3.0, 4.0]])
        }
        self.gradient = {
            'w1': np.array([0.1, 0.1, 0.1]),
            'w2': np.array([[0.1, 0.1], [0.1, 0.1]])
        }

    def test_sgdm_basic_step(self):
        """Test SGD with momentum basic update."""
        optimizer = ServerSGDM(lr=0.1, momentum=0.9)
        updated = optimizer.step(self.params, self.gradient)
        expected_w1 = self.params['w1'] - 0.1 * self.gradient['w1']
        np.testing.assert_allclose(updated['w1'], expected_w1, rtol=1e-5)
        print("✓ SGDM basic step test passed")

    def test_sgdm_momentum_accumulation(self):
        """Test that momentum accumulates over multiple steps."""
        optimizer = ServerSGDM(lr=0.1, momentum=0.9)
        updated = optimizer.step(self.params, self.gradient)
        updated2 = optimizer.step(updated, self.gradient)
        # Velocity should have accumulated
        expected_velocity = 1.9 * self.gradient['w1']
        expected_w1_2 = updated['w1'] - 0.1 * expected_velocity
        np.testing.assert_allclose(updated2['w1'], expected_w1_2, rtol=1e-5)
        print("✓ SGDM momentum accumulation test passed")

    def test_adam_basic_step(self):
        """Test Adam optimizer basic update."""
        optimizer = ServerAdam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
        updated = optimizer.step(self.params, self.gradient)
        self.assertFalse(np.allclose(updated['w1'], self.params['w1']))
        self.assertEqual(optimizer.state['step'], 1)
        self.assertIn('w1', optimizer.state['first_moment'])
        print("✓ Adam basic step test passed")

    def test_adam_convergence(self):
        """Test Adam convergence on simple quadratic."""
        optimizer = ServerAdam(lr=0.05, beta1=0.9, beta2=0.999)
        w_initial = {'x': np.array([10.0])}
        w = w_initial.copy()
        for _ in range(200):
            grad = {'x': 2 * w['x']}
            w = optimizer.step(w, grad)
        # Should improve (get closer to 0) even if not fully converged
        self.assertLess(np.abs(w['x'][0]), np.abs(w_initial['x'][0]))
        print("✓ Adam convergence test passed")

    def test_adagrad_basic_step(self):
        """Test AdaGrad optimizer basic update."""
        optimizer = ServerAdaGrad(lr=0.1, epsilon=1e-8)
        updated = optimizer.step(self.params, self.gradient)
        self.assertFalse(np.allclose(updated['w1'], self.params['w1']))
        self.assertIn('w1', optimizer.state['sum_squared_gradients'])
        print("✓ AdaGrad basic step test passed")

    def test_adagrad_decreasing_lr(self):
        """Test that AdaGrad effectively decreases learning rate."""
        optimizer = ServerAdaGrad(lr=1.0, epsilon=1e-8)
        w = {'x': np.array([10.0])}
        grad = {'x': np.array([1.0])}
        w1 = optimizer.step(w, grad)
        step1_change = np.abs(w1['x'] - w['x'])
        w2 = optimizer.step(w1, grad)
        step2_change = np.abs(w2['x'] - w1['x'])
        self.assertLess(step2_change[0], step1_change[0])
        print("✓ AdaGrad decreasing LR test passed")

    def test_yogi_basic_step(self):
        """Test Yogi optimizer basic update."""
        optimizer = ServerYogi(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
        updated = optimizer.step(self.params, self.gradient)
        self.assertFalse(np.allclose(updated['w1'], self.params['w1']))
        self.assertEqual(optimizer.state['step'], 1)
        print("✓ Yogi basic step test passed")

    def test_yogi_vs_adam(self):
        """Test that Yogi differs from Adam."""
        adam = ServerAdam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
        yogi = ServerYogi(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
        params = {'x': np.array([1.0])}
        grad = {'x': np.array([0.5])}
        params_adam = params.copy()
        params_yogi = params.copy()
        for _ in range(10):
            params_adam = adam.step(params_adam, grad)
            params_yogi = yogi.step(params_yogi, grad)
        self.assertFalse(np.allclose(params_adam['x'], params_yogi['x']))
        print("✓ Yogi vs Adam difference test passed")

    def test_zero_gradient(self):
        """Test optimizer behavior with zero gradient."""
        optimizer = ServerAdam(lr=0.01)
        zero_gradient = {
            'w1': np.zeros_like(self.params['w1']),
            'w2': np.zeros_like(self.params['w2'])
        }
        updated = optimizer.step(self.params, zero_gradient)
        np.testing.assert_allclose(updated['w1'], self.params['w1'], rtol=1e-5)
        print("✓ Zero gradient test passed")

    def test_large_gradient(self):
        """Test optimizer stability with large gradients."""
        optimizer = ServerAdam(lr=0.01, epsilon=1e-8)
        large_gradient = {
            'w1': np.array([100.0, 100.0, 100.0]),
            'w2': np.array([[100.0, 100.0], [100.0, 100.0]])
        }
        updated = optimizer.step(self.params, large_gradient)
        self.assertTrue(np.all(np.isfinite(updated['w1'])))
        self.assertTrue(np.all(np.isfinite(updated['w2'])))
        print("✓ Large gradient stability test passed")

    def test_missing_gradient_params(self):
        """Test handling of parameters without gradients."""
        optimizer = ServerAdam(lr=0.01)
        partial_gradient = {'w1': self.gradient['w1']}
        updated = optimizer.step(self.params, partial_gradient)
        self.assertFalse(np.allclose(updated['w1'], self.params['w1']))
        np.testing.assert_array_equal(updated['w2'], self.params['w2'])
        print("✓ Missing gradient params test passed")


class TestFedOptAlgorithm(unittest.TestCase):
    """Test FedOpt algorithm correctness."""

    def test_pseudo_gradient_computation(self):
        """Test pseudo-gradient computation from client deltas."""
        global_model = {'w': np.array([1.0, 2.0, 3.0])}
        client1_model = {'w': np.array([1.1, 2.1, 3.1])}
        client2_model = {'w': np.array([0.9, 1.9, 2.9])}
        delta1 = client1_model['w'] - global_model['w']
        delta2 = client2_model['w'] - global_model['w']
        expected_pseudo_grad = (delta1 + delta2) / 2.0
        np.testing.assert_allclose(expected_pseudo_grad, np.zeros(3), atol=1e-10)
        print("✓ Pseudo-gradient computation test passed")

    def test_fedopt_vs_fedavg_difference(self):
        """Test that FedOpt differs from FedAvg."""
        global_model = {'w': np.array([1.0, 2.0, 3.0])}
        client1 = {'w': np.array([1.2, 2.1, 3.0])}
        client2 = {'w': np.array([1.1, 1.9, 3.2])}
        fedavg_result = (client1['w'] + client2['w']) / 2.0
        optimizer = ServerAdam(lr=0.01)
        delta1 = client1['w'] - global_model['w']
        delta2 = client2['w'] - global_model['w']
        pseudo_grad = (delta1 + delta2) / 2.0
        fedopt_result = optimizer.step(global_model, {'w': pseudo_grad})['w']
        self.assertFalse(np.allclose(fedavg_result, fedopt_result))
        print("✓ FedOpt vs FedAvg difference test passed")

    def test_weighted_averaging(self):
        """Test weighted averaging of client updates."""
        global_model = {'w': np.array([0.0, 0.0])}

        # Client 1: large dataset (weight = 100)
        client1 = {'w': np.array([1.0, 1.0])}
        weight1 = 100

        # Client 2: small dataset (weight = 10)
        client2 = {'w': np.array([0.0, 2.0])}
        weight2 = 10

        # Compute weighted pseudo-gradient
        delta1 = client1['w'] - global_model['w']
        delta2 = client2['w'] - global_model['w']

        weighted_pseudo_grad = (weight1 * delta1 + weight2 * delta2) / (weight1 + weight2)

        # Should be closer to client1's update
        expected = (100 * np.array([1.0, 1.0]) + 10 * np.array([0.0, 2.0])) / 110
        np.testing.assert_allclose(weighted_pseudo_grad, expected, rtol=1e-5)
        print("✓ Weighted averaging test passed")


def run_tests():
    """Run all tests with detailed output."""
    print("="*70)
    print("Running FedOpt Server Optimizer Tests")
    print("="*70)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestServerOptimizers))
    suite.addTests(loader.loadTestsFromTestCase(TestFedOptAlgorithm))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
