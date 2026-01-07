"""
Unit Tests for Attention-Based Deep Learning Module

Tests all components with and without GPU availability.

Author: Generated for Haploblock Analysis Pipeline
Date: 2026-01-07
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import tempfile
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, '/Users/saltfish/Files/Coding/Haploblock_Clusters_ElixirBH25')

from snp_deconvolution.attention_dl.gpu_optimized_models import (
    GPUOptimizedSNPModel,
    create_gpu_optimized_model,
    create_multi_gpu_model,
    DataParallelSNPModel
)
from snp_deconvolution.attention_dl.gpu_trainer import (
    MultiGPUSNPTrainer,
    FocalLoss
)
from snp_deconvolution.attention_dl.memory_manager import (
    GPUMemoryManager,
    MemoryEfficientDataLoader,
    MemoryStats
)
from dl_models.snp_interpretable_models import InterpretableSNPModel


class TestGPUOptimizedModels(unittest.TestCase):
    """Test GPU-optimized model components"""

    def setUp(self):
        """Set up test fixtures"""
        self.n_snps = 100
        self.encoding_dim = 8
        self.num_classes = 2
        self.batch_size = 4

        # Create sample data
        self.X = torch.randn(self.batch_size, self.n_snps, self.encoding_dim)
        self.y = torch.randint(0, self.num_classes, (self.batch_size,))

    def test_gpu_optimized_model_creation(self):
        """Test creating GPU-optimized model"""
        base_model = InterpretableSNPModel(
            n_snps=self.n_snps,
            encoding_dim=self.encoding_dim,
            num_classes=self.num_classes,
            architecture='cnn_transformer'
        )

        model = GPUOptimizedSNPModel(
            base_model=base_model,
            use_amp=True,
            amp_dtype=torch.bfloat16
        )

        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.n_snps, self.n_snps)
        self.assertEqual(model.encoding_dim, self.encoding_dim)

    def test_forward_pass(self):
        """Test forward pass through model"""
        model = create_gpu_optimized_model(
            n_snps=self.n_snps,
            encoding_dim=self.encoding_dim,
            num_classes=self.num_classes,
            architecture='cnn_transformer',
            device='cpu'  # Use CPU for testing
        )

        output = model(self.X)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertTrue(torch.isfinite(output).all())

    def test_forward_with_amp(self):
        """Test forward pass with AMP"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model = create_gpu_optimized_model(
            n_snps=self.n_snps,
            encoding_dim=self.encoding_dim,
            use_amp=True,
            amp_dtype=torch.bfloat16,
            device='cuda'
        )

        X_cuda = self.X.cuda()
        output = model(X_cuda)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertTrue(torch.isfinite(output).all())

    def test_interpretability_methods(self):
        """Test predict_with_interpretation"""
        model = create_gpu_optimized_model(
            n_snps=self.n_snps,
            encoding_dim=self.encoding_dim,
            architecture='cnn_transformer',
            device='cpu'
        )

        results = model.predict_with_interpretation(
            self.X,
            methods=['attention']
        )

        self.assertIn('predictions', results)
        self.assertIn('probabilities', results)
        self.assertEqual(results['predictions'].shape, (self.batch_size,))
        self.assertEqual(results['probabilities'].shape, (self.batch_size, self.num_classes))

    def test_identify_causal_snps(self):
        """Test causal SNP identification"""
        model = create_gpu_optimized_model(
            n_snps=self.n_snps,
            encoding_dim=self.encoding_dim,
            architecture='cnn_transformer',
            device='cpu'
        )

        top_k = 5
        indices, scores = model.identify_causal_snps(self.X, top_k=top_k)

        self.assertEqual(indices.shape, (top_k,))
        self.assertEqual(scores.shape, (top_k,))
        self.assertTrue((scores >= 0).all())

    def test_different_architectures(self):
        """Test different model architectures"""
        architectures = ['cnn', 'cnn_transformer']

        for arch in architectures:
            with self.subTest(architecture=arch):
                model = create_gpu_optimized_model(
                    n_snps=self.n_snps,
                    encoding_dim=self.encoding_dim,
                    architecture=arch,
                    device='cpu'
                )

                output = model(self.X)
                self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestGPUTrainer(unittest.TestCase):
    """Test GPU trainer components"""

    def setUp(self):
        """Set up test fixtures"""
        self.n_snps = 100
        self.encoding_dim = 8
        self.num_classes = 2
        self.batch_size = 8
        self.num_samples = 32

        # Create sample data
        X = torch.randn(self.num_samples, self.n_snps, self.encoding_dim)
        y = torch.randint(0, self.num_classes, (self.num_samples,))

        # Split into train/val
        train_dataset = TensorDataset(X[:24], y[:24])
        val_dataset = TensorDataset(X[24:], y[24:])

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Create model
        self.model = create_gpu_optimized_model(
            n_snps=self.n_snps,
            encoding_dim=self.encoding_dim,
            num_classes=self.num_classes,
            device='cpu'
        )

    def test_focal_loss(self):
        """Test focal loss computation"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        inputs = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))

        loss = focal_loss(inputs, targets)

        self.assertIsInstance(loss.item(), float)
        self.assertTrue(loss.item() >= 0)

    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = MultiGPUSNPTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            gpu_ids=[0] if torch.cuda.is_available() else None,
            learning_rate=1e-4
        )

        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)

    def test_train_epoch(self):
        """Test training one epoch"""
        trainer = MultiGPUSNPTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            use_amp=False  # Disable AMP for CPU testing
        )

        train_loss, train_acc = trainer.train_epoch()

        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)
        self.assertTrue(train_loss >= 0)
        self.assertTrue(0 <= train_acc <= 100)

    def test_validate(self):
        """Test validation"""
        trainer = MultiGPUSNPTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            use_amp=False
        )

        val_loss, val_acc = trainer.validate()

        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        self.assertTrue(val_loss >= 0)
        self.assertTrue(0 <= val_acc <= 100)

    def test_full_training_loop(self):
        """Test complete training loop"""
        trainer = MultiGPUSNPTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            use_amp=False
        )

        history = trainer.train(
            num_epochs=2,
            early_stopping_patience=10,
            checkpoint_dir=None
        )

        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('train_acc', history)
        self.assertIn('val_acc', history)
        self.assertEqual(len(history['train_loss']), 2)

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        trainer = MultiGPUSNPTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            use_amp=False
        )

        # Train for 1 epoch
        trainer.train_epoch()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"
            trainer.save_checkpoint(checkpoint_path)

            self.assertTrue(checkpoint_path.exists())

            # Load checkpoint
            new_trainer = MultiGPUSNPTrainer(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                use_amp=False
            )
            new_trainer.load_checkpoint(checkpoint_path)

            self.assertEqual(trainer.current_epoch, new_trainer.current_epoch)

    def test_get_snp_importance(self):
        """Test SNP importance extraction"""
        trainer = MultiGPUSNPTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            use_amp=False
        )

        importance_dict = trainer.get_snp_importance(
            data_loader=self.val_loader,
            method='attention',
            aggregate='mean'
        )

        self.assertEqual(len(importance_dict), self.n_snps)
        self.assertTrue(all(isinstance(v, float) for v in importance_dict.values()))


class TestMemoryManager(unittest.TestCase):
    """Test GPU memory management utilities"""

    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation"""
        batch_size = GPUMemoryManager.get_optimal_batch_size(
            n_snps=1000,
            encoding_dim=8,
            gpu_memory_gb=40.0,  # Simulate 40GB GPU
            dtype=torch.bfloat16
        )

        self.assertIsInstance(batch_size, int)
        self.assertTrue(batch_size >= 1)
        self.assertTrue(batch_size <= 512)

    def test_batch_size_range_suggestion(self):
        """Test batch size range suggestion"""
        min_bs, opt_bs, max_bs = GPUMemoryManager.suggest_batch_size_range(
            n_snps=5000,
            encoding_dim=8,
            gpu_memory_gb=40.0,
            dtype=torch.bfloat16
        )

        self.assertTrue(min_bs <= opt_bs <= max_bs)
        self.assertTrue(all(isinstance(bs, int) for bs in [min_bs, opt_bs, max_bs]))

    def test_memory_usage_estimation(self):
        """Test memory usage estimation"""
        mem_est = GPUMemoryManager.estimate_memory_usage(
            batch_size=32,
            n_snps=10000,
            encoding_dim=8,
            dtype=torch.bfloat16
        )

        required_keys = ['input_gb', 'model_gb', 'activations_gb', 'gradients_gb', 'optimizer_gb', 'total_gb']
        for key in required_keys:
            self.assertIn(key, mem_est)
            self.assertIsInstance(mem_est[key], float)
            self.assertTrue(mem_est[key] >= 0)

    def test_clear_cache(self):
        """Test cache clearing"""
        # Should not raise exception
        GPUMemoryManager.clear_cache()

    def test_memory_stats(self):
        """Test getting memory stats"""
        stats = GPUMemoryManager.get_memory_stats(device=0)

        self.assertIsInstance(stats, MemoryStats)
        self.assertTrue(stats.total_gb >= 0)
        self.assertTrue(stats.free_gb >= 0)
        self.assertTrue(0 <= stats.utilization_pct <= 100)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_gpu_memory_detection(self):
        """Test GPU memory detection"""
        total_gb, free_gb = GPUMemoryManager.get_gpu_memory_gb(device=0)

        self.assertTrue(total_gb > 0)
        self.assertTrue(free_gb >= 0)
        self.assertTrue(free_gb <= total_gb)

    def test_log_memory_usage(self):
        """Test memory usage logging"""
        # Should not raise exception
        GPUMemoryManager.log_memory_usage(tag="test")

    def test_optimal_num_workers(self):
        """Test optimal number of workers calculation"""
        num_workers = MemoryEfficientDataLoader.get_optimal_num_workers()

        self.assertIsInstance(num_workers, int)
        self.assertTrue(num_workers >= 2)

    def test_create_dataloader(self):
        """Test data loader creation"""
        X = torch.randn(32, 100, 8)
        y = torch.randint(0, 2, (32,))
        dataset = TensorDataset(X, y)

        loader = MemoryEfficientDataLoader.create_dataloader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )

        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.batch_size, 8)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""

    def test_end_to_end_workflow(self):
        """Test complete workflow from data to trained model"""
        # Configuration
        n_snps = 100
        encoding_dim = 8
        num_samples = 32
        batch_size = 8

        # Generate data
        X = torch.randn(num_samples, n_snps, encoding_dim)
        y = torch.randint(0, 2, (num_samples,))

        # Create datasets
        train_dataset = TensorDataset(X[:24], y[:24])
        val_dataset = TensorDataset(X[24:], y[24:])

        # Create data loaders
        train_loader = MemoryEfficientDataLoader.create_dataloader(
            train_dataset, batch_size=batch_size, num_workers=0
        )
        val_loader = MemoryEfficientDataLoader.create_dataloader(
            val_dataset, batch_size=batch_size, num_workers=0
        )

        # Create model
        model = create_gpu_optimized_model(
            n_snps=n_snps,
            encoding_dim=encoding_dim,
            architecture='cnn_transformer',
            device='cpu'
        )

        # Train
        trainer = MultiGPUSNPTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            use_amp=False
        )

        history = trainer.train(num_epochs=2)

        # Extract importance
        importance_dict = trainer.get_snp_importance(
            data_loader=val_loader,
            method='attention'
        )

        # Assertions
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(importance_dict), n_snps)
        self.assertTrue(trainer.best_val_loss > 0)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGPUOptimizedModels))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
