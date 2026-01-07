"""
Unit Tests for NVFlare Base Module

Tests for:
    - Base executor interface
    - Model serialization/deserialization
    - XGBoost executor functionality
    - PyTorch executor functionality
    - Federated averaging
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from base_executor import SNPDeconvExecutor, ExecutorMetrics
from model_shareable import (
    serialize_pytorch_weights,
    deserialize_pytorch_weights,
    serialize_xgboost_model,
    deserialize_xgboost_model,
    validate_model_weights,
    _compute_checksum,
    _compute_string_checksum,
)

if XGBOOST_AVAILABLE:
    from xgb_nvflare_wrapper import XGBoostNVFlareExecutor

if PYTORCH_AVAILABLE:
    from dl_nvflare_wrapper import DLNVFlareExecutor


# Test fixtures
@pytest.fixture
def synthetic_snp_data():
    """Generate synthetic SNP data for testing"""
    np.random.seed(42)
    num_samples = 100
    num_snps = 50
    num_populations = 3

    X = np.random.randint(0, 3, (num_samples, num_snps)).astype(np.float32)
    y = np.random.randint(0, num_populations, num_samples).astype(np.int64)

    return X, y


@pytest.fixture
def simple_pytorch_model():
    """Create a simple PyTorch model for testing"""
    if not PYTORCH_AVAILABLE:
        pytest.skip("PyTorch not available")

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_snps = 50
            self.num_populations = 3
            self.fc1 = nn.Linear(50, 32)
            self.fc2 = nn.Linear(32, 3)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleNet()


class TestExecutorMetrics:
    """Test ExecutorMetrics dataclass"""

    def test_metrics_creation(self):
        """Test creating ExecutorMetrics"""
        metrics = ExecutorMetrics(
            loss=0.5,
            accuracy=0.85,
            num_samples=100,
            additional_metrics={'auc': 0.9},
        )

        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.85
        assert metrics.num_samples == 100
        assert metrics.additional_metrics['auc'] == 0.9

    def test_metrics_to_dict(self):
        """Test conversion to dictionary"""
        metrics = ExecutorMetrics(loss=0.5, accuracy=0.85, num_samples=100)
        d = metrics.to_dict()

        assert d['loss'] == 0.5
        assert d['accuracy'] == 0.85
        assert d['num_samples'] == 100
        assert isinstance(d, dict)


class TestModelSerialization:
    """Test model serialization utilities"""

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_serialization(self, simple_pytorch_model):
        """Test PyTorch model serialization and deserialization"""
        # Get state dict
        state_dict = simple_pytorch_model.state_dict()

        # Serialize
        serialized = serialize_pytorch_weights(state_dict, include_metadata=True)

        assert 'weights' in serialized
        assert 'format' in serialized
        assert serialized['format'] == 'pytorch_numpy'
        assert 'checksum' in serialized
        assert 'metadata' in serialized

        # Deserialize
        restored = deserialize_pytorch_weights(serialized, verify_checksum=True)

        # Check all keys present
        assert set(restored.keys()) == set(state_dict.keys())

        # Check values match
        for key in state_dict.keys():
            assert torch.allclose(state_dict[key], restored[key])

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_checksum_verification(self, simple_pytorch_model):
        """Test checksum verification fails on corrupted data"""
        state_dict = simple_pytorch_model.state_dict()
        serialized = serialize_pytorch_weights(state_dict)

        # Corrupt data
        first_key = list(serialized['weights'].keys())[0]
        serialized['weights'][first_key][0] = 999.0

        # Should raise error on checksum verification
        with pytest.raises(Exception):
            deserialize_pytorch_weights(serialized, verify_checksum=True)

        # Should work without verification
        restored = deserialize_pytorch_weights(serialized, verify_checksum=False)
        assert restored is not None

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_serialization(self, synthetic_snp_data):
        """Test XGBoost model serialization"""
        X, y = synthetic_snp_data

        # Train simple model
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 3,
        }
        model = xgb.train(params, dtrain, num_boost_round=5)

        # Serialize
        serialized = serialize_xgboost_model(model, include_metadata=True)

        assert 'model_json' in serialized
        assert 'config' in serialized
        assert 'format' in serialized
        assert serialized['format'] == 'xgboost_json'
        assert 'metadata' in serialized

        # Deserialize
        restored_model = deserialize_xgboost_model(serialized, verify_checksum=True)

        # Check predictions match
        pred_original = model.predict(dtrain)
        pred_restored = restored_model.predict(dtrain)
        np.testing.assert_array_almost_equal(pred_original, pred_restored)

    def test_checksum_computation(self):
        """Test checksum computation for numpy arrays"""
        data = {
            'array1': np.array([1, 2, 3]),
            'array2': np.array([4, 5, 6]),
        }

        checksum1 = _compute_checksum(data)
        checksum2 = _compute_checksum(data)

        assert checksum1 == checksum2
        assert isinstance(checksum1, str)
        assert len(checksum1) == 64  # SHA256 hex digest

        # Different data should have different checksum
        data['array1'][0] = 999
        checksum3 = _compute_checksum(data)
        assert checksum1 != checksum3

    def test_string_checksum(self):
        """Test string checksum computation"""
        text = "test string"
        checksum1 = _compute_string_checksum(text)
        checksum2 = _compute_string_checksum(text)

        assert checksum1 == checksum2
        assert isinstance(checksum1, str)

        checksum3 = _compute_string_checksum("different string")
        assert checksum1 != checksum3


class TestWeightValidation:
    """Test model weight validation"""

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_validate_pytorch_weights(self, simple_pytorch_model):
        """Test PyTorch weight validation"""
        state_dict = simple_pytorch_model.state_dict()
        serialized = serialize_pytorch_weights(state_dict)

        # Should pass validation
        assert validate_model_weights(serialized, expected_model_type='pytorch')

        # Should fail with wrong model type
        assert not validate_model_weights(serialized, expected_model_type='xgboost')

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_validate_xgboost_weights(self, synthetic_snp_data):
        """Test XGBoost weight validation"""
        X, y = synthetic_snp_data
        dtrain = xgb.DMatrix(X, label=y)
        params = {'objective': 'multi:softprob', 'num_class': 3}
        model = xgb.train(params, dtrain, num_boost_round=5)

        serialized = serialize_xgboost_model(model)

        # Should pass validation
        assert validate_model_weights(serialized, expected_model_type='xgboost')
        assert validate_model_weights(
            serialized,
            expected_model_type='xgboost',
            expected_num_snps=50,
        )

        # Should fail with wrong dimensions
        assert not validate_model_weights(
            serialized,
            expected_model_type='xgboost',
            expected_num_snps=100,
        )


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
class TestXGBoostExecutor:
    """Test XGBoost NVFlare executor"""

    def test_executor_initialization(self):
        """Test executor initialization"""
        executor = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=100,
            num_populations=5,
        )

        assert executor.model_type == 'xgboost'
        assert executor.num_snps == 100
        assert executor.num_populations == 5
        assert executor.get_current_round() == 0

    def test_data_preparation(self, synthetic_snp_data):
        """Test data preparation"""
        X, y = synthetic_snp_data
        executor = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=X.shape[1],
            num_populations=3,
        )

        executor.prepare_data(X, y)
        assert executor._dtrain is not None

        executor.prepare_data(X, y, is_validation=True)
        assert executor._dval is not None

    def test_local_training(self, synthetic_snp_data):
        """Test local training"""
        X, y = synthetic_snp_data
        executor = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=X.shape[1],
            num_populations=3,
            xgb_params={'tree_method': 'hist', 'device': 'cpu'},
        )

        executor.prepare_data(X, y)
        metrics = executor.local_train(num_epochs=5)

        assert isinstance(metrics, ExecutorMetrics)
        assert metrics.loss > 0
        assert 0 <= metrics.accuracy <= 1
        assert metrics.num_samples == len(X)

    def test_weight_export_import(self, synthetic_snp_data):
        """Test model weight export and import"""
        X, y = synthetic_snp_data
        executor = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=X.shape[1],
            num_populations=3,
            xgb_params={'tree_method': 'hist', 'device': 'cpu'},
        )

        executor.prepare_data(X, y)
        executor.local_train(num_epochs=5)

        # Export weights
        weights = executor.get_model_weights()
        assert 'model_json' in weights
        assert 'num_snps' in weights
        assert weights['model_type'] == 'xgboost'

        # Import weights
        executor.set_model_weights(weights)
        assert executor.model is not None

    def test_validation(self, synthetic_snp_data):
        """Test model validation"""
        X, y = synthetic_snp_data
        executor = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=X.shape[1],
            num_populations=3,
            xgb_params={'tree_method': 'hist', 'device': 'cpu'},
        )

        executor.prepare_data(X, y)
        executor.prepare_data(X, y, is_validation=True)
        executor.local_train(num_epochs=5)

        val_metrics = executor.validate()
        assert isinstance(val_metrics, ExecutorMetrics)
        assert val_metrics.loss > 0
        assert 0 <= val_metrics.accuracy <= 1

    def test_feature_importance(self, synthetic_snp_data):
        """Test feature importance extraction"""
        X, y = synthetic_snp_data
        executor = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=X.shape[1],
            num_populations=3,
            xgb_params={'tree_method': 'hist', 'device': 'cpu'},
        )

        executor.prepare_data(X, y)
        executor.local_train(num_epochs=10)

        importance = executor.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
        assert all(isinstance(k, int) for k in importance.keys())
        assert all(isinstance(v, float) for v in importance.values())

        # Scores should be normalized
        total = sum(importance.values())
        assert abs(total - 1.0) < 1e-5


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
class TestPyTorchExecutor:
    """Test PyTorch NVFlare executor"""

    def test_executor_initialization(self, simple_pytorch_model):
        """Test executor initialization"""
        executor = DLNVFlareExecutor(
            model=simple_pytorch_model,
            aggregation_strategy='fedavg',
        )

        assert executor.model_type == 'pytorch'
        assert executor.aggregation_strategy == 'fedavg'
        assert executor.get_current_round() == 0

    def test_fedprox_initialization(self, simple_pytorch_model):
        """Test FedProx initialization"""
        executor = DLNVFlareExecutor(
            model=simple_pytorch_model,
            aggregation_strategy='fedprox',
            fedprox_mu=0.1,
        )

        assert executor.aggregation_strategy == 'fedprox'
        assert executor.fedprox_mu == 0.1

    def test_weight_export_import(self, simple_pytorch_model):
        """Test model weight export and import"""
        executor = DLNVFlareExecutor(model=simple_pytorch_model)

        # Export weights
        weights = executor.get_model_weights()
        assert 'weights' in weights
        assert 'model_type' in weights
        assert weights['model_type'] == 'pytorch'

        # Import weights
        executor.set_model_weights(weights)

    def test_data_loader_setup(self, simple_pytorch_model, synthetic_snp_data):
        """Test data loader setup"""
        X, y = synthetic_snp_data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        executor = DLNVFlareExecutor(model=simple_pytorch_model)
        executor.set_data_loaders(loader, loader)

        assert executor._train_loader is not None
        assert executor._val_loader is not None

    def test_local_training(self, simple_pytorch_model, synthetic_snp_data):
        """Test local training"""
        X, y = synthetic_snp_data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        executor = DLNVFlareExecutor(model=simple_pytorch_model)
        executor.set_data_loaders(loader, loader)

        metrics = executor.local_train(num_epochs=2)

        assert isinstance(metrics, ExecutorMetrics)
        assert metrics.loss > 0
        assert 0 <= metrics.accuracy <= 1

    def test_validation(self, simple_pytorch_model, synthetic_snp_data):
        """Test validation"""
        X, y = synthetic_snp_data
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        executor = DLNVFlareExecutor(model=simple_pytorch_model)
        executor.set_data_loaders(loader, loader)

        val_metrics = executor.validate()
        assert isinstance(val_metrics, ExecutorMetrics)


class TestCheckpointing:
    """Test executor checkpointing"""

    @pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="XGBoost not available")
    def test_xgboost_checkpoint(self, synthetic_snp_data, tmp_path):
        """Test XGBoost executor checkpoint save/load"""
        X, y = synthetic_snp_data
        executor = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=X.shape[1],
            num_populations=3,
            xgb_params={'tree_method': 'hist', 'device': 'cpu'},
        )

        executor.prepare_data(X, y)
        executor.local_train(num_epochs=5)
        executor.increment_round()

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pkl"
        executor.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()

        # Create new executor and load checkpoint
        executor2 = XGBoostNVFlareExecutor(
            trainer=None,
            num_snps=X.shape[1],
            num_populations=3,
        )
        executor2.prepare_data(X, y)
        executor2.load_checkpoint(checkpoint_path)

        assert executor2.get_current_round() == executor.get_current_round()

    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_checkpoint(self, simple_pytorch_model, tmp_path):
        """Test PyTorch executor checkpoint save/load"""
        executor = DLNVFlareExecutor(model=simple_pytorch_model)
        executor.increment_round()

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pkl"
        executor.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()

        # Create new executor and load checkpoint
        executor2 = DLNVFlareExecutor(model=simple_pytorch_model)
        executor2.load_checkpoint(checkpoint_path)

        assert executor2.get_current_round() == executor.get_current_round()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
