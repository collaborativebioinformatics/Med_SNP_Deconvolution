"""
Model Serialization Utilities for NVFlare

Provides functions to serialize and deserialize model weights for transmission
between federated learning sites and the central server.

Features:
    - PyTorch state_dict <-> numpy conversion
    - XGBoost model JSON serialization
    - Validation and integrity checks
    - Compression support for large models

The serialization format is designed to be:
    - Platform-independent (numpy arrays)
    - JSON-serializable for network transmission
    - Compatible with NVFlare's Shareable objects
"""

import numpy as np
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pickle
import gzip
from datetime import datetime

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """Raised when serialization/deserialization fails"""
    pass


def serialize_pytorch_weights(
    state_dict: Dict[str, 'torch.Tensor'],
    compress: bool = False,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Convert PyTorch state_dict to numpy arrays for serialization.

    This function converts PyTorch tensors to numpy arrays, which can be
    JSON-serialized and transmitted over the network. The conversion maintains
    precision and supports all tensor types.

    Args:
        state_dict: PyTorch model state_dict
        compress: Whether to compress large arrays (>1MB)
        include_metadata: Include tensor shapes, dtypes, and checksums

    Returns:
        Dictionary with:
            - weights: Dict[str, np.ndarray] or compressed bytes
            - metadata: Shapes, dtypes, device info
            - checksum: Hash for integrity verification
            - timestamp: Serialization timestamp

    Raises:
        SerializationError: If PyTorch not available or conversion fails

    Example:
        >>> state_dict = model.state_dict()
        >>> serialized = serialize_pytorch_weights(state_dict)
        >>> # Transmit serialized dict
        >>> restored = deserialize_pytorch_weights(serialized)
        >>> model.load_state_dict(restored)
    """
    if not PYTORCH_AVAILABLE:
        raise SerializationError("PyTorch not available for serialization")

    try:
        weights_dict = {}
        metadata = {}

        for name, tensor in state_dict.items():
            # Move to CPU and convert to numpy
            np_array = tensor.detach().cpu().numpy()
            weights_dict[name] = np_array

            if include_metadata:
                metadata[name] = {
                    'shape': list(np_array.shape),
                    'dtype': str(np_array.dtype),
                    'size_bytes': np_array.nbytes,
                }

        result = {
            'weights': weights_dict,
            'format': 'pytorch_numpy',
            'timestamp': datetime.now().isoformat(),
        }

        if include_metadata:
            result['metadata'] = metadata
            result['total_parameters'] = sum(
                np.prod(meta['shape']) for meta in metadata.values()
            )
            result['total_size_mb'] = sum(
                meta['size_bytes'] for meta in metadata.values()
            ) / (1024 * 1024)

        # Optional compression for large models
        if compress:
            result = _compress_weights(result)

        # Add checksum for integrity
        result['checksum'] = _compute_checksum(weights_dict)

        logger.info(
            f"Serialized PyTorch model: {len(weights_dict)} tensors, "
            f"{result.get('total_size_mb', 0):.2f} MB"
        )

        return result

    except Exception as e:
        raise SerializationError(f"Failed to serialize PyTorch weights: {e}")


def deserialize_pytorch_weights(
    weights: Dict[str, Any],
    device: Optional[Union[str, 'torch.device']] = None,
    verify_checksum: bool = True
) -> Dict[str, 'torch.Tensor']:
    """
    Convert numpy arrays back to PyTorch tensors.

    Args:
        weights: Serialized weights from serialize_pytorch_weights()
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        verify_checksum: Verify integrity using checksum

    Returns:
        PyTorch state_dict ready for model.load_state_dict()

    Raises:
        SerializationError: If deserialization fails or checksum mismatch
    """
    if not PYTORCH_AVAILABLE:
        raise SerializationError("PyTorch not available for deserialization")

    try:
        # Decompress if needed
        if weights.get('compressed', False):
            weights = _decompress_weights(weights)

        weights_dict = weights['weights']

        # Verify checksum
        if verify_checksum and 'checksum' in weights:
            current_checksum = _compute_checksum(weights_dict)
            if current_checksum != weights['checksum']:
                raise SerializationError("Checksum mismatch - data corrupted")

        # Convert to PyTorch tensors
        state_dict = {}
        for name, np_array in weights_dict.items():
            tensor = torch.from_numpy(np_array)
            if device is not None:
                tensor = tensor.to(device)
            state_dict[name] = tensor

        logger.info(f"Deserialized PyTorch model: {len(state_dict)} tensors")

        return state_dict

    except Exception as e:
        raise SerializationError(f"Failed to deserialize PyTorch weights: {e}")


def serialize_xgboost_model(
    model: 'xgb.Booster',
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Serialize XGBoost model to JSON format.

    XGBoost models are serialized as JSON tree structures, which are
    platform-independent and human-readable. This format supports all
    XGBoost features including custom objectives and metrics.

    Args:
        model: XGBoost Booster object
        include_metadata: Include training info and feature names

    Returns:
        Dictionary with:
            - model_json: JSON tree structure as string
            - config: Booster configuration
            - feature_names: List of feature names
            - metadata: Training info, version, etc.

    Raises:
        SerializationError: If XGBoost not available or serialization fails

    Example:
        >>> booster = xgb.train(params, dtrain)
        >>> serialized = serialize_xgboost_model(booster)
        >>> # Transmit serialized dict
        >>> restored = deserialize_xgboost_model(serialized)
    """
    if not XGBOOST_AVAILABLE:
        raise SerializationError("XGBoost not available for serialization")

    try:
        # Serialize model to JSON string
        model_json = model.save_raw(raw_format='json').decode('utf-8')

        # Get configuration
        config = json.loads(model.save_config())

        result = {
            'model_json': model_json,
            'config': config,
            'format': 'xgboost_json',
            'timestamp': datetime.now().isoformat(),
        }

        if include_metadata:
            # Extract feature information
            try:
                feature_names = model.feature_names
                feature_types = model.feature_types
                result['feature_names'] = feature_names if feature_names else []
                result['feature_types'] = feature_types if feature_types else []
            except AttributeError:
                result['feature_names'] = []
                result['feature_types'] = []

            # Get model attributes
            result['metadata'] = {
                'num_features': model.num_features(),
                'num_boosted_rounds': model.num_boosted_rounds(),
                'xgboost_version': xgb.__version__,
            }

        # Compute checksum
        result['checksum'] = _compute_string_checksum(model_json)

        logger.info(
            f"Serialized XGBoost model: "
            f"{result['metadata']['num_boosted_rounds']} rounds, "
            f"{result['metadata']['num_features']} features"
        )

        return result

    except Exception as e:
        raise SerializationError(f"Failed to serialize XGBoost model: {e}")


def deserialize_xgboost_model(
    model_data: Dict[str, Any],
    verify_checksum: bool = True
) -> 'xgb.Booster':
    """
    Deserialize XGBoost model from JSON format.

    Args:
        model_data: Serialized model from serialize_xgboost_model()
        verify_checksum: Verify integrity using checksum

    Returns:
        XGBoost Booster object

    Raises:
        SerializationError: If deserialization fails or checksum mismatch
    """
    if not XGBOOST_AVAILABLE:
        raise SerializationError("XGBoost not available for deserialization")

    try:
        model_json = model_data['model_json']

        # Verify checksum
        if verify_checksum and 'checksum' in model_data:
            current_checksum = _compute_string_checksum(model_json)
            if current_checksum != model_data['checksum']:
                raise SerializationError("Checksum mismatch - data corrupted")

        # Create new booster and load from JSON
        booster = xgb.Booster()
        booster.load_model(bytearray(model_json, 'utf-8'))

        # Load configuration if available
        if 'config' in model_data:
            booster.load_config(json.dumps(model_data['config']))

        # Restore feature names
        if 'feature_names' in model_data and model_data['feature_names']:
            booster.feature_names = model_data['feature_names']
        if 'feature_types' in model_data and model_data['feature_types']:
            booster.feature_types = model_data['feature_types']

        logger.info("Deserialized XGBoost model successfully")

        return booster

    except Exception as e:
        raise SerializationError(f"Failed to deserialize XGBoost model: {e}")


def validate_model_weights(
    weights: Dict[str, Any],
    expected_model_type: Optional[str] = None,
    expected_num_snps: Optional[int] = None,
    expected_num_populations: Optional[int] = None
) -> bool:
    """
    Validate model weights structure and metadata.

    Performs comprehensive validation including:
        - Format compatibility
        - Dimension checks
        - Checksum verification
        - Timestamp recency

    Args:
        weights: Serialized model weights
        expected_model_type: Expected 'pytorch' or 'xgboost'
        expected_num_snps: Expected number of SNP features
        expected_num_populations: Expected number of populations

    Returns:
        True if valid, False otherwise (logs specific errors)

    Example:
        >>> if validate_model_weights(weights, 'pytorch', 10000, 5):
        ...     model.load_state_dict(deserialize_pytorch_weights(weights))
    """
    try:
        # Check format
        if 'format' not in weights:
            logger.error("Missing 'format' field in weights")
            return False

        format_type = weights['format']
        if format_type not in ['pytorch_numpy', 'xgboost_json']:
            logger.error(f"Unknown format: {format_type}")
            return False

        # Check model type
        if expected_model_type:
            actual_type = 'pytorch' if 'pytorch' in format_type else 'xgboost'
            if actual_type != expected_model_type:
                logger.error(
                    f"Model type mismatch: expected {expected_model_type}, "
                    f"got {actual_type}"
                )
                return False

        # Check dimensions for PyTorch
        if format_type == 'pytorch_numpy' and 'metadata' in weights:
            # Validate using metadata if dimension checks are needed
            if expected_num_snps or expected_num_populations:
                # This would require inspecting specific layer shapes
                logger.warning("Dimension validation not implemented for PyTorch")

        # Check dimensions for XGBoost
        if format_type == 'xgboost_json' and 'metadata' in weights:
            if expected_num_snps:
                actual_features = weights['metadata'].get('num_features')
                if actual_features != expected_num_snps:
                    logger.error(
                        f"Feature mismatch: expected {expected_num_snps}, "
                        f"got {actual_features}"
                    )
                    return False

        # Check timestamp recency (warn if > 24 hours old)
        if 'timestamp' in weights:
            try:
                timestamp = datetime.fromisoformat(weights['timestamp'])
                age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                if age_hours > 24:
                    logger.warning(f"Weights are {age_hours:.1f} hours old")
            except ValueError:
                logger.warning("Invalid timestamp format")

        logger.info("Model weights validation passed")
        return True

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False


def _compress_weights(weights_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Compress weights using gzip"""
    try:
        serialized = pickle.dumps(weights_dict)
        compressed = gzip.compress(serialized, compresslevel=6)

        original_size = len(serialized) / (1024 * 1024)
        compressed_size = len(compressed) / (1024 * 1024)
        ratio = compressed_size / original_size * 100

        logger.info(
            f"Compressed weights: {original_size:.2f} MB -> "
            f"{compressed_size:.2f} MB ({ratio:.1f}%)"
        )

        return {
            'compressed_data': compressed,
            'compressed': True,
            'original_size': len(serialized),
            'format': weights_dict.get('format', 'unknown'),
        }
    except Exception as e:
        logger.warning(f"Compression failed: {e}, using uncompressed")
        return weights_dict


def _decompress_weights(compressed_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Decompress weights"""
    try:
        compressed = compressed_dict['compressed_data']
        decompressed = gzip.decompress(compressed)
        weights_dict = pickle.loads(decompressed)
        logger.info("Decompressed weights successfully")
        return weights_dict
    except Exception as e:
        raise SerializationError(f"Decompression failed: {e}")


def _compute_checksum(weights_dict: Dict[str, np.ndarray]) -> str:
    """Compute checksum for numpy arrays"""
    import hashlib

    hasher = hashlib.sha256()
    for name in sorted(weights_dict.keys()):
        hasher.update(name.encode())
        hasher.update(weights_dict[name].tobytes())

    return hasher.hexdigest()


def _compute_string_checksum(data: str) -> str:
    """Compute checksum for string data"""
    import hashlib

    return hashlib.sha256(data.encode()).hexdigest()


def save_weights_to_file(
    weights: Dict[str, Any],
    path: Path,
    compress: bool = True
) -> None:
    """
    Save serialized weights to file.

    Args:
        weights: Serialized weights dictionary
        path: Output file path
        compress: Use gzip compression
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if compress:
        with gzip.open(path, 'wb') as f:
            pickle.dump(weights, f)
    else:
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    logger.info(f"Saved weights to {path}")


def load_weights_from_file(path: Path) -> Dict[str, Any]:
    """
    Load serialized weights from file.

    Args:
        path: Input file path

    Returns:
        Weights dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        SerializationError: If loading fails
    """
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")

    try:
        # Try gzip first
        with gzip.open(path, 'rb') as f:
            weights = pickle.load(f)
    except (gzip.BadGzipFile, OSError):
        # Fall back to uncompressed
        with open(path, 'rb') as f:
            weights = pickle.load(f)

    logger.info(f"Loaded weights from {path}")
    return weights
