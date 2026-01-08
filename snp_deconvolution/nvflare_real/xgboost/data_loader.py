"""
XGBoost数据加载器 - 用于NVFlare联邦训练

官方要求：继承XGBDataLoader基类，实现load_data()方法返回XGBoost DMatrix
Reference: https://nvflare.readthedocs.io/en/2.5.1/user_guide/federated_xgboost/

This data loader is called by FedXGBHistogramExecutor during federated training.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

# Import NVFlare XGBDataLoader base class
try:
    from nvflare.app_opt.xgboost.data_loader import XGBDataLoader
except ImportError:
    # Fallback for older versions
    XGBDataLoader = object

logger = logging.getLogger(__name__)


class SNPXGBDataLoader(XGBDataLoader):
    """
    被 FedXGBHistogramExecutor 调用的数据加载器

    必须实现 load_data() 方法返回 XGBoost DMatrix

    NVFlare要求：
    - load_data() 必须返回 xgb.DMatrix 或 (train_dmatrix, val_dmatrix)
    - 数据加载器会在客户端初始化时被调用
    """

    def __init__(
        self,
        data_dir: str,
        site_name: str,
        use_cluster_features: bool = True,
        validation_split: float = 0.2,
        random_seed: int = 42,
        feature_prefix: str = "snp_",
        label_column: str = "label",
        enable_categorical: bool = False
    ):
        """
        初始化XGBoost数据加载器

        Args:
            data_dir: 数据目录路径
            site_name: 站点名称（用于加载对应的数据文件）
            use_cluster_features: 是否使用聚类特征（默认True）
            validation_split: 验证集比例（默认0.2）
            random_seed: 随机种子
            feature_prefix: 特征列前缀
            label_column: 标签列名
            enable_categorical: 是否启用XGBoost的categorical特征支持
        """
        self.data_dir = Path(data_dir)
        self.site_name = site_name
        self.use_cluster_features = use_cluster_features
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.feature_prefix = feature_prefix
        self.label_column = label_column
        self.enable_categorical = enable_categorical

        # Normalize site name (site-1 -> site1)
        self.normalized_site_name = self._normalize_site_name(site_name)

        logger.info(f"Initialized SNPXGBDataLoader for site: {site_name} (normalized: {self.normalized_site_name})")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Use cluster features: {use_cluster_features}")
        logger.info(f"Validation split: {validation_split}")

    def _normalize_site_name(self, site_name: str) -> str:
        """Normalize site name: site-1 -> site1"""
        normalized = site_name.replace('-', '')
        # Check if normalized version exists
        if (self.data_dir / normalized).exists():
            return normalized
        return site_name

    def _load_raw_data(self) -> pd.DataFrame:
        """
        加载原始数据文件

        支持的文件格式：
        1. NPZ格式: {site_name}/train_cluster.npz
        2. CSV格式: {site_name}_train.csv, {site_name}.csv

        Returns:
            DataFrame with features and labels
        """
        # 首先尝试 NPZ 格式 (from prepare_federated_data.py)
        npz_train = self.data_dir / self.normalized_site_name / "train_cluster.npz"
        if npz_train.exists():
            logger.info(f"Loading NPZ data from: {npz_train}")
            train_data = np.load(npz_train)
            X = train_data['X']
            y = train_data['y']

            # Convert to DataFrame format
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            df['label'] = y
            logger.info(f"Loaded {len(df)} samples with {X.shape[1]} features from NPZ")
            return df

        # 尝试多种CSV文件命名模式
        possible_files = [
            self.data_dir / f"{self.normalized_site_name}_train.csv",
            self.data_dir / f"{self.normalized_site_name}.csv",
            self.data_dir / f"{self.site_name}_train.csv",
            self.data_dir / f"{self.site_name}.csv",
        ]

        data_file = None
        for file_path in possible_files:
            if file_path.exists():
                data_file = file_path
                break

        if data_file is None:
            raise FileNotFoundError(
                f"Could not find data file for site {self.site_name}. "
                f"Tried NPZ: {npz_train}, CSV: {[str(f) for f in possible_files]}"
            )

        logger.info(f"Loading CSV data from: {data_file}")
        df = pd.read_csv(data_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

        return df

    def _load_cluster_features(self) -> Optional[pd.DataFrame]:
        """
        加载聚类特征（如果可用）

        Returns:
            DataFrame with cluster features or None
        """
        if not self.use_cluster_features:
            return None

        cluster_file = self.data_dir / f"{self.site_name}_cluster_features.csv"
        if not cluster_file.exists():
            logger.warning(f"Cluster features file not found: {cluster_file}")
            return None

        logger.info(f"Loading cluster features from: {cluster_file}")
        cluster_df = pd.read_csv(cluster_file)
        logger.info(f"Loaded {len(cluster_df.columns)} cluster features")

        return cluster_df

    def _prepare_features_and_labels(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和标签

        Args:
            df: 原始数据框

        Returns:
            (features, labels) as numpy arrays
        """
        # 提取标签
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in data")

        labels = df[self.label_column].values

        # 提取SNP特征
        snp_columns = [col for col in df.columns if col.startswith(self.feature_prefix)]
        if not snp_columns:
            logger.warning(f"No SNP features found with prefix '{self.feature_prefix}'")
            # 尝试使用所有数值列（除了标签）
            snp_columns = [
                col for col in df.columns
                if col != self.label_column and pd.api.types.is_numeric_dtype(df[col])
            ]

        features = df[snp_columns].values
        logger.info(f"Prepared {features.shape[1]} SNP features")

        # 加载并合并聚类特征
        cluster_df = self._load_cluster_features()
        if cluster_df is not None:
            # 确保行数匹配
            if len(cluster_df) != len(df):
                logger.warning(
                    f"Cluster features length mismatch: {len(cluster_df)} vs {len(df)}"
                )
            else:
                cluster_features = cluster_df.values
                features = np.hstack([features, cluster_features])
                logger.info(
                    f"Combined features: {features.shape[1]} "
                    f"(SNP: {len(snp_columns)}, Cluster: {cluster_features.shape[1]})"
                )

        # 数据验证
        if np.any(np.isnan(features)):
            logger.warning("Found NaN values in features, filling with 0")
            features = np.nan_to_num(features, nan=0.0)

        if np.any(np.isnan(labels)):
            raise ValueError("Found NaN values in labels")

        logger.info(f"Feature shape: {features.shape}, Label shape: {labels.shape}")
        logger.info(f"Label distribution: {np.bincount(labels.astype(int))}")

        return features, labels

    def _create_train_val_split(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        创建训练集和验证集划分

        Args:
            features: 特征数组
            labels: 标签数组

        Returns:
            ((X_train, y_train), (X_val, y_val))
        """
        if self.validation_split <= 0 or self.validation_split >= 1:
            logger.info("No validation split, using all data for training")
            return (features, labels), (None, None)

        # 分层采样
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=self.validation_split,
            random_state=self.random_seed,
            stratify=labels
        )

        logger.info(f"Train split: {len(X_train)} samples")
        logger.info(f"Validation split: {len(X_val)} samples")
        logger.info(f"Train label distribution: {np.bincount(y_train.astype(int))}")
        logger.info(f"Val label distribution: {np.bincount(y_val.astype(int))}")

        return (X_train, y_train), (X_val, y_val)

    def load_data(self) -> Tuple[xgb.DMatrix, Optional[xgb.DMatrix]]:
        """
        加载数据并返回XGBoost DMatrix

        NVFlare required method - 被FedXGBHistogramExecutor调用

        Returns:
            训练DMatrix 或 (训练DMatrix, 验证DMatrix)
        """
        logger.info(f"Loading data for site: {self.site_name}")

        try:
            # 1. 加载原始数据
            df = self._load_raw_data()

            # 2. 准备特征和标签
            features, labels = self._prepare_features_and_labels(df)

            # 3. 创建训练/验证划分
            (X_train, y_train), (X_val, y_val) = self._create_train_val_split(
                features, labels
            )

            # 4. 创建DMatrix
            enable_categorical = self.enable_categorical

            train_dmatrix = xgb.DMatrix(
                X_train,
                label=y_train,
                enable_categorical=enable_categorical
            )
            logger.info(f"Created training DMatrix: {train_dmatrix.num_row()} samples")

            # 5. 创建验证DMatrix（如果需要）
            val_dmatrix = None
            if X_val is not None and y_val is not None:
                val_dmatrix = xgb.DMatrix(
                    X_val,
                    label=y_val,
                    enable_categorical=enable_categorical
                )
                logger.info(f"Created validation DMatrix: {val_dmatrix.num_row()} samples")

            # 6. 返回结果
            if val_dmatrix is not None:
                return train_dmatrix, val_dmatrix
            else:
                return train_dmatrix

        except Exception as e:
            logger.error(f"Failed to load data: {e}", exc_info=True)
            raise

    def get_data_info(self) -> Dict[str, any]:
        """
        获取数据集信息（用于调试和监控）

        Returns:
            数据集统计信息
        """
        try:
            df = self._load_raw_data()
            features, labels = self._prepare_features_and_labels(df)

            return {
                "site_name": self.site_name,
                "num_samples": len(df),
                "num_features": features.shape[1],
                "num_classes": len(np.unique(labels)),
                "label_distribution": np.bincount(labels.astype(int)).tolist(),
                "use_cluster_features": self.use_cluster_features,
                "validation_split": self.validation_split,
            }
        except Exception as e:
            logger.error(f"Failed to get data info: {e}")
            return {"error": str(e)}


# 兼容性别名
XGBDataLoader = SNPXGBDataLoader
