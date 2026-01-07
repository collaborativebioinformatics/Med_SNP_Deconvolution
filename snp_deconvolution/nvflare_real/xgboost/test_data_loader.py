"""
测试和验证XGBoost数据加载器

用于验证数据格式、加载逻辑和DMatrix创建
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

from .data_loader import SNPXGBDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_loader(
    data_dir: str,
    site_name: str,
    use_cluster_features: bool = True,
    validation_split: float = 0.2
):
    """
    测试数据加载器功能

    Args:
        data_dir: 数据目录
        site_name: 站点名称
        use_cluster_features: 是否使用聚类特征
        validation_split: 验证集比例
    """
    logger.info("=" * 80)
    logger.info("Testing SNPXGBDataLoader")
    logger.info("=" * 80)

    try:
        # 1. 初始化数据加载器
        logger.info(f"\n1. Initializing data loader for site: {site_name}")
        data_loader = SNPXGBDataLoader(
            data_dir=data_dir,
            site_name=site_name,
            use_cluster_features=use_cluster_features,
            validation_split=validation_split
        )
        logger.info("✓ Data loader initialized successfully")

        # 2. 获取数据信息
        logger.info("\n2. Getting data info...")
        info = data_loader.get_data_info()
        logger.info("Data Info:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")

        # 3. 加载数据
        logger.info("\n3. Loading data and creating DMatrix...")
        result = data_loader.load_data()

        if isinstance(result, tuple):
            train_dmatrix, val_dmatrix = result
            logger.info(f"✓ Training DMatrix created: {train_dmatrix.num_row()} samples, "
                       f"{train_dmatrix.num_col()} features")
            if val_dmatrix is not None:
                logger.info(f"✓ Validation DMatrix created: {val_dmatrix.num_row()} samples, "
                           f"{val_dmatrix.num_col()} features")
        else:
            train_dmatrix = result
            val_dmatrix = None
            logger.info(f"✓ Training DMatrix created: {train_dmatrix.num_row()} samples, "
                       f"{train_dmatrix.num_col()} features")

        # 4. 验证标签分布
        logger.info("\n4. Validating label distribution...")
        train_labels = train_dmatrix.get_label()
        unique_labels = np.unique(train_labels)
        logger.info(f"Unique labels: {unique_labels}")
        logger.info(f"Label counts: {np.bincount(train_labels.astype(int))}")

        if val_dmatrix is not None:
            val_labels = val_dmatrix.get_label()
            logger.info(f"Validation label counts: {np.bincount(val_labels.astype(int))}")

        # 5. 测试基本XGBoost训练
        logger.info("\n5. Testing basic XGBoost training (5 rounds)...")
        params = {
            'objective': 'multi:softprob',
            'num_class': len(unique_labels),
            'max_depth': 3,
            'eta': 0.3,
            'eval_metric': 'mlogloss'
        }

        if val_dmatrix is not None:
            evals = [(train_dmatrix, 'train'), (val_dmatrix, 'val')]
        else:
            evals = [(train_dmatrix, 'train')]

        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=5,
            evals=evals,
            verbose_eval=True
        )
        logger.info("✓ Basic training completed successfully")

        # 6. 测试预测
        logger.info("\n6. Testing predictions...")
        preds = bst.predict(train_dmatrix)
        logger.info(f"Prediction shape: {preds.shape}")
        logger.info(f"Sample predictions (first 3):\n{preds[:3]}")

        logger.info("\n" + "=" * 80)
        logger.info("All tests passed successfully!")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"\n{'=' * 80}")
        logger.error(f"Test failed with error: {e}", exc_info=True)
        logger.error("=" * 80)
        return False


def validate_data_files(data_dir: str, site_names: list):
    """
    验证所有站点的数据文件是否存在

    Args:
        data_dir: 数据目录
        site_names: 站点名称列表
    """
    logger.info("=" * 80)
    logger.info("Validating data files")
    logger.info("=" * 80)

    data_path = Path(data_dir)
    all_valid = True

    for site_name in site_names:
        logger.info(f"\nChecking site: {site_name}")

        # 检查训练数据文件
        possible_files = [
            data_path / f"{site_name}_train.csv",
            data_path / f"{site_name}.csv",
            data_path / f"site_{site_name}.csv",
        ]

        found = False
        for file_path in possible_files:
            if file_path.exists():
                logger.info(f"  ✓ Training data found: {file_path}")
                found = True
                break

        if not found:
            logger.error(f"  ✗ No training data found for {site_name}")
            logger.error(f"    Tried: {[str(f) for f in possible_files]}")
            all_valid = False

        # 检查聚类特征文件（可选）
        cluster_file = data_path / f"{site_name}_cluster_features.csv"
        if cluster_file.exists():
            logger.info(f"  ✓ Cluster features found: {cluster_file}")
        else:
            logger.info(f"  - Cluster features not found (optional): {cluster_file}")

    logger.info("\n" + "=" * 80)
    if all_valid:
        logger.info("All required data files are present")
    else:
        logger.error("Some data files are missing!")
    logger.info("=" * 80)

    return all_valid


def main():
    parser = argparse.ArgumentParser(
        description="Test and validate SNPXGBDataLoader"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory"
    )
    parser.add_argument(
        "--site_name",
        type=str,
        help="Site name to test (single site test)"
    )
    parser.add_argument(
        "--site_names",
        type=str,
        nargs='+',
        help="List of site names to validate (validation mode)"
    )
    parser.add_argument(
        "--no_cluster_features",
        action="store_true",
        help="Disable cluster features"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate file existence, don't test loading"
    )

    args = parser.parse_args()

    # 验证模式：检查所有站点的文件
    if args.validate_only:
        if not args.site_names:
            logger.error("Please provide --site_names for validation mode")
            sys.exit(1)

        success = validate_data_files(args.data_dir, args.site_names)
        sys.exit(0 if success else 1)

    # 测试模式：测试单个站点的数据加载
    if not args.site_name:
        logger.error("Please provide --site_name for test mode")
        sys.exit(1)

    success = test_data_loader(
        data_dir=args.data_dir,
        site_name=args.site_name,
        use_cluster_features=not args.no_cluster_features,
        validation_split=args.validation_split
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
