#!/usr/bin/env python3
"""
集成测试：验证XGBoost联邦学习完整流程

测试从数据加载到模型训练的完整pipeline
"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_snp_data(
    n_samples: int = 1000,
    n_snps: int = 100,
    n_clusters: int = 5,
    n_classes: int = 3,
    random_seed: int = 42
) -> tuple:
    """
    生成合成SNP数据用于测试

    Args:
        n_samples: 样本数量
        n_snps: SNP数量
        n_clusters: 聚类特征数量
        n_classes: 类别数量
        random_seed: 随机种子

    Returns:
        (snp_data, cluster_data, labels)
    """
    np.random.seed(random_seed)

    # 生成SNP数据（0, 1, 2）
    snp_data = np.random.randint(0, 3, size=(n_samples, n_snps))

    # 生成聚类特征（0-1范围）
    cluster_data = np.random.rand(n_samples, n_clusters)

    # 生成标签（基于SNP模式）
    # 使用前几个SNP的组合来决定类别
    label_scores = snp_data[:, :10].sum(axis=1)
    labels = np.digitize(label_scores, bins=[7, 14]) % n_classes

    return snp_data, cluster_data, labels


def create_test_dataset(
    output_dir: Path,
    site_names: List[str],
    samples_per_site: int = 500,
    n_snps: int = 100,
    n_clusters: int = 5
):
    """
    创建测试数据集

    Args:
        output_dir: 输出目录
        site_names: 站点名称列表
        samples_per_site: 每个站点的样本数
        n_snps: SNP数量
        n_clusters: 聚类特征数量
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating test dataset in: {output_dir}")

    for idx, site_name in enumerate(site_names):
        logger.info(f"Generating data for {site_name}...")

        # 生成数据（每个站点使用不同的随机种子）
        snp_data, cluster_data, labels = generate_synthetic_snp_data(
            n_samples=samples_per_site,
            n_snps=n_snps,
            n_clusters=n_clusters,
            random_seed=42 + idx
        )

        # 创建SNP DataFrame
        snp_columns = [f"snp_{i}" for i in range(n_snps)]
        snp_df = pd.DataFrame(snp_data, columns=snp_columns)
        snp_df['label'] = labels

        # 保存训练数据
        train_file = output_dir / f"{site_name}_train.csv"
        snp_df.to_csv(train_file, index=False)
        logger.info(f"  Saved training data: {train_file}")

        # 创建聚类特征DataFrame
        cluster_columns = [f"cluster_{i}" for i in range(n_clusters)]
        cluster_df = pd.DataFrame(cluster_data, columns=cluster_columns)

        # 保存聚类特征
        cluster_file = output_dir / f"{site_name}_cluster_features.csv"
        cluster_df.to_csv(cluster_file, index=False)
        logger.info(f"  Saved cluster features: {cluster_file}")

        # 打印统计信息
        logger.info(f"  Samples: {len(snp_df)}")
        logger.info(f"  Features: SNP={n_snps}, Cluster={n_clusters}")
        logger.info(f"  Label distribution: {np.bincount(labels)}")


def test_data_loader_integration(data_dir: Path, site_name: str):
    """
    测试数据加载器集成

    Args:
        data_dir: 数据目录
        site_name: 站点名称
    """
    logger.info("=" * 80)
    logger.info("Test 1: Data Loader Integration")
    logger.info("=" * 80)

    try:
        from .data_loader import SNPXGBDataLoader

        # 初始化数据加载器
        data_loader = SNPXGBDataLoader(
            data_dir=str(data_dir),
            site_name=site_name,
            use_cluster_features=True,
            validation_split=0.2
        )

        # 加载数据
        train_dmatrix, val_dmatrix = data_loader.load_data()

        # 验证
        assert train_dmatrix is not None, "Training DMatrix is None"
        assert val_dmatrix is not None, "Validation DMatrix is None"
        assert train_dmatrix.num_row() > 0, "No training samples"
        assert val_dmatrix.num_row() > 0, "No validation samples"

        logger.info(f"✓ Training samples: {train_dmatrix.num_row()}")
        logger.info(f"✓ Validation samples: {val_dmatrix.num_row()}")
        logger.info(f"✓ Features: {train_dmatrix.num_col()}")

        return True

    except Exception as e:
        logger.error(f"✗ Data loader test failed: {e}", exc_info=True)
        return False


def test_xgboost_training(data_dir: Path, site_name: str):
    """
    测试XGBoost训练

    Args:
        data_dir: 数据目录
        site_name: 站点名称
    """
    logger.info("\n" + "=" * 80)
    logger.info("Test 2: XGBoost Training")
    logger.info("=" * 80)

    try:
        from .data_loader import SNPXGBDataLoader

        # 加载数据
        data_loader = SNPXGBDataLoader(
            data_dir=str(data_dir),
            site_name=site_name,
            use_cluster_features=True,
            validation_split=0.2
        )
        train_dmatrix, val_dmatrix = data_loader.load_data()

        # XGBoost参数（与NVFlare配置一致）
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eta': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'seed': 42
        }

        # 训练
        evals = [(train_dmatrix, 'train'), (val_dmatrix, 'val')]
        evals_result = {}
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=20,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=5,
            verbose_eval=False
        )

        # 验证
        train_loss = evals_result['train']['mlogloss'][-1]
        val_loss = evals_result['val']['mlogloss'][-1]

        logger.info(f"✓ Training completed")
        logger.info(f"✓ Final train loss: {train_loss:.4f}")
        logger.info(f"✓ Final val loss: {val_loss:.4f}")
        logger.info(f"✓ Best iteration: {bst.best_iteration}")

        # 测试预测
        preds = bst.predict(val_dmatrix)
        assert preds.shape[0] == val_dmatrix.num_row()
        assert preds.shape[1] == 3  # 3 classes

        logger.info(f"✓ Prediction shape: {preds.shape}")

        return True

    except Exception as e:
        logger.error(f"✗ Training test failed: {e}", exc_info=True)
        return False


def test_multi_site_simulation(data_dir: Path, site_names: List[str]):
    """
    测试多站点训练模拟

    Args:
        data_dir: 数据目录
        site_names: 站点名称列表
    """
    logger.info("\n" + "=" * 80)
    logger.info("Test 3: Multi-Site Training Simulation")
    logger.info("=" * 80)

    try:
        from .data_loader import SNPXGBDataLoader

        site_metrics = {}

        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eta': 0.1,
            'max_depth': 6,
            'eval_metric': 'mlogloss',
            'seed': 42
        }

        for site_name in site_names:
            logger.info(f"\nTraining on {site_name}...")

            # 加载站点数据
            data_loader = SNPXGBDataLoader(
                data_dir=str(data_dir),
                site_name=site_name,
                use_cluster_features=True,
                validation_split=0.2
            )
            train_dmatrix, val_dmatrix = data_loader.load_data()

            # 训练
            evals = [(train_dmatrix, 'train'), (val_dmatrix, 'val')]
            evals_result = {}
            bst = xgb.train(
                params,
                train_dmatrix,
                num_boost_round=20,
                evals=evals,
                evals_result=evals_result,
                verbose_eval=False
            )

            # 记录指标
            train_loss = evals_result['train']['mlogloss'][-1]
            val_loss = evals_result['val']['mlogloss'][-1]

            site_metrics[site_name] = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'samples': train_dmatrix.num_row()
            }

            logger.info(f"  Train samples: {train_dmatrix.num_row()}")
            logger.info(f"  Train loss: {train_loss:.4f}")
            logger.info(f"  Val loss: {val_loss:.4f}")

        # 汇总
        logger.info("\n" + "-" * 80)
        logger.info("Multi-Site Training Summary:")
        logger.info("-" * 80)

        avg_train_loss = np.mean([m['train_loss'] for m in site_metrics.values()])
        avg_val_loss = np.mean([m['val_loss'] for m in site_metrics.values()])
        total_samples = sum([m['samples'] for m in site_metrics.values()])

        logger.info(f"Total sites: {len(site_names)}")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Average train loss: {avg_train_loss:.4f}")
        logger.info(f"Average val loss: {avg_val_loss:.4f}")

        logger.info("\n✓ Multi-site simulation completed")

        return True

    except Exception as e:
        logger.error(f"✗ Multi-site test failed: {e}", exc_info=True)
        return False


def test_config_validation(config_dir: Path):
    """
    测试NVFlare配置文件有效性

    Args:
        config_dir: 配置文件目录
    """
    logger.info("\n" + "=" * 80)
    logger.info("Test 4: Configuration Validation")
    logger.info("=" * 80)

    try:
        config_files = [
            "config_fed_server.json",
            "config_fed_client.json",
            "config_fed_server_gpu.json",
            "config_fed_client_gpu.json"
        ]

        for config_file in config_files:
            config_path = config_dir / config_file
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

                # 基本验证
                assert 'format_version' in config, f"Missing format_version in {config_file}"

                logger.info(f"✓ {config_file} is valid")
            else:
                logger.warning(f"- {config_file} not found (optional)")

        return True

    except Exception as e:
        logger.error(f"✗ Config validation failed: {e}", exc_info=True)
        return False


def run_integration_tests(
    use_temp_data: bool = True,
    data_dir: str = None,
    site_names: List[str] = None,
    config_dir: str = None
):
    """
    运行完整集成测试

    Args:
        use_temp_data: 是否使用临时测试数据
        data_dir: 数据目录（如果不使用临时数据）
        site_names: 站点名称列表
        config_dir: 配置文件目录
    """
    logger.info("=" * 80)
    logger.info("XGBoost Federated Learning Integration Tests")
    logger.info("=" * 80)

    results = {}

    if site_names is None:
        site_names = ["site1", "site2", "site3"]

    # 准备数据
    if use_temp_data:
        temp_dir = tempfile.mkdtemp(prefix="xgboost_test_")
        test_data_dir = Path(temp_dir)
        logger.info(f"\nUsing temporary data directory: {test_data_dir}")

        create_test_dataset(
            test_data_dir,
            site_names=site_names,
            samples_per_site=500,
            n_snps=100,
            n_clusters=5
        )
    else:
        test_data_dir = Path(data_dir)
        logger.info(f"\nUsing existing data directory: {test_data_dir}")

    # 运行测试
    results['data_loader'] = test_data_loader_integration(test_data_dir, site_names[0])
    results['training'] = test_xgboost_training(test_data_dir, site_names[0])
    results['multi_site'] = test_multi_site_simulation(test_data_dir, site_names)

    # 配置验证
    if config_dir:
        results['config'] = test_config_validation(Path(config_dir))

    # 汇总结果
    logger.info("\n" + "=" * 80)
    logger.info("Integration Test Results")
    logger.info("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False

    logger.info("=" * 80)

    if all_passed:
        logger.info("✓ All integration tests passed!")
        return 0
    else:
        logger.error("✗ Some tests failed")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Run XGBoost federated learning integration tests"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Existing data directory (if not using synthetic data)"
    )
    parser.add_argument(
        "--site_names",
        type=str,
        nargs='+',
        default=["site1", "site2", "site3"],
        help="Site names to test"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        help="Configuration directory to validate"
    )
    parser.add_argument(
        "--no_synthetic",
        action="store_true",
        help="Use existing data instead of generating synthetic data"
    )

    args = parser.parse_args()

    # 验证参数
    if args.no_synthetic and not args.data_dir:
        logger.error("Must provide --data_dir when using --no_synthetic")
        sys.exit(1)

    # 运行测试
    exit_code = run_integration_tests(
        use_temp_data=not args.no_synthetic,
        data_dir=args.data_dir,
        site_names=args.site_names,
        config_dir=args.config_dir
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
