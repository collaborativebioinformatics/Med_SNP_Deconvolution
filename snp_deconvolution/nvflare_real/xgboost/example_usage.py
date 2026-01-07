"""
示例：如何使用SNPXGBDataLoader

展示数据加载器的各种用法
"""
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from .data_loader import SNPXGBDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """
    示例1: 基本用法
    """
    print("\n" + "=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # 初始化数据加载器
    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1",
        use_cluster_features=True,
        validation_split=0.2
    )

    # 加载数据
    train_dmatrix, val_dmatrix = data_loader.load_data()

    print(f"Training samples: {train_dmatrix.num_row()}")
    print(f"Features: {train_dmatrix.num_col()}")
    print(f"Validation samples: {val_dmatrix.num_row()}")


def example_training_with_validation():
    """
    示例2: 使用验证集训练
    """
    print("\n" + "=" * 80)
    print("Example 2: Training with Validation")
    print("=" * 80)

    # 加载数据
    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1",
        validation_split=0.2
    )
    train_dmatrix, val_dmatrix = data_loader.load_data()

    # XGBoost参数
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': 'mlogloss'
    }

    # 训练模型
    evals = [(train_dmatrix, 'train'), (val_dmatrix, 'val')]
    bst = xgb.train(
        params,
        train_dmatrix,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    # 保存模型
    bst.save_model("model.json")
    print("Model saved to model.json")


def example_no_cluster_features():
    """
    示例3: 不使用聚类特征
    """
    print("\n" + "=" * 80)
    print("Example 3: Without Cluster Features")
    print("=" * 80)

    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1",
        use_cluster_features=False,  # 禁用聚类特征
        validation_split=0.2
    )

    train_dmatrix, val_dmatrix = data_loader.load_data()
    print(f"Features (SNP only): {train_dmatrix.num_col()}")


def example_custom_split():
    """
    示例4: 自定义验证集比例
    """
    print("\n" + "=" * 80)
    print("Example 4: Custom Validation Split")
    print("=" * 80)

    # 30% 验证集
    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1",
        validation_split=0.3
    )

    train_dmatrix, val_dmatrix = data_loader.load_data()

    total = train_dmatrix.num_row() + val_dmatrix.num_row()
    print(f"Total samples: {total}")
    print(f"Train: {train_dmatrix.num_row()} ({train_dmatrix.num_row()/total:.1%})")
    print(f"Val: {val_dmatrix.num_row()} ({val_dmatrix.num_row()/total:.1%})")


def example_no_validation():
    """
    示例5: 不使用验证集
    """
    print("\n" + "=" * 80)
    print("Example 5: No Validation Split")
    print("=" * 80)

    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1",
        validation_split=0.0  # 不划分验证集
    )

    train_dmatrix = data_loader.load_data()
    print(f"All samples used for training: {train_dmatrix.num_row()}")


def example_get_data_info():
    """
    示例6: 获取数据信息
    """
    print("\n" + "=" * 80)
    print("Example 6: Get Data Information")
    print("=" * 80)

    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1"
    )

    # 获取数据统计信息
    info = data_loader.get_data_info()

    print("Data Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


def example_cross_validation():
    """
    示例7: 手动K折交叉验证
    """
    print("\n" + "=" * 80)
    print("Example 7: Manual K-Fold Cross Validation")
    print("=" * 80)

    from sklearn.model_selection import KFold

    # 先加载所有数据（不划分验证集）
    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1",
        validation_split=0.0
    )

    # 获取所有数据
    full_dmatrix = data_loader.load_data()
    X = full_dmatrix.get_data()
    y = full_dmatrix.get_label()

    # K折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': 'mlogloss'
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/5")

        # 创建DMatrix
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx])

        # 训练
        evals = [(dtrain, 'train'), (dval, 'val')]
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=50,
            evals=evals,
            verbose_eval=False
        )

        # 评估
        preds = bst.predict(dval)
        from sklearn.metrics import log_loss
        score = log_loss(y[val_idx], preds)
        scores.append(score)
        print(f"  Val mlogloss: {score:.4f}")

    print(f"\nMean CV mlogloss: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")


def example_gpu_training():
    """
    示例8: GPU训练
    """
    print("\n" + "=" * 80)
    print("Example 8: GPU Training")
    print("=" * 80)

    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1",
        validation_split=0.2
    )

    train_dmatrix, val_dmatrix = data_loader.load_data()

    # GPU参数
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.1,
        'tree_method': 'gpu_hist',      # GPU histogram
        'predictor': 'gpu_predictor',   # GPU predictor
        'gpu_id': 0,                    # GPU device ID
        'eval_metric': 'mlogloss'
    }

    evals = [(train_dmatrix, 'train'), (val_dmatrix, 'val')]
    bst = xgb.train(
        params,
        train_dmatrix,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    print("GPU training completed")


def example_feature_importance():
    """
    示例9: 特征重要性分析
    """
    print("\n" + "=" * 80)
    print("Example 9: Feature Importance Analysis")
    print("=" * 80)

    data_loader = SNPXGBDataLoader(
        data_dir="/path/to/data",
        site_name="site1",
        validation_split=0.2
    )

    train_dmatrix, val_dmatrix = data_loader.load_data()

    # 训练模型
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'eta': 0.1,
        'eval_metric': 'mlogloss'
    }

    bst = xgb.train(params, train_dmatrix, num_boost_round=100)

    # 获取特征重要性
    importance = bst.get_score(importance_type='weight')

    # 排序并显示Top 10
    sorted_importance = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 10 Most Important Features:")
    for i, (feat, score) in enumerate(sorted_importance[:10], 1):
        print(f"  {i}. {feat}: {score:.2f}")


def example_multi_site_training():
    """
    示例10: 模拟多站点训练（本地聚合）
    """
    print("\n" + "=" * 80)
    print("Example 10: Multi-Site Training Simulation")
    print("=" * 80)

    sites = ["site1", "site2", "site3"]

    # 每个站点训练本地模型
    site_models = []
    for site in sites:
        print(f"\nTraining on {site}...")

        data_loader = SNPXGBDataLoader(
            data_dir="/path/to/data",
            site_name=site,
            validation_split=0.0
        )

        train_dmatrix = data_loader.load_data()

        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'eta': 0.1,
            'eval_metric': 'mlogloss'
        }

        bst = xgb.train(params, train_dmatrix, num_boost_round=50, verbose_eval=False)
        site_models.append(bst)
        print(f"  {site} training completed")

    print(f"\nTrained {len(site_models)} site models")
    print("In real federated learning, these models would be aggregated by NVFlare")


if __name__ == "__main__":
    # 运行示例（需要修改数据路径）

    print("SNPXGBDataLoader Usage Examples")
    print("================================")
    print("\nNote: Update data paths before running these examples")

    # 取消注释要运行的示例
    # example_basic_usage()
    # example_training_with_validation()
    # example_no_cluster_features()
    # example_custom_split()
    # example_no_validation()
    # example_get_data_info()
    # example_cross_validation()
    # example_gpu_training()
    # example_feature_importance()
    # example_multi_site_training()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
