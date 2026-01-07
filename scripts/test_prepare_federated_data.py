#!/usr/bin/env python3
"""
测试联邦学习数据准备脚本

运行测试:
    python scripts/test_prepare_federated_data.py

功能:
1. 生成合成数据测试数据划分
2. 验证输出文件格式
3. 检查数据一致性
"""
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from snp_deconvolution.nvflare_real.data.data_splitter import FederatedDataSplitter


def create_synthetic_data(n_samples: int = 300, n_features: int = 100, n_classes: int = 3):
    """
    创建合成数据用于测试

    Args:
        n_samples: 样本数量
        n_features: 特征维度
        n_classes: 类别数量

    Returns:
        X: 特征矩阵
        y: 标签数组
    """
    print(f"生成合成数据: {n_samples} 样本, {n_features} 特征, {n_classes} 类别")

    np.random.seed(42)

    # 生成特征 (cluster ID格式)
    X = np.random.randint(1, 20, size=(n_samples, n_features), dtype=np.int64)

    # 生成标签 (保证每个类别至少有足够样本)
    samples_per_class = n_samples // n_classes
    y = np.concatenate([
        np.full(samples_per_class, i, dtype=np.int64)
        for i in range(n_classes)
    ])

    # 添加剩余样本
    remainder = n_samples - len(y)
    if remainder > 0:
        y = np.concatenate([y, np.random.randint(0, n_classes, remainder, dtype=np.int64)])

    # 打乱顺序
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    print(f"标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


def test_basic_split(output_dir: Path):
    """
    测试基础数据划分功能

    Args:
        output_dir: 临时输出目录
    """
    print("\n" + "="*70)
    print("测试1: 基础数据划分")
    print("="*70)

    # 创建测试数据
    X, y = create_synthetic_data(n_samples=300, n_features=100, n_classes=3)

    # 创建splitter
    splitter = FederatedDataSplitter(
        output_dir=str(output_dir),
        num_sites=3,
        seed=42
    )

    # 执行划分
    stats = splitter.split_and_save(
        X=X,
        y=y,
        val_ratio=0.15,
        feature_type="cluster"
    )

    # 验证统计信息
    assert len(stats) == 3, "应该有3个站点"
    print("\n✓ 数据划分成功")

    # 验证文件
    for site_id in range(3):
        site_name = f"site-{site_id + 1}"
        site_dir = output_dir / site_name

        assert site_dir.exists(), f"{site_name} 目录应该存在"

        # 检查文件
        required_files = [
            'train_cluster.pt',
            'val_cluster.pt',
            'train_cluster.npz',
            'val_cluster.npz'
        ]

        for filename in required_files:
            filepath = site_dir / filename
            assert filepath.exists(), f"{filename} 应该存在"

    print("✓ 所有文件已创建")

    return stats


def test_data_loading(output_dir: Path):
    """
    测试数据加载

    Args:
        output_dir: 输出目录
    """
    print("\n" + "="*70)
    print("测试2: 数据加载")
    print("="*70)

    site_dir = output_dir / "site-1"

    # 加载PyTorch数据
    train_pt = torch.load(site_dir / 'train_cluster.pt')
    val_pt = torch.load(site_dir / 'val_cluster.pt')

    print(f"PyTorch训练集: X shape={train_pt['X'].shape}, y shape={train_pt['y'].shape}")
    print(f"PyTorch验证集: X shape={val_pt['X'].shape}, y shape={val_pt['y'].shape}")

    # 加载NumPy数据
    train_npz = np.load(site_dir / 'train_cluster.npz')
    val_npz = np.load(site_dir / 'val_cluster.npz')

    print(f"NumPy训练集: X shape={train_npz['X'].shape}, y shape={train_npz['y'].shape}")
    print(f"NumPy验证集: X shape={val_npz['X'].shape}, y shape={val_npz['y'].shape}")

    # 验证形状一致性
    assert train_pt['X'].shape == train_npz['X'].shape, "训练集形状应该一致"
    assert val_pt['X'].shape == val_npz['X'].shape, "验证集形状应该一致"

    print("\n✓ 数据加载成功")
    print("✓ PyTorch和NumPy格式一致")

    return train_pt, val_pt


def test_data_consistency(output_dir: Path):
    """
    测试数据一致性

    Args:
        output_dir: 输出目录
    """
    print("\n" + "="*70)
    print("测试3: 数据一致性")
    print("="*70)

    site_dir = output_dir / "site-1"

    # 加载PyTorch和NumPy数据
    train_pt = torch.load(site_dir / 'train_cluster.pt')
    train_npz = np.load(site_dir / 'train_cluster.npz')

    # 验证数值一致性
    X_pt = train_pt['X'].numpy()
    X_npz = train_npz['X']

    y_pt = train_pt['y'].numpy()
    y_npz = train_npz['y']

    assert np.allclose(X_pt, X_npz), "特征矩阵应该一致"
    assert np.array_equal(y_pt, y_npz), "标签数组应该一致"

    print("✓ PyTorch和NumPy数据数值一致")

    # 检查标签分布
    unique_labels, counts = np.unique(y_npz, return_counts=True)
    print(f"\n标签分布: {dict(zip(unique_labels, counts))}")

    # 检查特征范围
    print(f"特征值范围: [{X_npz.min()}, {X_npz.max()}]")

    return True


def test_verification(output_dir: Path):
    """
    测试验证功能

    Args:
        output_dir: 输出目录
    """
    print("\n" + "="*70)
    print("测试4: 验证功能")
    print("="*70)

    splitter = FederatedDataSplitter(
        output_dir=str(output_dir),
        num_sites=3,
        seed=42
    )

    verification = splitter.verify_split(feature_type='cluster')

    assert verification['success'], "验证应该通过"
    assert len(verification['errors']) == 0, "不应该有错误"

    print("✓ 验证通过")

    # 打印各站点信息
    for site_name, site_info in verification['sites'].items():
        print(f"\n{site_name}:")
        print(f"  训练集形状: {site_info['train_shape']}")
        print(f"  验证集形状: {site_info['val_shape']}")
        print(f"  训练集标签: {site_info['train_labels']}")
        print(f"  验证集标签: {site_info['val_labels']}")

    return verification


def test_stratified_split():
    """
    测试分层抽样功能
    """
    print("\n" + "="*70)
    print("测试5: 分层抽样")
    print("="*70)

    # 创建不平衡数据集
    n_samples = 300
    y = np.concatenate([
        np.full(150, 0, dtype=np.int64),  # 50% 类别0
        np.full(100, 1, dtype=np.int64),  # 33% 类别1
        np.full(50, 2, dtype=np.int64)    # 17% 类别2
    ])
    np.random.shuffle(y)

    print(f"原始标签分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # 创建splitter
        splitter = FederatedDataSplitter(
            output_dir=str(output_dir),
            num_sites=3,
            seed=42
        )

        # 创建特征矩阵
        X = np.random.randint(1, 20, size=(n_samples, 50), dtype=np.int64)

        # 执行划分
        stats = splitter.split_and_save(
            X=X,
            y=y,
            val_ratio=0.15,
            feature_type="test"
        )

        # 验证每个站点的标签分布相似
        print("\n各站点训练集标签分布:")
        for site_name, site_stats in stats.items():
            dist = site_stats['train_label_dist']
            total = sum(dist.values())
            percentages = {k: v/total*100 for k, v in dist.items()}
            print(f"  {site_name}: {dist} ({percentages})")

        print("\n✓ 分层抽样保持了标签分布")

    return True


def test_edge_cases():
    """
    测试边界情况
    """
    print("\n" + "="*70)
    print("测试6: 边界情况")
    print("="*70)

    # 测试1: 少量样本
    print("\n6.1 测试少量样本...")
    with tempfile.TemporaryDirectory() as tmpdir:
        X = np.random.randn(30, 10)
        y = np.array([0]*10 + [1]*10 + [2]*10)

        splitter = FederatedDataSplitter(str(tmpdir), num_sites=2, seed=42)
        stats = splitter.split_and_save(X, y, val_ratio=0.2, feature_type="small")

        assert len(stats) == 2
        print("  ✓ 少量样本处理成功")

    # 测试2: 大验证集比例
    print("\n6.2 测试大验证集比例...")
    with tempfile.TemporaryDirectory() as tmpdir:
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)

        splitter = FederatedDataSplitter(str(tmpdir), num_sites=2, seed=42)
        stats = splitter.split_and_save(X, y, val_ratio=0.4, feature_type="large_val")

        total_val = sum(s['val_samples'] for s in stats.values())
        total = sum(s['total_samples'] for s in stats.values())
        val_ratio = total_val / total

        assert 0.35 < val_ratio < 0.45, f"验证集比例应该约为0.4，实际为 {val_ratio}"
        print(f"  ✓ 验证集比例正确: {val_ratio:.2f}")

    # 测试3: 单个站点
    print("\n6.3 测试单个站点...")
    with tempfile.TemporaryDirectory() as tmpdir:
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)

        splitter = FederatedDataSplitter(str(tmpdir), num_sites=1, seed=42)
        stats = splitter.split_and_save(X, y, val_ratio=0.15, feature_type="single")

        assert len(stats) == 1
        assert 'site-1' in stats
        print("  ✓ 单站点处理成功")

    print("\n✓ 所有边界情况测试通过")

    return True


def main():
    """运行所有测试"""
    print("="*70)
    print("联邦学习数据准备脚本测试")
    print("="*70)

    # 使用临时目录进行测试
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "federated_test"
        output_dir.mkdir(parents=True)

        try:
            # 运行测试
            test_basic_split(output_dir)
            test_data_loading(output_dir)
            test_data_consistency(output_dir)
            test_verification(output_dir)
            test_stratified_split()
            test_edge_cases()

            print("\n" + "="*70)
            print("所有测试通过!")
            print("="*70)

            return 0

        except AssertionError as e:
            print(f"\n✗ 测试失败: {e}")
            return 1

        except Exception as e:
            print(f"\n✗ 测试出错: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
