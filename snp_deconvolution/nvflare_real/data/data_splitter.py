"""
联邦学习数据划分工具

将数据随机均分到多个站点，支持水平联邦学习（行分割）
"""
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class FederatedDataSplitter:
    """
    将数据随机均分到多个站点

    支持：
    - PyTorch格式输出（Lightning用）
    - NumPy格式输出（XGBoost用）
    - 分层抽样保证各站点标签分布一致
    """

    def __init__(self, output_dir: str, num_sites: int = 3, seed: int = 42):
        """
        Args:
            output_dir: 输出目录
            num_sites: 站点数量
            seed: 随机种子
        """
        self.output_dir = Path(output_dir)
        self.num_sites = num_sites
        self.seed = seed

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置随机种子
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        logger.info(
            f"初始化FederatedDataSplitter: "
            f"num_sites={num_sites}, seed={seed}, output_dir={output_dir}"
        )

    def split_and_save(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.15,
        feature_type: str = "cluster"  # "cluster" 或 "snp"
    ) -> Dict[str, Dict[str, Any]]:
        """
        划分数据并保存到各站点目录

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            val_ratio: 验证集比例
            feature_type: 特征类型，用于文件命名

        Returns:
            各站点的样本统计信息
        """
        logger.info(
            f"开始数据划分: X.shape={X.shape}, y.shape={y.shape}, "
            f"val_ratio={val_ratio}, feature_type={feature_type}"
        )

        # 验证输入
        assert len(X) == len(y), "X和y的样本数必须一致"
        assert 0 < val_ratio < 1, "val_ratio必须在(0, 1)之间"
        assert self.num_sites > 0, "num_sites必须大于0"

        n_samples = len(X)
        logger.info(f"总样本数: {n_samples}")
        logger.info(f"标签分布: {dict(Counter(y))}")

        # 第一步: 使用分层抽样将数据随机打乱并均分到各站点
        site_indices = self._stratified_split_to_sites(y, n_samples)

        # 第二步: 为每个站点划分训练/验证集并保存
        stats = {}

        for site_id in range(self.num_sites):
            site_name = f"site{site_id + 1}"  # 使用site1, site2格式（不带连字符）
            site_dir = self.output_dir / site_name
            site_dir.mkdir(parents=True, exist_ok=True)

            # 获取该站点的数据索引
            indices = site_indices[site_id]
            X_site = X[indices]
            y_site = y[indices]

            logger.info(f"\n{'='*60}")
            logger.info(f"处理 {site_name}")
            logger.info(f"站点样本数: {len(indices)}")
            logger.info(f"站点标签分布: {dict(Counter(y_site))}")

            # 划分训练/验证集（使用分层抽样）
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_site, y_site,
                    test_size=val_ratio,
                    stratify=y_site,
                    random_state=self.seed + site_id  # 每个站点使用不同的种子
                )
            except ValueError as e:
                # 如果某些类别样本太少无法分层，则不使用分层
                logger.warning(
                    f"{site_name} 无法使用分层抽样 (某些类别样本太少): {e}, "
                    f"改用随机抽样"
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_site, y_site,
                    test_size=val_ratio,
                    random_state=self.seed + site_id
                )

            logger.info(f"训练集: {len(X_train)} 样本, 标签分布: {dict(Counter(y_train))}")
            logger.info(f"验证集: {len(X_val)} 样本, 标签分布: {dict(Counter(y_val))}")

            # 保存数据
            self._save_site_data(site_dir, X_train, y_train, X_val, y_val, feature_type)

            # 收集统计信息
            stats[site_name] = {
                'total_samples': len(indices),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'train_label_dist': dict(Counter(y_train)),
                'val_label_dist': dict(Counter(y_val)),
                'feature_shape': X_site.shape,
            }

        # 打印总体统计
        self._print_summary_stats(stats, feature_type)

        return stats

    def _stratified_split_to_sites(
        self,
        y: np.ndarray,
        n_samples: int
    ) -> Dict[int, np.ndarray]:
        """
        使用分层抽样将样本随机均分到各站点

        Args:
            y: 标签数组
            n_samples: 总样本数

        Returns:
            Dict[site_id, indices]: 每个站点对应的样本索引
        """
        # 获取每个类别的样本索引
        unique_labels = np.unique(y)
        label_indices = {label: np.where(y == label)[0] for label in unique_labels}

        # 对每个类别的索引进行随机打乱
        for label in unique_labels:
            np.random.shuffle(label_indices[label])

        # 初始化每个站点的索引列表
        site_indices = {i: [] for i in range(self.num_sites)}

        # 对每个类别，将样本尽可能均匀地分配到各站点
        for label, indices in label_indices.items():
            n_label_samples = len(indices)
            samples_per_site = n_label_samples // self.num_sites
            remainder = n_label_samples % self.num_sites

            start_idx = 0
            for site_id in range(self.num_sites):
                # 前remainder个站点多分配一个样本
                end_idx = start_idx + samples_per_site + (1 if site_id < remainder else 0)
                site_indices[site_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx

        # 将每个站点的索引转换为numpy数组并打乱
        for site_id in range(self.num_sites):
            site_indices[site_id] = np.array(site_indices[site_id])
            np.random.shuffle(site_indices[site_id])

        return site_indices

    def _save_site_data(
        self,
        site_dir: Path,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_type: str
    ) -> None:
        """
        保存单个站点数据

        Args:
            site_dir: 站点目录
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            feature_type: 特征类型
        """
        # 保存PyTorch格式（用于Lightning）
        torch.save({
            'X': torch.tensor(X_train, dtype=torch.float32),
            'y': torch.tensor(y_train, dtype=torch.long)
        }, site_dir / f'train_{feature_type}.pt')

        torch.save({
            'X': torch.tensor(X_val, dtype=torch.float32),
            'y': torch.tensor(y_val, dtype=torch.long)
        }, site_dir / f'val_{feature_type}.pt')

        # 保存NumPy格式（用于XGBoost）
        np.savez(
            site_dir / f'train_{feature_type}.npz',
            X=X_train,
            y=y_train
        )
        np.savez(
            site_dir / f'val_{feature_type}.npz',
            X=X_val,
            y=y_val
        )

        logger.info(f"已保存数据到 {site_dir}:")
        logger.info(f"  - train_{feature_type}.pt / train_{feature_type}.npz")
        logger.info(f"  - val_{feature_type}.pt / val_{feature_type}.npz")

    def _print_summary_stats(
        self,
        stats: Dict[str, Dict[str, Any]],
        feature_type: str
    ) -> None:
        """
        打印总体统计信息

        Args:
            stats: 统计信息字典
            feature_type: 特征类型
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"数据划分完成 - {feature_type} 特征")
        logger.info(f"{'='*60}")

        total_train = sum(s['train_samples'] for s in stats.values())
        total_val = sum(s['val_samples'] for s in stats.values())
        total = sum(s['total_samples'] for s in stats.values())

        logger.info(f"\n总体统计:")
        logger.info(f"  总样本数: {total}")
        logger.info(f"  总训练集: {total_train} ({total_train/total*100:.1f}%)")
        logger.info(f"  总验证集: {total_val} ({total_val/total*100:.1f}%)")

        logger.info(f"\n各站点详细信息:")
        for site_name, site_stats in stats.items():
            logger.info(f"\n  {site_name}:")
            logger.info(f"    总样本: {site_stats['total_samples']}")
            logger.info(f"    训练集: {site_stats['train_samples']}")
            logger.info(f"    验证集: {site_stats['val_samples']}")
            logger.info(f"    特征维度: {site_stats['feature_shape']}")
            logger.info(f"    训练集标签分布: {site_stats['train_label_dist']}")
            logger.info(f"    验证集标签分布: {site_stats['val_label_dist']}")

        logger.info(f"\n输出目录: {self.output_dir}")
        logger.info(f"{'='*60}\n")

    def verify_split(self, feature_type: str = "cluster") -> Dict[str, Any]:
        """
        验证数据划分的正确性

        Args:
            feature_type: 特征类型

        Returns:
            验证结果字典
        """
        logger.info(f"验证数据划分: feature_type={feature_type}")

        verification = {
            'success': True,
            'errors': [],
            'warnings': [],
            'sites': {}
        }

        for site_id in range(self.num_sites):
            site_name = f"site{site_id + 1}"  # 与split_and_save保持一致
            site_dir = self.output_dir / site_name

            if not site_dir.exists():
                verification['success'] = False
                verification['errors'].append(f"{site_name} 目录不存在")
                continue

            # 检查文件是否存在
            required_files = [
                f'train_{feature_type}.pt',
                f'val_{feature_type}.pt',
                f'train_{feature_type}.npz',
                f'val_{feature_type}.npz'
            ]

            site_verification = {'files_exist': True, 'files': {}}

            for filename in required_files:
                filepath = site_dir / filename
                if not filepath.exists():
                    verification['success'] = False
                    verification['errors'].append(f"{site_name}/{filename} 不存在")
                    site_verification['files_exist'] = False
                else:
                    site_verification['files'][filename] = str(filepath)

            # 验证数据一致性（PyTorch vs NumPy）
            try:
                # 加载PyTorch数据
                train_pt = torch.load(site_dir / f'train_{feature_type}.pt')
                val_pt = torch.load(site_dir / f'val_{feature_type}.pt')

                # 加载NumPy数据
                train_npz = np.load(site_dir / f'train_{feature_type}.npz')
                val_npz = np.load(site_dir / f'val_{feature_type}.npz')

                # 验证形状一致性
                if train_pt['X'].shape != train_npz['X'].shape:
                    verification['warnings'].append(
                        f"{site_name} 训练集PyTorch和NumPy形状不一致"
                    )

                if val_pt['X'].shape != val_npz['X'].shape:
                    verification['warnings'].append(
                        f"{site_name} 验证集PyTorch和NumPy形状不一致"
                    )

                # 验证数值一致性
                if not np.allclose(train_pt['X'].numpy(), train_npz['X']):
                    verification['warnings'].append(
                        f"{site_name} 训练集PyTorch和NumPy数值不一致"
                    )

                site_verification['train_shape'] = train_pt['X'].shape
                site_verification['val_shape'] = val_pt['X'].shape
                site_verification['train_labels'] = dict(Counter(train_pt['y'].numpy()))
                site_verification['val_labels'] = dict(Counter(val_pt['y'].numpy()))

            except Exception as e:
                verification['success'] = False
                verification['errors'].append(f"{site_name} 数据加载失败: {str(e)}")

            verification['sites'][site_name] = site_verification

        # 打印验证结果
        if verification['success']:
            logger.info("数据划分验证通过")
        else:
            logger.error("数据划分验证失败")
            for error in verification['errors']:
                logger.error(f"  - {error}")

        if verification['warnings']:
            logger.warning("发现以下警告:")
            for warning in verification['warnings']:
                logger.warning(f"  - {warning}")

        return verification


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 示例使用
    print("FederatedDataSplitter 使用示例:\n")
    print("```python")
    print("from data_splitter import FederatedDataSplitter")
    print()
    print("# 创建splitter")
    print("splitter = FederatedDataSplitter(")
    print("    output_dir='data/federated',")
    print("    num_sites=3,")
    print("    seed=42")
    print(")")
    print()
    print("# 加载数据")
    print("X = np.random.randn(1000, 100)  # 示例数据")
    print("y = np.random.randint(0, 3, 1000)  # 示例标签")
    print()
    print("# 划分并保存")
    print("stats = splitter.split_and_save(")
    print("    X=X,")
    print("    y=y,")
    print("    val_ratio=0.15,")
    print("    feature_type='cluster'")
    print(")")
    print()
    print("# 验证划分")
    print("verification = splitter.verify_split(feature_type='cluster')")
    print("```")
