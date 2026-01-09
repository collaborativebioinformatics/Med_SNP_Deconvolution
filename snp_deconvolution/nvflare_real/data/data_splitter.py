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
    将数据划分到多个站点，支持多种划分策略

    支持的功能：
    - PyTorch格式输出（Lightning用）
    - NumPy格式输出（XGBoost用）

    支持的划分策略：
    - iid: 分层均匀划分，保证各站点标签分布一致
    - dirichlet: 使用Dirichlet分布创建异构标签分布（Non-IID）
    - quantity_skew: 数量偏斜划分，各站点拥有不同数量的样本
    - label_skew: 标签偏斜划分，各站点只拥有部分类别的样本
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

        # 创建独立的随机数生成器，避免污染全局状态
        self.rng = np.random.default_rng(self.seed)
        # Note: sklearn's train_test_split uses random_state parameter,
        # and torch operations use torch.manual_seed within specific scopes if needed

        logger.info(
            f"初始化FederatedDataSplitter: "
            f"num_sites={num_sites}, seed={seed}, output_dir={output_dir}"
        )

    def split_and_save(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_ratio: float = 0.15,
        feature_type: str = "cluster",  # "cluster" 或 "snp"
        split_type: str = "iid",  # "iid", "dirichlet", "label_skew", "quantity_skew"
        alpha: float = 0.5,  # Dirichlet分布的alpha参数，仅在split_type="dirichlet"时使用
        min_ratio: float = 0.1,  # quantity_skew模式下的最小比例
        labels_per_site: int = 2  # label_skew模式下每个站点的类别数
    ) -> Dict[str, Dict[str, Any]]:
        """
        划分数据并保存到各站点目录

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            val_ratio: 验证集比例
            feature_type: 特征类型，用于文件命名
            split_type: 数据划分类型
                - "iid": 独立同分布，使用分层抽样保证各站点标签分布一致
                - "dirichlet": 使用Dirichlet分布创建异构标签分布（Non-IID）
                - "label_skew": 标签偏斜，每个站点只包含部分类别
                - "quantity_skew": 数量偏斜，各站点数据量不均衡
            alpha: Dirichlet分布的alpha参数（仅在split_type="dirichlet"时使用）
                - 较小的alpha（如0.1）产生更高的异构性（标签分布更不均匀）
                - 较大的alpha（如10.0）产生更接近均匀的分布
                - alpha=1.0时为标准的对称Dirichlet分布
            min_ratio: quantity_skew模式下每个站点的最小数据比例
            labels_per_site: label_skew模式下每个站点包含的类别数

        Returns:
            各站点的样本统计信息
        """
        logger.info(
            f"开始数据划分: X.shape={X.shape}, y.shape={y.shape}, "
            f"val_ratio={val_ratio}, feature_type={feature_type}, "
            f"split_type={split_type}"
        )

        if split_type == "dirichlet":
            logger.info(f"Dirichlet alpha参数: {alpha}")

        # 验证输入
        assert len(X) == len(y), "X和y的样本数必须一致"
        assert 0 < val_ratio < 1, "val_ratio必须在(0, 1)之间"
        assert self.num_sites > 0, "num_sites必须大于0"
        assert split_type in ["iid", "dirichlet", "label_skew", "quantity_skew"], \
            f"split_type必须是'iid'、'dirichlet'、'label_skew'或'quantity_skew'"

        n_samples = len(X)
        logger.info(f"总样本数: {n_samples}")
        logger.info(f"标签分布: {dict(Counter(y))}")

        # 第一步: 根据选择的方法将数据分配到各站点
        if split_type == "iid":
            site_indices = self._stratified_split_to_sites(y, n_samples)
        elif split_type == "dirichlet":
            site_indices = self._dirichlet_split_to_sites(y, alpha)
        elif split_type == "quantity_skew":
            site_indices = self._quantity_skew_split_to_sites(y, n_samples, min_ratio)
        elif split_type == "label_skew":
            site_indices = self._label_skew_split_to_sites(y, labels_per_site)
        else:
            raise ValueError(f"不支持的划分方法: {split_type}")

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
        使用分层抽样将样本随机均分到各站点（IID）

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
            self.rng.shuffle(label_indices[label])

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
            self.rng.shuffle(site_indices[site_id])

        return site_indices

    def _dirichlet_split_to_sites(
        self,
        y: np.ndarray,
        alpha: float
    ) -> Dict[int, np.ndarray]:
        """
        使用Dirichlet分布将样本分配到各站点（Non-IID）

        Dirichlet分布用于创建异构的标签分布，模拟现实中数据分布不均的场景。
        对于每个类别，从Dirichlet(alpha, alpha, ..., alpha)中采样，得到该类别
        在各站点的分布比例，然后按比例分配样本。

        Args:
            y: 标签数组
            alpha: Dirichlet分布的concentration参数
                - alpha < 1: 产生稀疏分布，某些站点会获得该类别的大部分样本
                - alpha = 1: 对称Dirichlet分布，产生中等程度的异构性
                - alpha > 1: 产生更均匀的分布，接近IID设置

        Returns:
            Dict[site_id, indices]: 每个站点对应的样本索引

        References:
            Hsu et al. "Measuring the Effects of Non-Identical Data Distribution for
            Federated Visual Classification" (2019)
        """
        logger.info(f"Dirichlet Non-IID划分: alpha={alpha}, num_sites={self.num_sites}")

        # 获取每个类别的样本索引
        unique_labels = np.unique(y)
        num_classes = len(unique_labels)
        label_indices = {label: np.where(y == label)[0] for label in unique_labels}

        logger.info(f"类别数: {num_classes}, 各类别样本数: {[len(indices) for indices in label_indices.values()]}")

        # 对每个类别的索引进行随机打乱（保证可重现性）
        for label in unique_labels:
            self.rng.shuffle(label_indices[label])

        # 初始化每个站点的索引列表
        site_indices = {i: [] for i in range(self.num_sites)}

        # 对每个类别，使用Dirichlet分布采样分配比例
        for label in unique_labels:
            indices = label_indices[label]
            n_label_samples = len(indices)

            # 处理边界情况：如果某个类别样本太少
            if n_label_samples < self.num_sites:
                logger.warning(
                    f"类别 {label} 只有 {n_label_samples} 个样本，少于站点数 {self.num_sites}，"
                    f"将随机分配到前 {n_label_samples} 个站点"
                )
                # 随机选择n_label_samples个站点，每个站点分配一个样本
                selected_sites = self.rng.choice(self.num_sites, n_label_samples, replace=False)
                for idx, site_id in enumerate(selected_sites):
                    site_indices[site_id].append(indices[idx])
                continue

            # 使用Dirichlet分布采样各站点的分配比例
            # alpha越小，分布越不均匀；alpha越大，分布越均匀
            proportions = self.rng.dirichlet([alpha] * self.num_sites)

            logger.info(f"类别 {label}: {n_label_samples} 个样本")
            logger.info(f"  Dirichlet采样比例: {[f'{p:.3f}' for p in proportions]}")

            # 根据比例计算每个站点应分配的样本数
            # 使用cumsum和int确保样本总数不变
            cumsum_proportions = np.cumsum(proportions)
            splits = (cumsum_proportions * n_label_samples).astype(int)

            # 确保最后一个站点获得所有剩余样本
            splits[-1] = n_label_samples

            # 将样本分配到各站点
            start_idx = 0
            samples_allocated = []
            for site_id in range(self.num_sites):
                end_idx = splits[site_id]
                n_allocated = end_idx - start_idx

                # 处理边界情况：如果某个站点分配到的样本数为0
                if n_allocated > 0:
                    site_indices[site_id].extend(indices[start_idx:end_idx])
                    samples_allocated.append(n_allocated)
                else:
                    samples_allocated.append(0)
                    logger.warning(
                        f"  站点 {site_id + 1} 未分配到类别 {label} 的样本 "
                        f"(alpha={alpha}可能太小)"
                    )

                start_idx = end_idx

            logger.info(f"  实际分配样本数: {samples_allocated}")

        # 将每个站点的索引转换为numpy数组并打乱
        for site_id in range(self.num_sites):
            if len(site_indices[site_id]) == 0:
                logger.warning(f"警告: 站点 {site_id + 1} 未分配到任何样本！")
            else:
                site_indices[site_id] = np.array(site_indices[site_id])
                self.rng.shuffle(site_indices[site_id])
                logger.info(
                    f"站点 {site_id + 1}: 总共 {len(site_indices[site_id])} 个样本, "
                    f"标签分布: {dict(Counter(y[site_indices[site_id]]))}"
                )

        # 验证所有样本都被分配
        total_allocated = sum(len(indices) for indices in site_indices.values())
        if total_allocated != len(y):
            logger.error(
                f"样本分配错误: 总样本数={len(y)}, 已分配={total_allocated}, "
                f"差异={len(y) - total_allocated}"
            )
            raise ValueError("Dirichlet划分后样本总数不匹配")

        return site_indices

    def _label_skew_split_to_sites(
        self,
        y: np.ndarray,
        labels_per_site: int = 2
    ) -> Dict[int, np.ndarray]:
        """
        使用标签偏斜策略将样本分配到各站点

        每个站点只接收部分类别的样本，以模拟标签分布的非IID场景。
        类别采用round-robin方式分配，确保所有类别在联邦学习系统中都有表示。

        例如：3个类别(0, 1, 2)和3个站点，labels_per_site=2:
        - Site 1: 类别 0, 1
        - Site 2: 类别 1, 2
        - Site 3: 类别 2, 0

        Args:
            y: 标签数组
            labels_per_site: 每个站点包含的类别数

        Returns:
            Dict[site_id, indices]: 每个站点对应的样本索引

        Raises:
            ValueError: 当labels_per_site超过可用类别数时
        """
        # 获取唯一类别
        unique_labels = np.unique(y)
        num_classes = len(unique_labels)

        logger.info(f"标签偏斜划分: num_classes={num_classes}, labels_per_site={labels_per_site}")

        # 验证和调整参数
        if labels_per_site > num_classes:
            logger.warning(
                f"labels_per_site ({labels_per_site}) 大于类别总数 ({num_classes})，"
                f"自动调整为 {num_classes}"
            )
            labels_per_site = num_classes

        if labels_per_site < 1:
            raise ValueError(f"labels_per_site必须至少为1，当前值: {labels_per_site}")

        # 获取每个类别的样本索引
        label_indices = {label: np.where(y == label)[0] for label in unique_labels}

        # 对每个类别的索引进行随机打乱（保证可重现性）
        for label in unique_labels:
            self.rng.shuffle(label_indices[label])

        # 为每个站点分配类别（round-robin方式）
        site_labels = {i: [] for i in range(self.num_sites)}

        # 使用滑动窗口方式分配类别，确保类别重叠和覆盖
        if labels_per_site >= num_classes:
            # 如果每个站点都有所有类别，则所有站点分配所有类别
            for site_id in range(self.num_sites):
                site_labels[site_id] = list(unique_labels)
        else:
            # 使用round-robin分配，每个站点获得labels_per_site个连续的类别
            # 计算最佳步长以确保覆盖和适度重叠
            #
            # 使用步长s，可覆盖的类别数 = labels_per_site + (num_sites - 1) * s
            # 1. 先尝试使用较小步长（允许重叠）: step = labels_per_site // 2
            # 2. 如果无法覆盖所有类别，则计算所需的最小步长

            # 计算允许重叠的步长
            step_with_overlap = max(1, labels_per_site // 2) if labels_per_site > 1 else 1
            max_coverage_with_overlap = labels_per_site + (self.num_sites - 1) * step_with_overlap

            if max_coverage_with_overlap >= num_classes:
                # 可以覆盖所有类别并允许重叠
                step = step_with_overlap
                logger.info(f"使用步长 {step} (允许类别重叠)")
            else:
                # 需要更大步长以覆盖所有类别
                if self.num_sites == 1:
                    step = 1
                else:
                    # 计算覆盖所有类别所需的最小步长
                    min_step = max(1, (num_classes - labels_per_site + self.num_sites - 2) // (self.num_sites - 1))
                    step = min_step
                logger.info(
                    f"使用步长 {step} 以最大化类别覆盖 "
                    f"(预计覆盖 {min(num_classes, labels_per_site + (self.num_sites - 1) * step)} 个类别)"
                )

            for site_id in range(self.num_sites):
                start_idx = (site_id * step) % num_classes
                selected_labels = []
                for i in range(labels_per_site):
                    label_idx = (start_idx + i) % num_classes
                    selected_labels.append(unique_labels[label_idx])
                site_labels[site_id] = selected_labels

        # 记录类别分配情况
        logger.info("各站点类别分配:")
        for site_id in range(self.num_sites):
            logger.info(f"  站点 {site_id + 1}: 类别 {site_labels[site_id]}")

        # 验证所有类别都被至少一个站点使用
        all_assigned_labels = set()
        for labels in site_labels.values():
            all_assigned_labels.update(labels)

        if len(all_assigned_labels) < num_classes:
            missing_labels = set(unique_labels) - all_assigned_labels
            logger.warning(f"警告: 以下类别未分配到任何站点: {missing_labels}")
        else:
            logger.info(f"所有 {num_classes} 个类别都已分配到至少一个站点")

        # 初始化每个站点的索引列表
        site_indices = {i: [] for i in range(self.num_sites)}

        # 将每个类别的样本分配到对应的站点
        for label in unique_labels:
            indices = label_indices[label]
            sites_with_label = [
                site_id for site_id, labels in site_labels.items()
                if label in labels
            ]

            n_label_samples = len(indices)
            n_sites_for_label = len(sites_with_label)

            logger.info(
                f"类别 {label}: {n_label_samples} 个样本分配到 {n_sites_for_label} 个站点"
            )

            # 将样本尽可能均匀地分配到拥有该类别的站点
            samples_per_site = n_label_samples // n_sites_for_label
            remainder = n_label_samples % n_sites_for_label

            start_idx = 0
            for i, site_id in enumerate(sites_with_label):
                # 前remainder个站点多分配一个样本
                end_idx = start_idx + samples_per_site + (1 if i < remainder else 0)
                site_indices[site_id].extend(indices[start_idx:end_idx])
                start_idx = end_idx

        # 将每个站点的索引转换为numpy数组并打乱
        for site_id in range(self.num_sites):
            if len(site_indices[site_id]) == 0:
                logger.warning(f"警告: 站点 {site_id + 1} 未分配到任何样本")
            site_indices[site_id] = np.array(site_indices[site_id])
            self.rng.shuffle(site_indices[site_id])
            logger.info(
                f"站点 {site_id + 1}: 总共 {len(site_indices[site_id])} 个样本"
            )

        return site_indices

    def _quantity_skew_split_to_sites(
        self,
        y: np.ndarray,
        n_samples: int,
        min_ratio: float = 0.1
    ) -> Dict[int, np.ndarray]:
        """
        使用数量偏斜策略将样本分配到各站点（Non-IID）

        各站点获得不同数量的样本，但保持标签分布相对均衡。
        使用Dirichlet分布采样各站点的数据量比例。

        Args:
            y: 标签数组
            n_samples: 总样本数
            min_ratio: 每个站点的最小数据比例

        Returns:
            Dict[site_id, indices]: 每个站点对应的样本索引
        """
        logger.info(f"数量偏斜划分: n_samples={n_samples}, min_ratio={min_ratio}")

        # 使用Dirichlet(1, 1, ..., 1)采样各站点的数据量比例
        proportions = self.rng.dirichlet([1.0] * self.num_sites)

        # 确保每个站点至少获得min_ratio的数据
        proportions = np.maximum(proportions, min_ratio)
        proportions = proportions / proportions.sum()  # 重新归一化

        logger.info(f"各站点数据量比例: {[f'{p:.3f}' for p in proportions]}")

        # 按比例分配样本数
        samples_per_site = (proportions * n_samples).astype(int)

        # 调整以确保总数相等
        diff = n_samples - samples_per_site.sum()
        if diff > 0:
            # 将剩余样本分配给比例最大的站点
            max_idx = np.argmax(proportions)
            samples_per_site[max_idx] += diff
        elif diff < 0:
            # 从比例最大的站点减少样本
            max_idx = np.argmax(proportions)
            samples_per_site[max_idx] += diff  # diff是负数

        logger.info(f"各站点实际样本数: {samples_per_site}")

        # 对所有样本进行分层打乱
        site_indices = self._stratified_split_to_sites(y, n_samples)

        # 根据数量比例重新分配
        all_indices = np.concatenate([site_indices[i] for i in range(self.num_sites)])
        self.rng.shuffle(all_indices)

        # 按照计算的samples_per_site重新划分
        new_site_indices = {}
        start_idx = 0
        for site_id in range(self.num_sites):
            end_idx = start_idx + samples_per_site[site_id]
            new_site_indices[site_id] = all_indices[start_idx:end_idx]
            start_idx = end_idx
            logger.info(
                f"站点 {site_id + 1}: {len(new_site_indices[site_id])} 个样本"
            )

        return new_site_indices

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
    print("# IID划分（默认）")
    print("stats = splitter.split_and_save(")
    print("    X=X, y=y, val_ratio=0.15,")
    print("    feature_type='cluster',")
    print("    split_type='iid'")
    print(")")
    print()
    print("# Dirichlet Non-IID划分")
    print("stats = splitter.split_and_save(")
    print("    X=X, y=y, val_ratio=0.15,")
    print("    feature_type='cluster',")
    print("    split_type='dirichlet',")
    print("    alpha=0.5  # 较小的alpha产生更高的异构性")
    print(")")
    print()
    print("# 验证划分")
    print("verification = splitter.verify_split(feature_type='cluster')")
    print("```")
