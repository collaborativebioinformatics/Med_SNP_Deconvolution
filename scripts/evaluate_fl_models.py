#!/usr/bin/env python3
"""
联邦学习模型评估脚本

用法:
    python evaluate_fl_models.py --exp_dir /ephemeral/exp/runs --num_sites 3 --data_dir /ephemeral/exp/data/iid

功能:
1. 加载各站点的最优 checkpoint
2. 在测试集上评估模型性能
3. 计算全局聚合模型的性能
4. 输出详细的评估指标
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dl_models.snp_interpretable_models import InterpretableSNPModel
from snp_deconvolution.attention_dl.lightning_trainer import SNPLightningModule

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(site_dir: Path) -> Optional[Path]:
    """
    找到站点目录中的最优 checkpoint

    优先级:
    1. val_loss 最小的 checkpoint
    2. last.ckpt
    """
    checkpoint_dir = site_dir / "checkpoints"

    # 搜索所有可能的 checkpoint 位置
    search_paths = [
        checkpoint_dir,
        site_dir / "checkpoints" / site_dir.name,
        site_dir,
    ]

    best_ckpt = None
    best_loss = float('inf')

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for ckpt_file in search_path.glob("*.ckpt"):
            if ckpt_file.name == "last.ckpt":
                if best_ckpt is None:
                    best_ckpt = ckpt_file
                continue

            # 解析文件名中的 val_loss
            # 格式: snp-round0-epoch=00-val_loss=0.0871.ckpt
            import re
            match = re.search(r'val_loss=(\d+\.\d+)', ckpt_file.name)
            if match:
                try:
                    loss = float(match.group(1))
                    if loss < best_loss:
                        best_loss = loss
                        best_ckpt = ckpt_file
                except ValueError:
                    continue

    return best_ckpt


def load_model_from_checkpoint(
    ckpt_path: Path,
    n_snps: int,
    num_classes: int,
    encoding_dim: int = 8,
    architecture: str = "cnn_transformer"
) -> nn.Module:
    """
    从 checkpoint 加载模型
    """
    # 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # 创建模型
    model = InterpretableSNPModel(
        n_snps=n_snps,
        num_classes=num_classes,
        encoding_dim=encoding_dim,
        architecture=architecture,
    )

    # 加载权重
    if 'state_dict' in checkpoint:
        # Lightning checkpoint 格式
        state_dict = checkpoint['state_dict']
        # 移除 'model.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    return model


def load_test_data(data_dir: Path, feature_type: str = "cluster") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    加载测试数据

    支持两种数据结构:
    1. 根目录下的 test_cluster.npz / val_cluster.npz
    2. 各站点目录下的 val_cluster.npz (合并所有站点)
    """
    # 方式1: 尝试根目录下的测试数据
    possible_files = [
        data_dir / f"test_{feature_type}.pt",
        data_dir / f"test_{feature_type}.npz",
        data_dir / f"val_{feature_type}.pt",
        data_dir / f"val_{feature_type}.npz",
    ]

    for file_path in possible_files:
        if file_path.exists():
            if file_path.suffix == '.pt':
                data = torch.load(file_path)
                X = data['X'] if isinstance(data, dict) else data[0]
                y = data['y'] if isinstance(data, dict) else data[1]
            else:  # .npz
                data = np.load(file_path)
                X = torch.from_numpy(data['X']).float()
                y = torch.from_numpy(data['y']).long()

            logger.info(f"加载测试数据: {file_path}")
            logger.info(f"  X shape: {X.shape}, y shape: {y.shape}")
            return X, y

    # 方式2: 合并各站点的验证集
    site_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('site')])

    if site_dirs:
        logger.info(f"从 {len(site_dirs)} 个站点目录合并验证数据")
        all_X = []
        all_y = []

        for site_dir in site_dirs:
            # 尝试加载该站点的验证数据
            site_files = [
                site_dir / f"val_{feature_type}.npz",
                site_dir / f"val_{feature_type}.pt",
                site_dir / f"test_{feature_type}.npz",
                site_dir / f"test_{feature_type}.pt",
            ]

            for file_path in site_files:
                if file_path.exists():
                    if file_path.suffix == '.pt':
                        data = torch.load(file_path)
                        X = data['X'] if isinstance(data, dict) else data[0]
                        y = data['y'] if isinstance(data, dict) else data[1]
                    else:  # .npz
                        data = np.load(file_path)
                        X = torch.from_numpy(data['X']).float()
                        y = torch.from_numpy(data['y']).long()

                    all_X.append(X)
                    all_y.append(y)
                    logger.info(f"  {site_dir.name}: {X.shape[0]} 样本")
                    break

        if all_X:
            X_combined = torch.cat(all_X, dim=0)
            y_combined = torch.cat(all_y, dim=0)
            logger.info(f"合并后: X shape: {X_combined.shape}, y shape: {y_combined.shape}")
            return X_combined, y_combined

    raise FileNotFoundError(f"未找到测试数据: {data_dir}")


def aggregate_models(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
    num_classes: int = 2
) -> nn.Module:
    """
    聚合多个模型的权重 (FedAvg 风格)
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    # 使用第一个模型作为基础
    aggregated_model = models[0].__class__(
        n_snps=models[0].n_snps,
        num_classes=num_classes,
        encoding_dim=models[0].encoding_dim,
        architecture=models[0].architecture,
    )

    # 聚合权重
    aggregated_state_dict = {}
    for key in models[0].state_dict().keys():
        # 初始化为零张量
        aggregated_state_dict[key] = torch.zeros_like(models[0].state_dict()[key].float())
        for w, m in zip(weights, models):
            aggregated_state_dict[key] += w * m.state_dict()[key].float()

    aggregated_model.load_state_dict(aggregated_state_dict)
    return aggregated_model


def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: int = 128
) -> Dict[str, float]:
    """
    评估模型性能
    """
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item() * len(batch_y)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    accuracy = (all_preds == all_labels).mean()
    avg_loss = total_loss / len(all_labels)

    # 每类准确率
    unique_labels = np.unique(all_labels)
    per_class_acc = {}
    for label in unique_labels:
        mask = all_labels == label
        if mask.sum() > 0:
            per_class_acc[int(label)] = (all_preds[mask] == all_labels[mask]).mean()

    # 混淆矩阵
    n_classes = len(unique_labels)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        confusion_matrix[true_label, pred_label] += 1

    # Macro F1
    precision_per_class = []
    recall_per_class = []
    for i in range(n_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_per_class.append(precision)
        recall_per_class.append(recall)

    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

    return {
        'accuracy': float(accuracy),
        'loss': float(avg_loss),
        'macro_f1': float(macro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion_matrix.tolist(),
        'n_samples': len(all_labels),
    }


def evaluate_experiment(
    exp_dir: Path,
    data_dir: Path,
    num_sites: int,
    feature_type: str = "cluster",
    architecture: str = "cnn_transformer",
) -> Dict[str, Any]:
    """
    评估单个实验
    """
    results = {
        'exp_name': exp_dir.name,
        'site_results': {},
        'aggregated_result': None,
        'best_site': None,
    }

    # 加载测试数据
    try:
        X_test, y_test = load_test_data(data_dir, feature_type)
    except FileNotFoundError as e:
        logger.error(f"无法加载测试数据: {e}")
        return results

    # 推断模型参数
    if len(X_test.shape) == 3:
        n_snps = X_test.shape[1]
        encoding_dim = X_test.shape[2]
    else:
        n_snps = X_test.shape[1]
        encoding_dim = 8

    num_classes = len(torch.unique(y_test))
    logger.info(f"模型参数: n_snps={n_snps}, num_classes={num_classes}, encoding_dim={encoding_dim}")

    # 加载各站点模型
    models = []
    site_sample_counts = []

    for site_id in range(1, num_sites + 1):
        site_name = f"site-{site_id}"
        site_dir = exp_dir / site_name

        if not site_dir.exists():
            logger.warning(f"站点目录不存在: {site_dir}")
            continue

        ckpt_path = find_best_checkpoint(site_dir)
        if ckpt_path is None:
            logger.warning(f"未找到 checkpoint: {site_dir}")
            continue

        logger.info(f"加载 {site_name} checkpoint: {ckpt_path.name}")

        try:
            model = load_model_from_checkpoint(
                ckpt_path, n_snps, num_classes, encoding_dim, architecture
            )
            models.append(model)

            # 评估单个站点模型
            site_result = evaluate_model(model, X_test, y_test)
            results['site_results'][site_name] = site_result

            # 获取站点训练样本数 (用于加权聚合)
            site_data_dir = data_dir / f"site{site_id}"
            if site_data_dir.exists():
                try:
                    train_file = site_data_dir / f"train_{feature_type}.npz"
                    if train_file.exists():
                        train_data = np.load(train_file)
                        site_sample_counts.append(len(train_data['y']))
                    else:
                        site_sample_counts.append(1)
                except:
                    site_sample_counts.append(1)
            else:
                site_sample_counts.append(1)

            logger.info(f"  {site_name}: accuracy={site_result['accuracy']:.4f}, f1={site_result['macro_f1']:.4f}")

        except Exception as e:
            logger.error(f"加载模型失败 {site_name}: {e}")
            continue

    # 聚合模型并评估
    if len(models) > 1:
        logger.info("聚合模型...")

        # 加权聚合 (按训练样本数)
        total_samples = sum(site_sample_counts)
        weights = [c / total_samples for c in site_sample_counts]

        aggregated_model = aggregate_models(models, weights, num_classes)
        aggregated_result = evaluate_model(aggregated_model, X_test, y_test)
        results['aggregated_result'] = aggregated_result

        logger.info(f"  聚合模型: accuracy={aggregated_result['accuracy']:.4f}, f1={aggregated_result['macro_f1']:.4f}")
    elif len(models) == 1:
        results['aggregated_result'] = results['site_results'].get('site-1')

    # 找出最佳站点
    if results['site_results']:
        best_site = max(results['site_results'].items(), key=lambda x: x[1]['accuracy'])
        results['best_site'] = {
            'name': best_site[0],
            'accuracy': best_site[1]['accuracy'],
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='评估联邦学习模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估单个实验
    python evaluate_fl_models.py \\
        --exp_dir /ephemeral/exp/runs/iid_fedavg \\
        --data_dir /ephemeral/exp/data/iid \\
        --num_sites 3

    # 评估所有实验
    python evaluate_fl_models.py \\
        --exp_dir /ephemeral/exp/runs \\
        --data_dir /ephemeral/exp/data \\
        --num_sites 3 \\
        --all_experiments

    # 指定输出文件
    python evaluate_fl_models.py \\
        --exp_dir /ephemeral/exp/runs \\
        --data_dir /ephemeral/exp/data \\
        --num_sites 3 \\
        --all_experiments \\
        --output /ephemeral/exp/evaluation_results.json
        """
    )

    parser.add_argument('--exp_dir', type=str, required=True,
                        help='实验目录 (单个实验或包含多个实验的目录)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录')
    parser.add_argument('--num_sites', type=int, required=True,
                        help='站点数量')
    parser.add_argument('--feature_type', type=str, default='cluster',
                        choices=['cluster', 'snp'],
                        help='特征类型 (默认: cluster)')
    parser.add_argument('--architecture', type=str, default='cnn_transformer',
                        help='模型架构 (默认: cnn_transformer)')
    parser.add_argument('--all_experiments', action='store_true',
                        help='评估目录下所有实验')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 JSON 文件路径')

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    data_dir = Path(args.data_dir)

    all_results = {}

    if args.all_experiments:
        # 评估所有实验
        exp_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and (d / 'site-1').exists()]
        logger.info(f"找到 {len(exp_dirs)} 个实验")

        for exp_subdir in sorted(exp_dirs):
            logger.info(f"\n{'='*60}")
            logger.info(f"评估实验: {exp_subdir.name}")
            logger.info(f"{'='*60}")

            # 推断数据目录
            # 实验名格式: iid_fedavg, dirichlet_0.1_fedprox 等
            exp_name = exp_subdir.name
            if 'iid' in exp_name.lower():
                exp_data_dir = data_dir / 'iid'
            elif 'dirichlet' in exp_name.lower() or 'dir' in exp_name.lower():
                import re
                match = re.search(r'(?:dirichlet|dir)[_]?([\d.]+)', exp_name.lower())
                if match:
                    alpha = match.group(1)
                    exp_data_dir = data_dir / f'dirichlet_{alpha}'
                else:
                    exp_data_dir = data_dir
            else:
                exp_data_dir = data_dir

            if not exp_data_dir.exists():
                logger.warning(f"数据目录不存在: {exp_data_dir}, 使用默认: {data_dir}")
                exp_data_dir = data_dir

            results = evaluate_experiment(
                exp_subdir, exp_data_dir, args.num_sites,
                args.feature_type, args.architecture
            )
            all_results[exp_subdir.name] = results
    else:
        # 评估单个实验
        results = evaluate_experiment(
            exp_dir, data_dir, args.num_sites,
            args.feature_type, args.architecture
        )
        all_results[exp_dir.name] = results

    # 打印汇总
    print("\n" + "="*80)
    print("评估结果汇总")
    print("="*80)

    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")

        if results['aggregated_result']:
            agg = results['aggregated_result']
            print(f"  聚合模型: Acc={agg['accuracy']:.4f}, F1={agg['macro_f1']:.4f}, Loss={agg['loss']:.4f}")

        if results['best_site']:
            print(f"  最佳站点: {results['best_site']['name']} (Acc={results['best_site']['accuracy']:.4f})")

        for site_name, site_result in results['site_results'].items():
            print(f"    {site_name}: Acc={site_result['accuracy']:.4f}, F1={site_result['macro_f1']:.4f}")

    print("="*80)

    # 保存结果
    output_path = args.output or (exp_dir / 'evaluation_results.json')
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
