#!/usr/bin/env python3
"""
联邦学习评估结果绘图脚本

用法:
    python plot_evaluation_results.py --results /ephemeral/exp/evaluation_results.json --output /ephemeral/exp/figures

功能:
1. 策略对比柱状图
2. 站点性能对比
3. 混淆矩阵热力图
4. 收敛曲线 (如果有训练日志)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try imports
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# Style configurations
STYLE_CONFIGS = {
    'default': {
        'figure.figsize': (10, 6),
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'figure.dpi': 100,
    },
    'paper': {
        'figure.figsize': (8, 5),
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 300,
        'lines.linewidth': 2,
    },
    'presentation': {
        'figure.figsize': (12, 7),
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'figure.dpi': 150,
        'lines.linewidth': 2.5,
    }
}

# Colors for strategies
STRATEGY_COLORS = {
    'FedAvg': '#1f77b4',
    'FedProx': '#ff7f0e',
    'Scaffold': '#2ca02c',
    'FedOpt': '#d62728',
    'fedavg': '#1f77b4',
    'fedprox': '#ff7f0e',
    'scaffold': '#2ca02c',
    'fedopt': '#d62728',
}

# Colors for splits
SPLIT_COLORS = {
    'IID': '#2ecc71',
    'iid': '#2ecc71',
    'Dirichlet_0.1': '#e74c3c',
    'dirichlet_0.1': '#e74c3c',
    'dir0.1': '#e74c3c',
    'Dirichlet_1.0': '#f39c12',
    'dirichlet_1.0': '#f39c12',
    'dir1.0': '#f39c12',
}


def parse_experiment_name(exp_name: str) -> tuple:
    """
    解析实验名称，提取 split_type 和 strategy
    """
    exp_name_lower = exp_name.lower()

    # 提取策略
    strategy = 'FedAvg'
    if 'fedavg' in exp_name_lower:
        strategy = 'FedAvg'
    elif 'fedprox' in exp_name_lower:
        strategy = 'FedProx'
    elif 'scaffold' in exp_name_lower:
        strategy = 'Scaffold'
    elif 'fedopt' in exp_name_lower:
        strategy = 'FedOpt'

    # 提取分割类型
    split_type = 'IID'
    if 'iid' in exp_name_lower and 'dirichlet' not in exp_name_lower:
        split_type = 'IID'
    elif 'dirichlet' in exp_name_lower or 'dir' in exp_name_lower:
        import re
        match = re.search(r'(?:dirichlet|dir)[_]?([\d.]+)', exp_name_lower)
        if match:
            alpha = match.group(1)
            split_type = f'Dirichlet_{alpha}'
        else:
            split_type = 'Dirichlet'

    return split_type, strategy


def plot_strategy_comparison(
    results: Dict[str, Any],
    output_dir: Path,
    metric: str = 'accuracy',
    output_format: str = 'png'
):
    """
    绘制策略对比柱状图
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available")
        return

    # 组织数据
    data = {}  # {split_type: {strategy: metric_value}}

    for exp_name, exp_results in results.items():
        split_type, strategy = parse_experiment_name(exp_name)

        if split_type not in data:
            data[split_type] = {}

        if exp_results.get('aggregated_result'):
            value = exp_results['aggregated_result'].get(metric, 0)
            data[split_type][strategy] = value

    if not data:
        logger.warning("No data for strategy comparison plot")
        return

    # 准备绘图数据
    split_types = sorted(data.keys())
    strategies = sorted(set(s for splits in data.values() for s in splits.keys()))

    x = np.arange(len(split_types))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, strategy in enumerate(strategies):
        values = [data[split].get(strategy, 0) for split in split_types]
        offset = (i - len(strategies) / 2) * width + width / 2
        color = STRATEGY_COLORS.get(strategy, None)
        bars = ax.bar(x + offset, values, width, label=strategy, color=color, alpha=0.8)

        # 添加数值标签
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Data Split')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Strategy Comparison: {metric.replace("_", " ").title()}')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', '\n') for s in split_types])
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)

    if metric == 'accuracy':
        ax.set_ylim([0, 1.0])

    plt.tight_layout()
    output_path = output_dir / f'strategy_comparison_{metric}.{output_format}'
    plt.savefig(output_path, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_site_performance(
    results: Dict[str, Any],
    output_dir: Path,
    output_format: str = 'png'
):
    """
    绘制各站点性能对比
    """
    if not HAS_MATPLOTLIB:
        return

    # 为每个实验绘制站点性能
    for exp_name, exp_results in results.items():
        site_results = exp_results.get('site_results', {})
        if not site_results:
            continue

        sites = sorted(site_results.keys())
        accuracies = [site_results[s]['accuracy'] for s in sites]
        f1_scores = [site_results[s]['macro_f1'] for s in sites]

        # 添加聚合模型
        if exp_results.get('aggregated_result'):
            sites.append('Aggregated')
            accuracies.append(exp_results['aggregated_result']['accuracy'])
            f1_scores.append(exp_results['aggregated_result']['macro_f1'])

        x = np.arange(len(sites))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', color='#e74c3c', alpha=0.8)

        # 添加数值标签
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Site')
        ax.set_ylabel('Score')
        ax.set_title(f'Site Performance: {exp_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(sites)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f'site_performance_{exp_name}.{output_format}'
        plt.savefig(output_path, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")


def plot_confusion_matrices(
    results: Dict[str, Any],
    output_dir: Path,
    output_format: str = 'png'
):
    """
    绘制混淆矩阵热力图
    """
    if not HAS_MATPLOTLIB:
        return

    for exp_name, exp_results in results.items():
        if not exp_results.get('aggregated_result'):
            continue

        cm = exp_results['aggregated_result'].get('confusion_matrix')
        if cm is None:
            continue

        cm = np.array(cm)
        n_classes = cm.shape[0]

        fig, ax = plt.subplots(figsize=(8, 6))

        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=range(n_classes), yticklabels=range(n_classes))
        else:
            im = ax.imshow(cm, cmap='Blues')
            plt.colorbar(im, ax=ax)

            # 添加数值标签
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center')

            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix: {exp_name}')

        plt.tight_layout()
        output_path = output_dir / f'confusion_matrix_{exp_name}.{output_format}'
        plt.savefig(output_path, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path}")


def plot_accuracy_heatmap(
    results: Dict[str, Any],
    output_dir: Path,
    output_format: str = 'png'
):
    """
    绘制策略 x 数据分割 的准确率热力图
    """
    if not HAS_MATPLOTLIB:
        return

    # 组织数据
    data = {}

    for exp_name, exp_results in results.items():
        split_type, strategy = parse_experiment_name(exp_name)

        if exp_results.get('aggregated_result'):
            acc = exp_results['aggregated_result'].get('accuracy', 0)

            if strategy not in data:
                data[strategy] = {}
            data[strategy][split_type] = acc

    if not data:
        return

    strategies = sorted(data.keys())
    split_types = sorted(set(s for strat in data.values() for s in strat.keys()))

    # 创建矩阵
    matrix = np.zeros((len(strategies), len(split_types)))
    for i, strategy in enumerate(strategies):
        for j, split in enumerate(split_types):
            matrix[i, j] = data[strategy].get(split, 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    if HAS_SEABORN:
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=[s.replace('_', '\n') for s in split_types],
                    yticklabels=strategies, ax=ax,
                    vmin=0.3, vmax=0.9)
    else:
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.3, vmax=0.9)
        plt.colorbar(im, ax=ax, label='Accuracy')

        for i in range(len(strategies)):
            for j in range(len(split_types)):
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center')

        ax.set_xticks(range(len(split_types)))
        ax.set_xticklabels([s.replace('_', '\n') for s in split_types])
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)

    ax.set_xlabel('Data Split')
    ax.set_ylabel('Strategy')
    ax.set_title('Accuracy Heatmap: Strategy vs Data Split')

    plt.tight_layout()
    output_path = output_dir / f'accuracy_heatmap.{output_format}'
    plt.savefig(output_path, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_summary_table(
    results: Dict[str, Any],
    output_dir: Path,
    output_format: str = 'png'
):
    """
    生成汇总表格图
    """
    if not HAS_MATPLOTLIB:
        return

    # 组织数据
    rows = []
    for exp_name, exp_results in results.items():
        split_type, strategy = parse_experiment_name(exp_name)

        if exp_results.get('aggregated_result'):
            agg = exp_results['aggregated_result']
            rows.append({
                'Strategy': strategy,
                'Split': split_type,
                'Accuracy': f"{agg['accuracy']:.4f}",
                'Macro F1': f"{agg['macro_f1']:.4f}",
                'Loss': f"{agg['loss']:.4f}",
                'Samples': agg['n_samples'],
            })

    if not rows:
        return

    # 排序
    rows = sorted(rows, key=lambda x: (x['Split'], x['Strategy']))

    # 创建表格
    fig, ax = plt.subplots(figsize=(12, len(rows) * 0.5 + 2))
    ax.axis('off')

    columns = ['Strategy', 'Split', 'Accuracy', 'Macro F1', 'Loss', 'Samples']
    cell_text = [[row[col] for col in columns] for row in rows]

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # 设置表头样式
    for j, col in enumerate(columns):
        table[(0, j)].set_facecolor('#4a90d9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # 交替行颜色
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.set_title('Evaluation Results Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = output_dir / f'summary_table.{output_format}'
    plt.savefig(output_path, dpi=plt.rcParams['figure.dpi'], bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def generate_all_plots(
    results: Dict[str, Any],
    output_dir: Path,
    output_format: str = 'png',
    style: str = 'default'
):
    """
    生成所有图表
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib not available. Install with: pip install matplotlib")
        return

    # 应用样式
    if style in STYLE_CONFIGS:
        plt.rcParams.update(STYLE_CONFIGS[style])

    if HAS_SEABORN:
        sns.set_style("whitegrid")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating plots...")

    # 1. 策略对比 - 准确率
    try:
        plot_strategy_comparison(results, output_dir, 'accuracy', output_format)
    except Exception as e:
        logger.error(f"Error generating strategy comparison (accuracy): {e}")

    # 2. 策略对比 - F1
    try:
        plot_strategy_comparison(results, output_dir, 'macro_f1', output_format)
    except Exception as e:
        logger.error(f"Error generating strategy comparison (F1): {e}")

    # 3. 站点性能
    try:
        plot_site_performance(results, output_dir, output_format)
    except Exception as e:
        logger.error(f"Error generating site performance: {e}")

    # 4. 混淆矩阵
    try:
        plot_confusion_matrices(results, output_dir, output_format)
    except Exception as e:
        logger.error(f"Error generating confusion matrices: {e}")

    # 5. 准确率热力图
    try:
        plot_accuracy_heatmap(results, output_dir, output_format)
    except Exception as e:
        logger.error(f"Error generating accuracy heatmap: {e}")

    # 6. 汇总表格
    try:
        plot_summary_table(results, output_dir, output_format)
    except Exception as e:
        logger.error(f"Error generating summary table: {e}")

    logger.info(f"All plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='绘制联邦学习评估结果',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 从评估结果 JSON 绘图
    python plot_evaluation_results.py \\
        --results /ephemeral/exp/evaluation_results.json \\
        --output /ephemeral/exp/figures

    # 指定格式和样式
    python plot_evaluation_results.py \\
        --results /ephemeral/exp/evaluation_results.json \\
        --output /ephemeral/exp/figures \\
        --format pdf \\
        --style paper
        """
    )

    parser.add_argument('--results', type=str, required=True,
                        help='评估结果 JSON 文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='输出格式 (默认: png)')
    parser.add_argument('--style', type=str, default='default',
                        choices=['default', 'paper', 'presentation'],
                        help='绘图样式 (默认: default)')

    args = parser.parse_args()

    # 加载结果
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"结果文件不存在: {results_path}")
        sys.exit(1)

    with open(results_path, 'r') as f:
        results = json.load(f)

    logger.info(f"加载 {len(results)} 个实验结果")

    # 生成图表
    output_dir = Path(args.output)
    generate_all_plots(results, output_dir, args.format, args.style)

    # 列出生成的文件
    print("\n" + "="*60)
    print("生成的图表:")
    print("="*60)
    for f in sorted(output_dir.glob(f'*.{args.format}')):
        print(f"  {f.name}")
    print("="*60)


if __name__ == '__main__':
    main()
