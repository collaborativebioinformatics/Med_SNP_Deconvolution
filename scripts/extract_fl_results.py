#!/usr/bin/env python3
"""
从 NVFlare 实验结果中提取数据并生成可视化

用法:
    python extract_fl_results.py --runs_dir /ephemeral/exp/runs --output_dir /ephemeral/exp/figures
"""

import argparse
import json
import logging
import re
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


def parse_experiment_name(exp_name: str) -> tuple[str, str]:
    """
    解析实验目录名称，提取 split_type 和 strategy

    Expected formats:
        - iid_fedavg
        - dirichlet_0.5_fedprox
        - dir0.1_scaffold
    """
    exp_name = exp_name.lower()

    # 提取策略
    strategy = 'FedAvg'
    if 'fedavg' in exp_name:
        strategy = 'FedAvg'
    elif 'fedprox' in exp_name:
        strategy = 'FedProx'
    elif 'scaffold' in exp_name:
        strategy = 'Scaffold'
    elif 'fedopt' in exp_name or 'fedadam' in exp_name:
        strategy = 'FedOpt'

    # 提取分割类型
    split_type = 'IID'
    if 'iid' in exp_name and 'dirichlet' not in exp_name:
        split_type = 'IID'
    elif 'dirichlet' in exp_name or 'dir' in exp_name:
        # 尝试提取 alpha 值
        match = re.search(r'(?:dirichlet|dir)[_]?(\d+\.?\d*)', exp_name)
        if match:
            alpha = match.group(1)
            split_type = f'Dirichlet_{alpha}'
        else:
            split_type = 'Dirichlet_0.5'
    elif 'label' in exp_name and 'skew' in exp_name:
        split_type = 'Label_Skew'
    elif 'quantity' in exp_name and 'skew' in exp_name:
        split_type = 'Quantity_Skew'

    return split_type, strategy


def extract_from_nvflare_workspace(workspace_dir: Path) -> Optional[Dict[str, Any]]:
    """
    从 NVFlare workspace 中提取实验结果

    NVFlare workspace 结构:
        workspace/
        ├── server/
        │   └── simulate_job/
        │       └── ...
        └── site-1/
            └── simulate_job/
                └── ...
    """
    result = {
        'rounds': [],
        'global_accuracy': [],
        'site_accuracies': {},
        'losses': [],
    }

    # 尝试查找日志文件
    log_patterns = [
        '**/log*.txt',
        '**/training*.log',
        '**/server/**/*.json',
        '**/*metrics*.json',
        '**/*results*.json',
    ]

    for pattern in log_patterns:
        log_files = list(workspace_dir.glob(pattern))
        if log_files:
            logger.info(f"  找到日志文件: {[f.name for f in log_files[:3]]}")
            break

    # 尝试解析 JSON 结果文件
    json_files = list(workspace_dir.glob('**/*.json'))
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # 检查是否包含训练结果
            if 'accuracy' in data or 'val_accuracy' in data or 'metrics' in data:
                logger.info(f"  解析 JSON: {json_file.name}")

                if isinstance(data, dict):
                    if 'rounds' in data:
                        result['rounds'] = data['rounds']
                    if 'accuracy' in data:
                        if isinstance(data['accuracy'], list):
                            result['global_accuracy'] = data['accuracy']
                        else:
                            result['global_accuracy'].append(data['accuracy'])
                    if 'val_accuracy' in data:
                        if isinstance(data['val_accuracy'], list):
                            result['global_accuracy'] = data['val_accuracy']

        except Exception as e:
            continue

    # 尝试解析文本日志
    txt_files = list(workspace_dir.glob('**/*.txt')) + list(workspace_dir.glob('**/*.log'))
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                content = f.read()

            # 解析训练日志中的 accuracy
            # 常见格式: "Round 1: accuracy=0.75" 或 "Epoch 1, val_acc: 0.75"
            acc_patterns = [
                r'[Rr]ound\s*(\d+).*?(?:accuracy|acc)[=:\s]+(\d+\.?\d*)',
                r'[Ee]poch\s*(\d+).*?(?:val_acc|accuracy)[=:\s]+(\d+\.?\d*)',
                r'(?:accuracy|acc)[=:\s]+(\d+\.?\d*)',
            ]

            for pattern in acc_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    if len(matches[0]) == 2:
                        for round_num, acc in matches:
                            if int(round_num) not in result['rounds']:
                                result['rounds'].append(int(round_num))
                                result['global_accuracy'].append(float(acc))
                    else:
                        for acc in matches:
                            result['global_accuracy'].append(float(acc))
                    break

        except Exception as e:
            continue

    # 如果没有找到数据，返回 None
    if not result['global_accuracy']:
        return None

    # 确保 rounds 有值
    if not result['rounds']:
        result['rounds'] = list(range(1, len(result['global_accuracy']) + 1))

    # 计算最终统计
    result['final_accuracy'] = result['global_accuracy'][-1] if result['global_accuracy'] else 0
    result['std'] = np.std(result['global_accuracy'][-5:]) if len(result['global_accuracy']) >= 5 else 0.01

    return result


def extract_from_log_file(log_file: Path) -> Optional[Dict[str, Any]]:
    """
    从实验日志文件中提取结果
    """
    result = {
        'rounds': [],
        'global_accuracy': [],
        'site_accuracies': {},
        'losses': [],
    }

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # 解析不同格式的日志
        # 格式1: "Round X: global_accuracy=Y"
        pattern1 = r'[Rr]ound\s*(\d+).*?(?:global_)?(?:accuracy|acc)[=:\s]+(\d+\.?\d*)'
        matches = re.findall(pattern1, content)

        if matches:
            for round_num, acc in matches:
                result['rounds'].append(int(round_num))
                result['global_accuracy'].append(float(acc))

        # 格式2: Lightning 训练日志
        pattern2 = r'(?:val|validation).*?(?:accuracy|acc)[=:\s]+(\d+\.?\d*)'
        matches2 = re.findall(pattern2, content)

        if matches2 and not result['global_accuracy']:
            for i, acc in enumerate(matches2, 1):
                result['rounds'].append(i)
                result['global_accuracy'].append(float(acc))

        if result['global_accuracy']:
            result['final_accuracy'] = result['global_accuracy'][-1]
            result['std'] = np.std(result['global_accuracy'][-5:]) if len(result['global_accuracy']) >= 5 else 0.01
            return result

    except Exception as e:
        logger.error(f"解析日志文件失败 {log_file}: {e}")

    return None


def extract_all_results(runs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    从实验目录中提取所有结果

    Returns:
        {
            'FedAvg': {
                'IID': {...},
                'Dirichlet_0.5': {...},
            },
            'FedProx': {...},
        }
    """
    results = {}

    if not runs_dir.exists():
        logger.error(f"实验目录不存在: {runs_dir}")
        return results

    # 遍历所有实验目录
    exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]

    logger.info(f"找到 {len(exp_dirs)} 个实验目录")

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        logger.info(f"处理: {exp_name}")

        split_type, strategy = parse_experiment_name(exp_name)
        logger.info(f"  -> Strategy: {strategy}, Split: {split_type}")

        # 尝试从 workspace 提取
        exp_result = extract_from_nvflare_workspace(exp_dir)

        # 如果失败，尝试从日志文件提取
        if exp_result is None:
            log_files = list(exp_dir.glob('*.log')) + list(exp_dir.glob('*.txt'))
            for log_file in log_files:
                exp_result = extract_from_log_file(log_file)
                if exp_result:
                    break

        if exp_result:
            if strategy not in results:
                results[strategy] = {}
            results[strategy][split_type] = exp_result
            logger.info(f"  成功提取 {len(exp_result['rounds'])} 轮数据")
        else:
            logger.warning(f"  未能提取数据")

    return results


def save_results(results: Dict[str, Dict], output_dir: Path):
    """
    保存提取的结果为 JSON 文件
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for strategy, splits in results.items():
        for split_type, data in splits.items():
            filename = f"{strategy.lower()}_{split_type.lower().replace('.', '_')}.json"
            filepath = output_dir / filename

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"保存: {filepath}")


def generate_synthetic_results(
    strategies: List[str] = ['FedAvg', 'FedProx', 'Scaffold', 'FedOpt'],
    split_types: List[str] = ['IID', 'Dirichlet_0.5', 'Dirichlet_0.1'],
    num_rounds: int = 10,
    num_sites: int = 3
) -> Dict[str, Dict[str, Any]]:
    """
    生成合成实验数据用于演示
    """
    logger.info("生成合成实验数据...")

    results = {}

    # 策略特性
    strategy_characteristics = {
        'FedAvg': {'convergence_rate': 0.15, 'final_boost': 0},
        'FedProx': {'convergence_rate': 0.18, 'final_boost': 0.02},
        'Scaffold': {'convergence_rate': 0.20, 'final_boost': 0.03},
        'FedOpt': {'convergence_rate': 0.22, 'final_boost': 0.04},
    }

    # 分割类型特性
    split_characteristics = {
        'IID': {'final_acc': 0.85, 'noise': 0.01},
        'Dirichlet_0.5': {'final_acc': 0.80, 'noise': 0.015},
        'Dirichlet_0.1': {'final_acc': 0.75, 'noise': 0.02},
        'Dirichlet_1.0': {'final_acc': 0.82, 'noise': 0.012},
        'Label_Skew': {'final_acc': 0.72, 'noise': 0.025},
    }

    for strategy in strategies:
        results[strategy] = {}
        s_char = strategy_characteristics.get(strategy, strategy_characteristics['FedAvg'])

        for split_type in split_types:
            sp_char = split_characteristics.get(split_type, split_characteristics['IID'])

            rounds = list(range(1, num_rounds + 1))
            final_acc = sp_char['final_acc'] + s_char['final_boost']

            # 生成收敛曲线
            global_accuracy = []
            for r in rounds:
                acc = final_acc * (1 - np.exp(-s_char['convergence_rate'] * r))
                acc += np.random.normal(0, sp_char['noise'])
                acc = np.clip(acc, 0.3, 0.95)
                global_accuracy.append(float(acc))

            # 生成站点数据
            site_accuracies = {}
            for site_id in range(num_sites):
                site_offset = np.random.normal(0, 0.02)
                site_accuracies[f'site_{site_id}'] = [
                    float(np.clip(acc + site_offset + np.random.normal(0, sp_char['noise'] * 0.5), 0.25, 0.95))
                    for acc in global_accuracy
                ]

            results[strategy][split_type] = {
                'rounds': rounds,
                'global_accuracy': global_accuracy,
                'site_accuracies': site_accuracies,
                'final_accuracy': float(global_accuracy[-1]),
                'std': float(np.std([site_accuracies[f'site_{i}'][-1] for i in range(num_sites)])),
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='从 NVFlare 实验结果中提取数据并可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 提取实验结果
    python extract_fl_results.py --runs_dir /ephemeral/exp/runs --output_dir /ephemeral/exp/results

    # 提取并直接绘图
    python extract_fl_results.py --runs_dir /ephemeral/exp/runs --output_dir /ephemeral/exp/figures --plot

    # 使用合成数据演示
    python extract_fl_results.py --synthetic --output_dir /ephemeral/exp/figures --plot
        """
    )

    parser.add_argument(
        '--runs_dir',
        type=str,
        help='实验运行目录 (包含各实验子目录)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./fl_results',
        help='输出目录 (默认: ./fl_results)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='提取后直接生成可视化图表'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='生成合成数据用于演示'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='图表输出格式 (默认: png)'
    )
    parser.add_argument(
        '--style',
        type=str,
        choices=['default', 'paper', 'presentation'],
        default='default',
        help='绘图风格 (默认: default)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 提取或生成结果
    if args.synthetic:
        results = generate_synthetic_results()
    elif args.runs_dir:
        runs_dir = Path(args.runs_dir)
        results = extract_all_results(runs_dir)
    else:
        logger.error("必须指定 --runs_dir 或 --synthetic")
        sys.exit(1)

    if not results:
        logger.error("没有提取到任何结果")
        sys.exit(1)

    # 保存结果
    results_dir = output_dir / 'results' if args.plot else output_dir
    save_results(results, results_dir)

    # 生成可视化
    if args.plot:
        logger.info("生成可视化图表...")

        # 导入可视化模块
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from visualize_fl_experiments import FLVisualizer, ExperimentDataLoader

            figures_dir = output_dir / 'figures'
            figures_dir.mkdir(parents=True, exist_ok=True)

            visualizer = FLVisualizer(
                data=results,
                output_dir=figures_dir,
                output_format=args.format,
                style=args.style
            )
            visualizer.generate_all_visualizations()

            logger.info(f"图表保存到: {figures_dir}")

        except ImportError as e:
            logger.error(f"无法导入可视化模块: {e}")
            logger.info("请手动运行: python scripts/visualize_fl_experiments.py --results_dir " + str(results_dir))

    # 打印汇总
    print("\n" + "=" * 60)
    print("提取结果汇总")
    print("=" * 60)
    for strategy, splits in results.items():
        print(f"\n{strategy}:")
        for split_type, data in splits.items():
            final_acc = data.get('final_accuracy', 0)
            num_rounds = len(data.get('rounds', []))
            print(f"  {split_type}: {final_acc:.4f} (Rounds: {num_rounds})")
    print("=" * 60)


if __name__ == '__main__':
    main()
