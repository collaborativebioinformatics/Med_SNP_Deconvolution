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


def parse_nvflare_jsonl(log_file: Path) -> Dict[str, Any]:
    """
    解析 NVFlare JSON Lines 格式的日志文件
    """
    rounds_data = {}
    current_round = 0  # 初始化 current_round

    try:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    message = entry.get('message', '')

                    # 解析 FL Round 信息
                    round_match = re.search(r'FL Round (\d+)', message)
                    if round_match:
                        current_round = int(round_match.group(1))
                        if current_round not in rounds_data:
                            rounds_data[current_round] = {'val_loss': [], 'val_acc': []}

                    # 解析 val_loss - 多种格式
                    loss_patterns = [
                        r"'val_loss' reached ([\d.]+)",
                        r"val_loss[=:\s]+([\d.]+)",
                        r"Metric val_loss improved.*score: ([\d.]+)",
                    ]
                    for pattern in loss_patterns:
                        loss_match = re.search(pattern, message)
                        if loss_match:
                            val_loss = float(loss_match.group(1))
                            if current_round not in rounds_data:
                                rounds_data[current_round] = {'val_loss': [], 'val_acc': []}
                            rounds_data[current_round]['val_loss'].append(val_loss)
                            break

                    # 解析 val_accuracy 或 accuracy
                    acc_patterns = [
                        r"(?:val_accuracy|val_acc)[=:\s]+([\d.]+)",
                        r"accuracy[=:\s]+([\d.]+)",
                        r"Metric (?:val_)?acc(?:uracy)? improved.*score: ([\d.]+)",
                    ]
                    for pattern in acc_patterns:
                        acc_match = re.search(pattern, message)
                        if acc_match:
                            val_acc = float(acc_match.group(1))
                            if current_round not in rounds_data:
                                rounds_data[current_round] = {'val_loss': [], 'val_acc': []}
                            rounds_data[current_round]['val_acc'].append(val_acc)
                            break

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        logger.error(f"解析 JSONL 失败: {e}")

    return rounds_data


def extract_from_nvflare_workspace(workspace_dir: Path) -> Optional[Dict[str, Any]]:
    """
    从 NVFlare workspace 中提取实验结果

    NVFlare workspace 结构:
        workspace/
        ├── server/
        │   └── log.json
        └── site-1/
            ├── log.json
            └── lightning_logs/
    """
    result = {
        'rounds': [],
        'global_accuracy': [],
        'site_accuracies': {},
        'losses': [],
    }

    # 1. 首先尝试从 Lightning CSV 日志提取
    # 结构: site-X/lightning_logs/version_Y/metrics.csv
    # version_Y 对应 FL round Y
    site_dirs = [d for d in workspace_dir.iterdir() if d.is_dir() and d.name.startswith('site-')]

    if site_dirs:
        logger.info(f"  找到 {len(site_dirs)} 个站点目录")

        # 收集每个 round 每个站点的数据
        round_site_data = {}  # {round: {site: {'val_acc': [], 'val_loss': []}}}

        for site_dir in site_dirs:
            site_name = site_dir.name
            lightning_logs = site_dir / 'lightning_logs'

            if not lightning_logs.exists():
                continue

            # 遍历所有 version 目录
            for version_dir in sorted(lightning_logs.iterdir()):
                if not version_dir.is_dir() or not version_dir.name.startswith('version_'):
                    continue

                # 提取 round 号 (version_0 -> round 0)
                try:
                    round_num = int(version_dir.name.split('_')[1])
                except (IndexError, ValueError):
                    continue

                metrics_file = version_dir / 'metrics.csv'
                if not metrics_file.exists():
                    continue

                try:
                    import pandas as pd
                    df = pd.read_csv(metrics_file)

                    if round_num not in round_site_data:
                        round_site_data[round_num] = {}
                    if site_name not in round_site_data[round_num]:
                        round_site_data[round_num][site_name] = {'val_acc': [], 'val_loss': []}

                    # 提取 val_acc
                    if 'val_acc' in df.columns:
                        val_accs = df['val_acc'].dropna().tolist()
                        if val_accs:
                            round_site_data[round_num][site_name]['val_acc'].extend(val_accs)

                    # 提取 val_loss
                    if 'val_loss' in df.columns:
                        val_losses = df['val_loss'].dropna().tolist()
                        if val_losses:
                            round_site_data[round_num][site_name]['val_loss'].extend(val_losses)

                except Exception as e:
                    logger.warning(f"  解析 {metrics_file} 失败: {e}")
                    continue

        # 汇总数据
        if round_site_data:
            sorted_rounds = sorted(round_site_data.keys())
            logger.info(f"  找到 {len(sorted_rounds)} 轮数据")

            for r in sorted_rounds:
                result['rounds'].append(r)
                sites_data = round_site_data[r]

                # 计算所有站点的平均 val_acc
                all_val_accs = []
                all_val_losses = []

                for site_name, site_metrics in sites_data.items():
                    if site_metrics['val_acc']:
                        # 取每个站点该轮的最后一个 val_acc (最终结果)
                        all_val_accs.append(site_metrics['val_acc'][-1])
                    if site_metrics['val_loss']:
                        all_val_losses.append(site_metrics['val_loss'][-1])

                    # 保存站点数据
                    if site_name not in result['site_accuracies']:
                        result['site_accuracies'][site_name] = []
                    if site_metrics['val_acc']:
                        result['site_accuracies'][site_name].append(site_metrics['val_acc'][-1])

                if all_val_accs:
                    result['global_accuracy'].append(float(np.mean(all_val_accs)))
                elif all_val_losses:
                    avg_loss = np.mean(all_val_losses)
                    result['losses'].append(avg_loss)
                    result['global_accuracy'].append(1.0 / (1.0 + avg_loss))

    # 2. 尝试从 NVFlare JSON Lines 日志提取
    if not result['global_accuracy']:
        jsonl_files = list(workspace_dir.glob('**/log.json'))
        logger.info(f"  找到 JSONL 日志: {len(jsonl_files)} 个文件")

        all_rounds_data = {}
        site_data = {}

        for jsonl_file in jsonl_files:
            # 判断是 server 还是 site 日志
            parent_name = jsonl_file.parent.name
            rounds_data = parse_nvflare_jsonl(jsonl_file)

            if 'server' in str(jsonl_file):
                # 服务端日志 - 聚合数据
                for r, data in rounds_data.items():
                    if r not in all_rounds_data:
                        all_rounds_data[r] = data
            else:
                # 站点日志
                site_name = parent_name
                if rounds_data:
                    site_data[site_name] = rounds_data

        # 如果有站点数据但没有服务端数据，使用站点数据平均值
        if site_data and not all_rounds_data:
            # 合并所有站点的数据
            for site_name, rounds_data in site_data.items():
                for r, data in rounds_data.items():
                    if r not in all_rounds_data:
                        all_rounds_data[r] = {'val_loss': [], 'val_acc': []}
                    all_rounds_data[r]['val_loss'].extend(data.get('val_loss', []))
                    all_rounds_data[r]['val_acc'].extend(data.get('val_acc', []))

        # 转换为结果格式
        if all_rounds_data:
            sorted_rounds = sorted(all_rounds_data.keys())
            for r in sorted_rounds:
                result['rounds'].append(r)
                data = all_rounds_data[r]

                if data.get('val_acc'):
                    result['global_accuracy'].append(np.mean(data['val_acc']))
                elif data.get('val_loss'):
                    # 从 loss 估算 accuracy
                    avg_loss = np.mean(data['val_loss'])
                    # 使用 sigmoid 转换: acc = 1 / (1 + loss)
                    estimated_acc = 1.0 / (1.0 + avg_loss)
                    result['global_accuracy'].append(estimated_acc)
                    result['losses'].append(avg_loss)

    # 3. 尝试从 checkpoints 目录提取
    if not result['global_accuracy']:
        ckpt_files = list(workspace_dir.glob('**/checkpoints/**/*.ckpt'))
        if ckpt_files:
            logger.info(f"  找到 checkpoints: {len(ckpt_files)} 个文件")
            # 从文件名提取信息，例如: snp-round0-epoch=00-val_loss=0.0871.ckpt
            for ckpt in sorted(ckpt_files):
                name = ckpt.stem
                round_match = re.search(r'round(\d+)', name)
                loss_match = re.search(r'val_loss=([\d.]+)', name)
                acc_match = re.search(r'val_acc(?:uracy)?=([\d.]+)', name)

                if round_match:
                    r = int(round_match.group(1))
                    if r not in result['rounds']:
                        result['rounds'].append(r)

                        if acc_match:
                            result['global_accuracy'].append(float(acc_match.group(1)))
                        elif loss_match:
                            loss = float(loss_match.group(1))
                            result['losses'].append(loss)
                            # 估算 accuracy
                            result['global_accuracy'].append(1.0 / (1.0 + loss))

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
