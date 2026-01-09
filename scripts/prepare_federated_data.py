#!/usr/bin/env python3
"""
联邦学习数据准备脚本

用法:
    python prepare_federated_data.py --num_sites 3 --output_dir data/federated

功能:
1. 加载Haploblock Pipeline输出的cluster特征
2. 加载人群标签
3. 使用FederatedDataSplitter划分到各站点
4. 支持cluster和SNP两种特征模式
5. 支持多种数据划分策略:
   - IID: 独立同分布
   - Dirichlet: 非独立同分布 (使用Dirichlet分布)
   - Label Skew: 标签偏斜 (每个站点只包含部分类别)
   - Quantity Skew: 数量偏斜 (每个站点的数据量不同)
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from snp_deconvolution.data_integration.unified_loader import UnifiedFeatureLoader
from snp_deconvolution.nvflare_real.data.data_splitter import FederatedDataSplitter


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_arguments(args: argparse.Namespace) -> None:
    """
    验证命令行参数

    Args:
        args: 解析后的参数

    Raises:
        ValueError: 参数验证失败
    """
    # 验证pipeline输出目录
    pipeline_dir = Path(args.pipeline_output)
    if not pipeline_dir.exists():
        raise ValueError(f"Pipeline输出目录不存在: {pipeline_dir}")

    # 验证人群标签文件
    for pop_file in args.population_files:
        pop_path = Path(pop_file)
        if not pop_path.exists():
            raise ValueError(f"人群标签文件不存在: {pop_path}")

    # 验证数值参数
    if args.num_sites < 1:
        raise ValueError(f"站点数量必须至少为1: {args.num_sites}")

    if not 0 < args.val_ratio < 1:
        raise ValueError(f"验证集比例必须在(0, 1)之间: {args.val_ratio}")

    # 验证模式
    if args.mode not in ['cluster', 'snp', 'both']:
        raise ValueError(f"无效的模式: {args.mode}，必须是 'cluster', 'snp', 或 'both'")

    # 如果使用SNP模式，验证VCF路径
    if args.mode in ['snp', 'both'] and not args.vcf_path:
        raise ValueError("使用SNP模式时必须提供 --vcf_path 参数")

    if args.vcf_path:
        vcf_path = Path(args.vcf_path)
        if not vcf_path.exists():
            raise ValueError(f"VCF文件不存在: {vcf_path}")

    # 验证数据划分类型相关参数
    if args.split_type == 'dirichlet' and args.alpha <= 0:
        raise ValueError(f"Dirichlet alpha参数必须大于0: {args.alpha}")

    if args.split_type == 'label_skew':
        if args.labels_per_site < 1:
            raise ValueError(f"每站点标签数必须至少为1: {args.labels_per_site}")

    if args.split_type == 'quantity_skew':
        if not 0 < args.min_ratio < 1:
            raise ValueError(f"最小数据比例必须在(0, 1)之间: {args.min_ratio}")


def load_and_prepare_data(
    loader: UnifiedFeatureLoader,
    population_files: List[str],
    mode: str
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    加载并准备数据

    Args:
        loader: UnifiedFeatureLoader实例
        population_files: 人群标签文件列表
        mode: 特征模式 ('cluster' 或 'snp')

    Returns:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        metadata: 数据集元信息
    """
    logger.info(f"开始加载数据 (mode={mode})...")

    # 加载特征
    if mode == 'cluster':
        X, vocab_sizes = loader.load_cluster_features()
        logger.info(f"加载cluster特征: {X.shape}, vocab_sizes: {vocab_sizes}")
        metadata = {
            'mode': 'cluster',
            'vocab_sizes': vocab_sizes,
            'n_haploblocks': X.shape[1]
        }
    elif mode == 'snp':
        X = loader.load_snp_features()
        if hasattr(X, 'toarray'):  # 如果是稀疏矩阵
            X = X.toarray()
        logger.info(f"加载SNP特征: {X.shape}")
        metadata = {
            'mode': 'snp',
            'n_snps': X.shape[1]
        }
    else:
        raise ValueError(f"无效的模式: {mode}")

    # 加载标签
    labels, label_map = loader.load_labels(population_files)
    logger.info(f"加载标签: {len(labels)} 样本")

    # 过滤到有标签的样本
    labeled_mask = labels >= 0
    X_labeled = X[labeled_mask]
    y_labeled = labels[labeled_mask]

    n_samples = len(X_labeled)
    n_classes = len(np.unique(y_labeled))

    logger.info(f"有效样本: {n_samples} 个")
    logger.info(f"类别数量: {n_classes}")

    # 统计每个类别的样本数
    unique_labels, counts = np.unique(y_labeled, return_counts=True)
    logger.info("类别分布:")
    for label, count in zip(unique_labels, counts):
        logger.info(f"  类别 {label}: {count} 样本 ({count/n_samples*100:.1f}%)")

    metadata.update({
        'n_samples': n_samples,
        'n_classes': n_classes,
        'n_features': X_labeled.shape[1],
        'class_distribution': dict(zip(unique_labels.tolist(), counts.tolist()))
    })

    return X_labeled, y_labeled, metadata


def save_metadata(
    output_dir: Path,
    metadata: Dict[str, Any],
    args: argparse.Namespace
) -> None:
    """
    保存数据集元信息

    Args:
        output_dir: 输出目录
        metadata: 元信息字典
        args: 命令行参数
    """
    import json

    metadata_file = output_dir / 'dataset_metadata.json'

    full_metadata = {
        'dataset': metadata,
        'split_config': {
            'num_sites': args.num_sites,
            'val_ratio': args.val_ratio,
            'seed': args.seed,
            'split_type': args.split_type,
            'alpha': args.alpha if args.split_type == 'dirichlet' else None,
            'labels_per_site': args.labels_per_site if args.split_type == 'label_skew' else None,
            'min_ratio': args.min_ratio if args.split_type == 'quantity_skew' else None
        },
        'source': {
            'pipeline_output': str(args.pipeline_output),
            'population_files': args.population_files,
            'vcf_path': args.vcf_path
        }
    }

    with open(metadata_file, 'w') as f:
        json.dump(full_metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"元信息已保存到: {metadata_file}")


def print_summary(
    stats: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    打印汇总统计信息

    Args:
        stats: 各站点统计信息
        metadata: 数据集元信息
        output_dir: 输出目录
    """
    print("\n" + "="*70)
    print("联邦学习数据准备完成")
    print("="*70)

    print("\n数据集信息:")
    print(f"  特征模式: {metadata['mode']}")
    print(f"  总样本数: {metadata['n_samples']}")
    print(f"  特征维度: {metadata['n_features']}")
    print(f"  类别数量: {metadata['n_classes']}")

    if metadata['mode'] == 'cluster':
        print(f"  Haploblock数量: {metadata['n_haploblocks']}")
        print(f"  Vocab sizes: {metadata['vocab_sizes'][:5]}..." if len(metadata['vocab_sizes']) > 5 else f"  Vocab sizes: {metadata['vocab_sizes']}")

    print("\n类别分布:")
    for label, count in metadata['class_distribution'].items():
        print(f"  类别 {label}: {count} 样本")

    print(f"\n站点配置:")
    print(f"  站点数量: {len(stats)}")
    print(f"  划分类型: {metadata.get('split_type', 'iid')}")

    total_train = sum(s['train_samples'] for s in stats.values())
    total_val = sum(s['val_samples'] for s in stats.values())

    print(f"\n各站点统计:")
    for site_name, site_stats in stats.items():
        print(f"\n  {site_name}:")
        print(f"    总样本: {site_stats['total_samples']}")
        print(f"    训练集: {site_stats['train_samples']} ({site_stats['train_samples']/site_stats['total_samples']*100:.1f}%)")
        print(f"    验证集: {site_stats['val_samples']} ({site_stats['val_samples']/site_stats['total_samples']*100:.1f}%)")
        print(f"    训练集标签分布: {site_stats['train_label_dist']}")
        print(f"    验证集标签分布: {site_stats['val_label_dist']}")

    print(f"\n总计:")
    print(f"  训练集样本: {total_train}")
    print(f"  验证集样本: {total_val}")

    print(f"\n输出目录: {output_dir}")
    print(f"\n文件结构:")
    print(f"  {output_dir}/")
    for site_name in stats.keys():
        print(f"    {site_name}/")
        print(f"      train_{metadata['mode']}.pt (PyTorch格式)")
        print(f"      train_{metadata['mode']}.npz (NumPy格式)")
        print(f"      val_{metadata['mode']}.pt (PyTorch格式)")
        print(f"      val_{metadata['mode']}.npz (NumPy格式)")
    print(f"    dataset_metadata.json (数据集元信息)")

    print("\n" + "="*70)


def prepare_federated_data(args: argparse.Namespace) -> None:
    """
    执行联邦学习数据准备

    Args:
        args: 命令行参数
    """
    try:
        # 验证参数
        logger.info("验证命令行参数...")
        validate_arguments(args)

        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")

        # 初始化加载器
        logger.info("初始化数据加载器...")
        loader = UnifiedFeatureLoader(
            pipeline_output_dir=args.pipeline_output,
            vcf_path=args.vcf_path
        )

        # 确定要处理的模式
        modes = ['cluster', 'snp'] if args.mode == 'both' else [args.mode]

        for mode in modes:
            logger.info(f"\n{'='*70}")
            logger.info(f"处理模式: {mode}")
            logger.info(f"{'='*70}")

            # 加载数据
            X, y, metadata = load_and_prepare_data(
                loader=loader,
                population_files=args.population_files,
                mode=mode
            )

            # 创建模式特定的输出目录
            mode_output_dir = output_dir / mode if args.mode == 'both' else output_dir

            # 初始化数据划分器
            logger.info("\n初始化联邦数据划分器...")
            splitter = FederatedDataSplitter(
                output_dir=str(mode_output_dir),
                num_sites=args.num_sites,
                seed=args.seed
            )

            # 划分并保存数据
            logger.info(f"\n开始数据划分 (split_type={args.split_type})...")

            # 准备split_and_save的参数
            split_kwargs = {
                'X': X,
                'y': y,
                'val_ratio': args.val_ratio,
                'feature_type': mode,
                'split_type': args.split_type
            }

            # 根据划分类型添加额外参数
            if args.split_type == 'dirichlet':
                split_kwargs['alpha'] = args.alpha
                logger.info(f"  Dirichlet concentration parameter: {args.alpha}")
            elif args.split_type == 'label_skew':
                split_kwargs['labels_per_site'] = args.labels_per_site
                logger.info(f"  Labels per site: {args.labels_per_site}")
            elif args.split_type == 'quantity_skew':
                split_kwargs['min_ratio'] = args.min_ratio
                logger.info(f"  Minimum data ratio per site: {args.min_ratio}")

            stats = splitter.split_and_save(**split_kwargs)

            # 将split_type添加到metadata中以便在print_summary中使用
            metadata['split_type'] = args.split_type

            # 验证划分
            if args.verify:
                logger.info("\n验证数据划分...")
                verification = splitter.verify_split(feature_type=mode)

                if not verification['success']:
                    logger.error("数据划分验证失败!")
                    for error in verification['errors']:
                        logger.error(f"  - {error}")
                    sys.exit(1)
                else:
                    logger.info("数据划分验证通过")

            # 保存元信息
            save_metadata(mode_output_dir, metadata, args)

            # 打印汇总
            print_summary(stats, metadata, mode_output_dir)

        logger.info("\n所有数据处理完成!")

    except Exception as e:
        logger.error(f"数据准备失败: {e}", exc_info=True)
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="准备联邦学习数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  1. 使用IID划分准备3个站点的数据 (默认):
     python prepare_federated_data.py \\
         --pipeline_output out_dir/TNFa \\
         --num_sites 3 \\
         --output_dir data/federated/TNFa

  2. 使用Dirichlet分布创建非IID数据划分:
     python prepare_federated_data.py \\
         --pipeline_output out_dir/TNFa \\
         --num_sites 3 \\
         --split_type dirichlet \\
         --alpha 0.1 \\
         --output_dir data/federated/TNFa_dirichlet

  3. 使用标签偏斜 (每个站点只有部分类别):
     python prepare_federated_data.py \\
         --pipeline_output out_dir/TNFa \\
         --num_sites 3 \\
         --split_type label_skew \\
         --labels_per_site 2 \\
         --output_dir data/federated/TNFa_label_skew

  4. 使用数量偏斜 (不同站点有不同数据量):
     python prepare_federated_data.py \\
         --pipeline_output out_dir/TNFa \\
         --num_sites 3 \\
         --split_type quantity_skew \\
         --min_ratio 0.1 \\
         --output_dir data/federated/TNFa_quantity_skew

  5. 同时准备cluster和SNP特征:
     python prepare_federated_data.py \\
         --pipeline_output out_dir/TNFa \\
         --vcf_path data/chr6.vcf.gz \\
         --mode both \\
         --num_sites 3 \\
         --split_type dirichlet \\
         --alpha 0.5 \\
         --output_dir data/federated/TNFa

  6. 自定义人群标签文件:
     python prepare_federated_data.py \\
         --pipeline_output out_dir/TNFa \\
         --population_files data/igsr-chb.tsv.tsv data/igsr-gbr.tsv.tsv \\
         --num_sites 2 \\
         --output_dir data/federated/TNFa
        """
    )

    # 输入数据参数
    parser.add_argument(
        "--pipeline_output",
        type=str,
        default="out_dir/TNFa",
        help="Haploblock pipeline输出目录 (默认: out_dir/TNFa)"
    )
    parser.add_argument(
        "--population_files",
        nargs="+",
        default=["data/igsr-chb.tsv.tsv", "data/igsr-gbr.tsv.tsv", "data/igsr-pur.tsv.tsv"],
        help="人群标签文件列表 (默认: CHB, GBR, PUR)"
    )
    parser.add_argument(
        "--vcf_path",
        type=str,
        default=None,
        help="VCF文件路径 (SNP模式需要)"
    )

    # 特征模式
    parser.add_argument(
        "--mode",
        type=str,
        default="cluster",
        choices=["cluster", "snp", "both"],
        help="特征模式: cluster (隐私保护), snp (基线), 或 both (默认: cluster)"
    )

    # 联邦学习配置
    parser.add_argument(
        "--num_sites",
        type=int,
        default=3,
        help="联邦学习站点数量 (默认: 3)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/federated",
        help="输出目录 (默认: data/federated)"
    )

    # 数据划分参数
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="验证集比例 (默认: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--split_type",
        "--split_method",
        type=str,
        default="iid",
        choices=["iid", "dirichlet", "label_skew", "quantity_skew"],
        dest="split_type",
        help=(
            "数据划分类型 (默认: iid):\n"
            "  - iid: 独立同分布，每个站点的数据分布相同\n"
            "  - dirichlet: 使用Dirichlet分布生成非独立同分布的数据划分\n"
            "  - label_skew: 标签偏斜，每个站点只包含部分类别\n"
            "  - quantity_skew: 数量偏斜，每个站点的数据量不同"
        )
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Dirichlet分布的浓度参数 (仅在split_type='dirichlet'时使用，默认: 0.5)。\n"
            "较小的alpha值(例如0.1)会产生更不平衡的分布，\n"
            "较大的alpha值(例如10.0)会产生更均衡的分布。"
        )
    )
    parser.add_argument(
        "--labels_per_site",
        type=int,
        default=2,
        help=(
            "每个站点包含的标签数量 (仅在split_type='label_skew'时使用，默认: 2)。\n"
            "此参数控制每个站点的类别多样性。"
        )
    )
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=0.1,
        help=(
            "每个站点的最小数据比例 (仅在split_type='quantity_skew'时使用，默认: 0.1)。\n"
            "此参数确保每个站点至少获得总数据的这个比例。"
        )
    )

    # 其他选项
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证数据划分的正确性"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志"
    )

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 执行数据准备
    prepare_federated_data(args)


if __name__ == "__main__":
    main()
