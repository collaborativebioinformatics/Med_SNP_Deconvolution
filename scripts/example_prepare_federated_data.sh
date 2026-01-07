#!/bin/bash
# 联邦学习数据准备示例脚本
#
# 用法: bash scripts/example_prepare_federated_data.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo "联邦学习数据准备示例"
echo "========================================"
echo ""

# 配置参数
PIPELINE_OUTPUT="data/out_dir/TNFa"
OUTPUT_DIR="data/federated/TNFa_example"
NUM_SITES=3
VAL_RATIO=0.15
SEED=42

# 检查pipeline输出是否存在
if [ ! -d "$PIPELINE_OUTPUT" ]; then
    echo "错误: Pipeline输出目录不存在: $PIPELINE_OUTPUT"
    echo "请先运行Haploblock Pipeline"
    exit 1
fi

# 检查人群标签文件
POPULATION_FILES=(
    "data/igsr-chb.tsv.tsv"
    "data/igsr-gbr.tsv.tsv"
    "data/igsr-pur.tsv.tsv"
)

for pop_file in "${POPULATION_FILES[@]}"; do
    if [ ! -f "$pop_file" ]; then
        echo "警告: 人群标签文件不存在: $pop_file"
    fi
done

echo "配置信息:"
echo "  Pipeline输出: $PIPELINE_OUTPUT"
echo "  输出目录: $OUTPUT_DIR"
echo "  站点数量: $NUM_SITES"
echo "  验证集比例: $VAL_RATIO"
echo "  随机种子: $SEED"
echo ""

# 示例1: 使用默认配置准备cluster特征
echo "========================================="
echo "示例1: 准备cluster特征 (推荐)"
echo "========================================="
python scripts/prepare_federated_data.py \
    --pipeline_output "$PIPELINE_OUTPUT" \
    --population_files "${POPULATION_FILES[@]}" \
    --mode cluster \
    --num_sites "$NUM_SITES" \
    --output_dir "$OUTPUT_DIR/cluster" \
    --val_ratio "$VAL_RATIO" \
    --seed "$SEED" \
    --verify

echo ""
echo "完成! 数据已保存到: $OUTPUT_DIR/cluster"
echo ""

# 示例2: 如果有VCF文件，也可以准备SNP特征
VCF_PATH="data/chr6.vcf.gz"
if [ -f "$VCF_PATH" ]; then
    echo "========================================="
    echo "示例2: 准备SNP特征 (基线对比)"
    echo "========================================="
    python scripts/prepare_federated_data.py \
        --pipeline_output "$PIPELINE_OUTPUT" \
        --population_files "${POPULATION_FILES[@]}" \
        --vcf_path "$VCF_PATH" \
        --mode snp \
        --num_sites "$NUM_SITES" \
        --output_dir "$OUTPUT_DIR/snp" \
        --val_ratio "$VAL_RATIO" \
        --seed "$SEED" \
        --verify

    echo ""
    echo "完成! 数据已保存到: $OUTPUT_DIR/snp"
    echo ""
else
    echo "跳过SNP模式 (VCF文件不存在: $VCF_PATH)"
    echo ""
fi

# 显示输出文件结构
echo "========================================="
echo "输出文件结构"
echo "========================================="
if [ -d "$OUTPUT_DIR/cluster" ]; then
    echo ""
    echo "Cluster特征:"
    tree "$OUTPUT_DIR/cluster" -L 2 2>/dev/null || find "$OUTPUT_DIR/cluster" -type f | head -20
fi

if [ -d "$OUTPUT_DIR/snp" ]; then
    echo ""
    echo "SNP特征:"
    tree "$OUTPUT_DIR/snp" -L 2 2>/dev/null || find "$OUTPUT_DIR/snp" -type f | head -20
fi

echo ""
echo "========================================="
echo "数据准备完成!"
echo "========================================="
echo ""
echo "下一步:"
echo "1. 查看元信息: cat $OUTPUT_DIR/cluster/dataset_metadata.json"
echo "2. 配置NVFlare使用这些数据"
echo "3. 启动联邦学习训练"
echo ""
