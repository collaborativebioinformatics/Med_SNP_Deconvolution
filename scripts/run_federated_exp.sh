#!/bin/bash
#
# 联邦学习实验脚本
#
# 用法:
#   ./run_federated_exp.sh --input /ephemeral/output --output /ephemeral/exp
#
# 功能:
#   1. 数据分割 (IID 和 Non-IID)
#   2. 运行多种联邦学习策略实验
#

set -e

# ============================================================================
# 默认参数
# ============================================================================
INPUT_DIR=""
OUTPUT_DIR=""
NUM_SITES=3
NUM_ROUNDS=10
LOCAL_EPOCHS=1
BATCH_SIZE=128
LEARNING_RATE=1e-4
MODE="cluster"
SKIP_SPLIT=false
SKIP_EXP=false
STRATEGIES="fedavg,fedprox,fedopt"
SPLIT_TYPES="iid,dirichlet_0.5"

# ============================================================================
# 帮助信息
# ============================================================================
usage() {
    cat << EOF
联邦学习实验脚本

用法:
    $0 --input <INPUT_DIR> --output <OUTPUT_DIR> [OPTIONS]

必需参数:
    --input, -i         Pipeline输出目录 (包含 haploblock_hashes.tsv 等)
    --output, -o        实验输出目录

可选参数:
    --num_sites         站点数量 (默认: 3)
    --num_rounds        联邦学习轮数 (默认: 10)
    --local_epochs      本地训练epochs (默认: 1)
    --batch_size        批大小 (默认: 128)
    --learning_rate     学习率 (默认: 1e-4)
    --mode              特征模式: cluster/snp (默认: cluster)
    --strategies        逗号分隔的策略列表 (默认: fedavg,fedprox,fedopt)
    --split_types       逗号分隔的分割类型 (默认: iid,dirichlet_0.5)
    --skip_split        跳过数据分割步骤
    --skip_exp          跳过实验运行步骤
    -h, --help          显示帮助信息

示例:
    # 完整实验
    $0 --input /ephemeral/output --output /ephemeral/exp

    # 只分割数据
    $0 --input /ephemeral/output --output /ephemeral/exp --skip_exp

    # 只运行实验 (数据已分割)
    $0 --input /ephemeral/output --output /ephemeral/exp --skip_split

    # 自定义策略和分割类型
    $0 --input /ephemeral/output --output /ephemeral/exp \\
       --strategies "fedavg,fedprox,scaffold,fedopt" \\
       --split_types "iid,dirichlet_0.1,dirichlet_1.0"

分割类型说明:
    iid                 独立同分布
    dirichlet_<alpha>   Dirichlet分布 (alpha越小越不均衡)
    label_skew_<n>      每站点n个类别
    quantity_skew_<r>   最小数据比例r

策略说明:
    fedavg              联邦平均 (基线)
    fedprox             联邦近端 (处理Non-IID)
    scaffold            方差缩减
    fedopt              服务端自适应优化
EOF
    exit 0
}

# ============================================================================
# 参数解析
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --input|-i)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_sites)
            NUM_SITES="$2"
            shift 2
            ;;
        --num_rounds)
            NUM_ROUNDS="$2"
            shift 2
            ;;
        --local_epochs)
            LOCAL_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        --split_types)
            SPLIT_TYPES="$2"
            shift 2
            ;;
        --skip_split)
            SKIP_SPLIT=true
            shift
            ;;
        --skip_exp)
            SKIP_EXP=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

# 验证必需参数
if [[ -z "$INPUT_DIR" ]]; then
    echo "错误: 必须指定 --input 参数"
    usage
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "错误: 必须指定 --output 参数"
    usage
fi

# ============================================================================
# 路径设置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${OUTPUT_DIR}/data"
RUNS_DIR="${OUTPUT_DIR}/runs"
LOGS_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$DATA_DIR" "$RUNS_DIR" "$LOGS_DIR"

# ============================================================================
# 日志函数
# ============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_section() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
}

# ============================================================================
# 数据分割函数
# ============================================================================
split_data() {
    local split_type=$1
    local output_subdir=$2
    local extra_args=$3

    log "分割数据: $split_type -> $output_subdir"

    python "${PROJECT_DIR}/scripts/prepare_federated_data.py" \
        --pipeline_output "$INPUT_DIR" \
        --num_sites "$NUM_SITES" \
        --output_dir "${DATA_DIR}/${output_subdir}" \
        --mode "$MODE" \
        --verify \
        $extra_args \
        2>&1 | tee "${LOGS_DIR}/split_${output_subdir}.log"
}

# ============================================================================
# 实验运行函数
# ============================================================================
run_experiment() {
    local strategy=$1
    local data_subdir=$2
    local extra_args=$3

    local exp_name="${data_subdir}_${strategy}"
    log "运行实验: $exp_name"

    python "${PROJECT_DIR}/snp_deconvolution/nvflare_real/lightning/job.py" \
        --mode poc \
        --strategy "$strategy" \
        --data_dir "${DATA_DIR}/${data_subdir}" \
        --num_clients "$NUM_SITES" \
        --num_rounds "$NUM_ROUNDS" \
        --local_epochs "$LOCAL_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE" \
        --feature_type "$MODE" \
        --workspace "${RUNS_DIR}/${exp_name}" \
        --run_now \
        $extra_args \
        2>&1 | tee "${LOGS_DIR}/exp_${exp_name}.log"
}

# ============================================================================
# 主流程
# ============================================================================
log_section "联邦学习实验"
log "输入目录: $INPUT_DIR"
log "输出目录: $OUTPUT_DIR"
log "站点数量: $NUM_SITES"
log "联邦轮数: $NUM_ROUNDS"
log "特征模式: $MODE"
log "策略列表: $STRATEGIES"
log "分割类型: $SPLIT_TYPES"

# ============================================================================
# Step 1: 数据分割
# ============================================================================
if [[ "$SKIP_SPLIT" == false ]]; then
    log_section "Step 1: 数据分割"

    IFS=',' read -ra SPLIT_ARRAY <<< "$SPLIT_TYPES"
    for split in "${SPLIT_ARRAY[@]}"; do
        split=$(echo "$split" | xargs)  # trim whitespace

        if [[ "$split" == "iid" ]]; then
            split_data "iid" "iid" "--split_type iid"

        elif [[ "$split" =~ ^dirichlet_([0-9.]+)$ ]]; then
            alpha="${BASH_REMATCH[1]}"
            split_data "dirichlet" "dirichlet_${alpha}" "--split_type dirichlet --alpha $alpha"

        elif [[ "$split" =~ ^label_skew_([0-9]+)$ ]]; then
            labels="${BASH_REMATCH[1]}"
            split_data "label_skew" "label_skew_${labels}" "--split_type label_skew --labels_per_site $labels"

        elif [[ "$split" =~ ^quantity_skew_([0-9.]+)$ ]]; then
            ratio="${BASH_REMATCH[1]}"
            split_data "quantity_skew" "quantity_skew_${ratio}" "--split_type quantity_skew --min_ratio $ratio"

        else
            log "警告: 未知的分割类型 '$split', 跳过"
        fi
    done

    log "数据分割完成"
else
    log "跳过数据分割步骤"
fi

# ============================================================================
# Step 2: 运行实验
# ============================================================================
if [[ "$SKIP_EXP" == false ]]; then
    log_section "Step 2: 运行联邦学习实验"

    IFS=',' read -ra STRATEGY_ARRAY <<< "$STRATEGIES"
    IFS=',' read -ra SPLIT_ARRAY <<< "$SPLIT_TYPES"

    for split in "${SPLIT_ARRAY[@]}"; do
        split=$(echo "$split" | xargs)

        # 确定数据子目录名称
        if [[ "$split" == "iid" ]]; then
            data_subdir="iid"
        elif [[ "$split" =~ ^dirichlet_([0-9.]+)$ ]]; then
            data_subdir="dirichlet_${BASH_REMATCH[1]}"
        elif [[ "$split" =~ ^label_skew_([0-9]+)$ ]]; then
            data_subdir="label_skew_${BASH_REMATCH[1]}"
        elif [[ "$split" =~ ^quantity_skew_([0-9.]+)$ ]]; then
            data_subdir="quantity_skew_${BASH_REMATCH[1]}"
        else
            continue
        fi

        # 检查数据目录是否存在
        if [[ ! -d "${DATA_DIR}/${data_subdir}" ]]; then
            log "警告: 数据目录不存在 ${DATA_DIR}/${data_subdir}, 跳过"
            continue
        fi

        for strategy in "${STRATEGY_ARRAY[@]}"; do
            strategy=$(echo "$strategy" | xargs)

            # 策略特定参数
            extra_args=""
            case "$strategy" in
                fedprox)
                    extra_args="--mu 0.01"
                    ;;
                fedopt)
                    extra_args="--server_optimizer adam --server_lr 0.01"
                    ;;
            esac

            run_experiment "$strategy" "$data_subdir" "$extra_args"
        done
    done

    log "所有实验完成"
else
    log "跳过实验运行步骤"
fi

# ============================================================================
# 汇总
# ============================================================================
log_section "实验完成"
log "数据目录: $DATA_DIR"
log "实验结果: $RUNS_DIR"
log "日志目录: $LOGS_DIR"

echo ""
echo "目录结构:"
echo "${OUTPUT_DIR}/"
if [[ -d "$DATA_DIR" ]]; then
    echo "├── data/"
    ls -1 "$DATA_DIR" 2>/dev/null | sed 's/^/│   ├── /'
fi
if [[ -d "$RUNS_DIR" ]]; then
    echo "├── runs/"
    ls -1 "$RUNS_DIR" 2>/dev/null | sed 's/^/│   ├── /'
fi
if [[ -d "$LOGS_DIR" ]]; then
    echo "└── logs/"
    ls -1 "$LOGS_DIR" 2>/dev/null | sed 's/^/    ├── /'
fi
