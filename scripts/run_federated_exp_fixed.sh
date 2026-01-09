#!/bin/bash
# ============================================
# 联邦学习实验脚本（修复版）
# ============================================
#
# 主要修复：
# 1. 学习率从 1e-4 提升到 2e-3（与单体一致）
# 2. weight_decay 从 1e-5 提升到 0.01（与单体一致）
# 3. transformer_dim 从 128 降低到 64（与单体一致）
# 4. local_epochs 从 1 提升到 5（增加每轮训练）
# 5. num_rounds 从 20 提升到 50（更多联邦轮次）
#
# ============================================

# ============================================
# 变量定义
# ============================================
PIPELINE_OUTPUT="/ephemeral/output2"
DATA_BASE="/ephemeral/exp6/data"
EXP_BASE="/ephemeral/exp6/runs"

NUM_SITES=3
NUM_ROUNDS=50           # 增加到 50 轮
LOCAL_EPOCHS=5          # 每轮训练 5 个 epoch

# 模型配置（匹配单体训练）
MODEL_TYPE="haploblock"
ARCHITECTURE="transformer"
EMBEDDING_DIM=32
TRANSFORMER_DIM=64      # 改为 64（单体配置）

# 训练超参数（匹配单体训练）
LEARNING_RATE=2e-3      # 提升学习率
WEIGHT_DECAY=0.01       # 提升正则化

# ============================================
# 第一步：准备数据
# ============================================

echo "================================================"
echo "准备联邦学习数据..."
echo "================================================"

# IID 基线
echo "准备 IID 数据..."
python scripts/prepare_federated_data.py \
    --pipeline_output $PIPELINE_OUTPUT \
    --num_sites $NUM_SITES --split_type iid \
    --output_dir $DATA_BASE/iid \
    --verify

# Non-IID (Dirichlet α=0.1)
echo "准备 Dirichlet α=0.1 数据..."
python scripts/prepare_federated_data.py \
    --pipeline_output $PIPELINE_OUTPUT \
    --num_sites $NUM_SITES --split_type dirichlet --alpha 0.1 \
    --output_dir $DATA_BASE/dirichlet_0.1 \
    --verify

# Non-IID (Dirichlet α=1.0)
echo "准备 Dirichlet α=1.0 数据..."
python scripts/prepare_federated_data.py \
    --pipeline_output $PIPELINE_OUTPUT \
    --num_sites $NUM_SITES --split_type dirichlet --alpha 1.0 \
    --output_dir $DATA_BASE/dirichlet_1.0 \
    --verify

echo "数据准备完成！"
echo ""

# ============================================
# 第二步：运行实验
# ============================================

echo "================================================"
echo "开始联邦学习实验（修复版配置）"
echo "================================================"
echo "配置摘要："
echo "  - 学习率: $LEARNING_RATE (vs 旧版 1e-4)"
echo "  - Weight decay: $WEIGHT_DECAY (vs 旧版 1e-5)"
echo "  - Transformer dim: $TRANSFORMER_DIM (vs 旧版 128)"
echo "  - Local epochs: $LOCAL_EPOCHS (vs 旧版 1)"
echo "  - FL rounds: $NUM_ROUNDS (vs 旧版 20)"
echo "================================================"
echo ""

# ===== IID 数据 =====

echo "----------------------------------------"
echo "实验 1: IID + FedAvg"
echo "----------------------------------------"
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedavg \
    --data_dir $DATA_BASE/iid \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY \
    --workspace $EXP_BASE/iid_fedavg \
    --run_now

echo "----------------------------------------"
echo "实验 2: IID + FedProx"
echo "----------------------------------------"
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedprox --mu 0.01 \
    --data_dir $DATA_BASE/iid \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY \
    --workspace $EXP_BASE/iid_fedprox \
    --run_now

echo "----------------------------------------"
echo "实验 3: IID + FedOpt (Adam)"
echo "----------------------------------------"
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedopt --server_optimizer adam --server_lr 0.01 \
    --data_dir $DATA_BASE/iid \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY \
    --workspace $EXP_BASE/iid_fedopt \
    --run_now

# ===== Non-IID (Dirichlet α=0.1) =====

echo "----------------------------------------"
echo "实验 4: Dirichlet α=0.1 + FedAvg"
echo "----------------------------------------"
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedavg \
    --data_dir $DATA_BASE/dirichlet_0.1 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY \
    --workspace $EXP_BASE/dir0.1_fedavg \
    --run_now

echo "----------------------------------------"
echo "实验 5: Dirichlet α=0.1 + FedProx"
echo "----------------------------------------"
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedprox --mu 0.01 \
    --data_dir $DATA_BASE/dirichlet_0.1 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY \
    --workspace $EXP_BASE/dir0.1_fedprox \
    --run_now

echo "----------------------------------------"
echo "实验 6: Dirichlet α=0.1 + SCAFFOLD"
echo "----------------------------------------"
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy scaffold \
    --data_dir $DATA_BASE/dirichlet_0.1 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY \
    --workspace $EXP_BASE/dir0.1_scaffold \
    --run_now

# ===== Non-IID (Dirichlet α=1.0) =====

echo "----------------------------------------"
echo "实验 7: Dirichlet α=1.0 + FedAvg"
echo "----------------------------------------"
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedavg \
    --data_dir $DATA_BASE/dirichlet_1.0 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY \
    --workspace $EXP_BASE/dir1.0_fedavg \
    --run_now

echo "----------------------------------------"
echo "实验 8: Dirichlet α=1.0 + FedProx"
echo "----------------------------------------"
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedprox --mu 0.01 \
    --data_dir $DATA_BASE/dirichlet_1.0 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS \
    --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY \
    --workspace $EXP_BASE/dir1.0_fedprox \
    --run_now

# ============================================
# 第三步：可视化结果
# ============================================

echo ""
echo "================================================"
echo "生成可视化..."
echo "================================================"
python scripts/visualize_fl_experiments.py \
      --runs_dir $EXP_BASE \
      --output_dir ./figures/exp6 \
      --plot \
      --target_accuracy 0.7 \
      --format png \
      --style paper

echo ""
echo "================================================"
echo "所有实验完成！"
echo "================================================"
echo "结果目录: $EXP_BASE"
echo "图表目录: ./figures/exp6"
echo "================================================"
