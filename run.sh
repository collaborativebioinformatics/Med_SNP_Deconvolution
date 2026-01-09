#!/bin/bash

# ============================================ 
# 变量定义
# ============================================ 
PIPELINE_OUTPUT="/ephemeral/output2"
DATA_BASE="/ephemeral/exp5/data"
EXP_BASE="/ephemeral/exp5/runs"

NUM_SITES=3
# Updated to 50 to match standalone total epochs (50)
NUM_ROUNDS=50
# Increased local epochs to allow more local learning per round
LOCAL_EPOCHS=5
# Updated LR to match standalone script (2e-3)
LEARNING_RATE=2e-3
# Updated Batch Size to match standalone script (32)
BATCH_SIZE=32

# 模型配置 (haploblock 用于 2D cluster ID 数据)
MODEL_TYPE="haploblock"
ARCHITECTURE="transformer"
EMBEDDING_DIM=32
# Updated to 64 to match standalone script
TRANSFORMER_DIM=64

# ============================================
# 第一步：准备数据
# ============================================

# IID 基线
python scripts/prepare_federated_data.py \
    --pipeline_output $PIPELINE_OUTPUT \
    --num_sites $NUM_SITES --split_type iid \
    --output_dir $DATA_BASE/iid \
    --verify

# Non-IID (Dirichlet α=0.1)
python scripts/prepare_federated_data.py \
    --pipeline_output $PIPELINE_OUTPUT \
    --num_sites $NUM_SITES --split_type dirichlet --alpha 0.1 \
    --output_dir $DATA_BASE/dirichlet_0.1 \
    --verify

# Non-IID (Dirichlet α=1.0)
python scripts/prepare_federated_data.py \
    --pipeline_output $PIPELINE_OUTPUT \
    --num_sites $NUM_SITES --split_type dirichlet --alpha 1.0 \
    --output_dir $DATA_BASE/dirichlet_1.0 \
    --verify

# ============================================
# 第二步：运行实验
# ============================================

# ===== IID 数据 =====

# FedAvg
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedavg \
    --data_dir $DATA_BASE/iid \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --workspace $EXP_BASE/iid_fedavg \
    --run_now

# FedProx
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedprox --mu 0.01 \
    --data_dir $DATA_BASE/iid \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --workspace $EXP_BASE/iid_fedprox \
    --run_now

# FedOpt (Adam)
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedopt --server_optimizer adam --server_lr 0.01 \
    --data_dir $DATA_BASE/iid \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --workspace $EXP_BASE/iid_fedopt \
    --run_now

# ===== Non-IID (Dirichlet α=0.1) =====

# FedAvg
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedavg \
    --data_dir $DATA_BASE/dirichlet_0.1 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --workspace $EXP_BASE/dir0.1_fedavg \
    --run_now

# FedProx
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedprox --mu 0.01 \
    --data_dir $DATA_BASE/dirichlet_0.1 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --workspace $EXP_BASE/dir0.1_fedprox \
    --run_now

# SCAFFOLD
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy scaffold \
    --data_dir $DATA_BASE/dirichlet_0.1 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --workspace $EXP_BASE/dir0.1_scaffold \
    --run_now

# ===== Non-IID (Dirichlet α=1.0) =====

# FedAvg
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedavg \
    --data_dir $DATA_BASE/dirichlet_1.0 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --workspace $EXP_BASE/dir1.0_fedavg \
    --run_now

# FedProx
python snp_deconvolution/nvflare_real/lightning/job.py \
    --mode poc --strategy fedprox --mu 0.01 \
    --data_dir $DATA_BASE/dirichlet_1.0 \
    --model_type $MODEL_TYPE --architecture $ARCHITECTURE \
    --embedding_dim $EMBEDDING_DIM --transformer_dim $TRANSFORMER_DIM \
    --num_clients $NUM_SITES --num_rounds $NUM_ROUNDS \
    --local_epochs $LOCAL_EPOCHS --learning_rate $LEARNING_RATE --batch_size $BATCH_SIZE \
    --workspace $EXP_BASE/dir1.0_fedprox \
    --run_now
python scripts/visualize_fl_experiments.py \
      --runs_dir /ephemeral/exp5/runs \
      --output_dir  ./figures/exp5 \
      --plot \
      --target_accuracy 0.5 \
      --format png \
      --style paper