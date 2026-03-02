#!/bin/bash
# PiM-IK 消融实验一键运行脚本
# 使用 4 个 GPU (3,5,6,7)，每个 GPU 运行 3 个实验（串行）
# 总共 12 个实验，预计时间 3-4 小时
#
# 使用方法:
#   cd /home/ygx/pim-ik
#   bash scripts/run_all_ablations.sh
#
# 查看进度:
#   tail -f logs/ablation_*.log
#
# 终止所有实验:
#   pkill -f torchrun

set -e

# 项目根目录
PROJECT_DIR="/home/ygx/pim-ik"

# Checkpoint 保存位置 (ygx_data)
CHECKPOINT_BASE="/data0/ygx_data/checkpoints/ablation_anti"

# 日志目录
LOG_DIR="${PROJECT_DIR}/logs"

# 清理旧日志
rm -f ${LOG_DIR}/ablation_*.log 2>/dev/null || true
echo "🧹 已清理旧日志"

# 创建日志目录
mkdir -p ${LOG_DIR}

# 创建本地 checkpoints 软链接
ln -sf /data0/ygx_data/checkpoints ${PROJECT_DIR}/checkpoints
echo "📁 Checkpoints 软链接: ${PROJECT_DIR}/checkpoints -> /data0/ygx_data/checkpoints"

# 公共参数 (抗过拟合配置)
COMMON_ARGS="--epochs 50 --batch_size 2056 --no_wandb --dropout 0.1 --train_stride 5 --t0 10 --t_mult 2 --backbone transformer --num_layers 4"

# 定义实验函数
run_experiment() {
    local gpu=$1
    local port=$2
    local window_size=$3
    local w_swivel=$4
    local w_elbow=$5
    local w_smooth=$6
    local save_dir=$7
    local log_name=$8

    echo "🚀 启动实验: ${log_name} (GPU ${gpu}, Port ${port})"

    CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=1 --master_port ${port} \
        training/trainer.py \
        ${COMMON_ARGS} \
        --window_size ${window_size} \
        --w_swivel ${w_swivel} \
        --w_elbow ${w_elbow} \
        --w_smooth ${w_smooth} \
        --save_dir ${save_dir} \
        2>&1 | tee "${LOG_DIR}/ablation_${log_name}.log"
}

# ============================================
# GPU 3: 窗口大小消融 (W1, W15, W30)
# 使用端口 29500
# ============================================
(
    run_experiment 3 29500 1  1.0 0.0 0.0 "${CHECKPOINT_BASE}/window/W1"  "W1"
    run_experiment 3 29500 15 1.0 0.0 0.0 "${CHECKPOINT_BASE}/window/W15" "W15"
    run_experiment 3 29500 30 1.0 0.0 0.0 "${CHECKPOINT_BASE}/window/W30" "W30"
    echo "✅ GPU 3 所有实验完成"
) &

# ============================================
# GPU 5: 损失函数消融 (swivel_only, elbow_only, full_loss)
# 使用端口 29501
# ============================================
(
    run_experiment 5 29501 15 1.0 0.0 0.0 "${CHECKPOINT_BASE}/loss/swivel_only" "swivel_only"
    run_experiment 5 29501 15 0.0 1.0 0.0 "${CHECKPOINT_BASE}/loss/elbow_only"  "elbow_only"
    run_experiment 5 29501 15 1.0 1.0 0.1 "${CHECKPOINT_BASE}/loss/full_loss"    "full_loss"
    echo "✅ GPU 5 所有实验完成"
) &

# ============================================
# GPU 6: 网络层数消融 (L2, L4, L8)
# 使用端口 29502
# ============================================
(
    run_experiment 6 29502 15 1.0 0.0 0.0 "${CHECKPOINT_BASE}/layers/L2" "L2"
    run_experiment 6 29502 15 1.0 0.0 0.0 "${CHECKPOINT_BASE}/layers/L4" "L4"
    run_experiment 6 29502 15 1.0 0.0 0.0 "${CHECKPOINT_BASE}/layers/L8" "L8"
    echo "✅ GPU 6 所有实验完成"
) &

# ============================================
# GPU 7: Backbone 消融 (transformer, mamba, lstm)
# 使用端口 29503
# ============================================
(
    CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port 29503 \
        training/trainer.py ${COMMON_ARGS} --window_size 15 \
        --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
        --backbone transformer \
        --save_dir "${CHECKPOINT_BASE}/backbone/transformer" \
        2>&1 | tee "${LOG_DIR}/ablation_transformer.log"

    CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port 29503 \
        training/trainer.py ${COMMON_ARGS} --window_size 15 \
        --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
        --backbone mamba \
        --save_dir "${CHECKPOINT_BASE}/backbone/mamba" \
        2>&1 | tee "${LOG_DIR}/ablation_mamba.log"

    CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port 29503 \
        training/trainer.py ${COMMON_ARGS} --window_size 15 \
        --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
        --backbone lstm \
        --save_dir "${CHECKPOINT_BASE}/backbone/lstm" \
        2>&1 | tee "${LOG_DIR}/ablation_lstm.log"

    echo "✅ GPU 7 所有实验完成"
) &

echo ""
echo "=========================================="
echo "🚀 所有 12 个消融实验已启动!"
echo "=========================================="
echo ""
echo "📁 Checkpoints 保存位置:"
echo "   ${CHECKPOINT_BASE}"
echo ""
echo "📊 查看进度:"
echo "   watch -n 5 'nvidia-smi'"
echo "   tail -f ${LOG_DIR}/ablation_*.log"
echo ""
echo "⏹️  终止所有实验:"
echo "   pkill -f torchrun"
echo ""

# 等待所有后台任务完成
wait

echo ""
echo "=========================================="
echo "🎉 所有消融实验完成!"
echo "=========================================="
