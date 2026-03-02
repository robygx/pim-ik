#!/bin/bash
# PiM-IK 消融实验 - 抗过拟合版本
# 
# 新增抗过拟合策略:
#   - train_stride 5: 降低相邻样本重叠
#   - add_noise: 物理噪声增强
#   - dropout 0.3: 多级 Dropout
#
# 可用 GPU: 0, 3, 5, 6, 7 (GPUs 1,2,4 被占用)
#
# 默认配置 (baseline): mamba + W15 + L4 + full_loss

set -e

# 清理旧日志
rm -f /home/ygx/ygx_hl_ik_v2/logs/ablation_*.log 2>/dev/null || true
mkdir -p /home/ygx/ygx_hl_ik_v2/logs

# 公共参数 - 新配置 (抗过拟合)
COMMON_ARGS="--epochs 50 --batch_size 2048 --no_wandb --train_stride 5 --add_noise --dropout 0.3 --weight_decay 1e-3 --t0 10 --t_mult 2 --num_layers 4"

echo "=========================================="
echo "🚀 启动抗过拟合消融实验"
echo "=========================================="
echo "配置: stride=5, noise=True, dropout=0.3"
echo ""

# 定义实验函数
run_experiment() {
    local gpu=$1
    local port=$2
    local window_size=$3
    local num_layers=$4
    local backbone=$5
    local w_swivel=$6
    local w_elbow=$7
    local w_smooth=$8
    local save_dir=$9
    local log_name=${10}

    echo "🚀 启动实验: ${log_name} (GPU ${gpu}, Port ${port})"

    CUDA_VISIBLE_DEVICES=${gpu} torchrun --nproc_per_node=1 --master_port ${port} \
        training/trainer.py \
        ${COMMON_ARGS} \
        --window_size ${window_size} \
        --num_layers ${num_layers} \
        --backbone ${backbone} \
        --w_swivel ${w_swivel} \
        --w_elbow ${w_elbow} \
        --w_smooth ${w_smooth} \
        --save_dir ${save_dir} \
        2>&1 | tee "/home/ygx/ygx_hl_ik_v2/logs/ablation_${log_name}.log"
}

# ============================================
# GPU 0: 损失函数消融 (baseline: full_loss)
# ============================================
(
    echo "=== GPU 0: 损失函数消融 ==="
    
    # swivel_only
    run_experiment 0 29500 15 4 mamba 1.0 0.0 0.0 \
        "checkpoints/ablation_anti/loss/swivel_only" \
        "loss_swivel"
    
    # elbow_only
    run_experiment 0 29500 15 4 mamba 0.0 1.0 0.0 \
        "checkpoints/ablation_anti/loss/elbow_only" \
        "loss_elbow"
    
    # full_loss (baseline)
    run_experiment 0 29500 15 4 mamba 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/loss/full_loss" \
        "loss_full"
    
    echo "✅ GPU 0 完成"
) &

# ============================================
# GPU 3: 窗口大小消融 (baseline: W15)
# ============================================
(
    echo "=== GPU 3: 窗口大小消融 ==="
    
    # W1 (无时序记忆基线)
    run_experiment 3 29503 1 4 mamba 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/window/W1" \
        "window_W1"
    
    # W15 (默认)
    run_experiment 3 29503 15 4 mamba 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/window/W15" \
        "window_W15"
    
    # W30 (长时序)
    run_experiment 3 29503 30 4 mamba 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/window/W30" \
        "window_W30"
    
    echo "✅ GPU 3 完成"
) &

# ============================================
# GPU 5: 层数消融 (baseline: L4)
# ============================================
(
    echo "=== GPU 5: 层数消融 ==="
    
    # L2 (浅层)
    run_experiment 5 29505 15 2 mamba 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/layers/L2" \
        "layers_L2"
    
    # L4 (默认)
    run_experiment 5 29505 15 4 mamba 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/layers/L4" \
        "layers_L4"
    
    # L8 (深层)
    run_experiment 5 29505 15 8 mamba 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/layers/L8" \
        "layers_L8"
    
    echo "✅ GPU 5 完成"
) &

# ============================================
# GPU 6: Backbone 消融 (baseline: mamba)
# ============================================
(
    echo "=== GPU 6: Backbone 消融 ==="
    
    # mamba (默认)
    run_experiment 6 29506 15 4 mamba 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/backbone/mamba" \
        "backbone_mamba"
    
    # transformer
    run_experiment 6 29506 15 4 transformer 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/backbone/transformer" \
        "backbone_transformer"
    
    # lstm
    run_experiment 6 29506 15 4 lstm 1.0 1.0 0.1 \
        "checkpoints/ablation_anti/backbone/lstm" \
        "backbone_lstm"
    
    echo "✅ GPU 6 完成"
) &

# ============================================
# GPU 7: 预留/备用 (可用于额外实验)
# ============================================
(
    echo "=== GPU 7: 预留 ==="
    echo "可用于额外实验或作为备用"
) &

echo ""
echo "=========================================="
echo "📊 查看进度:"
echo "   watch -n 5 'nvidia-smi'"
echo "   tail -f /home/ygx/ygx_hl_ik_v2/logs/ablation_*.log"
echo ""
echo "⏹️  终止所有实验:"
echo "   pkill -f 'ablation_anti'"
echo ""
echo "=========================================="

# 等待所有后台任务完成
wait

echo ""
echo "=========================================="
echo "🎉 所有抗过拟合消融实验完成!"
echo "=========================================="
