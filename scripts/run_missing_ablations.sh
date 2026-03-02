#!/bin/bash
# 缺失消融实验补充脚本 - 新配置 (dropout 0.3, weight_decay 1e-3, batch_size 2056)
# 
# 只需要重训旧配置(10:21启动)的实验:
# - swivel_only, L2, transformer
# 
# 以及未开始的实验:
# - lstm

set -e

mkdir -p /home/ygx/ygx_hl_ik_v2/logs

COMMON_ARGS="--epochs 50 --batch_size 2056 --no_wandb --dropout 0.3 --weight_decay 1e-3 --t0 10 --t_mult 2 --num_layers 4"

echo "=========================================="
echo "🚀 启动缺失的消融实验 (新配置)"
echo "=========================================="
echo ""

# ============================================
# GPU 0: 损失函数消融补充 (swivel_only) + L2
# ============================================
(
    echo "=== GPU 0: swivel_only → L2 ==="
    
    # swivel_only (旧配置只跑到 epoch 6)
    echo "[GPU 0] 启动 swivel_only..."
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29510 \
        training/trainer.py ${COMMON_ARGS} \
        --window_size 15 --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
        --backbone transformer \
        --save_dir "checkpoints/loss_ablation/do0.3_wd1e-3/swivel_only" \
        2>&1 | tee "/home/ygx/ygx_hl_ik_v2/logs/ablation_swivel_only_new.log"
    
    # L2 (旧配置只跑到 epoch 10)
    echo "[GPU 0] 启动 L2..."
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29510 \
        training/trainer.py ${COMMON_ARGS} \
        --window_size 15 --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
        --backbone transformer --num_layers 2 \
        --save_dir "checkpoints/layers_ablation/do0.3_wd1e-3/L2" \
        2>&1 | tee "/home/ygx/ygx_hl_ik_v2/logs/ablation_L2_new.log"
    
    echo "✅ GPU 0 完成"
) &

# ============================================
# GPU 1: Backbone 消融补充 (transformer, lstm)
# ============================================
(
    echo "=== GPU 1: transformer → lstm ==="
    
    # transformer (旧配置只跑到 epoch 6)
    echo "[GPU 1] 启动 transformer..."
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29511 \
        training/trainer.py ${COMMON_ARGS} \
        --window_size 15 --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
        --backbone transformer \
        --save_dir "checkpoints/backbone_ablation/do0.3_wd1e-3/transformer" \
        2>&1 | tee "/home/ygx/ygx_hl_ik_v2/logs/ablation_transformer_new.log"
    
    # lstm (未开始)
    echo "[GPU 1] 启动 lstm..."
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29511 \
        training/trainer.py ${COMMON_ARGS} \
        --window_size 15 --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
        --backbone lstm \
        --save_dir "checkpoints/backbone_ablation/do0.3_wd1e-3/lstm" \
        2>&1 | tee "/home/ygx/ygx_hl_ik_v2/logs/ablation_lstm_new.log"
    
    echo "✅ GPU 1 完成"
) &

echo ""
echo "=========================================="
echo "📊 查看进度:"
echo "   watch -n 5 'nvidia-smi'"
echo "   tail -f /home/ygx/ygx_hl_ik_v2/logs/ablation_*_new.log"
echo ""
echo "⏹️  终止所有补充实验:"
echo "   pkill -f '_new.log'"
echo ""
echo "=========================================="

wait

echo ""
echo "=========================================="
echo "🎉 所有缺失消融实验完成!"
echo "=========================================="
