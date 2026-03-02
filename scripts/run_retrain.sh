#!/bin/bash
cd /home/ygx/ygx_hl_ik_v2

COMMON_ARGS="--epochs 50 --batch_size 2056 --no_wandb --dropout 0.3 --weight_decay 1e-3 --t0 10 --t_mult 2 --backbone transformer --num_layers 4"

echo "=== GPU 0: swivel_only ==="
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29520 \
    training/trainer.py ${COMMON_ARGS} \
    --window_size 15 --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
    --save_dir "checkpoints/loss_ablation/do0.3_wd1e-3/swivel_only" \
    2>&1 | tee "logs/ablation_swivel_only_new.log"

echo "=== GPU 0: L2 ==="
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29522 \
    training/trainer.py ${COMMON_ARGS} \
    --window_size 15 --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
    --num_layers 2 \
    --save_dir "checkpoints/layers_ablation/do0.3_wd1e-3/L2" \
    2>&1 | tee "logs/ablation_L2_new.log"

echo "=== GPU 5: transformer ==="
CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port 29521 \
    training/trainer.py ${COMMON_ARGS} \
    --window_size 15 --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 \
    --save_dir "checkpoints/backbone_ablation/do0.3_wd1e-3/transformer" \
    2>&1 | tee "logs/ablation_transformer_new.log"
