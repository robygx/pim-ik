#!/bin/bash
# PiM-IK 消融实验批量评测脚本
# 使用多 GPU 并行评测所有消融实验

set -e  # 遇到错误立即退出

CHECKPOINT_DIR="./checkpoints/ablation_anti"
OUTPUT_DIR="./evaluation/ablation_anti"
DATA_PATH="/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data_with_swivel.npz"

# 可用 GPU 列表
GPU_LIST=(0 3 5 6 7)

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  PiM-IK 消融实验批量评测"
echo "=========================================="
echo "Checkpoint 目录: $CHECKPOINT_DIR"
echo "数据集: GRAB"
echo "采样方式: 随机采样 50,000 帧"
echo "IK 类型: Real IK (HierarchicalIKSolver)"
echo "输出目录: $OUTPUT_DIR"
echo "可用 GPU: ${GPU_LIST[@]}"
echo "=========================================="
echo

# 定义实验和对应的 GPU
declare -A experiments
experiments["loss_ablation"]="${GPU_LIST[0]}"
experiments["window_size_ablation"]="${GPU_LIST[1]}"
experiments["backbone_ablation"]="${GPU_LIST[2]}"
experiments["layers_ablation"]="${GPU_LIST[3]}"

# 存储进程 PID
pids=()

# 启动所有评测任务
for exp in "${!experiments[@]}"; do
    gpu="${experiments[$exp]}"
    echo "[$exp] 启动评测 (GPU $gpu)..."

    CUDA_VISIBLE_DEVICES=$gpu python ablation/comprehensive_eval.py \
        --experiment $exp \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --data_path "$DATA_PATH" \
        --output "$OUTPUT_DIR/${exp}_results.json" \
        --num_frames 50000 \
        --random_sample \
        --use-real-ik &

    pids+=($!)
done

echo
echo "所有评测任务已启动，等待完成..."
echo

# 等待所有评测完成
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    wait $pid
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ 任务完成 (PID: $pid)"
    else
        echo "✗ 任务失败 (PID: $pid, 退出码: $exit_code)"
    fi
done

echo
echo "=========================================="
echo "所有评测完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "=========================================="
echo

# 列出生成的结果文件
echo "生成的结果文件:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "未找到结果文件"

echo
echo "运行以下命令生成汇总报告:"
echo "  python evaluation/generate_report.py"
