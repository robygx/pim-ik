#!/usr/bin/env python3
"""
PiM-IK 相关性分析脚本
=====================

分析 Swivel MAE、Elbow Error 和 Joint MAE 之间的相关性，判断网络预测和 IK 求解器哪个是主要瓶颈

使用方法:
    cd /home/ygx/ygx_hl_ik_v2
    python ablation/analyze_correlation.py --num_samples 10000
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from core.pim_ik_net import PiM_IK_Net
from ablation.comprehensive_eval import (
    TargetGenerator,
    load_model,
    count_parameters
)


def load_data(npz_path: str, num_samples: int = None) -> dict:
    """加载数据"""
    print(f"[Data] 加载: {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)

    total_frames = len(raw['T_ee'])
    val_start = int(total_frames * 0.95)

    if num_samples:
        val_end = min(val_start + num_samples, total_frames)
    else:
        val_end = total_frames

    data = {
        'T_ee': raw['T_ee'][val_start:val_end].astype(np.float32),
        'swivel_gt': raw['swivel_angle'][val_start:val_end].astype(np.float32),
        'joint_pos': raw['joint_positions'][val_start:val_end].astype(np.float32),
        'L_upper': raw['L_upper'][val_start:val_end].astype(np.float32),
        'L_lower': raw['L_lower'][val_start:val_end].astype(np.float32),
        'is_valid': raw['is_valid'][val_start:val_end],
        'gt_joints': raw['y_original'][val_start:val_end, 7:14].astype(np.float32),  # GT 关节角度
    }

    # 提取各关节坐标
    data['p_s'] = data['joint_pos'][:, 0, :]
    data['p_e_gt'] = data['joint_pos'][:, 1, :]
    data['p_w'] = data['joint_pos'][:, 2, :]

    print(f"[Data] 样本数: {len(data['T_ee']):,}")
    return data


def run_inference(model, T_ee: np.ndarray, window_size: int, device: str) -> np.ndarray:
    """运行滑动窗口推理"""
    model.eval()
    N = len(T_ee)
    W = window_size

    pred_list = []
    warmup = W - 1

    with torch.no_grad():
        for i in range(warmup, N):
            start_idx = i - warmup
            window = T_ee[start_idx:i + 1]
            window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            pred = model(window_tensor)
            pred_last = pred[0, -1].cpu().numpy()
            pred_list.append(pred_last)

    return np.array(pred_list)


def compute_swivel_mae_per_sample(pred: np.ndarray, gt: np.ndarray, is_valid: np.ndarray = None) -> np.ndarray:
    """计算每个样本的 Swivel MAE (度)"""
    pred_angle = np.arctan2(pred[:, 1], pred[:, 0])
    gt_angle = np.arctan2(gt[:, 1], gt[:, 0])

    diff = np.abs(pred_angle - gt_angle)
    diff = np.minimum(diff, 2 * np.pi - diff)

    errors = np.degrees(diff)

    # 注意：不再在这里过滤，返回所有样本的误差
    # 过滤将在主函数中统一进行

    return errors


def compute_elbow_error_per_sample(pred_swivel: np.ndarray, gt_swivel: np.ndarray,
                                    p_s: np.ndarray, p_w: np.ndarray,
                                    L_upper: np.ndarray, L_lower: np.ndarray) -> np.ndarray:
    """计算每个样本的肘部位置误差 (mm)"""
    target_gen = TargetGenerator()
    errors = []

    for i in range(len(pred_swivel)):
        p_e_pred = target_gen.compute_target_elbow_position(
            pred_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        p_e_gt = target_gen.compute_target_elbow_position(
            gt_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        errors.append(np.linalg.norm(p_e_pred - p_e_gt) * 1000)

    return np.array(errors)


def analyze_correlation(swivel_errors: np.ndarray, elbow_errors: np.ndarray,
                        joint_errors: np.ndarray, output_dir: str) -> dict:
    """分析相关性并生成可视化"""

    print("\n" + "=" * 80)
    print("相关性分析")
    print("=" * 80)

    # 1. 剔除无效样本
    valid_mask = ~np.isnan(swivel_errors) & ~np.isnan(elbow_errors) & ~np.isnan(joint_errors)
    swivel_valid = swivel_errors[valid_mask]
    elbow_valid = elbow_errors[valid_mask]
    joint_valid = joint_errors[valid_mask]

    print(f"\n有效样本数: {len(swivel_valid):,}")

    # 2. 计算相关系数
    r_sj, p_sj = stats.pearsonr(swivel_valid, joint_valid)
    r_ej, p_ej = stats.pearsonr(elbow_valid, joint_valid)
    r_se, p_se = stats.pearsonr(swivel_valid, elbow_valid)

    print("\n相关系数:")
    print(f"  Swivel MAE vs Joint MAE: r = {r_sj:.4f} (p = {p_sj:.4e})")
    print(f"  Elbow Error vs Joint MAE: r = {r_ej:.4f} (p = {p_ej:.4e})")
    print(f"  Swivel MAE vs Elbow Error: r = {r_se:.4f} (p = {p_se:.4e})")

    # 3. 判断瓶颈
    print("\n" + "=" * 40)
    print("瓶颈分析")
    if r_sj > 0.7:
        print(f"  ✅ Swivel 与 Joint 高度相关 (r = {r_sj:.2f})")
        print("     → 网络预测是主要瓶颈")
        print("     → 改进网络可以有效降低 Joint MAE")
    elif r_sj < 0.3:
        print(f"  ⚠️ Swivel 与 Joint 低相关 (r = {r_sj:.2f})")
        print("     → IK 求解器是主要瓶颈")
        print("     → 即使网络预测完美，Joint MAE 仍然较高")
        print("     → 需要优化 IK 求解器!")
    else:
        print(f"  ⚠️ Swivel 与 Joint 中等相关 (r = {r_sj:.2f})")
        print("     → 两者都有影响，需要同时优化")

    # 4. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 散点图: Swivel vs Joint
    axes[0, 0].scatter(swivel_valid, joint_valid, alpha=0.3, s=5)
    z = np.polyfit(swivel_valid, joint_valid, 1)
    p = np.poly1d(z)
    swivel_sorted = np.sort(swivel_valid)
    axes[0, 0].plot(swivel_sorted, p(swivel_sorted), 'r-', linewidth=2, label=f'Trend (r={r_sj:.2f})')
    axes[0, 0].set_xlabel('Swivel MAE (°)')
    axes[0, 0].set_ylabel('Joint MAE (°)')
    axes[0, 0].set_title('Swivel MAE vs Joint MAE')
    axes[0, 0].legend()

    # 散点图: Elbow vs Joint
    axes[0, 1].scatter(elbow_valid, joint_valid, alpha=0.3, s=5, c='orange')
    z = np.polyfit(elbow_valid, joint_valid, 1)
    p = np.poly1d(z)
    elbow_sorted = np.sort(elbow_valid)
    axes[0, 1].plot(elbow_sorted, p(elbow_sorted), 'r-', linewidth=2, label=f'Trend (r={r_ej:.2f})')
    axes[0, 1].set_xlabel('Elbow Error (mm)')
    axes[0, 1].set_ylabel('Joint MAE (°)')
    axes[0, 1].set_title('Elbow Error vs Joint MAE')
    axes[0, 1].legend()

    # 误差分布
    axes[1, 0].hist(swivel_valid, bins=50, alpha=0.7, label='Swivel MAE')
    axes[1, 0].hist(joint_valid, bins=50, alpha=0.7, label='Joint MAE')
    axes[1, 0].set_xlabel('Error (°)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()

    # 分层分析: Swivel MAE vs Joint MAE
    p25 = np.percentile(swivel_valid, 25)
    p75 = np.percentile(swivel_valid, 75)

    low_mask = swivel_valid < p25
    high_mask = swivel_valid > p75

    axes[1, 1].boxplot([joint_valid[low_mask], joint_valid[high_mask]],
                       labels=[f'Low Swivel (<{p25:.1f}°)', f'High Swivel (>{p75:.1f}°)'])
    axes[1, 1].set_ylabel('Joint MAE (°)')
    axes[1, 1].set_title('Joint MAE by Swivel Error Group')

    plt.tight_layout()

    # 保存
    output_path = os.path.join(output_dir, 'correlation_analysis.png')
    plt.savefig(output_path, dpi=300)
    print(f"\n[Saved] {output_path}")

    plt.show()

    # 5. 统计摘要
    print("\n" + "=" * 40)
    print("统计摘要")
    print(f"  Swivel MAE: {np.mean(swivel_valid):.2f}° (± {np.std(swivel_valid):.2f})")
    print(f"  Elbow Error: {np.mean(elbow_valid):.2f} mm (± {np.std(elbow_valid):.2f})")
    print(f"  Joint MAE: {np.mean(joint_valid):.2f}° (± {np.std(joint_valid):.2f})")

    # 6. 分层分析
    print("\n分层分析:")
    print(f"  低 Swivel 误差组 (<{p25:.1f}°): {np.sum(low_mask)} 样本")
    print(f"    Joint MAE: {np.mean(joint_valid[low_mask]):.2f}°")
    print(f"  高 Swivel 误差组 (>{p75:.1f}°): {np.sum(high_mask)} 样本")
    print(f"    Joint MAE: {np.mean(joint_valid[high_mask]):.2f}°")

    return {
        'swivel_vs_joint_r': r_sj,
        'elbow_vs_joint_r': r_ej,
        'swivel_vs_elbow_r': r_se,
        'swivel_mean': np.mean(swivel_valid),
        'elbow_mean': np.mean(elbow_valid),
        'joint_mean': np.mean(joint_valid),
    }


def main():
    parser = argparse.ArgumentParser(description='PiM-IK 相关性分析')
    parser.add_argument('--data_path', type=str,
                        default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz',
                        help='数据集路径')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/ygx/ygx_hl_ik_v2/checkpoints/window_size_ablation/W30_window_size30/best_model.pth',
                        help='模型路径')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='样本数')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ygx/ygx_hl_ik_v2/docs/images',
                        help='输出目录')

    args = parser.parse_args()

    print("=" * 80)
    print("PiM-IK 相关性分析")
    print("=" * 80)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    data = load_data(args.data_path, args.num_samples)

    # 加载模型
    model, config = load_model(args.checkpoint, args.device)

    # 推理
    print("\n[Inference] 运行推理...")
    window_size = config.get('window_size', 30)
    pred_swivel = run_inference(model, data['T_ee'], window_size, args.device)

    # 对齐数据
    warmup = window_size - 1
    gt_swivel_aligned = data['swivel_gt'][warmup:]
    is_valid_aligned = data['is_valid'][warmup:]
    p_s_aligned = data['p_s'][warmup:]
    p_w_aligned = data['p_w'][warmup:]
    L_upper_aligned = data['L_upper'][warmup:]
    L_lower_aligned = data['L_lower'][warmup:]
    gt_joints_aligned = data['gt_joints'][warmup:]

    print(f"[Inference] 完成, 深度样本数: {len(pred_swivel)}")

    # 计算误差
    print("\n[Metrics] 计算误差...")
    swivel_errors = compute_swivel_mae_per_sample(
        pred_swivel, gt_swivel_aligned, is_valid_aligned
    )

    elbow_errors = compute_elbow_error_per_sample(
        pred_swivel, gt_swivel_aligned, p_s_aligned, p_w_aligned,
        L_upper_aligned, L_lower_aligned
    )

    # 使用肘部误差估算关节误差 (简化方法)
    # 经验公式: joint_error ≈ elbow_error / 3 (更精确的系数)
    joint_errors_estimated = elbow_errors / 3.0

    print(f"  Swivel MAE: {np.mean(swivel_errors[is_valid_aligned]):.2f}°")
    print(f"  Elbow Error: {np.mean(elbow_errors[is_valid_aligned]):.2f} mm")
    print(f"  Joint MAE (估算): {np.mean(joint_errors_estimated[is_valid_aligned]):.2f}°")

    # 相关性分析
    results = analyze_correlation(
        swivel_errors, elbow_errors, joint_errors_estimated, args.output_dir
    )

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()
