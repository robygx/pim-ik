#!/usr/bin/env python3
"""
PiM-IK 时序窗口长度消融实验分析脚本
======================================

本脚本用于评估不同窗口大小对模型预测平滑度的影响，验证 Mamba 时序建模的有效性。

功能:
1. 加载三个不同窗口大小的模型 (W=30, W=15, W=1)
2. 在 GRAB 数据集的验证集上进行推理
3. 计算角度误差 (MAE) 和平滑度指标 (Jerk)
4. 生成学术水准的可视化图表

使用方法:
    python ablation_window_size.py --data_path /path/to/GRAB_data.npz

作者: PiM-IK Project
日期: 2025-02-27
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持 (可选)
rcParams['font.family'] = 'DejaVu Sans'
rcParams['figure.dpi'] = 100
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 12

# 导入自定义模块
from pim_ik_net import PiM_IK_Net


# ============================================================================
# 模型加载
# ============================================================================

def load_model_with_window_size(
    checkpoint_path: str,
    window_size: int,
    device: str = 'cuda:0'
) -> nn.Module:
    """
    加载指定窗口大小的模型

    Args:
        checkpoint_path: 模型检查点路径
        window_size: 窗口大小 (用于验证)
        device: 计算设备

    Returns:
        model: 加载好权重的模型
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    print(f"[Model] 加载模型: {checkpoint_path}")

    # 创建模型
    model = PiM_IK_Net(d_model=256, num_layers=4).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 处理 DDP 包装的权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 移除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

        # 验证窗口大小
        if 'window_size' in checkpoint:
            assert checkpoint['window_size'] == window_size, \
                f"窗口大小不匹配: checkpoint={checkpoint['window_size']}, 期望={window_size}"
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"[Model] 模型加载完成 (W={window_size})")

    return model


# ============================================================================
# 数据处理
# ============================================================================

class AblationDataset:
    """
    消融实验数据集
    从验证集抽取一段连续轨迹用于评估
    """

    def __init__(
        self,
        npz_path: str,
        num_frames: int = 100,
        train_split: float = 0.95
    ):
        print(f"[Data] 加载数据: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        # 加载全部数据
        self.T_ee = data['T_ee'].astype(np.float32)           # (N, 4, 4)
        self.swivel_angle = data['swivel_angle'].astype(np.float32)  # (N, 2)
        self.is_valid = data['is_valid'].astype(np.float32)   # (N,)

        self.total_frames = len(self.T_ee)
        print(f"[Data] 总帧数: {self.total_frames:,}")

        # 计算验证集起始位置
        val_start = int(self.total_frames * train_split)

        # 从验证集中抽取 num_frames 帧 (取最后 num_frames 帧，确保动作变化丰富)
        val_frames = self.total_frames - val_start
        self.num_frames = min(num_frames, val_frames)

        self.start_idx = self.total_frames - self.num_frames
        self.end_idx = self.total_frames

        print(f"[Data] 评估帧范围: [{self.start_idx}, {self.end_idx})")

        # 提取评估数据
        self.T_ee_eval = self.T_ee[self.start_idx:self.end_idx]  # (num_frames, 4, 4)
        self.swivel_eval = self.swivel_angle[self.start_idx:self.end_idx]  # (num_frames, 2)
        self.valid_eval = self.is_valid[self.start_idx:self.end_idx]  # (num_frames,)

    def get_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取评估轨迹

        Returns:
            T_ee_traj: (T, 4, 4) 末端位姿轨迹
            swivel_traj: (T, 2) 真实臂角轨迹
        """
        return self.T_ee_eval, self.swivel_eval


# ============================================================================
# 推理函数
# ============================================================================

def infer_with_window(
    model: nn.Module,
    T_ee_traj: np.ndarray,
    window_size: int,
    device: str = 'cuda:0'
) -> tuple:
    """
    根据窗口大小进行单帧推理

    对于时序模型 (W>1)，使用滑动窗口；
    对于单帧模型 (W=1)，直接推理当前帧。

    【修复】跳过预热期，只返回窗口填满后的有效预测

    Args:
        model: 神经网络模型
        T_ee_traj: (T, 4, 4) 末端位姿轨迹
        window_size: 窗口大小
        device: 计算设备

    Returns:
        predictions: (T - warmup, 2) 预测的臂角 [cos, sin]（跳过预热期）
        warmup_frames: 预热帧数 = window_size - 1
    """
    model.eval()
    T = len(T_ee_traj)
    predictions = []
    warmup_frames = window_size - 1

    with torch.no_grad():
        for t in range(T):
            if window_size == 1:
                # 单帧推理：直接使用当前帧
                window = T_ee_traj[t:t+1]  # (1, 4, 4)
            else:
                # 滑动窗口推理
                if t < warmup_frames:
                    # 【修复】预热期：用当前帧填满整个窗口（更平滑）
                    # 这样模型至少能看到一致的输入，而不是突然变化
                    window = np.repeat(T_ee_traj[t:t+1], window_size, axis=0)
                else:
                    window = T_ee_traj[t-warmup_frames:t+1]

            # 转换为 tensor 并添加 batch 维度
            window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, W, 4, 4)

            # 前向传播
            pred = model(window_tensor)  # (1, W, 2)

            # 取最后一帧的预测
            pred_last = pred[0, -1].cpu().numpy()  # (2,)

            # 【修复】跳过预热期，只记录稳定期预测
            if t >= warmup_frames:
                predictions.append(pred_last)

    return np.array(predictions), warmup_frames  # (T - warmup, 2), warmup


# ============================================================================
# 指标计算
# ============================================================================

def compute_angle_mae(
    pred_cos_sin: np.ndarray,
    gt_cos_sin: np.ndarray,
    is_valid: Optional[np.ndarray] = None
) -> float:
    """
    计算角度 MAE（考虑角度周期性）

    Args:
        pred_cos_sin: (N, 2) 预测臂角 [cos, sin]
        gt_cos_sin: (N, 2) 真实臂角 [cos, sin]
        is_valid: (N,) 有效性掩码

    Returns:
        mae_deg: 平均角度误差（度）
    """
    # 转换为角度
    pred_angle = np.arctan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0])
    gt_angle = np.arctan2(gt_cos_sin[:, 1], gt_cos_sin[:, 0])

    # 计算角度差（考虑周期性）
    diff = np.abs(pred_angle - gt_angle)
    diff = np.minimum(diff, 2 * np.pi - diff)

    # 应用有效性掩码
    if is_valid is not None:
        diff = diff[is_valid > 0.5]

    # 转换为度
    mae_deg = np.degrees(diff).mean()
    return mae_deg


def compute_jerk(
    pred_cos_sin: np.ndarray,
    is_valid: Optional[np.ndarray] = None
) -> float:
    """
    计算平滑度指标 (Jerk)

    Jerk = 二阶差分的均方值
    值越小表示轨迹越平滑

    Args:
        pred_cos_sin: (N, 2) 预测臂角 [cos, sin]
        is_valid: (N,) 有效性掩码

    Returns:
        jerk: 二阶差分均方值
    """
    # 转换为角度（度）
    phi = np.degrees(np.arctan2(pred_cos_sin[:, 1], pred_cos_sin[:, 0]))

    # 应用有效性掩码
    if is_valid is not None:
        valid_indices = is_valid > 0.5
        phi = phi[valid_indices]

    # 计算二阶差分
    if len(phi) < 3:
        return 0.0

    diff2 = phi[2:] - 2 * phi[1:-1] + phi[:-2]

    # 均方值
    jerk = np.mean(diff2 ** 2)
    return jerk


def compute_elbow_mae_from_swivel(
    pred_swivel: np.ndarray,
    gt_swivel: np.ndarray,
    p_s: np.ndarray,
    p_w: np.ndarray,
    L_upper: np.ndarray,
    L_lower: np.ndarray,
    is_valid: Optional[np.ndarray] = None
) -> float:
    """
    从臂角计算肘部位置 MAE（辅助指标）

    Args:
        pred_swivel: (N, 2) 预测臂角 [cos, sin]
        gt_swivel: (N, 2) 真实臂角 [cos, sin]
        p_s: (N, 3) 肩部位置
        p_w: (N, 3) 腕部位置
        L_upper: (N,) 上臂长度
        L_lower: (N,) 前臂长度
        is_valid: (N,) 有效性掩码

    Returns:
        elbow_mae_mm: 肘部位置 MAE (毫米)
    """
    from inference import TargetGenerator

    target_gen = TargetGenerator()

    # 计算预测肘部位置
    p_e_pred = target_gen.compute_target_elbow_position(
        swivel_angle=pred_swivel,
        p_s=p_s,
        p_w=p_w,
        L_upper=L_upper,
        L_lower=L_lower
    )

    # 计算真实肘部位置
    p_e_gt = target_gen.compute_target_elbow_position(
        swivel_angle=gt_swivel,
        p_s=p_s,
        p_w=p_w,
        L_upper=L_upper,
        L_lower=L_lower
    )

    # 计算误差
    errors = np.linalg.norm(p_e_pred - p_e_gt, axis=-1) * 1000  # 转换为毫米

    if is_valid is not None:
        errors = errors[is_valid > 0.5]

    return errors.mean()


# ============================================================================
# 可视化
# ============================================================================

def plot_ablation_results(
    gt_phi: np.ndarray,
    pred_w30: np.ndarray,
    pred_w15: np.ndarray,
    pred_w1: np.ndarray,
    jerk_w30: float,
    jerk_w15: float,
    jerk_w1: float,
    mae_w30: float,
    mae_w15: float,
    mae_w1: float,
    output_path: str = 'ablation_window_size.png'
):
    """
    绘制消融实验结果

    Args:
        gt_phi: (T,) 真实臂角（度）
        pred_w30: (T,) W=30 预测臂角（度）
        pred_w15: (T,) W=15 预测臂角（度）
        pred_w1: (T,) W=1 预测臂角（度）
        jerk_w30, jerk_w15, jerk_w1: 各模型的 Jerk 值
        mae_w30, mae_w15, mae_w1: 各模型的 MAE 值
        output_path: 输出图片路径
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    frames = np.arange(len(gt_phi))

    # 绘制曲线
    ax.plot(frames, gt_phi, 'k--', label='Ground Truth', linewidth=2.5, alpha=0.7, zorder=4)
    ax.plot(frames, pred_w30, 'b-', label=f'W=30 (MAE: {mae_w30:.2f}°, Jerk: {jerk_w30:.2f})',
            linewidth=2.5, alpha=0.9, zorder=3)
    ax.plot(frames, pred_w15, 'g-', label=f'W=15 (MAE: {mae_w15:.2f}°, Jerk: {jerk_w15:.2f})',
            linewidth=2.5, alpha=0.9, zorder=2)
    ax.plot(frames, pred_w1, 'r-', label=f'W=1 (MAE: {mae_w1:.2f}°, Jerk: {jerk_w1:.2f})',
            linewidth=2, alpha=0.85, zorder=1)

    # 设置标签和标题
    ax.set_xlabel('Time Step (Frames)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Swivel Angle $\\phi$ (Degrees)', fontsize=16, fontweight='bold')
    ax.set_title(
        'Ablation Study: Impact of Temporal Window Size on Motion Smoothness\n' +
        'Lower Jerk $\\Rightarrow$ Smoother Motion (Better Temporal Modeling)',
        fontsize=18, fontweight='bold', pad=15
    )

    # 图例
    legend = ax.legend(fontsize=13, loc='best', framealpha=0.95)
    legend.set_title('Model Configuration', prop={'size': 14, 'weight': 'bold'})

    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 添加统计信息文本框
    stats_text = (
        f"Statistics Summary:\n"
        f"W=30:  MAE={mae_w30:.2f}°, Jerk={jerk_w30:.2f}\n"
        f"W=15:  MAE={mae_w15:.2f}°, Jerk={jerk_w15:.2f}\n"
        f"W=1:   MAE={mae_w1:.2f}°, Jerk={jerk_w1:.2f}\n\n"
        f"Jerk Reduction (W=30 vs W=1): {(1 - jerk_w30/jerk_w1)*100:.1f}%"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[Vis] 图表已保存: {output_path}")


def print_comparison_table(results: Dict[str, Dict]):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("消融实验结果对比表")
    print("=" * 80)
    print(f"{'模型':<10} {'窗口大小':<10} {'MAE (°)':<12} {'Jerk':<12} {'Jerk 降低率':<15}")
    print("-" * 80)

    # 找到基准 (W=1)
    jerk_base = results['W=1']['jerk']

    for name in ['W=30', 'W=15', 'W=1']:
        r = results[name]
        jerk_reduction = (1 - r['jerk'] / jerk_base) * 100 if jerk_base > 0 else 0
        print(f"{name:<10} {name:<10} {r['mae']:<12.4f} {r['jerk']:<12.4f} {jerk_reduction:<15.1f}%")

    print("=" * 80)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PiM-IK 窗口长度消融实验')
    parser.add_argument('--data_path', type=str,
                        default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz',
                        help='训练数据集路径 (ACCAD_CMU)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='检查点目录 (应包含 W30, W15, W1 子目录)')
    parser.add_argument('--checkpoint_w30', type=str, default=None,
                        help='W=30 模型路径 (如未指定则使用默认路径)')
    parser.add_argument('--checkpoint_w15', type=str, default=None,
                        help='W=15 模型路径')
    parser.add_argument('--checkpoint_w1', type=str, default=None,
                        help='W=1 模型路径')
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='评估帧数 (默认1000，跳过预热期后约970帧有效)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    parser.add_argument('--output', type=str, default='ablation_window_size.png',
                        help='输出图片路径')

    args = parser.parse_args()

    print("=" * 80)
    print("PiM-IK 时序窗口长度消融实验")
    print("=" * 80)

    # 检查设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] 使用设备: {device}")

    # ================================================================
    # 加载数据
    # ================================================================
    dataset = AblationDataset(
        npz_path=args.data_path,
        num_frames=args.num_frames
    )
    T_ee_traj, swivel_traj = dataset.get_trajectory()
    valid_mask = dataset.valid_eval

    # 转换为角度（用于可视化）
    gt_phi = np.degrees(np.arctan2(swivel_traj[:, 1], swivel_traj[:, 0]))

    print(f"\n[Data] 评估轨迹长度: {len(T_ee_traj)} 帧")

    # ================================================================
    # 加载模型
    # ================================================================
    # 构建默认检查点路径
    if args.checkpoint_w30 is None:
        # 尝试查找 W=30 检查点
        default_path = Path(args.checkpoint_dir)
        w30_paths = list(default_path.glob("**/best_model_w30.pth"))
        if w30_paths:
            args.checkpoint_w30 = str(w30_paths[0])
        else:
            args.checkpoint_w30 = './checkpoints/20260227_111614/best_model.pth'

    if args.checkpoint_w15 is None:
        default_path = Path(args.checkpoint_dir)
        w15_paths = list(default_path.glob("**/best_model_w15.pth"))
        args.checkpoint_w15 = str(w15_paths[0]) if w15_paths else './checkpoints/W15_xxx/best_model_w15.pth'

    if args.checkpoint_w1 is None:
        default_path = Path(args.checkpoint_dir)
        w1_paths = list(default_path.glob("**/best_model_w1.pth"))
        args.checkpoint_w1 = str(w1_paths[0]) if w1_paths else './checkpoints/W1_xxx/best_model_w1.pth'

    print(f"\n[Model] 检查点路径:")
    print(f"  W=30: {args.checkpoint_w30}")
    print(f"  W=15: {args.checkpoint_w15}")
    print(f"  W=1:  {args.checkpoint_w1}")

    # 加载模型
    try:
        model_w30 = load_model_with_window_size(args.checkpoint_w30, 30, device)
        model_w15 = load_model_with_window_size(args.checkpoint_w15, 15, device)
        model_w1 = load_model_with_window_size(args.checkpoint_w1, 1, device)
    except FileNotFoundError as e:
        print(f"\n[Error] 模型文件未找到: {e}")
        print("\n请先运行训练脚本生成模型:")
        print("  torchrun --nproc_per_node=2 trainer.py --window_size 30 --epochs 50")
        print("  torchrun --nproc_per_node=2 trainer.py --window_size 15 --epochs 50")
        print("  torchrun --nproc_per_node=2 trainer.py --window_size 1 --epochs 50")
        sys.exit(1)

    # ================================================================
    # 推理
    # ================================================================
    print("\n[Infer] 开始推理...")

    pred_w30, warmup_w30 = infer_with_window(model_w30, T_ee_traj, 30, device)
    print(f"  W=30: 完成 (预热期: {warmup_w30} 帧)")

    pred_w15, warmup_w15 = infer_with_window(model_w15, T_ee_traj, 15, device)
    print(f"  W=15: 完成 (预热期: {warmup_w15} 帧)")

    pred_w1, warmup_w1 = infer_with_window(model_w1, T_ee_traj, 1, device)
    print(f"  W=1:  完成 (预热期: {warmup_w1} 帧)")

    # ================================================================
    # 【修复】对齐数据：使用最大预热期，确保所有模型评估相同的时间段
    # ================================================================
    max_warmup = max(warmup_w30, warmup_w15, warmup_w1)  # 29
    print(f"\n[Metrics] 使用预热期: {max_warmup} 帧 (取最大值)")

    # 截取对应的 GT 数据和有效掩码
    swivel_aligned = swivel_traj[max_warmup:]  # 跳过预热期
    valid_aligned = valid_mask[max_warmup:]

    print(f"  有效评估帧数: {len(swivel_aligned)}")

    # ================================================================
    # 计算指标
    # ================================================================
    print("\n[Metrics] 计算评估指标...")

    # MAE
    mae_w30 = compute_angle_mae(pred_w30, swivel_aligned, valid_aligned)
    mae_w15 = compute_angle_mae(pred_w15, swivel_aligned, valid_aligned)
    mae_w1 = compute_angle_mae(pred_w1, swivel_aligned, valid_aligned)

    # Jerk
    jerk_w30 = compute_jerk(pred_w30, valid_aligned)
    jerk_w15 = compute_jerk(pred_w15, valid_aligned)
    jerk_w1 = compute_jerk(pred_w1, valid_aligned)

    # 结果汇总
    results = {
        'W=30': {'mae': mae_w30, 'jerk': jerk_w30},
        'W=15': {'mae': mae_w15, 'jerk': jerk_w15},
        'W=1':  {'mae': mae_w1,  'jerk': jerk_w1}
    }

    # 打印结果
    print_comparison_table(results)

    # ================================================================
    # 可视化
    # ================================================================
    print("\n[Vis] 生成可视化图表...")

    # 【修复】使用对齐后的 GT 数据
    gt_phi_aligned = np.degrees(np.arctan2(swivel_aligned[:, 1], swivel_aligned[:, 0]))

    # 转换预测为角度
    pred_w30_phi = np.degrees(np.arctan2(pred_w30[:, 1], pred_w30[:, 0]))
    pred_w15_phi = np.degrees(np.arctan2(pred_w15[:, 1], pred_w15[:, 0]))
    pred_w1_phi = np.degrees(np.arctan2(pred_w1[:, 1], pred_w1[:, 0]))

    plot_ablation_results(
        gt_phi=gt_phi_aligned,
        pred_w30=pred_w30_phi,
        pred_w15=pred_w15_phi,
        pred_w1=pred_w1_phi,
        jerk_w30=jerk_w30,
        jerk_w15=jerk_w15,
        jerk_w1=jerk_w1,
        mae_w30=mae_w30,
        mae_w15=mae_w15,
        mae_w1=mae_w1,
        output_path=args.output
    )

    print("\n" + "=" * 80)
    print("消融实验完成!")
    print("=" * 80)

    # 实验结论
    print("\n[Conclusion] 实验结论:")

    if jerk_w30 < jerk_w15 and jerk_w15 < jerk_w1:
        print("  ✓ 时序建模有效: Jerk 随窗口大小增加而降低")
        print(f"    W=30 相比 W=1 的平滑度提升: {(1 - jerk_w30/jerk_w1)*100:.1f}%")
    else:
        print("  ⚠ 时序建模效果不明显，需要进一步分析")

    if abs(mae_w30 - mae_w1) < 2:
        print("  ✓ 单帧预测能力相近: 验证了 MAE 不是主要差异来源")
    else:
        print("  ⚠ MAE 差异较大，可能存在其他因素影响")


if __name__ == '__main__':
    main()
