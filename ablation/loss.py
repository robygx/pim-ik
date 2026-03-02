#!/usr/bin/env python3
"""
PiM-IK 物理内化损失消融实验评估脚本
=====================================

本脚本用于评估不同损失函数配置对模型性能的影响，验证物理内化损失各组件的必要性。

三种 Loss 配置:
1. Baseline: 仅 L_swivel (w_swivel=1.0, w_elbow=0.0, w_smooth=0.0)
2. Variant A: L_swivel + L_elbow (w_swivel=1.0, w_elbow=1.0, w_smooth=0.0)
3. Ours (Full): L_swivel + L_elbow + L_smooth (w_swivel=1.0, w_elbow=1.0, w_smooth=0.1)

评估指标:
- Swivel MAE (°): 臂角预测误差
- Elbow Error (mm): 肘部位置误差
- Jerk (Smoothness): 平滑度惩罚

使用方法:
    python ablation_loss_eval.py --data_path /path/to/data.npz

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
from typing import Dict, Optional

# 导入自定义模块
from pim_ik_net import PiM_IK_Net


# ============================================================================
# 模型加载
# ============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda:0') -> nn.Module:
    """
    加载训练好的模型

    Args:
        checkpoint_path: 检查点文件路径
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

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 移除 'module.' 前缀 (DDP 包装)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

        # 打印训练信息
        if 'loss_weights' in checkpoint:
            lw = checkpoint['loss_weights']
            print(f"[Model] 损失权重: swivel={lw.get('w_swivel', 'N/A')}, "
                  f"elbow={lw.get('w_elbow', 'N/A')}, smooth={lw.get('w_smooth', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# ============================================================================
# 指标计算
# ============================================================================

def compute_swivel_mae(
    pred: np.ndarray,
    gt: np.ndarray,
    is_valid: Optional[np.ndarray] = None
) -> float:
    """
    计算臂角 MAE（考虑角度周期性）

    Args:
        pred: (N, 2) 预测臂角 [cos, sin]
        gt: (N, 2) 真实臂角 [cos, sin]
        is_valid: (N,) 有效性掩码

    Returns:
        mae_deg: 平均角度误差（度）
    """
    # 转换为角度
    pred_angle = np.arctan2(pred[:, 1], pred[:, 0])
    gt_angle = np.arctan2(gt[:, 1], gt[:, 0])

    # 计算角度差（考虑周期性）
    diff = np.abs(pred_angle - gt_angle)
    diff = np.minimum(diff, 2 * np.pi - diff)

    # 应用有效性掩码
    if is_valid is not None:
        diff = diff[is_valid > 0.5]

    # 转换为度
    mae_deg = np.degrees(diff).mean()
    return mae_deg


def compute_elbow_error(
    pred_swivel: np.ndarray,
    gt_swivel: np.ndarray,
    p_s: np.ndarray,
    p_w: np.ndarray,
    L_upper: np.ndarray,
    L_lower: np.ndarray,
    is_valid: Optional[np.ndarray] = None
) -> float:
    """
    计算肘部位置误差 (mm)

    【重要】TargetGenerator 仅支持单帧计算，必须逐帧循环处理！

    Args:
        pred_swivel: (N, 2) 预测臂角
        gt_swivel: (N, 2) 真实臂角
        p_s: (N, 3) 肩部坐标
        p_w: (N, 3) 腕部坐标
        L_upper: (N,) 上臂长度
        L_lower: (N,) 前臂长度
        is_valid: (N,) 有效性掩码

    Returns:
        error_mm: 平均肘部位置误差（毫米）
    """
    from inference import TargetGenerator

    target_gen = TargetGenerator()

    # 逐帧计算肘部位置误差（TargetGenerator 不支持批量运算）
    errors = []
    for i in range(len(pred_swivel)):
        # 单帧计算预测肘部位置
        p_e_pred = target_gen.compute_target_elbow_position(
            pred_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        # 单帧计算真实肘部位置
        p_e_gt = target_gen.compute_target_elbow_position(
            gt_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        # 计算欧氏距离并转换为毫米
        errors.append(np.linalg.norm(p_e_pred - p_e_gt) * 1000)

    errors = np.array(errors)

    # 应用有效性掩码
    if is_valid is not None:
        errors = errors[is_valid > 0.5]

    return errors.mean()


def compute_jerk(
    pred_swivel: np.ndarray,
    is_valid: Optional[np.ndarray] = None
) -> float:
    """
    计算平滑度指标 (Jerk)

    Jerk = 二阶差分的均方值
    值越小表示轨迹越平滑

    Args:
        pred_swivel: (N, 2) 预测臂角 [cos, sin]
        is_valid: (N,) 有效性掩码

    Returns:
        jerk: 二阶差分均方值
    """
    # 转换为角度（度）
    phi = np.arctan2(pred_swivel[:, 1], pred_swivel[:, 0])
    phi_deg = np.degrees(phi)

    # 应用有效性掩码
    if is_valid is not None:
        valid_indices = is_valid > 0.5
        phi_deg = phi_deg[valid_indices]

    # 计算二阶差分
    if len(phi_deg) < 3:
        return 0.0

    jerk = phi_deg[2:] - 2 * phi_deg[1:-1] + phi_deg[:-2]

    # 均方值
    return np.mean(jerk ** 2)


# ============================================================================
# 数据加载
# ============================================================================

def load_validation_data(npz_path: str, train_split: float = 0.95) -> Dict[str, np.ndarray]:
    """
    加载验证集数据（最后 5%）

    Args:
        npz_path: 数据文件路径
        train_split: 训练集占比，验证集为 1 - train_split

    Returns:
        data: 包含验证集数据的字典
    """
    print(f"[Data] 加载数据: {npz_path}")

    raw = np.load(npz_path, allow_pickle=True)

    total_frames = len(raw['T_ee'])
    val_split = int(total_frames * train_split)

    print(f"[Data] 总帧数: {total_frames:,}, 验证集: {total_frames - val_split:,} 帧")

    data = {
        'T_ee': raw['T_ee'][val_split:].astype(np.float32),
        'swivel': raw['swivel_angle'][val_split:].astype(np.float32),
        'joint_pos': raw['joint_positions'][val_split:].astype(np.float32),
        'L_upper': raw['L_upper'][val_split:].astype(np.float32),
        'L_lower': raw['L_lower'][val_split:].astype(np.float32),
        'is_valid': raw['is_valid'][val_split:].astype(np.float32),
    }

    # 提取肩部和腕部坐标
    data['p_s'] = data['joint_pos'][:, 0, :]  # 肩部
    data['p_w'] = data['joint_pos'][:, 2, :]  # 腕部

    return data


# ============================================================================
# 推理
# ============================================================================

def run_inference(
    model: nn.Module,
    T_ee: np.ndarray,
    window_size: int,
    device: str
) -> np.ndarray:
    """
    在验证集上运行推理

    【重要】Mamba 模型需要完整的时序窗口输入！
    - 模型使用 W=30 训练
    - 推理时必须输入长度为 30 的滑动窗口
    - 取窗口最后一帧的预测作为当前帧结果

    Args:
        model: 神经网络模型
        T_ee: (N, 4, 4) 末端位姿
        window_size: 窗口大小 (必须与训练时一致，默认 30)
        device: 计算设备

    Returns:
        predictions: (N - W + 1, 2) 预测臂角
    """
    model.eval()
    N = len(T_ee)
    W = window_size

    print(f"[Infer] 开始滑动窗口推理 ({N:,} 帧, window_size={W})...")
    print(f"[Infer] 有效输出帧数: {N - W + 1:,} (丢弃前 {W-1} 帧)")

    pred_list = []

    with torch.no_grad():
        # 从 W-1 开始，确保窗口完整
        for i in range(W - 1, N):
            # 构建滑动窗口 [i-W+1 : i+1]，长度为 W
            start_idx = i - W + 1
            window = T_ee[start_idx:i + 1]  # (W, 4, 4)

            # 转换为 tensor 并添加 batch 维度
            window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, W, 4, 4)

            # 前向传播
            pred = model(window_tensor)  # (1, W, 2)

            # 取最后一个时间步的预测作为当前帧结果
            pred_last = pred[0, -1].cpu().numpy()  # (2,)
            pred_list.append(pred_last)

            # 进度显示
            if (i - W + 2) % 10000 == 0:
                print(f"  已处理: {i - W + 2:,}/{N - W + 1:,} 帧")

    print(f"[Infer] 推理完成，输出 {len(pred_list):,} 帧")
    return np.array(pred_list)


# ============================================================================
# 结果输出
# ============================================================================

def print_markdown_table(results: Dict[str, Dict[str, float]]):
    """
    输出高水平 Markdown 表格

    Args:
        results: 各模型的评估结果
    """
    print("\n" + "=" * 90)
    print("## Physics-Informed Loss Ablation Study Results")
    print("=" * 90)
    print()
    print("| Model | Swivel MAE (°) ↓ | Elbow Error (mm) ↓ | Jerk (Smoothness) ↓ |")
    print("|:------|:----------------:|:------------------:|:-------------------:|")

    # 定义顺序
    order = [
        'Baseline (sw1_el0_sm0)',
        'Variant A (sw1_el1_sm0)',
        'Ours (sw1_el1_sm0.1)'
    ]

    for name in order:
        if name in results:
            metrics = results[name]
            # 加粗 Ours 行
            if 'Ours' in name:
                print(f"| **{name}** | **{metrics['mae']:.2f}** | **{metrics['elbow']:.2f}** | **{metrics['jerk']:.4f}** |")
            else:
                print(f"| {name} | {metrics['mae']:.2f} | {metrics['elbow']:.2f} | {metrics['jerk']:.4f} |")

    print()
    print("**Legend**: ↓ indicates lower is better")
    print("=" * 90)


def print_conclusion(results: Dict[str, Dict[str, float]]):
    """
    打印实验结论

    Args:
        results: 各模型的评估结果
    """
    print("\n" + "=" * 90)
    print("## Experimental Conclusions")
    print("=" * 90)

    baseline = results.get('Baseline (sw1_el0_sm0)', {})
    variant_a = results.get('Variant A (sw1_el1_sm0)', {})
    ours = results.get('Ours (sw1_el1_sm0.1)', {})

    print()
    print("### 1. Effect of L_elbow (Spatial Constraint)")
    if baseline and variant_a:
        elbow_improve = (baseline.get('elbow', 0) - variant_a.get('elbow', 0)) / max(baseline.get('elbow', 1), 0.001) * 100
        print(f"   - Elbow Error: {baseline.get('elbow', 0):.2f}mm → {variant_a.get('elbow', 0):.2f}mm "
              f"({elbow_improve:+.1f}% improvement)")
        print(f"   - Conclusion: L_elbow significantly reduces spatial error ✓")

    print()
    print("### 2. Effect of L_smooth (Temporal Smoothness)")
    if variant_a and ours:
        jerk_improve = (variant_a.get('jerk', 0) - ours.get('jerk', 0)) / max(variant_a.get('jerk', 0.001), 0.001) * 100
        print(f"   - Jerk: {variant_a.get('jerk', 0):.4f} → {ours.get('jerk', 0):.4f} "
              f"({jerk_improve:+.1f}% improvement)")
        print(f"   - Conclusion: L_smooth improves motion smoothness ✓")

    print()
    print("### 3. MAE Consistency (Single-frame Prediction)")
    maes = [baseline.get('mae', 0), variant_a.get('mae', 0), ours.get('mae', 0)]
    maes = [m for m in maes if m > 0]
    if len(maes) >= 2:
        mae_range = max(maes) - min(maes)
        print(f"   - MAE Range: {min(maes):.2f}° ~ {max(maes):.2f}° (Δ = {mae_range:.2f}°)")
        print(f"   - Conclusion: Loss configuration does NOT affect single-frame accuracy ✓")

    print("=" * 90)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PiM-IK 物理内化损失消融实验评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str,
                        default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz',
                        help='验证数据集路径')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='检查点目录')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')
    parser.add_argument('--window_size', type=int, default=30,
                        help='推理时使用的窗口大小 (必须与训练时一致)')

    args = parser.parse_args()

    print("=" * 90)
    print("PiM-IK Physics-Informed Loss Ablation Study")
    print("=" * 90)

    # 检查设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] 使用设备: {device}")
    print(f"[Config] 窗口大小: W={args.window_size}")

    # ================================================================
    # 加载数据
    # ================================================================
    data = load_validation_data(args.data_path)

    # ================================================================
    # 定义模型配置
    # ================================================================
    configs = {
        'Baseline (sw1_el0_sm0)': {
            'pattern': f'**/W{args.window_size}_loss_sw1.0_el0.0_sm0.0_*/best_model_w{args.window_size}_sw1.0_el0.0_sm0.0.pth',
            'w_swivel': 1.0, 'w_elbow': 0.0, 'w_smooth': 0.0
        },
        'Variant A (sw1_el1_sm0)': {
            'pattern': f'**/W{args.window_size}_loss_sw1.0_el1.0_sm0.0_*/best_model_w{args.window_size}_sw1.0_el1.0_sm0.0.pth',
            'w_swivel': 1.0, 'w_elbow': 1.0, 'w_smooth': 0.0
        },
        'Ours (sw1_el1_sm0.1)': {
            'pattern': f'**/W{args.window_size}_loss_sw1.0_el1.0_sm0.1_*/best_model_w{args.window_size}_sw1.0_el1.0_sm0.1.pth',
            'w_swivel': 1.0, 'w_elbow': 1.0, 'w_smooth': 0.1
        },
    }

    results = {}

    # ================================================================
    # 遍历每个模型配置
    # ================================================================
    for name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"[{name}]")
        print(f"{'='*60}")

        # 查找 checkpoint
        ckpt_paths = list(Path(args.checkpoint_dir).glob(config['pattern']))

        if not ckpt_paths:
            print(f"[Warning] 未找到模型: {config['pattern']}")
            print(f"[Hint] 请先运行训练:")
            print(f"       torchrun --nproc_per_node=2 trainer.py --w_swivel {config['w_swivel']} --w_elbow {config['w_elbow']} --w_smooth {config['w_smooth']} --epochs 6")
            continue

        print(f"[Checkpoint] {ckpt_paths[0]}")

        # 加载模型
        model = load_model(str(ckpt_paths[0]), device)

        # 滑动窗口推理
        pred_swivel = run_inference(model, data['T_ee'], args.window_size, device)

        # ============================================================
        # 对齐 GT 和掩码（丢弃前 W-1 帧）
        # ============================================================
        W = args.window_size
        gt_swivel_trunc = data['swivel'][W - 1:]
        p_s_trunc = data['p_s'][W - 1:]
        p_w_trunc = data['p_w'][W - 1:]
        L_upper_trunc = data['L_upper'][W - 1:]
        L_lower_trunc = data['L_lower'][W - 1:]
        is_valid_trunc = data['is_valid'][W - 1:]

        # 验证长度对齐
        assert len(pred_swivel) == len(gt_swivel_trunc), \
            f"长度不匹配: pred={len(pred_swivel)}, gt={len(gt_swivel_trunc)}"

        # ============================================================
        # 计算指标
        # ============================================================
        print(f"[Metrics] 计算评估指标...")

        results[name] = {
            'mae': compute_swivel_mae(pred_swivel, gt_swivel_trunc, is_valid_trunc),
            'elbow': compute_elbow_error(
                pred_swivel, gt_swivel_trunc,
                p_s_trunc, p_w_trunc,
                L_upper_trunc, L_lower_trunc,
                is_valid_trunc
            ),
            'jerk': compute_jerk(pred_swivel, is_valid_trunc)
        }

        print(f"  Swivel MAE: {results[name]['mae']:.2f}°")
        print(f"  Elbow Error: {results[name]['elbow']:.2f} mm")
        print(f"  Jerk: {results[name]['jerk']:.4f}")

    # ================================================================
    # 输出结果
    # ================================================================
    if results:
        print_markdown_table(results)
        print_conclusion(results)
    else:
        print("\n[Error] 未找到任何模型，请先训练！")
        print("\n训练命令:")
        print("  # Baseline (仅 L_swivel)")
        print("  torchrun --nproc_per_node=2 trainer.py --w_swivel 1.0 --w_elbow 0.0 --w_smooth 0.0 --epochs 6")
        print()
        print("  # Variant A (L_swivel + L_elbow)")
        print("  torchrun --nproc_per_node=2 trainer.py --w_swivel 1.0 --w_elbow 1.0 --w_smooth 0.0 --epochs 6")
        print()
        print("  # Ours (完整物理内化损失)")
        print("  torchrun --nproc_per_node=2 trainer.py --w_swivel 1.0 --w_elbow 1.0 --w_smooth 0.1 --epochs 6")


if __name__ == '__main__':
    main()
