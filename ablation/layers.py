#!/usr/bin/env python3
"""
PiM-IK Mamba 骨干网络深度消融实验评估脚本
=============================================

本脚本用于评估不同 Mamba 层数（num_layers = 2, 3, 4, 6）对模型性能的影响，
验证网络深度与精度/参数量的关系。

评估指标:
- Swivel MAE (°): 臂角预测误差
- Elbow Error (mm): 肘部位置误差
- Jerk (Smoothness): 平滑度惩罚
- Params (K): 可训练参数量

使用方法:
    python ablation_layers_eval.py --data_path /path/to/data.npz

作者: PiM-IK Project
日期: 2025-02-28
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List

# 导入自定义模块
from pim_ik_net import PiM_IK_Net


# ============================================================================
# 模型加载
# ============================================================================

def load_model(checkpoint_path: str, num_layers: int, device: str = 'cuda:0') -> nn.Module:
    """
    加载训练好的模型

    Args:
        checkpoint_path: 检查点文件路径
        num_layers: Mamba 堆叠层数
        device: 计算设备

    Returns:
        model: 加载好权重的模型
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    print(f"[Model] 加载模型: {checkpoint_path}")

    # 创建模型（指定层数）
    model = PiM_IK_Net(d_model=256, num_layers=num_layers, backbone_type='mamba').to(device)

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
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 指标计算
# ============================================================================

def compute_swivel_mae(
    pred: np.ndarray,
    gt: np.ndarray,
    is_valid: np.ndarray = None
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
    is_valid: np.ndarray = None
) -> float:
    """
    计算肘部位置误差 (mm)

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

    # 逐帧计算肘部位置误差
    errors = []
    for i in range(len(pred_swivel)):
        # 预测肘部位置
        p_e_pred = target_gen.compute_target_elbow_position(
            pred_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        # 真实肘部位置
        p_e_gt = target_gen.compute_target_elbow_position(
            gt_swivel[i], p_s[i], p_w[i], L_upper[i], L_lower[i]
        )
        # 欧氏距离（毫米）
        errors.append(np.linalg.norm(p_e_pred - p_e_gt) * 1000)

    errors = np.array(errors)

    if is_valid is not None:
        errors = errors[is_valid > 0.5]

    return errors.mean()


def compute_jerk(
    pred_swivel: np.ndarray,
    is_valid: np.ndarray = None
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

    if is_valid is not None:
        valid_indices = is_valid > 0.5
        phi_deg = phi_deg[valid_indices]

    # 计算二阶差分
    if len(phi_deg) < 3:
        return 0.0

    jerk = phi_deg[2:] - 2 * phi_deg[1:-1] + phi_deg[:-2]
    return np.mean(jerk ** 2)


# ============================================================================
# 数据加载
# ============================================================================

def load_validation_data(npz_path: str, train_split: float = 0.95) -> Dict[str, np.ndarray]:
    """
    加载验证集数据（最后 5%）

    Args:
        npz_path: 数据文件路径
        train_split: 训练集占比

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

    Args:
        model: 神经网络模型
        T_ee: (N, 4, 4) 末端位姿
        window_size: 窗口大小
        device: 计算设备

    Returns:
        predictions: (N - W + 1, 2) 预测臂角
    """
    model.eval()
    N = len(T_ee)
    W = window_size

    print(f"[Infer] 开始滑动窗口推理 ({N:,} 帧, window_size={W})...")
    print(f"[Infer] 有效输出帧数: {N - W + 1:,}")

    pred_list = []

    with torch.no_grad():
        for i in range(W - 1, N):
            # 构建滑动窗口
            start_idx = i - W + 1
            window = T_ee[start_idx:i + 1]
            window_tensor = torch.from_numpy(window).unsqueeze(0).to(device)

            # 前向传播
            pred = model(window_tensor)
            pred_last = pred[0, -1].cpu().numpy()
            pred_list.append(pred_last)

            # 进度显示
            if (i - W + 2) % 10000 == 0:
                print(f"  已处理: {i - W + 2:,}/{N - W + 1:,} 帧")

    print(f"[Infer] 推理完成")
    return np.array(pred_list)


# ============================================================================
# 结果输出
# ============================================================================

def print_markdown_table(results: List[Dict]):
    """
    输出 Markdown 表格

    Args:
        results: 各层数配置的评估结果
    """
    print("\n" + "=" * 100)
    print("## Mamba Backbone Depth Ablation Study Results")
    print("=" * 100)
    print()
    print("| Layers | Params (K) | Swivel MAE (°) ↓ | Elbow Error (mm) ↓ | Jerk ↓ |")
    print("|:------:|:---------:|:----------------:|:------------------:|:------:|")

    for r in results:
        layers = r['layers']
        params_k = r['params'] / 1000
        mae = r['mae']
        elbow = r['elbow']
        jerk = r['jerk']

        # 加粗最佳结果
        best_mae = min(results, key=lambda x: x['mae'])['mae']
        best_elbow = min(results, key=lambda x: x['elbow'])['elbow']
        best_jerk = min(results, key=lambda x: x['jerk'])['jerk']

        mae_str = f"**{mae:.2f}**" if mae == best_mae else f"{mae:.2f}"
        elbow_str = f"**{elbow:.2f}**" if elbow == best_elbow else f"{elbow:.2f}"
        jerk_str = f"**{jerk:.2f}**" if jerk == best_jerk else f"{jerk:.2f}"

        print(f"| {layers} | {params_k:.0f} | {mae_str} | {elbow_str} | {jerk_str} |")

    print()
    print("**Legend**: ↓ indicates lower is better, **bold** = best result")
    print("=" * 100)


def print_conclusion(results: List[Dict]):
    """
    打印实验结论

    Args:
        results: 各层数配置的评估结果
    """
    print("\n" + "=" * 100)
    print("## Experimental Conclusions")
    print("=" * 100)

    # 参数量对比
    print("\n### 1. Parameter Count vs Network Depth")
    for r in results:
        params_k = r['params'] / 1000
        print(f"   - L{r['layers']}: {params_k:.0f}K params")

    # 找出最佳配置
    best_overall = min(results, key=lambda x: x['mae'])
    print(f"\n### 2. Best Configuration")
    print(f"   - Best MAE: L{best_overall['layers']} ({best_overall['mae']:.2f}°)")

    # 性能/参数比
    print(f"\n### 3. Efficiency (Accuracy per 1K params)")
    for r in results:
        efficiency = (100 - r['mae']) / (r['params'] / 1000)
        print(f"   - L{r['layers']}: {efficiency:.2f}")

    print("=" * 100)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PiM-IK Mamba 骨干网络深度消融实验评估',
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
                        help='推理时使用的窗口大小')
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 3, 4, 6],
                        help='要测试的层数列表')

    args = parser.parse_args()

    print("=" * 100)
    print("PiM-IK Mamba Backbone Depth Ablation Study")
    print("=" * 100)

    # 检查设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\n[Device] 使用设备: {device}")
    print(f"[Config] 窗口大小: W={args.window_size}")
    print(f"[Config] 测试层数: {args.layers}")

    # 加载数据
    data = load_validation_data(args.data_path)

    results = []

    # 遍历每个层数配置
    for num_layers in args.layers:
        print(f"\n{'='*60}")
        print(f"[Testing L{num_layers}]")
        print(f"{'='*60}")

        # 查找 checkpoint
        pattern = f"**/mamba_L{num_layers}_W{args.window_size}_*/best_model_mamba_L{num_layers}_w{args.window_size}_*.pth"
        ckpt_paths = list(Path(args.checkpoint_dir).glob(pattern))

        if not ckpt_paths:
            print(f"[Warning] 未找到 L{num_layers} 模型")
            print(f"[Hint] 请先运行训练:")
            print(f"       torchrun --nproc_per_node=2 trainer.py --num_layers {num_layers} --epochs 6")
            continue

        print(f"[Checkpoint] {ckpt_paths[0]}")

        # 加载模型
        model = load_model(str(ckpt_paths[0]), num_layers, device)

        # 统计参数量
        num_params = count_parameters(model)
        print(f"[Model] 参数量: {num_params:,}")

        # 滑动窗口推理
        pred_swivel = run_inference(model, data['T_ee'], args.window_size, device)

        # 对齐 GT 和掩码
        W = args.window_size
        gt_swivel_trunc = data['swivel'][W - 1:]
        p_s_trunc = data['p_s'][W - 1:]
        p_w_trunc = data['p_w'][W - 1:]
        L_upper_trunc = data['L_upper'][W - 1:]
        L_lower_trunc = data['L_lower'][W - 1:]
        is_valid_trunc = data['is_valid'][W - 1:]

        # 计算指标
        print(f"[Metrics] 计算评估指标...")

        results.append({
            'layers': num_layers,
            'params': num_params,
            'mae': compute_swivel_mae(pred_swivel, gt_swivel_trunc, is_valid_trunc),
            'elbow': compute_elbow_error(
                pred_swivel, gt_swivel_trunc,
                p_s_trunc, p_w_trunc,
                L_upper_trunc, L_lower_trunc,
                is_valid_trunc
            ),
            'jerk': compute_jerk(pred_swivel, is_valid_trunc)
        })

        print(f"  Swivel MAE: {results[-1]['mae']:.2f}°")
        print(f"  Elbow Error: {results[-1]['elbow']:.2f} mm")
        print(f"  Jerk: {results[-1]['jerk']:.2f}")

    # 输出结果
    if results:
        print_markdown_table(results)
        print_conclusion(results)
    else:
        print("\n[Error] 未找到任何模型，请先训练！")
        print("\n训练命令:")
        for layers in args.layers:
            print(f"  torchrun --nproc_per_node=2 trainer.py --num_layers {layers} --epochs 6")


if __name__ == '__main__':
    main()
