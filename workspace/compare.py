#!/usr/bin/env python3
"""
工作空间对比可视化脚本
用于对比 VR 控制器工作空间和机器人训练数据集工作空间

功能：
1. 读取 VR 原始数据和训练数据集
2. 在同一 3D 图中绘制两个点云
3. 计算并打印中心点偏移和范围比例
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_vr_data(vr_npy_path: str) -> np.ndarray:
    """
    加载 VR 原始位置数据

    Args:
        vr_npy_path: VR .npy 文件路径

    Returns:
        (N, 3) VR 位置数组
    """
    print(f"正在加载 VR 数据: {vr_npy_path}")
    if not os.path.exists(vr_npy_path):
        raise FileNotFoundError(f"VR 数据文件不存在: {vr_npy_path}")

    vr_positions = np.load(vr_npy_path)
    print(f"  VR 数据形状: {vr_positions.shape}")
    return vr_positions


def load_dataset_positions(dataset_npz_path: str) -> np.ndarray:
    """
    从训练数据集 NPZ 文件中提取末端位姿位置

    Args:
        dataset_npz_path: 数据集 .npz 文件路径

    Returns:
        (N, 3) 数据集位置数组
    """
    print(f"正在加载数据集: {dataset_npz_path}")
    if not os.path.exists(dataset_npz_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_npz_path}")

    data = np.load(dataset_npz_path, allow_pickle=False)
    if 'T_ee' not in data:
        raise ValueError("数据集中未找到 'T_ee' 字段")

    T_ee = data['T_ee']
    positions = T_ee[:, :3, 3].astype(np.float64)
    print(f"  数据集形状: {positions.shape}")
    return positions


def compute_robust_center_and_range(positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于鲁棒边界 (1% - 99% 分位数) 计算中心和范围

    Args:
        positions: (N, 3) 位置数组

    Returns:
        center: (3,) 鲁棒中心点
        range: (3,) 活动范围
    """
    p1 = np.percentile(positions, 1, axis=0)
    p99 = np.percentile(positions, 99, axis=0)

    center = (p99 + p1) / 2.0
    range_val = p99 - p1

    return center, range_val


def print_comparison_panel(
    vr_center: np.ndarray,
    vr_range: np.ndarray,
    dataset_center: np.ndarray,
    dataset_range: np.ndarray,
    vr_count: int,
    dataset_count: int
):
    """
    打印格式化的对比统计面板

    Args:
        vr_center: VR 鲁棒中心
        vr_range: VR 活动范围
        dataset_center: 数据集鲁棒中心
        dataset_range: 数据集活动范围
        vr_count: VR 点数
        dataset_count: 数据集点数
    """
    H_DOUBLE = "═"
    V_DOUBLE = "║"
    TL = "╔"
    TR = "╗"
    BL = "╚"
    BR = "╝"
    TC = "╦"
    BC = "╩"
    LC = "╠"
    RC = "╣"

    # 中心偏移
    center_offset = dataset_center - vr_center

    # 范围比例
    range_ratio = dataset_range / vr_range

    print("\n")
    print(TL + H_DOUBLE * 76 + TR)
    print(V_DOUBLE + " " * 20 + "工作空间对比分析报告" + " " * 34 + V_DOUBLE)
    print(TC + H_DOUBLE * 76 + BC)

    # 数据量
    print(V_DOUBLE + f"  VR 点数: {vr_count:>10,}".ljust(38) + f"数据集点数: {dataset_count:>10,}".ljust(38) + V_DOUBLE)
    print(LC + H_DOUBLE * 76 + RC)

    # 鲁棒中心对比
    print(V_DOUBLE + " " * 15 + "鲁棒中心点 (Robust Center) 对比" + " " * 27 + V_DOUBLE)
    print(LC + "────────────┬────────────────┬────────────────┬────────────────┬────────────────" + RC)
    print(V_DOUBLE + "    轴向    │     VR 中心    │   数据集中心    │    偏移量      │    偏移距离    " + V_DOUBLE)
    print(LC + "────────────┼────────────────┼────────────────┼────────────────┼────────────────" + RC)

    axes = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes):
        offset = center_offset[i]
        dist = abs(offset)
        print(V_DOUBLE +
              f"     {axis}     │  {vr_center[i]:>10.4f}  │  {dataset_center[i]:>10.4f}  │  {offset:>+10.4f}  │  {dist:>10.4f}  " +
              V_DOUBLE)

    # 总偏移距离
    total_offset = np.linalg.norm(center_offset)
    print(LC + "────────────┴────────────────┴────────────────┴────────────────┴────────────────" + RC)
    print(V_DOUBLE + f"  中心点总偏移距离: {total_offset:.4f} 米".ljust(77) + V_DOUBLE)
    print(TC + H_DOUBLE * 76 + BC)

    # 活动范围对比
    print(V_DOUBLE + " " * 15 + "活动范围 (99% - 1%) 对比" + " " * 31 + V_DOUBLE)
    print(LC + "────────────┬────────────────┬────────────────┬────────────────" + RC)
    print(V_DOUBLE + "    轴向    │    VR 范围     │   数据集范围    │    缩放比例    " + V_DOUBLE)
    print(LC + "────────────┼────────────────┼────────────────┼────────────────" + RC)

    for i, axis in enumerate(axes):
        ratio = range_ratio[i]
        print(V_DOUBLE +
              f"     {axis}     │  {vr_range[i]:>10.4f}  │  {dataset_range[i]:>10.4f}  │  {ratio:>10.4f}x  " +
              V_DOUBLE)

    print(LC + "────────────┴────────────────┴────────────────┴────────────────" + RC)

    # 推荐缩放
    min_scale = range_ratio.min()
    print(V_DOUBLE + f"  推荐统一缩放系数 (防变形): {min_scale:.4f}x".ljust(77) + V_DOUBLE)

    print(BL + H_DOUBLE * 76 + BR)
    print()


def plot_workspace_comparison(
    vr_positions: np.ndarray,
    dataset_positions: np.ndarray,
    output_path: str = 'workspace_comparison.png',
    max_points: int = 50000
):
    """
    绘制工作空间对比 3D 图

    Args:
        vr_positions: (N_vr, 3) VR 位置数组
        dataset_positions: (N_ds, 3) 数据集位置数组
        output_path: 输出图片路径
        max_points: 每个点云最大采样点数
    """
    print(f"\n正在生成工作空间对比图...")

    # 降采样
    if len(vr_positions) > max_points:
        vr_indices = np.random.choice(len(vr_positions), max_points, replace=False)
        vr_sample = vr_positions[vr_indices]
        print(f"  VR 数据降采样: {len(vr_positions):,} → {max_points:,}")
    else:
        vr_sample = vr_positions

    if len(dataset_positions) > max_points:
        ds_indices = np.random.choice(len(dataset_positions), max_points, replace=False)
        ds_sample = dataset_positions[ds_indices]
        print(f"  数据集降采样: {len(dataset_positions):,} → {max_points:,}")
    else:
        ds_sample = dataset_positions

    # 创建图形
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制数据集点云（半透明红色）
    ax.scatter(ds_sample[:, 0], ds_sample[:, 1], ds_sample[:, 2],
               c='red', s=0.5, alpha=0.3, label='Dataset Workspace', rasterized=True)

    # 绘制 VR 点云（半透明蓝色）
    ax.scatter(vr_sample[:, 0], vr_sample[:, 1], vr_sample[:, 2],
               c='blue', s=0.5, alpha=0.3, label='VR Controller Workspace', rasterized=True)

    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_zlabel('Z (m)', fontsize=14)

    # 设置标题
    ax.set_title('Workspace Comparison: VR Controller vs Training Dataset',
                 fontsize=16, fontweight='bold')

    # 设置等比例
    ax.set_box_aspect([1, 1, 1])

    # 图例
    ax.legend(loc='upper right', fontsize=12)

    # 网格
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  对比图已保存: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='工作空间对比分析 - VR 控制器 vs 训练数据集',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--vr_npy',
        type=str,
        default='/home/ygx/VR_data_any/vr_workspace_raw.npy',
        help='VR 原始位置数据 .npy 文件路径'
    )

    parser.add_argument(
        '--dataset_npz',
        type=str,
        default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz',
        help='训练数据集 .npz 文件路径'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='workspace_comparison.png',
        help='输出对比图路径'
    )

    parser.add_argument(
        '--max_points',
        type=int,
        default=50000,
        help='每个点云的最大采样点数'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    print("=" * 78)
    print("工作空间对比分析".center(78))
    print("=" * 78)

    try:
        # 1. 加载数据
        vr_positions = load_vr_data(args.vr_npy)
        dataset_positions = load_dataset_positions(args.dataset_npz)

        # 2. 计算鲁棒中心和范围
        print("\n正在计算统计指标...")
        vr_center, vr_range = compute_robust_center_and_range(vr_positions)
        dataset_center, dataset_range = compute_robust_center_and_range(dataset_positions)

        # 3. 打印对比面板
        print_comparison_panel(
            vr_center, vr_range,
            dataset_center, dataset_range,
            len(vr_positions), len(dataset_positions)
        )

        # 4. 生成对比图
        plot_workspace_comparison(
            vr_positions, dataset_positions,
            args.output, args.max_points
        )

        print("=" * 78)
        print("分析完成!".center(78))
        print("=" * 78)

        return 0

    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
