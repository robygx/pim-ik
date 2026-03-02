#!/usr/bin/env python3
"""
工作空间分析脚本 - 增强版
用于统计和可视化训练数据集中左臂末端位姿的三维平移活动范围
支持完整数据可视化、2D 投影密度图、轴向分布直方图和综合分析报告
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d


# ============================================
# 数据加载
# ============================================

def load_data(data_path: str) -> tuple:
    """
    加载 NPZ 文件并提取 T_ee 位置数据

    Args:
        data_path: NPZ 文件路径

    Returns:
        positions: (N, 3) 三维平移向量
        metadata: 数据元信息
    """
    print(f"正在加载数据: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    data = np.load(data_path, allow_pickle=False)

    # 检查必需字段
    if 'T_ee' not in data:
        raise ValueError("NPZ 文件中未找到 'T_ee' 字段")

    T_ee = data['T_ee']
    N = T_ee.shape[0]

    print(f"  T_ee shape: {T_ee.shape}")
    print(f"  总帧数: {N:,}")

    # 提取平移向量 (N, 3)
    positions = T_ee[:, :3, 3].astype(np.float64)

    # 收集元数据
    metadata = {
        'num_samples': int(N),
        'data_path': data_path,
    }

    # 尝试获取更多信息
    if 'window_size' in data:
        metadata['window_size'] = int(data['window_size'])
    if 'num_samples' in data:
        metadata['total_samples'] = int(data['num_samples'])

    return positions, metadata


# ============================================
# 统计指标计算
# ============================================

def compute_statistics(positions: np.ndarray) -> Dict[str, Any]:
    """
    计算工作空间统计指标

    Args:
        positions: (N, 3) 三维位置数组

    Returns:
        stats: 统计指标字典
    """
    print("\n正在计算统计指标...")

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    stats = {
        'absolute_bounds': {
            'x': {'min': float(x.min()), 'max': float(x.max())},
            'y': {'min': float(y.min()), 'max': float(y.max())},
            'z': {'min': float(z.min()), 'max': float(z.max())},
        },
        'robust_bounds': {
            'x': {'p1': float(np.percentile(x, 1)), 'p99': float(np.percentile(x, 99))},
            'y': {'p1': float(np.percentile(y, 1)), 'p99': float(np.percentile(y, 99))},
            'z': {'p1': float(np.percentile(z, 1)), 'p99': float(np.percentile(z, 99))},
        },
        'center': {
            'x': float(x.mean()),
            'y': float(y.mean()),
            'z': float(z.mean()),
        },
        'range': {
            'x': float(np.percentile(x, 99) - np.percentile(x, 1)),
            'y': float(np.percentile(y, 99) - np.percentile(y, 1)),
            'z': float(np.percentile(z, 99) - np.percentile(z, 1)),
        }
    }

    return stats


def save_statistics(stats: Dict[str, Any], metadata: Dict[str, Any],
                    output_path: str = 'dataset_workspace_limits.json'):
    """
    保存统计结果到 JSON 文件

    Args:
        stats: 统计指标字典
        metadata: 元数据字典
        output_path: 输出文件路径
    """
    print(f"\n正在保存统计结果到: {output_path}")

    output_data = {
        'metadata': metadata,
        'statistics': stats
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("  保存完成")


# ============================================
# 可视化功能
# ============================================

def plot_workspace_3d_full(positions: np.ndarray, stats: Dict[str, Any],
                           output_path: str = 'dataset_workspace_3d_full.png',
                           use_all_data: bool = True):
    """
    创建完整 3D 散点图可视化

    Args:
        positions: (N, 3) 三维位置数组
        stats: 统计指标字典
        output_path: 输出图片路径
        use_all_data: 是否使用全部数据（可能导致渲染变慢）
    """
    print(f"\n正在生成完整 3D 可视化...")

    sample = positions
    if not use_all_data and len(positions) > 50000:
        indices = np.random.choice(len(positions), 50000, replace=False)
        sample = positions[indices]
        print(f"  采样 {len(sample):,} 点")
    else:
        print(f"  使用全部 {len(sample):,} 点")

    # 创建图形
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图 - 使用极小的点和低透明度
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2],
               s=0.05, alpha=0.05, c='steelblue', edgecolors='none', rasterized=True)

    # 绘制鲁棒边界框 (1% - 99%)
    rb = stats['robust_bounds']
    x_range = (rb['x']['p1'], rb['x']['p99'])
    y_range = (rb['y']['p1'], rb['y']['p99'])
    z_range = (rb['z']['p1'], rb['z']['p99'])

    # 绘制边界框的 8 个顶点
    corners_x = [x_range[0], x_range[1], x_range[1], x_range[0],
                 x_range[0], x_range[1], x_range[1], x_range[0]]
    corners_y = [y_range[0], y_range[0], y_range[1], y_range[1],
                 y_range[0], y_range[0], y_range[1], y_range[1]]
    corners_z = [z_range[0], z_range[0], z_range[0], z_range[0],
                 z_range[1], z_range[1], z_range[1], z_range[1]]

    # 绘制边界框的 12 条边
    edges = [
        ([0, 1], [0, 0], [0, 0]), ([0, 3], [1, 1], [0, 0]),
        ([4, 5], [0, 0], [1, 1]), ([6, 7], [1, 1], [1, 1]),
        ([0, 3], [0, 1], [0, 0]), ([1, 2], [0, 1], [0, 0]),
        ([4, 7], [0, 1], [1, 1]), ([5, 6], [0, 1], [1, 1]),
        ([0, 4], [0, 0], [0, 1]), ([1, 5], [1, 1], [0, 1]),
        ([2, 6], [1, 1], [1, 1]), ([3, 7], [0, 0], [1, 1]),
    ]

    for edge in edges:
        xs = [corners_x[i] for i in edge[0]]
        ys = [corners_y[i] for i in edge[1]]
        zs = [corners_z[i] for i in edge[2]]
        ax.plot(xs, ys, zs, 'r-', linewidth=2, alpha=0.8)

    # 绘制几何中心
    center = stats['center']
    ax.scatter([center['x']], [center['y']], [center['z']],
               s=200, c='red', marker='*', zorder=10,
               label=f"Center ({center['x']:.2f}, {center['y']:.2f}, {center['z']:.2f})")

    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=14)
    ax.set_ylabel('Y (m)', fontsize=14)
    ax.set_zlabel('Z (m)', fontsize=14)

    # 设置标题
    ax.set_title(f'Full Workspace Distribution (N={len(positions):,})\nRobust Bounds: 1% - 99% Percentile',
                 fontsize=16, fontweight='bold')

    # 设置三轴等比例
    ax.set_box_aspect([1, 1, 1])

    # 设置网格
    ax.grid(True, alpha=0.3)

    # 图例
    ax.legend(loc='upper right', fontsize=12)

    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  图片已保存: {output_path}")

    plt.close()


def plot_2d_projections(positions: np.ndarray, stats: Dict[str, Any],
                        output_path: str = 'dataset_workspace_2d_projections.png'):
    """
    创建 2D 投影密度图 (XY, XZ, YZ 平面)

    Args:
        positions: (N, 3) 三维位置数组
        stats: 统计指标字典
        output_path: 输出图片路径
    """
    print(f"\n正在生成 2D 投影密度图...")

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # 创建 3 子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 颜色映射
    cmap = 'Blues'
    alpha_val = 0.3

    # XY 平面 (俯视图)
    ax0 = axes[0]
    hb0 = ax0.hexbin(x, y, gridsize=50, cmap=cmap, alpha=alpha_val, edgecolors='none')
    ax0.set_xlabel('X (m)', fontsize=12)
    ax0.set_ylabel('Y (m)', fontsize=12)
    ax0.set_title('XY Projection (Top View)', fontsize=14, fontweight='bold')
    ax0.set_aspect('equal')
    cb0 = plt.colorbar(hb0, ax=ax0)
    cb0.set_label('Count', rotation=270, labelpad=15)

    # 绘制鲁棒边界框
    rb = stats['robust_bounds']
    rect0 = Rectangle((rb['x']['p1'], rb['y']['p1']),
                      rb['x']['p99'] - rb['x']['p1'],
                      rb['y']['p99'] - rb['y']['p1'],
                      fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax0.add_patch(rect0)

    # XZ 平面 (侧视图)
    ax1 = axes[1]
    hb1 = ax1.hexbin(x, z, gridsize=50, cmap=cmap, alpha=alpha_val, edgecolors='none')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Z (m)', fontsize=12)
    ax1.set_title('XZ Projection (Side View)', fontsize=14, fontweight='bold')
    ax1.set_aspect('equal')
    cb1 = plt.colorbar(hb1, ax=ax1)
    cb1.set_label('Count', rotation=270, labelpad=15)

    # 绘制鲁棒边界框
    rect1 = Rectangle((rb['x']['p1'], rb['z']['p1']),
                      rb['x']['p99'] - rb['x']['p1'],
                      rb['z']['p99'] - rb['z']['p1'],
                      fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax1.add_patch(rect1)

    # YZ 平面 (侧视图)
    ax2 = axes[2]
    hb2 = ax2.hexbin(y, z, gridsize=50, cmap=cmap, alpha=alpha_val, edgecolors='none')
    ax2.set_xlabel('Y (m)', fontsize=12)
    ax2.set_ylabel('Z (m)', fontsize=12)
    ax2.set_title('YZ Projection (Side View)', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    cb2 = plt.colorbar(hb2, ax=ax2)
    cb2.set_label('Count', rotation=270, labelpad=15)

    # 绘制鲁棒边界框
    rect2 = Rectangle((rb['y']['p1'], rb['z']['p1']),
                      rb['y']['p99'] - rb['y']['p1'],
                      rb['z']['p99'] - rb['z']['p1'],
                      fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax2.add_patch(rect2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  图片已保存: {output_path}")

    plt.close()


def plot_axis_histograms(positions: np.ndarray, stats: Dict[str, Any],
                         output_path: str = 'dataset_workspace_histograms.png'):
    """
    创建轴向分布直方图

    Args:
        positions: (N, 3) 三维位置数组
        stats: 统计指标字典
        output_path: 输出图片路径
    """
    print(f"\n正在生成轴向分布直方图...")

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # 创建 3 子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes_list = [(axes[0], x, 'X', colors[0]), (axes[1], y, 'Y', colors[1]), (axes[2], z, 'Z', colors[2])]

    for ax, data, axis_name, color in axes_list:
        # 计算直方图
        n, bins, patches = ax.hist(data, bins=100, color=color, alpha=0.6, edgecolor='black', linewidth=0.5)

        # 标注分位数
        axis_lower = axis_name.lower()
        p1 = stats['robust_bounds'][axis_lower]['p1']
        p50 = np.percentile(data, 50)
        p99 = stats['robust_bounds'][axis_lower]['p99']

        ax.axvline(p1, color='red', linestyle='--', linewidth=2, label=f'1%: {p1:.3f}m')
        ax.axvline(p50, color='green', linestyle='-.', linewidth=2, label=f'50%: {p50:.3f}m')
        ax.axvline(p99, color='red', linestyle='--', linewidth=2, label=f'99%: {p99:.3f}m')

        # 填充中间区域
        ax.axvspan(p1, p99, alpha=0.1, color='gray', label=f'Range: {p99-p1:.3f}m')

        ax.set_xlabel(f'{axis_name} Position (m)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{axis_name} Axis Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  图片已保存: {output_path}")

    plt.close()


def plot_comprehensive_summary(positions: np.ndarray, stats: Dict[str, Any],
                                output_path: str = 'dataset_workspace_comprehensive.png'):
    """
    创建综合分析图 (4合1布局)

    Args:
        positions: (N, 3) 三维位置数组
        stats: 统计指标字典
        output_path: 输出图片路径
    """
    print(f"\n正在生成综合分析图...")

    # 随机采样用于 3D 图
    n_samples = min(50000, len(positions))
    indices = np.random.choice(len(positions), n_samples, replace=False)
    sample = positions[indices]

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    rb = stats['robust_bounds']
    center = stats['center']

    # 创建 2x2 布局
    fig = plt.figure(figsize=(16, 14))

    # 左上: 3D 散点图
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.scatter(sample[:, 0], sample[:, 1], sample[:, 2],
                 s=0.1, alpha=0.1, c='steelblue', rasterized=True)
    # 边界框 - 底部和顶部
    bottom_x = [rb['x']['p1'], rb['x']['p99'], rb['x']['p99'], rb['x']['p1'], rb['x']['p1']]
    bottom_y = [rb['y']['p1'], rb['y']['p1'], rb['y']['p99'], rb['y']['p99'], rb['y']['p1']]
    bottom_z = [rb['z']['p1']] * 5
    top_x = [rb['x']['p1'], rb['x']['p99'], rb['x']['p99'], rb['x']['p1'], rb['x']['p1']]
    top_y = [rb['y']['p1'], rb['y']['p1'], rb['y']['p99'], rb['y']['p99'], rb['y']['p1']]
    top_z = [rb['z']['p99']] * 5
    ax3d.plot(bottom_x, bottom_y, bottom_z, 'r-', linewidth=1.5)
    ax3d.plot(top_x, top_y, top_z, 'r-', linewidth=1.5)
    # 连接竖线
    for i in range(4):
        ax3d.plot([bottom_x[i], top_x[i]], [bottom_y[i], top_y[i]], [bottom_z[i], top_z[i]], 'r-', linewidth=1.5)

    ax3d.scatter([center['x']], [center['y']], [center['z']],
                 s=100, c='red', marker='*')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Workspace', fontweight='bold')
    ax3d.set_box_aspect([1, 1, 1])

    # 右上: XY 投影
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.hexbin(x, y, gridsize=40, cmap='Blues', alpha=0.5, edgecolors='none')
    ax_xy.add_patch(Rectangle((rb['x']['p1'], rb['y']['p1']),
                              rb['x']['p99'] - rb['x']['p1'],
                              rb['y']['p99'] - rb['y']['p1'],
                              fill=False, edgecolor='red', linewidth=2))
    ax_xy.scatter([center['x']], [center['y']], s=100, c='red', marker='*')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('XY Projection (Top View)', fontweight='bold')
    ax_xy.set_aspect('equal')

    # 左下: XZ 投影
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.hexbin(x, z, gridsize=40, cmap='Blues', alpha=0.5, edgecolors='none')
    ax_xz.add_patch(Rectangle((rb['x']['p1'], rb['z']['p1']),
                              rb['x']['p99'] - rb['x']['p1'],
                              rb['z']['p99'] - rb['z']['p1'],
                              fill=False, edgecolor='red', linewidth=2))
    ax_xz.scatter([center['x']], [center['z']], s=100, c='red', marker='*')
    ax_xz.set_xlabel('X (m)')
    ax_xz.set_ylabel('Z (m)')
    ax_xz.set_title('XZ Projection (Side View)', fontweight='bold')
    ax_xz.set_aspect('equal')

    # 右下: 统计信息文本
    ax_info = fig.add_subplot(2, 2, 4)
    ax_info.axis('off')

    info_text = f"""
WORKSPACE STATISTICS
{'='*40}

Total Samples: {len(positions):,}

Absolute Bounds (m):
  X: [{stats['absolute_bounds']['x']['min']:.3f}, {stats['absolute_bounds']['x']['max']:.3f}]
  Y: [{stats['absolute_bounds']['y']['min']:.3f}, {stats['absolute_bounds']['y']['max']:.3f}]
  Z: [{stats['absolute_bounds']['z']['min']:.3f}, {stats['absolute_bounds']['z']['max']:.3f}]

Robust Bounds [1%, 99%] (m):
  X: [{rb['x']['p1']:.3f}, {rb['x']['p99']:.3f}]
  Y: [{rb['y']['p1']:.3f}, {rb['y']['p99']:.3f}]
  Z: [{rb['z']['p1']:.3f}, {rb['z']['p99']:.3f}]

Workspace Range (m):
  X: {stats['range']['x']:.3f}
  Y: {stats['range']['y']:.3f}
  Z: {stats['range']['z']:.3f}

Geometric Center (m):
  ({center['x']:.3f}, {center['y']:.3f}, {center['z']:.3f})

Recommended VR Mapping Limits:
  X: [{rb['x']['p1']:.3f}, {rb['x']['p99']:.3f}]
  Y: [{rb['y']['p1']:.3f}, {rb['y']['p99']:.3f}]
  Z: [{rb['z']['p1']:.3f}, {rb['z']['p99']:.3f}]
"""

    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  图片已保存: {output_path}")

    plt.close()


# ============================================
# 终端输出面板
# ============================================

def print_statistics_panel(stats: Dict[str, Any], metadata: Dict[str, Any]):
    """
    打印格式化的统计面板到终端

    Args:
        stats: 统计指标字典
        metadata: 元数据字典
    """
    # 单线边框
    H_SINGLE = "─"
    V_SINGLE = "│"
    # 双线边框
    H_DOUBLE = "═"
    V_DOUBLE = "║"
    # 交叉
    TC_SINGLE = "┬"
    BC_SINGLE = "┴"
    LC_SINGLE = "├"
    RC_SINGLE = "┤"
    CROSS_SINGLE = "┼"

    def format_float(val: float, width: int = 12) -> str:
        """格式化浮点数，右对齐"""
        return f"{val:>12.4f}"

    # 标题
    title = "Training Dataset Workspace Limits Analysis"

    print("\n")
    print("╔" + H_DOUBLE * 78 + "╗")
    print(V_DOUBLE + " " * 15 + title + " " * (78 - 15 - len(title)) + V_DOUBLE)
    print("╦" + H_SINGLE * 78 + "╩")

    # 元数据
    num_frames = metadata.get('num_samples', 0)
    data_source = metadata.get('data_path', 'Unknown')
    if len(data_source) > 55:
        data_source = "..." + data_source[-52:]

    print(V_DOUBLE + f"  Total Frames:  {num_frames:>15,}".ljust(78) + V_DOUBLE)
    print(V_DOUBLE + f"  Data Source: {data_source}".ljust(78) + V_DOUBLE)
    print("╦" + H_SINGLE * 78 + "╩")

    # 绝对边界
    print(V_DOUBLE + " " * 23 + "Absolute Bounds (m)" + " " * 40 + V_DOUBLE)
    print("╠" + H_SINGLE * 20 + TC_SINGLE + H_SINGLE * 16 + TC_SINGLE + H_SINGLE * 22 + "╣")
    print(V_DOUBLE + " " * 20 + V_SINGLE + " " * 7 + "Min" + " " * 7 + V_SINGLE + " " * 9 + "Max" + " " * 11 + V_DOUBLE)
    print("├" + H_SINGLE * 20 + CROSS_SINGLE + H_SINGLE * 16 + CROSS_SINGLE + H_SINGLE * 22 + "┤")

    abs_bounds = stats['absolute_bounds']
    for axis in ['X', 'Y', 'Z']:
        axis_lower = axis.lower()
        row = (V_DOUBLE + f" {axis} Axis" + " " * 14 +
               V_SINGLE + format_float(abs_bounds[axis_lower]['min']) + " " + V_SINGLE +
               format_float(abs_bounds[axis_lower]['max']) + " " * 11 + V_DOUBLE)
        print(row)

    print("╦" + H_SINGLE * 78 + "╩")

    # 鲁棒边界
    print(V_DOUBLE + " " * 19 + "Robust Bounds [1%, 99%] (m)" + " " * 36 + V_DOUBLE)
    print("╠" + H_SINGLE * 20 + TC_SINGLE + H_SINGLE * 16 + TC_SINGLE + H_SINGLE * 22 + "╣")
    print(V_DOUBLE + " " * 20 + V_SINGLE + " " * 6 + "1%" + " " * 7 + V_SINGLE + " " * 8 + "99%" + " " * 10 + V_DOUBLE)
    print("├" + H_SINGLE * 20 + CROSS_SINGLE + H_SINGLE * 16 + CROSS_SINGLE + H_SINGLE * 22 + "┤")

    robust_bounds = stats['robust_bounds']
    for axis in ['X', 'Y', 'Z']:
        axis_lower = axis.lower()
        row = (V_DOUBLE + f" {axis} Axis" + " " * 14 +
               V_SINGLE + format_float(robust_bounds[axis_lower]['p1']) + " " + V_SINGLE +
               format_float(robust_bounds[axis_lower]['p99']) + " " * 11 + V_DOUBLE)
        print(row)

    print("╦" + H_SINGLE * 78 + "╩")

    # 活动范围
    print(V_DOUBLE + " " * 18 + "Workspace Range (99% - 1%)" + " " * 36 + V_DOUBLE)
    print("╠" + H_SINGLE * 20 + CROSS_SINGLE + H_SINGLE * 42 + "╣")

    range_stats = stats['range']
    for axis in ['X', 'Y', 'Z']:
        axis_lower = axis.lower()
        row = (V_DOUBLE + f" {axis} Range" + " " * 12 +
               V_SINGLE + " " * 10 + format_float(range_stats[axis_lower]) + " " * 18 + V_DOUBLE)
        print(row)

    print("╦" + H_SINGLE * 78 + "╩")

    # 几何中心
    print(V_DOUBLE + " " * 21 + "Geometric Center (m)" + " " * 39 + V_DOUBLE)
    print("╠" + H_SINGLE * 78 + "╣")
    center = stats['center']
    center_str = f"Center:  ({center['x']:.4f}, {center['y']:.4f}, {center['z']:.4f})"
    print(V_DOUBLE + " " * 4 + center_str + " " * (78 - 4 - len(center_str)) + V_DOUBLE)

    print("╚" + H_DOUBLE * 78 + "╝")

    # 输出文件说明
    print("\nOutput files:")
    print("  - dataset_workspace_limits.json")
    print("  - dataset_workspace_3d_full.png")
    print("  - dataset_workspace_2d_projections.png")
    print("  - dataset_workspace_histograms.png")
    print("  - dataset_workspace_comprehensive.png")
    print()


# ============================================
# 主函数
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='工作空间分析脚本增强版 - 统计和可视化末端执行器位置分布',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz',
        help='训练数据 NPZ 文件路径'
    )

    parser.add_argument(
        '--json_output',
        type=str,
        default='dataset_workspace_limits.json',
        help='JSON 输出文件路径'
    )

    parser.add_argument(
        '--full_3d_output',
        type=str,
        default='dataset_workspace_3d_full.png',
        help='完整 3D 可视化输出路径'
    )

    parser.add_argument(
        '--projections_output',
        type=str,
        default='dataset_workspace_2d_projections.png',
        help='2D 投影图输出路径'
    )

    parser.add_argument(
        '--histograms_output',
        type=str,
        default='dataset_workspace_histograms.png',
        help='直方图输出路径'
    )

    parser.add_argument(
        '--comprehensive_output',
        type=str,
        default='dataset_workspace_comprehensive.png',
        help='综合分析图输出路径'
    )

    parser.add_argument(
        '--skip_3d',
        action='store_true',
        help='跳过 3D 可视化生成'
    )

    parser.add_argument(
        '--skip_projections',
        action='store_true',
        help='跳过 2D 投影图生成'
    )

    parser.add_argument(
        '--skip_histograms',
        action='store_true',
        help='跳过直方图生成'
    )

    parser.add_argument(
        '--skip_comprehensive',
        action='store_true',
        help='跳过综合分析图生成'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    print("=" * 80)
    print("工作空间分析脚本增强版".center(80))
    print("=" * 80)

    try:
        # 1. 加载数据
        positions, metadata = load_data(args.data_path)

        # 2. 计算统计指标
        stats = compute_statistics(positions)

        # 3. 保存 JSON
        save_statistics(stats, metadata, args.json_output)

        # 4. 生成各种可视化
        if not args.skip_3d:
            plot_workspace_3d_full(positions, stats, args.full_3d_output, use_all_data=True)

        if not args.skip_projections:
            plot_2d_projections(positions, stats, args.projections_output)

        if not args.skip_histograms:
            plot_axis_histograms(positions, stats, args.histograms_output)

        if not args.skip_comprehensive:
            plot_comprehensive_summary(positions, stats, args.comprehensive_output)

        # 5. 打印终端面板
        print_statistics_panel(stats, metadata)

        print("=" * 80)
        print("分析完成!".center(80))
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
