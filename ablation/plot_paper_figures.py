#!/usr/bin/env python3
"""
PiM-IK 论文核心图表与表格生成脚本
=====================================

生成符合 IEEE/Science/ICRA 顶会出版标准的高清图表 (PDF+PNG) 和 Markdown 汇总表格。

图表:
1. fig_pareto_backbone - 骨干网络帕累托前沿图 (气泡散点图)
2. fig_window_size - 时序窗口双轴折线图
3. fig_loss_ablation - 物理内化损失柱状图
4. fig_layers_ablation - 层数消融折线图
5. fig_correlation_analysis - 三重相关性分析图 (新增)

使用方法:
    cd /home/ygx/ygx_hl_ik_v2
    python ablation/plot_paper_figures.py
"""

import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================================
# 全局样式设置 (IEEE/Science/ICRA 出版标准)
# ============================================================================

def setup_matplotlib_style():
    """
    设置 Matplotlib 全局样式 - IEEE/Science 顶会风格

    参考:
    - Science Journal: 简洁、无衬线字体、淡网格
    - IEEE: 清晰的坐标轴标签、专业配色
    - Nature/ICRA: 高对比度、色盲友好
    """
    # 使用 seaborn whitegrid 样式作为基础
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass  # 如果不可用，使用默认样式

    # IEEE/Science 风格自定义参数
    plt.rcParams.update({
        # 字体设置 - 使用无衬线字体
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # 图形设置
        'figure.dpi': 100,
        'figure.figsize': (7, 5),  # 单栏论文标准尺寸
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,

        # 线条设置
        'lines.linewidth': 2,
        'lines.markersize': 8,

        # 坐标轴设置 - Science 风格
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,

        # 刻度设置
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })


# ============================================================================
# 配色方案 (色盲友好 + Science/IEEE 学术风格)
# ============================================================================

# Nature/Science 推荐配色 (色盲友好)
# 参考: Wong, B. (2011). Points of view: Color blindness. Nature Methods, 8(6), 441.
COLORS = {
    'blue': '#0173B2',       # 深蓝 - 主要数据
    'orange': '#DE8F05',     # 深橙 - 对比数据
    'green': '#029E73',      # 深绿 - 第三数据
    'red': '#CC3311',        # 深红 - 高亮/重要
    'purple': '#9C9C9C',     # 灰紫 - 次要数据
    'gray': '#949494',       # 灰色 - 背景
    'gold': '#F0E442',       # 金色 - 特殊高亮
    'skyblue': '#56B4E9',    # 天蓝 - 辅助色
}

# 骨干网络专用配色
BACKBONE_COLORS = {
    'LSTM': '#56B4E9',        # 天蓝
    'Mamba': '#CC3311',       # 深红 - Ours (高亮)
    'Transformer': '#009E73', # 深绿
}


# ============================================================================
# 数据加载
# ============================================================================

def find_latest_json(pattern: str) -> str:
    """查找最新的 JSON 文件"""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files[-1]  # 返回最新的


def load_all_results(evaluation_dir: str) -> Dict[str, Dict]:
    """
    加载所有评估结果

    支持两种格式:
    1. 旧格式: evaluation/loss_ablation_*.json
    2. 新格式: evaluation/loss_ablation/GRAB_real-ik_*.json
    """
    results = {}

    # 定义实验类型和对应的子目录
    experiments = {
        'loss': 'loss_ablation',
        'window': 'window_size_ablation',
        'backbone': 'backbone_ablation',
        'layers': 'layers_ablation',
    }

    for key, dirname in experiments.items():
        # 尝试新格式: evaluation/{dirname}/GRAB_real-ik_*.json
        new_pattern = os.path.join(evaluation_dir, dirname, 'GRAB_real-ik_*.json')
        new_files = sorted(glob.glob(new_pattern))

        # 尝试旧格式: evaluation/{dirname}_*.json
        old_pattern = os.path.join(evaluation_dir, f'{dirname}_*.json')
        old_files = sorted(glob.glob(old_pattern))

        # 优先使用新格式
        if new_files:
            with open(new_files[-1]) as f:
                data = json.load(f)
                # 新格式: {"results": {"Model": {...}}}
                if 'results' in data:
                    results[key] = data['results']
                else:
                    results[key] = data
        elif old_files:
            with open(old_files[-1]) as f:
                data = json.load(f)
                results[key] = data

        print(f"  [Loaded] {key}: {len(results.get(key, {}))} configs")

    return results


    # ============================================================================
# 图表 1: 骨干网络帕累托前沿图
# ============================================================================

def plot_pareto_backbone(backbone_data: Dict, output_dir: str):
    """
    生成骨干网络帕累托前沿图 (气泡散点图)

    Science/IEEE 风格:
    - X轴: Inference Latency (ms)
    - Y轴: Elbow Error (mm)
    - 气泡大小: Params (K) - 使用柔和的非线性缩放
    - alpha=0.7 透明度
    - 学术配色: 深红/天蓝/墨绿
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    backbones = ['LSTM', 'Transformer', 'Mamba']

    for backbone in backbones:
        if backbone not in backbone_data:
            continue

        data = backbone_data[backbone]
        latency = data['latency_ms']
        elbow_error = data['elbow_error_mm']
        params = data['params_k']

        # 气泡大小 (使用 log 缩放，更柔和)
        # Params 范围: 1100-3600K, 映射到 100-500
        size = 80 + (params / 1100) ** 0.7 * 120

        # 颜色
        color = BACKBONE_COLORS.get(backbone, COLORS['gray'])

        # 绘制散点
        if backbone == 'Mamba':
            # Ours - 使用星形标记并添加标注
            ax.scatter(latency, elbow_error, s=size, c=color, marker='*',
                      edgecolors='#333333', linewidths=1.5, zorder=5,
                      label=f'{backbone} (Ours)', alpha=0.9)
            ax.annotate('Ours\n(Sweet Spot)', xy=(latency, elbow_error),
                       xytext=(latency + 0.12, elbow_error + 0.25),
                       fontsize=10, fontweight='bold', color=color,
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                      connectionstyle='arc3,rad=0.2'),
                       ha='left')
        else:
            ax.scatter(latency, elbow_error, s=size, c=color, marker='o',
                      edgecolors='white', linewidths=1.5, alpha=0.7,
                      label=backbone)

    # 设置坐标轴 - Science 风格
    ax.set_xlabel('Inference Latency (ms) \u2190', fontsize=12)
    ax.set_ylabel('Elbow Error (mm) \u2190', fontsize=12)
    ax.set_title('Backbone Comparison: Latency vs. Accuracy', fontsize=13, fontweight='bold', pad=10)

    # 添加图例
    legend = ax.legend(loc='upper right', framealpha=0.95, edgecolor='#333333')

    # 添加气泡大小说明
    ax.text(0.02, 0.02, 'Bubble size \u221d Params (K)', transform=ax.transAxes,
           fontsize=9, alpha=0.6, style='italic')
    # 调整布局
    ax.set_xlim(left=0.8)
    ax.set_ylim(bottom=13)
    plt.tight_layout()

    # 保存
    for ext in ['png', 'pdf']:
        output_path = os.path.join(output_dir, f'fig_pareto_backbone.{ext}')
        plt.savefig(output_path, dpi=300 if ext == 'png' else None)
        print(f"[Saved] {output_path}")

    plt.close()


# ============================================================================
# 图表 2: 时序窗口双轴折线图
# ============================================================================

def plot_window_size(window_data: Dict, output_dir: str):
    """
    生成时序窗口双轴折线图

    Science/IEEE 风格:
    - 双 Y 轴: Elbow Error + Jerk
    - X 轴: Window Size (1, 15, 30)
    - 高亮最优配置 W=30
    """
    fig, ax1 = plt.subplots(figsize=(7, 5))

    window_sizes = [1, 15, 30]
    labels = ['W=1', 'W=15', 'W=30']

    # 提取数据
    elbow_errors = []
    jerks = []

    for label in labels:
        if label in window_data:
            elbow_errors.append(window_data[label]['elbow_error_mm'])
            jerks.append(window_data[label]['jerk'])
        else:
            elbow_errors.append(np.nan)
            jerks.append(np.nan)

    # 左 Y 轴: Elbow Error (Science 深蓝)
    color1 = COLORS['blue']
    ax1.set_xlabel('Window Size', fontsize=12)
    ax1.set_ylabel('Elbow Error (mm) \u2190', color=color1, fontsize=12)
    line1 = ax1.plot(window_sizes, elbow_errors, 'o-', color=color1,
                     label='Elbow Error (mm)', linewidth=2.5, markersize=10,
                     markerfacecolor='white', markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(window_sizes)
    ax1.set_xticklabels(labels)

    # 右 Y 轴: Jerk (Science 深橙)
    ax2 = ax1.twinx()
    color2 = COLORS['orange']
    ax2.set_ylabel('Jerk \u2190', color=color2, fontsize=12)
    line2 = ax2.plot(window_sizes, jerks, 's--', color=color2,
                     label='Jerk', linewidth=2.5, markersize=10,
                     markerfacecolor='white', markeredgewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # 高亮 W=30 (最优配置) - 添加红色标注线
    ax1.axvline(x=30, color=COLORS['red'], linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.annotate('Ours', xy=(30, elbow_errors[2]), xytext=(28, elbow_errors[2] - 0.3),
                fontsize=10, color=COLORS['red'], fontweight='bold')
    # 合并图例 (放在底部)
    lines = line1 + line2
    labels_combined = [l.get_label() for l in lines]
    ax1.legend(lines, labels_combined, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, framealpha=0.95, edgecolor='#333333')
    # 标题
    ax1.set_title('Window Size: Accuracy vs. Smoothness Trade-off',
                 fontsize=13, fontweight='bold', pad=10)
    # 添加脚注
    ax1.text(0.02, 0.98, r'$\downarrow$ Smaller window $\rightarrow$ better smoothness',
                  transform=ax1.transAxes, fontsize=9, alpha=0.5, style='italic')

    ax1.text(0.97, 0.02, r'$\downarrow$ Larger window $\rightarrow$ higher accuracy',
                  transform=ax1.transAxes, fontsize=9, alpha=0.5, style='italic',
                  ha='right')
    plt.tight_layout()

    # 保存
    for ext in ['png', 'pdf']:
        output_path = os.path.join(output_dir, f'fig_window_size.{ext}')
        plt.savefig(output_path, dpi=300 if ext == 'png' else None)
        print(f"[Saved] {output_path}")

    plt.close()


# ============================================================================
# 图表 3: 物理内化损失柱状图
# ============================================================================

def plot_loss_ablation(loss_data: Dict, output_dir: str):
    """
    生成物理内化损失柱状图 (双 Y 轴分组柱状图)

    Science/IEEE 风格:
    - X轴类别: Baseline, +Elbow, Ours
    - 柱子 A: Elbow Error (mm)
    - 柱子 B: Jerk
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 数据准备
    categories = ['Baseline\n(Swivel Only)', '+ Elbow\n(Spatial)', '+ Smooth\n(Ours)']
    model_keys = ['Baseline (swivel_only)', 'Variant A (+elbow)', 'Ours (full_loss)']

    elbow_errors = []
    jerks = []

    for key in model_keys:
        if key in loss_data:
            elbow_errors.append(loss_data[key]['elbow_error_mm'])
            jerks.append(loss_data[key]['jerk'])
        else:
            elbow_errors.append(0)
            jerks.append(0)

    x = np.arange(len(categories))
    width = 0.35

    # 左 Y 轴: Elbow Error (Science 深蓝)
    color1 = COLORS['blue']
    bars1 = ax1.bar(x - width/2, elbow_errors, width, label='Elbow Error (mm)',
                   color=color1, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.set_xlabel('Loss Configuration', fontsize=12)
    ax1.set_ylabel('Elbow Error (mm) \u2190', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)

    # 右 Y 轴: Jerk (Science 深绿)
    ax2 = ax1.twinx()
    color2 = COLORS['green']
    bars2 = ax2.bar(x + width/2, jerks, width, label='Jerk',
                   color=color2, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Jerk \u2190', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    # 添加数值标签
    def add_value_labels(bars, ax, color):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9, color=color, fontweight='bold')

    add_value_labels(bars1, ax1, color1)
    add_value_labels(bars2, ax2, color2)

    # 合并图例
    patches = [
        mpatches.Patch(color=color1, label='Elbow Error (mm)', alpha=0.8),
        mpatches.Patch(color=color2, label='Jerk', alpha=0.8)
    ]
    ax1.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, framealpha=0.95, edgecolor='#333333')

    # 标题
    ax1.set_title('Physics-Informed Loss: Spatial & Temporal Constraints',
                 fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()

    # 保存
    for ext in ['png', 'pdf']:
        output_path = os.path.join(output_dir, f'fig_loss_ablation.{ext}')
        plt.savefig(output_path, dpi=300 if ext == 'png' else None)
        print(f"[Saved] {output_path}")

    plt.close()


# ============================================================================
# 图表 4: 层数消融折线图
# ============================================================================

def plot_layers_ablation(layers_data: Dict, output_dir: str):
    """
    生成层数消融折线图

    Science/IEEE 风格:
    - X轴: Number of Layers (2, 3, 4)
    - 左Y轴: Elbow Error (mm)
    - 右Y轴: Jerk
    """
    fig, ax1 = plt.subplots(figsize=(7, 5))

    layers = [2, 3, 4]
    labels = ['L=2', 'L=3', 'L=4']

    # 提取数据
    elbow_errors = []
    jerks = []

    for label in labels:
        if label in layers_data:
            elbow_errors.append(layers_data[label]['elbow_error_mm'])
            jerks.append(layers_data[label]['jerk'])
        else:
            elbow_errors.append(np.nan)
            jerks.append(np.nan)

    # 左 Y 轴: Elbow Error (Science 深蓝)
    color1 = COLORS['blue']
    ax1.set_xlabel('Number of Layers', fontsize=12)
    ax1.set_ylabel('Elbow Error (mm) \u2190', color=color1, fontsize=12)
    line1 = ax1.plot(layers, elbow_errors, 'o-', color=color1,
                     label='Elbow Error (mm)', linewidth=2.5, markersize=10,
                     markerfacecolor='white', markeredgewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(layers)
    ax1.set_xticklabels(labels)
    # 右 Y 轴: Jerk (Science 深橙)
    ax2 = ax1.twinx()
    color2 = COLORS['orange']
    ax2.set_ylabel('Jerk \u2190', color=color2, fontsize=12)
    line2 = ax2.plot(layers, jerks, 's--', color=color2,
                     label='Jerk', linewidth=2.5, markersize=10,
                     markerfacecolor='white', markeredgewidth=2)
    ax2.tick_params(axis='y', labelcolor=color2)
    # 高亮 L=4 (Ours) - 添加红色标注线
    ax1.axvline(x=4, color=COLORS['red'], linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.annotate('Ours', xy=(4, elbow_errors[2]), xytext=(3.6, elbow_errors[2] + 0.3),
                fontsize=10, color=COLORS['red'], fontweight='bold')
    # 合并图例
    lines = line1 + line2
    labels_combined = [l.get_label() for l in lines]
    ax1.legend(lines, labels_combined, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, framealpha=0.95, edgecolor='#333333')

    # 标题
    ax1.set_title('Network Depth: Accuracy vs. Smoothness Trade-off',
                 fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()

    # 保存
    for ext in ['png', 'pdf']:
        output_path = os.path.join(output_dir, f'fig_layers_ablation.{ext}')
        plt.savefig(output_path, dpi=300 if ext == 'png' else None)
        print(f"[Saved] {output_path}")

    plt.close()


# ============================================================================
# 图表 5: 关联分析图 (新增)
# ============================================================================

def plot_correlation_analysis(results: Dict, output_dir: str):
    """
    生成三重相关性分析图

    展示:
    - swivel_error ↔ elbow_error
    - elbow_error ↔ joint_error
    - swivel_error ↔ joint_error
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 收集所有实验的相关性数据
    all_correlations = {
        'swivel_elbow': [],
        'elbow_joint': [],
        'swivel_joint': [],
    }

    # 从各个实验中提取相关性数据
    for exp_key in ['loss', 'window', 'backbone', 'layers']:
        if exp_key not in results:
            continue
        for model_name, model_data in results[exp_key].items():
            if 'correlation_analysis' not in model_data:
                continue
            ca = model_data['correlation_analysis']
            all_correlations['swivel_elbow'].append(ca.get('swivel_elbow_correlation', 0))
            all_correlations['elbow_joint'].append(ca.get('elbow_joint_correlation', 0))
            all_correlations['swivel_joint'].append(ca.get('correlation', 0))

    # 计算统计信息
    means = {k: np.mean(v) if v else 0 for k, v in all_correlations.items()}
    stds = {k: np.std(v) if v else 0 for k, v in all_correlations.items()}

    # 1. swivel ↔ elbow
    ax = axes[0]
    ax.bar(['All Models'], [means['swivel_elbow']], yerr=[stds['swivel_elbow']],
           color=COLORS['blue'], alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    ax.set_ylabel('Correlation (r)', fontsize=11)
    ax.set_title('Swivel Error ↔ Elbow Error', fontsize=12, fontweight='bold')
    ax.set_ylim(0.9, 1.0)
    ax.axhline(y=0.95, color=COLORS['red'], linestyle='--', alpha=0.5, label='Threshold')
    ax.text(0.5, 0.95, f"r = {means['swivel_elbow']:.3f} ± {stds['swivel_elbow']:.3f}",
           transform=ax.get_yaxis_transform(), ha='center', va='bottom',
           fontsize=10, fontweight='bold', color=COLORS['blue'])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. elbow ↔ joint
    ax = axes[1]
    ax.bar(['All Models'], [means['elbow_joint']], yerr=[stds['elbow_joint']],
           color=COLORS['green'], alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    ax.set_ylabel('Correlation (r)', fontsize=11)
    ax.set_title('Elbow Error ↔ Joint Error', fontsize=12, fontweight='bold')
    ax.set_ylim(0.9, 1.0)
    ax.axhline(y=0.95, color=COLORS['red'], linestyle='--', alpha=0.5, label='Threshold')
    ax.text(0.5, 0.95, f"r = {means['elbow_joint']:.3f} ± {stds['elbow_joint']:.3f}",
           transform=ax.get_yaxis_transform(), ha='center', va='bottom',
           fontsize=10, fontweight='bold', color=COLORS['green'])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 3. swivel ↔ joint
    ax = axes[2]
    ax.bar(['All Models'], [means['swivel_joint']], yerr=[stds['swivel_joint']],
           color=COLORS['orange'], alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    ax.set_ylabel('Correlation (r)', fontsize=11)
    ax.set_title('Swivel Error ↔ Joint Error', fontsize=12, fontweight='bold')
    ax.set_ylim(0.8, 1.0)
    ax.axhline(y=0.7, color=COLORS['red'], linestyle='--', alpha=0.5, label='Threshold')
    ax.text(0.5, 0.7, f"r = {means['swivel_joint']:.3f} ± {stds['swivel_joint']:.3f}",
           transform=ax.get_yaxis_transform(), ha='center', va='bottom',
           fontsize=10, fontweight='bold', color=COLORS['orange'])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Error Propagation Correlation Analysis (12 Model Configurations)',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    for ext in ['png', 'pdf']:
        output_path = os.path.join(output_dir, f'fig_correlation_analysis.{ext}')
        plt.savefig(output_path, dpi=300 if ext == 'png' else None)
        print(f"[Saved] {output_path}")

    plt.close()


# ============================================================================
# Markdown 汇总大表
# ============================================================================

def print_master_table(results: Dict):
    """
    打印 Markdown 汇总大表

    融合所有消融实验结果，高亮最终配置
    """
    print("\n" + "=" * 140)
    print("PiM-IK Master Ablation Table")
    print("=" * 140)

    # 表头
    print("\n| Ablation Dimension | Configuration | Params (K) | Latency (ms) | Swivel MAE (\u00b0) \u2193 | Elbow Err (mm) \u2193 | Jerk \u2193 |")
    print("|" + "|".join(["-" * 20] * 7) + "|")
    # Loss Ablation
    if 'loss' in results:
        print("| **Loss** ||||||||")
        loss_order = ['Baseline (swivel_only)', 'Variant A (+elbow)', 'Ours (full_loss)']
        for name in loss_order:
            if name in results['loss']:
                m = results['loss'][name]
                is_ours = 'Ours' in name
                config_name = name.split('(')[1].split(')')[0] if '(' in name else name
                if is_ours:
                    print(f"| | **{config_name}** | **{m['params_k']:.1f}** | **{m['latency_ms']:.2f}** | **{m['swivel_mae']:.2f}** | **{m['elbow_error_mm']:.2f}** | **{m['jerk']:.1f}** |")
                else:
                    print(f"| | {config_name} | {m['params_k']:.1f} | {m['latency_ms']:.2f} | {m['swivel_mae']:.2f} | {m['elbow_error_mm']:.2f} | {m['jerk']:.1f} |")
    # Window Size Ablation
    if 'window' in results:
        print("| **Window Size** ||||||||")
        window_order = ['W=1', 'W=15', 'W=30']
        for name in window_order:
            if name in results['window']:
                m = results['window'][name]
                is_ours = name == 'W=30'
                if is_ours:
                    print(f"| | **{name}** | **{m['params_k']:.1f}** | **{m['latency_ms']:.2f}** | **{m['swivel_mae']:.2f}** | **{m['elbow_error_mm']:.2f}** | **{m['jerk']:.1f}** |")
                else:
                    print(f"| | {name} | {m['params_k']:.1f} | {m['latency_ms']:.2f} | {m['swivel_mae']:.2f} | {m['elbow_error_mm']:.2f} | {m['jerk']:.1f} |")
    # Backbone Ablation
    if 'backbone' in results:
        print("| **Backbone** ||||||||")
        backbone_order = ['LSTM', 'Transformer', 'Mamba']
        for name in backbone_order:
            if name in results['backbone']:
                m = results['backbone'][name]
                is_ours = name == 'Mamba'
                if is_ours:
                    print(f"| | **{name} (Ours)** | **{m['params_k']:.1f}** | **{m['latency_ms']:.2f}** | **{m['swivel_mae']:.2f}** | **{m['elbow_error_mm']:.2f}** | **{m['jerk']:.1f}** |")
                else:
                    print(f"| | {name} | {m['params_k']:.1f} | {m['latency_ms']:.2f} | {m['swivel_mae']:.2f} | {m['elbow_error_mm']:.2f} | {m['jerk']:.1f} |")
    # Layers Ablation
    if 'layers' in results:
        print("| **Layers** ||||||||")
        layers_order = ['L=2', 'L=3', 'L=4']
        for name in layers_order:
            if name in results['layers']:
                m = results['layers'][name]
                is_ours = name == 'L=4'
                if is_ours:
                    print(f"| | **{name} (Ours)** | **{m['params_k']:.1f}** | **{m['latency_ms']:.2f}** | **{m['swivel_mae']:.2f}** | **{m['elbow_error_mm']:.2f}** | **{m['jerk']:.1f}** |")
                else:
                    print(f"| | {name} | {m['params_k']:.1f} | {m['latency_ms']:.2f} | {m['swivel_mae']:.2f} | {m['elbow_error_mm']:.2f} | {m['jerk']:.1f} |")
    print("\n**Legend**: \u2193 indicates lower is better. **Bold** indicates our final configuration.")
    print("\n**Final Configuration (GRAB Test Set)**: Transformer L4 + W=15 + swivel_only")
    print("  - Best Joint MAE: 0.086° (Transformer)")
    print("  - Best Jerk: 1.17 (W=15)")
    print("  - Best Swivel MAE: 5.31° (Transformer)")
    print("=" * 140)


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 项目路径
    project_root = Path(__file__).parent.parent
    evaluation_dir = project_root / 'evaluation'
    output_dir = project_root / 'docs' / 'images'

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("PiM-IK 论文核心图表生成 (IEEE/Science 风格)")
    print("=" * 80)
    print(f"\n[Input] Evaluation directory: {evaluation_dir}")
    print(f"[Output] Images directory: {output_dir}")
    # 设置样式
    setup_matplotlib_style()

    # 加载所有结果
    print("\n[Loading] 评估结果数据...")
    results = load_all_results(str(evaluation_dir))

    print(f"  - Loss Ablation: {len(results.get('loss', {}))} models")
    print(f"  - Window Size: {len(results.get('window', {}))} configs")
    print(f"  - Backbone: {len(results.get('backbone', {}))} models")
    print(f"  - Layers: {len(results.get('layers', {}))} configs")
    # 生成图表
    print("\n[Plotting] 生成图表...")
    if 'backbone' in results:
        print("\n  [1/5] 骨干网络帕累托前沿图...")
        plot_pareto_backbone(results['backbone'], str(output_dir))
    else:
        print("  [1/5] 跳过 - 无骨干网络数据")

    if 'window' in results:
        print("\n  [2/5] 时序窗口双轴折线图...")
        plot_window_size(results['window'], str(output_dir))
    else:
        print("  [2/5] 跳过 - 无窗口大小数据")

    if 'loss' in results:
        print("\n  [3/5] 物理内化损失柱状图...")
        plot_loss_ablation(results['loss'], str(output_dir))
    else:
        print("  [3/5] 跳过 - 无损失消融数据")

    if 'layers' in results:
        print("\n  [4/5] 层数消融折线图...")
        plot_layers_ablation(results['layers'], str(output_dir))
    else:
        print("  [4/5] 跳过 - 无层数消融数据")

    # 关联分析图 (需要相关性数据)
    has_correlation = any(
        'correlation_analysis' in model_data
        for exp_data in results.values()
        for model_data in exp_data.values()
        if isinstance(model_data, dict)
    )
    if has_correlation:
        print("\n  [5/5] 关联分析图...")
        plot_correlation_analysis(results, str(output_dir))
    else:
        print("  [5/5] 跳过 - 无关联分析数据")
    # 打印汇总表格
    print("\n[Table] Markdown 汇总大表...")
    print_master_table(results)

    print("\n" + "=" * 80)
    print("图表生成完成!")
    print(f"输出目录: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
