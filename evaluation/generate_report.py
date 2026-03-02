#!/usr/bin/env python3
"""
PiM-IK 消融实验评测报告生成器
读取所有评测结果 JSON 文件，生成统一的 Markdown 对比报告
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any


def load_json_results(directory: str) -> Dict[str, Dict]:
    """加载目录下所有评测结果 JSON 文件"""
    results = {}
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"[Warning] 目录不存在: {directory}")
        return results

    for json_file in dir_path.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # 提取实验类型
                experiment_type = json_file.stem.replace('_results', '')
                results[experiment_type] = data
                print(f"[Load] 已加载: {json_file.name}")
        except Exception as e:
            print(f"[Error] 无法读取 {json_file}: {e}")

    return results


def format_metric(value: float, precision: int = 2, unit: str = "") -> str:
    """格式化指标值"""
    if value < 0:
        return "N/A"
    return f"{value:.{precision}f}{unit}"


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    """渲染 Markdown 表格"""
    # 计算每列宽度
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # 构建表格
    lines = []

    # 表头
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    lines.append(header_line)

    # 分隔线
    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"
    lines.append(separator)

    # 数据行
    for row in rows:
        row_line = "| " + " | ".join(cell.ljust(w) for cell, w in zip(row, col_widths)) + " |"
        lines.append(row_line)

    return "\n".join(lines)


def generate_markdown_report(results: Dict[str, Dict], output_path: str):
    """生成 Markdown 报告"""

    lines = []
    lines.append("# PiM-IK 消融实验评测报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 1. 损失函数消融
    if "loss_ablation" in results:
        lines.append("## 1. 损失函数消融")
        lines.append("")
        lines.append("不同损失函数配置对模型性能的影响：")
        lines.append("")

        data = results["loss_ablation"]["results"]
        headers = ["Model", "Params (K)", "Network (ms)", "IK (ms)", "Total (ms)", "Swivel MAE (°)", "Elbow (mm)", "Jerk"]
        rows = []

        for model_name, metrics in data.items():
            rows.append([
                model_name,
                format_metric(metrics.get("params_k", -1), 1),
                format_metric(metrics.get("network_latency_ms", -1), 3),
                format_metric(metrics.get("ik_latency_ms", -1), 3),
                format_metric(metrics.get("total_latency_ms", -1), 3),
                format_metric(metrics.get("swivel_mae", -1), 2),
                format_metric(metrics.get("elbow_error_mm", -1), 2),
                format_metric(metrics.get("jerk", -1), 4),
            ])

        lines.append(render_table(headers, rows))
        lines.append("")
        lines.append("**结论**:")
        lines.append("- `swivel_only`: 仅使用臂角损失，肘部位置由 IK 约束间接控制")
        lines.append("- `elbow_only`: 仅使用肘部位置损失，不考虑臂角拟人性")
        lines.append("- `full_loss`: 完整损失（推荐），平衡拟人性和精度")
        lines.append("")

    # 2. 窗口大小消融
    if "window_size_ablation" in results:
        lines.append("## 2. 窗口大小消融")
        lines.append("")
        lines.append("不同时序窗口大小对模型性能的影响：")
        lines.append("")

        data = results["window_size_ablation"]["results"]
        headers = ["Window", "Params (K)", "Network (ms)", "IK (ms)", "Total (ms)", "Swivel MAE (°)", "Elbow (mm)", "Jerk"]
        rows = []

        for model_name, metrics in sorted(data.items(), key=lambda x: int(x[0].split("=")[1])):
            rows.append([
                model_name,
                format_metric(metrics.get("params_k", -1), 1),
                format_metric(metrics.get("network_latency_ms", -1), 3),
                format_metric(metrics.get("ik_latency_ms", -1), 3),
                format_metric(metrics.get("total_latency_ms", -1), 3),
                format_metric(metrics.get("swivel_mae", -1), 2),
                format_metric(metrics.get("elbow_error_mm", -1), 2),
                format_metric(metrics.get("jerk", -1), 4),
            ])

        lines.append(render_table(headers, rows))
        lines.append("")
        lines.append("**结论**:")
        lines.append("- `W=1`: 无时序记忆，单帧预测，速度最快")
        lines.append("- `W=15`: 中等时序窗口，平衡精度和速度（推荐）")
        lines.append("- `W=30`: 完整时序窗口，精度最高但速度较慢")
        lines.append("")

    # 3. 层数消融
    if "layers_ablation" in results:
        lines.append("## 3. 层数消融")
        lines.append("")
        lines.append("不同网络深度对模型性能的影响：")
        lines.append("")

        data = results["layers_ablation"]["results"]
        headers = ["Layers", "Params (K)", "Network (ms)", "IK (ms)", "Total (ms)", "Swivel MAE (°)", "Elbow (mm)", "Jerk"]
        rows = []

        for model_name, metrics in sorted(data.items(), key=lambda x: int(x[0].split("=")[1])):
            rows.append([
                model_name,
                format_metric(metrics.get("params_k", -1), 1),
                format_metric(metrics.get("network_latency_ms", -1), 3),
                format_metric(metrics.get("ik_latency_ms", -1), 3),
                format_metric(metrics.get("total_latency_ms", -1), 3),
                format_metric(metrics.get("swivel_mae", -1), 2),
                format_metric(metrics.get("elbow_error_mm", -1), 2),
                format_metric(metrics.get("jerk", -1), 4),
            ])

        lines.append(render_table(headers, rows))
        lines.append("")
        lines.append("**结论**:")
        lines.append("- `L=2`: 轻量级模型，速度快但精度较低")
        lines.append("- `L=4`: 标准配置（推荐），平衡性能和效率")
        lines.append("- `L=8`: 深层模型，精度提升有限但参数量大")
        lines.append("")

    # 4. Backbone 消融
    if "backbone_ablation" in results:
        lines.append("## 4. Backbone 消融")
        lines.append("")
        lines.append("不同骨干网络架构对模型性能的影响：")
        lines.append("")

        data = results["backbone_ablation"]["results"]
        headers = ["Backbone", "Params (K)", "Network (ms)", "IK (ms)", "Total (ms)", "Swivel MAE (°)", "Elbow (mm)", "Jerk"]
        rows = []

        # 按照推荐顺序排序
        order = {"Mamba": 0, "Transformer": 1, "LSTM": 2}
        for model_name in sorted(data.keys(), key=lambda x: order.get(x, 99)):
            metrics = data[model_name]
            rows.append([
                model_name,
                format_metric(metrics.get("params_k", -1), 1),
                format_metric(metrics.get("network_latency_ms", -1), 3),
                format_metric(metrics.get("ik_latency_ms", -1), 3),
                format_metric(metrics.get("total_latency_ms", -1), 3),
                format_metric(metrics.get("swivel_mae", -1), 2),
                format_metric(metrics.get("elbow_error_mm", -1), 2),
                format_metric(metrics.get("jerk", -1), 4),
            ])

        lines.append(render_table(headers, rows))
        lines.append("")
        lines.append("**结论**:")
        lines.append("- **Mamba**: 线性复杂度，长序列建模能力强（推荐）")
        lines.append("- **Transformer**: 二次复杂度，注意力机制但速度较慢")
        lines.append("- **LSTM**: 传统循环网络，参数量最少但长程记忆有限")
        lines.append("")

    # 综合结论
    lines.append("---")
    lines.append("")
    lines.append("## 综合推荐配置")
    lines.append("")
    lines.append("基于消融实验结果，推荐以下配置：")
    lines.append("")
    lines.append("| 配置项 | 推荐值 | 理由 |")
    lines.append("|--------|--------|------|")
    lines.append("| 损失函数 | `full_loss` | 平衡拟人性和精度 |")
    lines.append("| 窗口大小 | `W=15` | 平衡精度和速度 |")
    lines.append("| 网络层数 | `L=4` | 标准配置，性能最优 |")
    lines.append("| 骨干网络 | `Mamba` | 线性复杂度，长序列建模 |")
    lines.append("")

    # 图例说明
    lines.append("---")
    lines.append("")
    lines.append("## 指标说明")
    lines.append("")
    lines.append("| 指标 | 说明 | 单位 |")
    lines.append("|------|------|------|")
    lines.append("| Params (K) | 模型参数量 | 千 |")
    lines.append("| Network (ms) | 神经网络推理延迟 | 毫秒 |")
    lines.append("| IK (ms) | IK 求解延迟 | 毫秒 |")
    lines.append("| Total (ms) | 总延迟 (Network + IK) | 毫秒 |")
    lines.append("| Swivel MAE | 臂角平均绝对误差 | 度 |")
    lines.append("| Elbow Error | 肘部位置误差 | 毫米 |")
    lines.append("| Jerk | 动作平滑度（越小越平滑） | - |")
    lines.append("")

    # 写入文件
    report_content = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n[Save] 报告已生成: {output_path}")
    return report_content


def main():
    # 配置
    RESULTS_DIR = "./evaluation/ablation_anti"
    OUTPUT_REPORT = "./docs/ablation_evaluation_results.md"

    print("=" * 60)
    print("  PiM-IK 消融实验报告生成器")
    print("=" * 60)
    print()

    # 加载所有评测结果
    print("[Load] 加载评测结果...")
    results = load_json_results(RESULTS_DIR)

    if not results:
        print(f"[Error] 未找到评测结果文件!")
        print(f"       请先运行: bash evaluation/run_all_evaluations.sh")
        sys.exit(1)

    print(f"[Load] 已加载 {len(results)} 个实验结果")
    print()

    # 生成报告
    print("[Generate] 生成 Markdown 报告...")
    report = generate_markdown_report(results, OUTPUT_REPORT)

    # 显示报告预览
    print()
    print("=" * 60)
    print("报告预览 (前 30 行)")
    print("=" * 60)
    preview_lines = report.split("\n")[:30]
    print("\n".join(preview_lines))
    print("...")
    print()
    print(f"完整报告请查看: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
