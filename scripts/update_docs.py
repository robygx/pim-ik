#!/usr/bin/env python3
"""
Documentation Update Script / 文档自动更新脚本

当代码更改或有新的训练/实验时，运行此脚本同步更新 README 文档。

Usage:
    python scripts/update_docs.py              # 更新所有文档
    python scripts/update_docs.py --core       # 仅更新 core/README.md
    python scripts/update_docs.py --experiments # 仅更新实验结果
"""

import argparse
import ast
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


# ============================================================================
# Configuration / 配置
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
EXPERIMENTS_MD = DOCS_DIR / "experiments.md"


# ============================================================================
# Parameter Extraction / 参数提取
# ============================================================================

def extract_function_params(file_path: str, func_name: str) -> Dict[str, Any]:
    """从函数定义中提取参数默认值"""
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.args:
            params = {}
            for arg, default in zip(node.args.args, node.args.defaults):
                # 只处理有默认值的参数
                if default is not None:
                    param_name = arg.arg
                    if isinstance(default, ast.Constant):
                        param_value = default.value
                    elif isinstance(default, ast.Num):  # Python < 3.8
                        param_value = default.n
                    elif isinstance(default, ast.Str):  # Python < 3.8
                        param_value = default.s
                    elif isinstance(default, ast.NameConstant):
                        param_value = default.value
                    elif isinstance(default, ast.UnaryOp) and isinstance(default.op, ast.USub):
                        if isinstance(default.operand, (ast.Num, ast.Constant)):
                            param_value = -default.operand.n if hasattr(default.operand, 'n') else -default.operand.value
                        else:
                            param_value = ast.unparse(default)
                    else:
                        param_value = ast.unparse(default) if hasattr(ast, 'unparse') else str(type(default).__name__)
                    params[param_name] = param_value
            return params
    return {}


def extract_class_init_params(file_path: str, class_name: str) -> Dict[str, Any]:
    """从类 __init__ 方法中提取参数默认值"""
    with open(file_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                    params = {}
                    # 跳过 self 参数
                    args = item.args.args[1:]
                    defaults = item.args.defaults
                    # 对齐参数和默认值
                    for i, arg in enumerate(args):
                        default_idx = i - len(args) + len(defaults)
                        if default_idx >= 0:
                            default = defaults[default_idx]
                            param_name = arg.arg
                            if isinstance(default, ast.Constant):
                                param_value = default.value
                            elif isinstance(default, ast.Num):
                                param_value = default.n
                            elif isinstance(default, ast.Str):
                                param_value = default.s
                            elif isinstance(default, ast.NameConstant):
                                param_value = default.value
                            elif isinstance(default, ast.UnaryOp) and isinstance(default.op, ast.USub):
                                if isinstance(default.operand, (ast.Num, ast.Constant)):
                                    param_value = -default.operand.n if hasattr(default.operand, 'n') else -default.operand.value
                                else:
                                    param_value = f"-{ast.unparse(default.operand) if hasattr(ast, 'unparse') else ''}"
                            else:
                                param_value = ast.unparse(default) if hasattr(ast, 'unparse') else str(type(default).__name__)
                            params[param_name] = param_value
                    return params
    return {}


# ============================================================================
# Module Update Functions / 模块更新函数
# ============================================================================

def update_core_readme():
    """更新 core/README.md - 从 PiM_IK_Net 提取参数"""
    print("\n[Core] Updating core/README.md...")

    params = extract_class_init_params(
        str(PROJECT_ROOT / "core/pim_ik_net.py"),
        "PiM_IK_Net"
    )

    readme_path = PROJECT_ROOT / "core/README.md"
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 更新参数表格
    param_table = "| 参数 | 默认值 | 说明 |\n|------|--------|------|\n"
    param_descriptions = {
        'd_model': '隐藏层维度',
        'num_layers': 'Mamba 堆叠层数',
        'd_state': 'SSM 状态维度',
        'd_conv': '卷积核大小',
        'expand': '扩展因子',
        'backbone': '主干网络类型',
    }

    for param, desc in param_descriptions.items():
        value = params.get(param, 'N/A')
        param_table += f"| `{param}` | {value} | {desc} |\n"

    # 替换参数表格部分
    pattern = r'\| 参数 \| 默认值 \|\n?\|------.*?\n?\|.*?\n?(?:\|.*?\n?)*'
    content = re.sub(
        r'(?<=### 构造参数\n\n)(.*?)(?=\n\n###)',
        param_table,
        content,
        flags=re.DOTALL
    )

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ Updated parameters: {list(params.keys())}")


def update_training_readme():
    """更新 training/README.md - 从 trainer.py 提取参数"""
    print("\n[Training] Updating training/README.md...")

    params = extract_function_params(
        str(PROJECT_ROOT / "training/trainer.py"),
        "main"
    )

    readme_path = PROJECT_ROOT / "training/README.md"
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 更新参数表格
    param_table = "| 参数 | 默认值 | 说明 |\n|------|--------|------|\n"
    param_descriptions = {
        'data_path': '数据集路径',
        'window_size': '时序窗口大小',
        'epochs': '训练轮数',
        'batch_size': '批次大小',
        'lr': '学习率',
        'd_model': '模型维度',
        'num_layers': '网络层数',
    }

    for param, desc in param_descriptions.items():
        value = params.get(param, 'N/A')
        if value == 'N/A' and param == 'data_path':
            value = '`/data0/.../ACCAD_CMU_merged_training_data_with_swivel.npz`'
        param_table += f"| `--{param}` | {value} | {desc} |\n"

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)  # 临时保存原内容

    print(f"  ✓ Found {len(params)} parameters")


def update_experiments_results(results: Dict[str, Any] = None):
    """更新实验结果 - experiments.md"""
    print("\n[Experiments] Updating docs/experiments.md...")

    if results is None:
        results = {
            'window_size': {
                'W=30': {'mae': 15.87, 'jerk': 0.304, 'jerk_reduction': 75},
                'W=15': {'mae': 12.14, 'jerk': 0.240, 'jerk_reduction': 80},
                'W=1': {'mae': 14.41, 'jerk': 1.207, 'jerk_reduction': 0},
            },
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }

    readme_path = EXPERIMENTS_MD
    if not readme_path.exists():
        print("  ⚠ experiments.md not found, skipping...")
        return

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 更新窗口大小实验结果表格
    ws_results = results.get('window_size', {})
    if ws_results:
        table_header = "| 窗口大小 | Swivel MAE (°) | Jerk | Jerk 降低率 |\n|----------|---------------|------|-------------|"
        table_rows = ""
        for w in ['W=30', 'W=15', 'W=1']:
            r = ws_results.get(w, {})
            table_rows += f"| {w} | {r.get('mae', '-')} | {r.get('jerk', '-')} | ↓ {r.get('jerk_reduction', 0)}% |\n"

        # 替换 2.1 节的结果表格
        pattern = r'(?<=#### Results / 结果\n\n)(.*?)(?=\n\n#### Analysis)'
        new_table = table_header + "\n" + table_rows
        content = re.sub(pattern, new_table, content, flags=re.DOTALL)

    # 更新日期
    content = re.sub(
        r'(?<=\*实验记录更新日期: )\d{4}-\d{2}-\d{2}',
        datetime.now().strftime('%Y-%m-%d'),
        content
    )

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ Updated experiment results")


# ============================================================================
# Changelog Generation / 更新日志生成
# ============================================================================

def generate_changelog(days: int = 7) -> str:
    """从 Git 历史生成更新日志"""
    print(f"\n[Changelog] Generating changelog for last {days} days...")

    since = datetime.now().strftime('%Y-%m-%d')
    cmd = f"git log --since='{since}' --pretty=format:'%h|%s|%an|%ad' --date=short"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)

    if not result.stdout:
        return "## 最近更新\n\n暂无更新记录\n"

    lines = result.stdout.strip().split('\n')
    changelog = "## 最近更新 / Recent Changes\n\n"

    for line in lines:
        parts = line.split('|')
        if len(parts) >= 3:
            commit_hash, message, author = parts[0], parts[1], parts[2]
            changelog += f"- **{message}** (`{commit_hash}` by {author})\n"

    return changelog


# ============================================================================
# Main Function / 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Update project documentation")
    parser.add_argument('--core', action='store_true', help='Update core/README.md only')
    parser.add_argument('--training', action='store_true', help='Update training/README.md only')
    parser.add_argument('--experiments', action='store_true', help='Update experiments.md only')
    parser.add_argument('--changelog', action='store_true', help='Generate changelog from git')
    parser.add_argument('--all', action='store_true', help='Update all documentation')

    args = parser.parse_args()

    print("=" * 60)
    print("PiM-IK Documentation Update Tool / 文档自动更新工具")
    print("=" * 60)

    # 如果没有指定任何选项，更新所有
    if not (args.core or args.training or args.experiments or args.changelog):
        args.all = True

    if args.all or args.core:
        update_core_readme()

    if args.all or args.training:
        update_training_readme()

    if args.all or args.experiments:
        update_experiments_results()

    if args.all or args.changelog:
        changelog = generate_changelog()
        print("\n" + changelog)

    print("\n" + "=" * 60)
    print("✓ Documentation update complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
