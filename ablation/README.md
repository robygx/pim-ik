# Ablation Module

消融实验模块包含用于验证模型各组件有效性的脚本和评估工具。

---

## 综合评估脚本

### `comprehensive_eval.py`

遵循全局评估标准，支持真实 IK 求解和关联分析。

#### 基本用法

```bash
# 基础评估 (近似方法)
python ablation/comprehensive_eval.py --experiment loss_ablation

# 真实 IK 评估
python ablation/comprehensive_eval.py \
  --experiment loss_ablation \
  --use-real-ik

# 关联分析 (VR 抽动诊断)
python ablation/comprehensive_eval.py \
  --experiment loss_ablation \
  --use-real-ik \
  --analyze-correlation \
  --sample-ratio 0.05
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--experiment` | 实验类型 (loss/window_size/backbone/layers) | 必需 |
| `--data_path` | 数据集路径 | GRAB 测试集 |
| `--num_frames` | 评估帧数 (None=全部) | None |
| `--use-real-ik` | 使用真实 IK 求解器 | False |
| `--analyze-correlation` | 启用关联分析 | False |
| `--sample-ratio` | 关联分析采样比例 | 0.1 |

#### 关联分析输出

```
evaluation/{experiment}/
├── GRAB_real-ik_*.json              # 评估结果
├── error_correlation_{model}.png     # 时间序列图
├── error_scatter_{model}.png         # 散点图
├── swivel_elbow_scatter_{model}.png  # Swivel-Elbow 散点图
├── lag_correlation_{model}.png       # 滞后相关图
└── error_timeseries_{model}.npz      # 原始数据
```

---

## 实验配置

### 1. 时序窗口消融 (Window Size Ablation)

验证 Mamba 时序建模对运动平滑度的影响。

| 窗口大小 | 说明 |
|---------|------|
| W=1 | 单帧预测，无时序信息 |
| W=15 | 中等时序窗口 |
| W=30 | 较长时序窗口 |

### 2. 骨干网络消融 (Backbone Ablation)

验证不同时序建模架构的性能。

| 网络 | 说明 |
|------|------|
| LSTM | 循环神经网络基线 |
| Mamba | 状态空间模型 (本项目) |
| Transformer | 自注意力基线 |

### 3. 层数消融 (Layers Ablation)

验证网络深度对性能的影响。

| 层数 | 参数量 (K) |
|-----|-----------|
| L=2 | 1109 |
| L=3 | 1547 |
| L=4 | 1985 |

### 4. 损失函数消融 (Loss Ablation)

验证各损失组件的贡献。

| 配置 | swivel | elbow | smooth |
|------|--------|-------|--------|
| Baseline (swivel_only) | 1.0 | 0.0 | 0.0 |
| Variant A (+elbow) | 1.0 | 1.0 | 0.0 |
| Ours (full_loss) | 1.0 | 1.0 | 0.1 |

---

## 实验结果 (GRAB 测试集)

### 最优配置

| 组件 | 选择 | 理由 |
|------|------|------|
| Backbone | **Transformer** | Joint MAE = 0.086° |
| Layers | **L=4** | 精度最优 |
| Window | **W=15** | Jerk = 1.17 |
| Loss | **swivel_only** | 综合最优 |

### 关联分析结论

```
模型预测 swivel 有误差
    ↓ r = 0.96 ✓ TargetGenerator 正常
肘部位置计算误差
    ↓ r = 0.98 ✓ HierarchicalIKSolver 正常
关节角度误差 → VR 抽动
```

**诊断**: VR 抽动由模型预测 swivel 误差导致，IK 求解器工作正常。

---

## 绘图脚本

### `plot_paper_figures.py`

生成符合 IEEE/Science 标准的论文图表。

```bash
cd /home/ygx/ygx_hl_ik_v2
python ablation/plot_paper_figures.py
```

输出图表：
- `fig_window_size.pdf/png`
- `fig_layers_ablation.pdf/png`
- `fig_loss_ablation.pdf/png`
- `fig_pareto_backbone.pdf/png`
- `fig_correlation_analysis.pdf/png` (新增)

---

## 使用示例

### 评估单个模型

```python
from ablation.comprehensive_eval import main as eval_main

import sys
sys.argv = [
    'comprehensive_eval.py',
    '--experiment', 'loss_ablation',
    '--use-real-ik',
    '--analyze-correlation'
]

eval_main()
```

### 加载评估结果

```python
import json

with open('evaluation/loss_ablation/GRAB_real-ik_*.json') as f:
    data = json.load(f)

results = data['results']
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Swivel MAE: {metrics['swivel_mae']:.2f}°")
    print(f"  Joint MAE: {metrics['joint_mae']:.2f}°")

    if 'correlation_analysis' in metrics:
        ca = metrics['correlation_analysis']
        print(f"  Correlation: r = {ca['correlation']:.3f}")
```
