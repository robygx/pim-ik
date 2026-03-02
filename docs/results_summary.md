# PiM-IK 消融实验结果总结

> 最新更新: 2026-03-02 - GRAB 数据集随机采样评测 (抗过拟合策略)

---

## 0. 最新评测结果 (2026-03-02)

### 0.1 实验配置

| 配置项 | 值 |
|--------|-----|
| **数据集** | GRAB training data with swivel |
| **采样方式** | 随机采样 50,000 帧 |
| **IK 类型** | Approx (elbow_error / 3) |
| **训练策略** | 抗过拟合 (stride=5, dropout=0.1) |

### 0.2 评估指标

| 指标 | 说明 | 单位 |
|------|------|------|
| **Swivel MAE** | 臂角绝对误差 | 度 (°) |
| **Elbow Error** | 肘部空间误差 | 毫米 (mm) |
| **Jerk** | 动作平滑度 (二阶差分均方) | - |
| **Joint MAE** | 关节角度误差 | 度 (°) |
| **Params** | 参数量 | 千 (K) |

### 0.3 消融实验结果

#### 损失函数消融

| Loss | Params (K) | Swivel MAE (°) | Elbow (mm) | Jerk | Joint MAE (°) |
|------|------------|----------------|------------|------|---------------|
| swivel_only | 1985.7 | 5.93 | 13.17 | **3.30** ✓ | 4.39 |
| **elbow_only** | 1985.7 | **5.75** ✓ | **12.33** ✓ | 6.56 | **4.11** ✓ |
| full_loss | 1985.7 | 5.82 | 12.74 | 5.82 | 4.25 |

**结论**: `elbow_only` 综合表现最优，`swivel_only` 平滑度最佳。

#### 窗口大小消融

| Window | Params (K) | Swivel MAE (°) | Elbow (mm) | Jerk | Joint MAE (°) |
|--------|------------|----------------|------------|------|---------------|
| W=1 | 1985.7 | 6.77 | 14.21 | **4.58** ✓ | 4.74 |
| **W=15** | 1985.7 | **5.79** ✓ | **12.36** ✓ | 8.39 | **4.12** ✓ |
| W=30 | 1985.7 | 6.04 | 12.94 | 5.06 | 4.31 |

**结论**: W=15 精度最优，但 Jerk 异常高（可能过拟合噪声）。

#### 层数消融

| Layers | Params (K) | Swivel MAE (°) | Elbow (mm) | Jerk | Joint MAE (°) |
|--------|------------|----------------|------------|------|---------------|
| L=2 | 1109.1 | 5.90 | 12.79 | 6.34 | 4.26 |
| **L=4** | 1985.7 | **5.64** ✓ | **12.41** ✓ | 6.05 | **4.14** ✓ |
| L=8 | 3738.8 | 5.93 | 13.02 | **3.35** ✓ | 4.34 |

**结论**: L=4 性价比最优，L=8 参数翻倍但精度下降。

#### Backbone 消融

| Backbone | Params (K) | Swivel MAE (°) | Elbow (mm) | Jerk | Joint MAE (°) |
|----------|------------|----------------|------------|------|---------------|
| LSTM | 2337.9 | 5.52 | 12.19 | 4.76 | 4.06 |
| Mamba | 1985.7 | 6.35 | 13.85 | 5.62 | 4.62 |
| **Transformer** | 3520.0 | **5.12** ✓ | **11.21** ✓ | **4.08** ✓ | **3.74** ✓ |

**结论**: **Transformer 全面碾压**，Mamba 表现最差（与预期不符）。

### 0.4 关键发现

1. **Transformer 全面领先** - 在所有指标上最优，Swivel MAE 达 5.12°
2. **Mamba 意外最差** - 可能 GRAB 数据集特性不适合 SSM 建模
3. **elbow_only 最优** - 简单损失反而效果更好
4. **W=15 Jerk 异常** - 时序模型可能过拟合局部噪声

### 0.5 综合推荐配置

| 配置项 | 推荐值 | 理由 |
|--------|--------|------|
| 损失函数 | `elbow_only` | 精度最优 |
| 窗口大小 | `W=15` | 精度最高（需关注 Jerk） |
| 网络层数 | `L=4` | 性价比最优 |
| 骨干网络 | **Transformer** | 全面最优 |

---

## 1. 训练策略：抗过拟合三板斧

### 1.1 滑动窗口步长 (Train Stride)

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--train_stride` | 5 | 降低训练样本相关性 |

**原理**:
- 原来: stride=1，相邻样本 29 帧重叠 (W=30)
- 现在: stride=5，相邻样本 24 帧重叠
- 效果: 训练样本减少 ~80%，但独立性显著提高

**使用方法**:
```bash
torchrun --nproc_per_node=2 training/trainer.py --train_stride 5
```

### 1.2 多级 Dropout

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--dropout` | 0.1 | 多级随机失活 |

**新增 Dropout 层**:
- **Stem 级**: `Dropout1d` 在 Conv1d + GELU 之后
- **Backbone 级**: `Dropout` 在每层 Mamba 输出后
- **Head 级**: 原有的 MLP Dropout

**使用方法**:
```bash
torchrun --nproc_per_node=2 training/trainer.py --dropout 0.1
```

### 1.3 物理噪声数据增强

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--add_noise` | False | 打破坐标记忆 |

**噪声配置**:
- 平移: `N(0, 0.002^2)` → 约 2mm 标准差
- 旋转: `N(0, 0.005^2)` → 极微弱扰动

**使用方法**:
```bash
torchrun --nproc_per_node=2 training/trainer.py --add_noise
```

---

## 2. 评测工具

### 2.1 批量评测脚本

```bash
# 使用 GRAB 数据集，随机采样 50,000 帧
bash evaluation/run_all_evaluations.sh
```

### 2.2 随机采样评测

```bash
# 随机采样（每次结果不同）
python ablation/comprehensive_eval.py \
    --experiment loss_ablation \
    --random_sample \
    --num_frames 50000

# 指定随机种子（可复现）
python ablation/comprehensive_eval.py \
    --experiment loss_ablation \
    --random_sample \
    --seed 42
```

---

> 历史结果: 基于_GRAB 测试集_的全面评估结果 (2026-03-01)

---

## 3. 实验配置 (历史)

### 3.1 数据集

| 属性 | 数值 |
|------|------|
| **测试集** | GRAB training data with swivel |
| **总帧数** | 80,874 frames |
| **IK 类型** | Real IK (HierarchicalIKSolver) |

---

## 4. 实验结果 (历史)

### 4.1 窗口大小消融 (Window Size Ablation)

| 窗口大小 | Params (K) | Latency (ms) | Swivel MAE (°) | Elbow Error (mm) | Jerk | Joint MAE (°) |
|---------|------------|-------------|---------------|-----------------|------|---------------|
| W=1 | 1985.7 | **0.14** | 5.99 | 13.06 | 2.83 | 0.10 |
| **W=15** | **1985.7** | 1.79 | **5.67** ✓ | 12.38 | **1.17** ✓ | **0.09** ✓ |
| W=30 | 1985.7 | 1.78 | 5.67 | 12.33 | 1.86 | 0.09 |

**结论**: W=15 在 Jerk 指标上显著最优 (1.17 vs 1.86)，同时保持精度领先。

---

### 4.2 骨干网络消融 (Backbone Ablation)

| 骨干网络 | Params (K) | Latency (ms) | Swivel MAE (°) | Elbow Error (mm) | Jerk | Joint MAE (°) |
|---------|------------|-------------|---------------|-----------------|------|---------------|
| LSTM | 2337.9 | 1.26 | 6.12 | 13.25 | 1.84 | 0.10 |
| Mamba | 1985.7 | 2.44 | 5.83 | 12.70 | 2.52 | 0.10 |
| **Transformer** | **3519.6** | **1.29** ✓ | **5.31** ✓ | **11.52** ✓ | 2.00 | **0.086** ✓ |

**结论**: Transformer 在所有精度指标上全面领先，泛化能力最强。

---

### 4.3 层数消融 (Layers Ablation)

| 层数 | Params (K) | Latency (ms) | Swivel MAE (°) | Elbow Error (mm) | Jerk | Joint MAE (°) |
|-----|------------|-------------|---------------|-----------------|------|---------------|
| L=2 | 1109.1 | **1.13** ✓ | 5.74 | 12.28 | 2.45 | 0.09 |
| L=3 | 1547.4 | 1.46 | 5.80 | 12.39 | 1.57 | 0.09 |
| **L=4** | **1985.7** | 2.52 | **5.62** ✓ | **12.22** ✓ | 1.70 | **0.09** ✓ |

**结论**: L=4 在精度指标上最优。

---

### 4.4 损失函数消融 (Loss Ablation)

| 损失配置 | Params (K) | Latency (ms) | Swivel MAE (°) | Elbow Error (mm) | Jerk | Joint MAE (°) |
|---------|------------|-------------|---------------|-----------------|------|---------------|
| **Baseline (swivel_only)** | **1985.7** | 1.83 | **5.47** ✓ | **11.99** ✓ | **2.04** ✓ | **0.09** ✓ |
| Variant A (+elbow) | 1985.7 | **1.76** ✓ | 6.03 | 12.85 | 2.46 | 0.10 |
| Ours (full_loss) | 1985.7 | 1.81 | 5.82 | 12.56 | 2.44 | 0.09 |

**结论**: swivel_only 在所有指标上表现最佳。

---

## 5. 关联分析结果 (历史)

### 5.1 三重相关性分析 (全部 12 个模型配置)

| 关联性 | 相关系数范围 | 平均值 | 标准差 | 诊断 |
|--------|-------------|--------|--------|------|
| **swivel ↔ elbow** | 0.946 ~ 0.968 | 0.960 | 0.007 | ✓ 转换正常 |
| **elbow ↔ joint** | 0.970 ~ 0.979 | 0.976 | 0.004 | ✓ IK 正常 |
| **swivel ↔ joint** | 0.856 ~ 0.910 | 0.886 | 0.021 | ✓ 误差传递确认 |

### 5.2 误差传递链路

```
模型预测 swivel 有误差
    ↓ r = 0.96 ✓ TargetGenerator 正常
肘部位置计算误差
    ↓ r = 0.98 ✓ HierarchicalIKSolver 正常
关节角度误差 → VR 抽动
```

---

## 6. 文件位置

### 评估结果 JSON

```
evaluation/ablation_anti/
├── loss_ablation_results.json
├── window_size_ablation_results.json
├── layers_ablation_results.json
└── backbone_ablation_results.json
```

### 可视化图表

```
docs/images/
├── fig_window_size.pdf/png
├── fig_layers_ablation.pdf/png
├── fig_loss_ablation.pdf/png
├── fig_backbone_ablation.pdf/png
└── fig_correlation_analysis.pdf/png
```
