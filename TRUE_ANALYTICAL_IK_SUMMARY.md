# TrueAnalyticalIKSolver 实现总结

## 实现状态

已成功实现 `TrueAnalyticalIKSolver` 类，位于 `/home/ygx/pim-ik/core/g1_analytical_ik.py`。

## 性能指标

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 位置误差 | < 1mm | **0.17mm** | ✅ 优秀 |
| 旋转误差 | < 1° | 0.15° (3/4 测试) | ✅ 大部分 |
| IK 时间 | < 0.1ms | ~20ms | ⚠️ 数值优化 |
| 成功率 | > 95% | 75% (3/4) | ⚠️ 需改进 |

## 核心算法

```
输入: T_ee_target (4x4), swivel_angle [cos(φ), sin(φ)]
输出: q (7 个关节角度)

步骤:
1. 计算目标肘部位置 (使用与 DifferentiableKinematicsLayer 相同的公式)
2. 使用 scipy.optimize.minimize 优化所有 7 个关节
   - 目标函数: 末端位姿误差 + 肘部位置误差
   - 约束: 关节限位
3. 验证精度 (使用 Pinocchio FK)
```

## 测试结果

### 使用正确的 swivel angle 计算方法

```
测试用例: 零位配置
  位置误差: 0.008 mm ✅
  旋转误差: 0.076° ✅
  IK 时间: 17.48 ms

测试用例: 中等伸展
  位置误差: 0.075 mm ✅
  旋转误差: 0.150° ✅
  IK 时间: 21.05 ms

测试用例: 负角度
  位置误差: 0.077 mm ✅
  旋转误差: 0.151° ✅
  IK 时间: 22.16 ms

测试用例: 大角度 (接近完全伸展)
  位置误差: 0.516 mm ✅
  旋转误差: 20.132° ❌ (边界情况)
```

## 关键发现

### 1. Swivel Angle ≠ Shoulder Roll

测试中发现，使用 `shoulder_roll` 作为 `swivel_angle` 是错误的。
正确的 `swivel_angle` 应该从肘部位置计算，使用与 `DifferentiableKinematicsLayer` 相同的公式。

### 2. 准球形手腕处理

G1 的手腕不是标准球形手腕（pitch 和 yaw 轴相距 46mm）。
"偏移吸收"方法（L_eff = 87.5mm）在数值优化中自动处理了这个复杂性。

### 3. 数值 vs 解析

真正的解析 IK（纯几何公式）对于 G1 这种复杂机械臂很难实现。
当前实现使用 scipy 数值优化，虽然比纯解析慢，但精度高且稳定。

## 文件清单

| 文件 | 描述 |
|------|------|
| `core/g1_analytical_ik.py` | TrueAnalyticalIKSolver 类实现 |
| `tests/test_true_analytical_ik.py` | 原始测试 (使用错误的 swivel angle) |
| `tests/test_true_analytical_ik_correct.py` | 改进测试 (使用正确的 swivel angle) |
| `core/pinocchio_ik_solver.py` | Pinocchio IK 参考 (12ms, 0.01mm) |

## 使用方法

```python
from core.g1_analytical_ik import TrueAnalyticalIKSolver
import numpy as np

solver = TrueAnalyticalIKSolver()

# 目标位姿
T_target = np.eye(4)
T_target[:3, 3] = [0.3, 0.2, 0.3]  # 位置

# Swivel angle (从网络预测或肘部位置计算)
swivel_angle = np.array([1.0, 0.0])  # [cos(φ), sin(φ)]

# 求解 IK
q_solved, info = solver.solve(
    T_ee_target=T_target,
    swivel_angle=swivel_angle,
    p_shoulder=p_shoulder,
    p_wrist=p_wrist  # 可选，用于肘部位置约束
)

print(f"位置误差: {info['pos_error_mm']:.3f} mm")
print(f"旋转误差: {info['rot_error_deg']:.3f}°")
```

## 下一步改进

1. **提高成功率**: 处理手臂接近完全伸展的边界情况
2. **加速优化**: 使用更好的初始猜测或更快的优化器
3. **集成到推理管线**: 与 PiM-IK 网络集成测试

## 对比

| IK 求解器 | 位置误差 | 时间 | 特点 |
|-----------|----------|------|------|
| Pinocchio IK | 0.01mm | ~12ms | 精确，稳定 |
| TrueAnalyticalIKSolver | 0.17mm | ~20ms | 兼容 swivel angle |
| HierarchicalIKSolver | - | ~1ms | 快速但依赖 p_e_target |
