# Core Module

核心模块包含 PiM-IK 网络定义、运动学层和推理逻辑。

## 文件说明

### `pim_ik_net.py`
**PiM-IK 神经网络定义**

```python
from pim_ik_net import PiM_IK_Net

model = PiM_IK_Net(d_model=256, num_layers=4)
T_ee = torch.randn(1, 30, 4, 4)  # (B, W, 4, 4)
pred_swivel = model(T_ee)        # (B, W, 2) [cos(φ), sin(φ)]
```

**网络结构**：
- 9D 连续位姿表征 (3D 平移 + 6D 旋转)
- Conv1d 时序扩张 (d_conv=4)
- Mamba 状态空间模型 × 4 层
- L2 归一化输出臂角

### `pim_ik_kinematics.py`
**运动学层和物理内化损失**

```python
from pim_ik_kinematics import (
    PhysicsInformedLoss,
    DifferentiableKinematicsLayer,
    TargetGenerator
)

# 损失函数
loss_fn = PhysicsInformedLoss(
    w_swivel=1.0,  # 拟人先验
    w_elbow=1.0,   # 空间约束
    w_smooth=0.1   # 平滑约束
)
```

### `inference.py`
**推理和分层 IK 求解**

```python
from inference import HierarchicalIKSolver, TargetGenerator

# 分层 IK 求解器
solver = HierarchicalIKSolver(model, ee_frame_name='L_ee')
q, info = solver.solve(T_ee_target, p_e_target, q_init)
```

## 使用示例

```python
import torch
from pim_ik_net import PiM_IK_Net
from pim_ik_kinematics import PhysicsInformedLoss

# 创建模型
model = PiM_IK_Net().cuda()

# 创建损失函数
loss_fn = PhysicsInformedLoss().cuda()

# 前向传播
T_ee = torch.randn(8, 30, 4, 4).cuda()
pred = model(T_ee)
loss, loss_dict = loss_fn(pred, gt, p_s, p_w, p_e_gt, L_upper, L_lower, is_valid)
```
