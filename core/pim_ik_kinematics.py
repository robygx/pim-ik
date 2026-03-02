"""
PiM-IK: 物理内化 Mamba 逆运动学网络
可微运动学层与物理内化损失函数

作者: PiM-IK 项目
日期: 2025-02-27

本模块包含:
1. DifferentiableKinematicsLayer: 无参数可微运动学层
2. PhysicsInformedLoss: 物理内化联合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DifferentiableKinematicsLayer(nn.Module):
    """
    无参数可微运动学层

    将网络预测的臂角 [cos(φ), sin(φ)] 结合肩腕坐标和动态臂长，
    通过纯张量运算可微地还原出肘部 3D 坐标。

    几何原理:
    - 肘部在以肩腕连线为主轴的轨道圆上运动
    - 臂角 φ 决定肘部在轨道圆上的角度位置
    - 轨道圆的半径和圆心由上下臂长度及肩腕距离决定（余弦定理）

    输入:
        pred_swivel: (B, W, 2) - 预测臂角 [cos(φ), sin(φ)]
        p_s: (B, W, 3) - 肩部 3D 坐标
        p_w: (B, W, 3) - 腕部 3D 坐标
        L_upper: (B, W) - 上臂长度
        L_lower: (B, W) - 前臂长度

    输出:
        p_e_pred: (B, W, 3) - 预测的肘部 3D 坐标
    """

    # 数值稳定性常数
    EPS: float = 1e-6

    def __init__(self):
        super().__init__()
        # 无参数层，仅定义几何运算

    def forward(
        self,
        pred_swivel: torch.Tensor,
        p_s: torch.Tensor,
        p_w: torch.Tensor,
        L_upper: torch.Tensor,
        L_lower: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：从臂角预测肘部坐标

        Args:
            pred_swivel: (B, W, 2) 预测臂角 [cos(φ), sin(φ)]
            p_s: (B, W, 3) 肩部坐标
            p_w: (B, W, 3) 腕部坐标
            L_upper: (B, W) 上臂长度
            L_lower: (B, W) 前臂长度

        Returns:
            p_e_pred: (B, W, 3) 预测的肘部坐标
        """
        # ============================================================
        # 步骤 1: L2 归一化预测臂角
        # ============================================================
        # 确保预测的 [cos, sin] 满足 cos² + sin² = 1
        pred_swivel_norm = F.normalize(pred_swivel, p=2, dim=-1)  # (B, W, 2)
        cos_phi = pred_swivel_norm[..., 0]  # (B, W)
        sin_phi = pred_swivel_norm[..., 1]  # (B, W)

        # ============================================================
        # 步骤 2: 构建轨道圆坐标系的正交基 (u, v, n)
        # ============================================================
        # 肩腕向量 sw = p_w - p_s
        sw = p_w - p_s  # (B, W, 3)

        # 肩腕距离 ||sw||
        sw_norm = torch.norm(sw, dim=-1)  # (B, W)

        # 主轴 n = sw / ||sw||（单位化，添加 EPS 防止除零）
        n = sw / (sw_norm.unsqueeze(-1) + self.EPS)  # (B, W, 3)

        # ============================================================
        # 构建轨道圆平面的正交基（带回退机制）
        # ============================================================
        # 主参考向量 v_ref = [-1, 0, 0]（指向胸腔正后方）
        v_ref = torch.tensor(
            [-1.0, 0.0, 0.0],
            dtype=sw.dtype,
            device=sw.device
        ).view(1, 1, 3).expand_as(n)  # (B, W, 3)

        # 备用参考向量 v_ref_alt = [0, 1, 0]（当主参考向量接近奇异时使用）
        v_ref_alt = torch.tensor(
            [0.0, 1.0, 0.0],
            dtype=sw.dtype,
            device=sw.device
        ).view(1, 1, 3).expand_as(n)  # (B, W, 3)

        # 计算 X 轴候选向量 u = v_ref - (v_ref · n) * n（Gram-Schmidt 正交化）
        v_ref_dot_n = torch.sum(v_ref * n, dim=-1, keepdim=True)  # (B, W, 1)
        u_candidate = v_ref - v_ref_dot_n * n  # (B, W, 3)
        u_norm = torch.norm(u_candidate, dim=-1, keepdim=True)  # (B, W, 1)

        # 备用向量的 Gram-Schmidt 投影
        v_ref_alt_dot_n = torch.sum(v_ref_alt * n, dim=-1, keepdim=True)  # (B, W, 1)
        u_alt = v_ref_alt - v_ref_alt_dot_n * n  # (B, W, 3)

        # 张量级回退机制：当 u_norm < 1e-5 时自动切换到备用向量
        # 使用 torch.where 保持可微和并行计算
        singularity_mask = (u_norm < 1e-5).expand_as(u_candidate)  # (B, W, 3)
        u_raw = torch.where(singularity_mask, u_alt, u_candidate)  # (B, W, 3)

        # 归一化 X 轴
        u = u_raw / (torch.norm(u_raw, dim=-1, keepdim=True) + self.EPS)  # (B, W, 3)

        # 计算 Y 轴 v = n × u（叉积构建正交右手系）
        v = torch.linalg.cross(n, u, dim=-1)  # (B, W, 3)

        # ============================================================
        # 步骤 3: 利用余弦定理计算轨道圆参数
        # ============================================================
        # 投影距离 d = (L_upper² - L_lower² + ||sw||²) / (2 * ||sw||)
        L_upper_sq = L_upper ** 2  # (B, W)
        L_lower_sq = L_lower ** 2  # (B, W)
        sw_norm_sq = sw_norm ** 2   # (B, W)

        d = (L_upper_sq - L_lower_sq + sw_norm_sq) / (2.0 * sw_norm + self.EPS)  # (B, W)

        # 轨道圆心 p_c = p_s + d * n
        p_c = p_s + d.unsqueeze(-1) * n  # (B, W, 3)

        # 轨道圆半径 R = sqrt(max(L_upper² - d², EPS))
        # 使用 clamp 防止负数开方导致 NaN
        R_sq = torch.clamp(L_upper_sq - d ** 2, min=self.EPS)  # (B, W)
        R = torch.sqrt(R_sq)  # (B, W)

        # ============================================================
        # 步骤 4: 计算预测肘部位置
        # ============================================================
        # p_e = p_c + R * (cos(φ) * u + sin(φ) * v)
        # 扩展 cos_phi, sin_phi, R 为 (B, W, 1) 以便广播
        cos_phi = cos_phi.unsqueeze(-1)  # (B, W, 1)
        sin_phi = sin_phi.unsqueeze(-1)  # (B, W, 1)
        R = R.unsqueeze(-1)              # (B, W, 1)

        # 肘部在轨道圆平面上的偏移向量
        offset = R * (cos_phi * u + sin_phi * v)  # (B, W, 3)

        # 最终肘部坐标
        p_e_pred = p_c + offset  # (B, W, 3)

        return p_e_pred


class PhysicsInformedLoss(nn.Module):
    """
    物理内化联合损失函数

    包含三个子损失的加权组合：
    1. L_swivel: 拟人先验约束 - pred_swivel 与 gt_swivel 的 L1 误差
    2. L_elbow: 三维空间物理约束 - 预测肘部与真实肘部的误差 (支持 MSE/RMSE/L1)
    3. L_smooth: 二阶平滑惩罚 - 惩罚时间维度上的加加速度 (Jerk)

    所有损失在计算均值前会乘以 is_valid mask，屏蔽奇异点。

    Args:
        w_swivel: 拟人先验约束权重 (默认 1.0)
        w_elbow: 三维空间约束权重 (默认 1.0)
        w_smooth: 平滑惩罚权重 (默认 0.1)
        elbow_loss_type: 肘部损失类型 ('mse', 'rmse', 'l1')，默认 'rmse'
            - 'mse': MSE 损失，单位 m²，量级 ~0.0001
            - 'rmse': RMSE 损失，单位 m，量级 ~0.01 (推荐)
            - 'l1': L1 损失，单位 m，量级 ~0.01
    """

    def __init__(
        self,
        w_swivel: float = 1.0,
        w_elbow: float = 1.0,
        w_smooth: float = 0.1,
        elbow_loss_type: str = 'rmse'  # 新增：肘部损失类型
    ):
        super().__init__()
        self.w_swivel = w_swivel
        self.w_elbow = w_elbow
        self.w_smooth = w_smooth
        self.elbow_loss_type = elbow_loss_type.lower()

        # 验证 elbow_loss_type
        assert self.elbow_loss_type in ['mse', 'rmse', 'l1'], \
            f"elbow_loss_type 必须是 'mse', 'rmse' 或 'l1'，当前: {self.elbow_loss_type}"

        # 实例化可微运动学层（用于计算 L_elbow）
        self.kinematics_layer = DifferentiableKinematicsLayer()

    def forward(
        self,
        pred_swivel: torch.Tensor,
        gt_swivel: torch.Tensor,
        p_s: torch.Tensor,
        p_w: torch.Tensor,
        p_e_gt: torch.Tensor,
        L_upper: torch.Tensor,
        L_lower: torch.Tensor,
        is_valid: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算物理内化联合损失

        Args:
            pred_swivel: (B, W, 2) 预测臂角 [cos, sin]
            gt_swivel: (B, W, 2) 真实臂角 [cos, sin]
            p_s: (B, W, 3) 肩部坐标
            p_w: (B, W, 3) 腕部坐标
            p_e_gt: (B, W, 3) 真实肘部坐标
            L_upper: (B, W) 上臂长度
            L_lower: (B, W) 前臂长度
            is_valid: (B, W) 奇异点掩码，True=有效，False=奇异点

        Returns:
            total_loss: 标量总损失
            loss_dict: 各子损失的字典，用于日志记录
        """
        B, W, _ = pred_swivel.shape

        # 确保 is_valid 为浮点型 mask
        if is_valid.dtype == torch.bool:
            valid_mask = is_valid.float()  # (B, W)
        else:
            valid_mask = is_valid.float()  # (B, W)

        # 计算有效样本总数（用于正确归一化）
        num_valid = valid_mask.sum() + 1e-6  # 防止除零

        # ============================================================
        # 损失 1: 拟人先验约束 L_swivel (L1 误差)
        # ============================================================
        # L1 误差对异常值更鲁棒
        swivel_l1 = torch.abs(pred_swivel - gt_swivel)  # (B, W, 2)
        swivel_l1 = swivel_l1.sum(dim=-1)  # (B, W) 每个时间步的总 L1

        # 乘以 mask 后取均值
        L_swivel = (swivel_l1 * valid_mask).sum() / num_valid

        # ============================================================
        # 损失 2: 三维空间物理约束 L_elbow (支持 MSE/RMSE/L1)
        # ============================================================
        # 通过可微运动学层计算预测肘部坐标
        p_e_pred = self.kinematics_layer(pred_swivel, p_s, p_w, L_upper, L_lower)  # (B, W, 3)

        # 根据 elbow_loss_type 计算肘部损失
        if self.elbow_loss_type == 'mse':
            # MSE 误差 (单位: m², 量级 ~0.0001)
            elbow_err = (p_e_pred - p_e_gt) ** 2  # (B, W, 3)
            elbow_err = elbow_err.sum(dim=-1)  # (B, W) 每个时间步的平方误差
            L_elbow = (elbow_err * valid_mask).sum() / num_valid
        elif self.elbow_loss_type == 'rmse':
            # RMSE 误差 (单位: m, 量级 ~0.01)
            # 先计算每个样本的 MSE，再开方
            elbow_sq_err = (p_e_pred - p_e_gt) ** 2  # (B, W, 3)
            elbow_sq_err = elbow_sq_err.sum(dim=-1)  # (B, W) 每个时间步的平方误差
            elbow_rmse = torch.sqrt(elbow_sq_err + 1e-8)  # (B, W) 数值稳定的开方
            L_elbow = (elbow_rmse * valid_mask).sum() / num_valid
        else:  # 'l1'
            # L1 误差 (单位: m, 量级 ~0.01)
            elbow_l1 = torch.abs(p_e_pred - p_e_gt)  # (B, W, 3)
            elbow_l1 = elbow_l1.sum(dim=-1)  # (B, W) 每个时间步的 L1 误差
            L_elbow = (elbow_l1 * valid_mask).sum() / num_valid

        # ============================================================
        # 损失 3: 二阶平滑惩罚 L_smooth (Jerk)
        # ============================================================
        # 计算时间维度 W 上的二阶差分: φ_t - 2*φ_{t-1} + φ_{t-2}
        # 仅当 W >= 3 时才计算

        if W >= 3:
            # 使用切片避免 for 循环
            # 二阶差分: pred[t] - 2*pred[t-1] + pred[t-2]
            phi_t = pred_swivel[:, 2:, :]      # (B, W-2, 2) t 时刻
            phi_t_1 = pred_swivel[:, 1:-1, :]  # (B, W-2, 2) t-1 时刻
            phi_t_2 = pred_swivel[:, :-2, :]   # (B, W-2, 2) t-2 时刻

            # 二阶差分（加速度的变化 = Jerk）
            jerk = phi_t - 2.0 * phi_t_1 + phi_t_2  # (B, W-2, 2)

            # L2 范数的平方
            jerk_sq = (jerk ** 2).sum(dim=-1)  # (B, W-2)

            # 严谨的平滑度掩码：只有连续三帧 (t-2, t-1, t) 都有效，才计算平滑度惩罚
            # 这避免了跨奇异点的无意义差分计算
            mask_t = valid_mask[:, 2:]      # t 时刻掩码
            mask_t_1 = valid_mask[:, 1:-1]  # t-1 时刻掩码
            mask_t_2 = valid_mask[:, :-2]   # t-2 时刻掩码
            valid_mask_smooth = mask_t * mask_t_1 * mask_t_2  # 逻辑与 (B, W-2)
            num_valid_smooth = valid_mask_smooth.sum() + 1e-6

            # 乘以 mask 后取均值
            L_smooth = (jerk_sq * valid_mask_smooth).sum() / num_valid_smooth
        else:
            # 窗口太短，不计算平滑损失
            L_smooth = torch.tensor(0.0, device=pred_swivel.device)

        # ============================================================
        # 加权组合总损失
        # ============================================================
        total_loss = (
            self.w_swivel * L_swivel +
            self.w_elbow * L_elbow +
            self.w_smooth * L_smooth
        )

        # 构建损失字典用于日志
        loss_dict = {
            'L_swivel': L_swivel.item(),
            'L_elbow': L_elbow.item(),
            'L_smooth': L_smooth.item() if isinstance(L_smooth, torch.Tensor) else L_smooth,
            'total_loss': total_loss.item()
        }

        return total_loss, loss_dict


# ============================================================================
# 测试模块
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PiM-IK 可微运动学层与物理内化损失函数测试")
    print("=" * 60)

    # 设置随机种子以确保可复现性
    torch.manual_seed(42)

    # 测试参数
    B = 4   # Batch size
    W = 10  # 时间窗口大小

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # ============================================================
    # 生成随机测试数据
    # ============================================================
    print("\n[1] 生成随机测试数据...")

    # 预测臂角 (需要梯度)
    pred_swivel = torch.randn(B, W, 2, device=device, requires_grad=True)

    # 真实臂角
    gt_swivel = torch.randn(B, W, 2, device=device)
    gt_swivel = F.normalize(gt_swivel, dim=-1)  # 确保是单位向量

    # 肩、腕、肘坐标（模拟真实数据范围，单位：米）
    p_s = torch.randn(B, W, 3, device=device) * 0.1 + torch.tensor([0.0, 0.2, 0.3], device=device)
    p_w = torch.randn(B, W, 3, device=device) * 0.1 + torch.tensor([0.3, 0.4, 0.5], device=device)
    p_e_gt = torch.randn(B, W, 3, device=device) * 0.1 + torch.tensor([0.15, 0.3, 0.4], device=device)

    # 动态臂长（单位：米，约 18cm）
    L_upper = torch.rand(B, W, device=device) * 0.05 + 0.16  # 16-21cm
    L_lower = torch.rand(B, W, device=device) * 0.03 + 0.16  # 16-19cm

    # 奇异点掩码（约 96% 有效）
    is_valid = torch.rand(B, W, device=device) > 0.04  # 布尔型

    print(f"  pred_swivel: {pred_swivel.shape}")
    print(f"  gt_swivel: {gt_swivel.shape}")
    print(f"  p_s: {p_s.shape}")
    print(f"  p_w: {p_w.shape}")
    print(f"  p_e_gt: {p_e_gt.shape}")
    print(f"  L_upper: {L_upper.shape}")
    print(f"  L_lower: {L_lower.shape}")
    print(f"  is_valid: {is_valid.shape}, 有效样本: {is_valid.sum().item()}/{B*W}")

    # ============================================================
    # 测试 DifferentiableKinematicsLayer
    # ============================================================
    print("\n[2] 测试 DifferentiableKinematicsLayer...")

    kinematics_layer = DifferentiableKinematicsLayer()
    p_e_pred = kinematics_layer(pred_swivel, p_s, p_w, L_upper, L_lower)

    print(f"  输出 p_e_pred 形状: {p_e_pred.shape}")
    assert p_e_pred.shape == (B, W, 3), f"形状错误: 期望 {(B, W, 3)}, 实际 {p_e_pred.shape}"
    print("  ✅ 肘部坐标形状正确")

    # 检查是否有 NaN
    assert not torch.isnan(p_e_pred).any(), "检测到 NaN!"
    print("  ✅ 无 NaN 值")

    # ============================================================
    # 测试 PhysicsInformedLoss
    # ============================================================
    print("\n[3] 测试 PhysicsInformedLoss...")

    loss_fn = PhysicsInformedLoss(w_swivel=1.0, w_elbow=1.0, w_smooth=0.1)
    total_loss, loss_dict = loss_fn(
        pred_swivel, gt_swivel, p_s, p_w, p_e_gt,
        L_upper, L_lower, is_valid
    )

    print(f"  L_swivel: {loss_dict['L_swivel']:.6f}")
    print(f"  L_elbow: {loss_dict['L_elbow']:.6f}")
    print(f"  L_smooth: {loss_dict['L_smooth']:.6f}")
    print(f"  total_loss: {loss_dict['total_loss']:.6f}")
    print("  ✅ 损失计算成功")

    # ============================================================
    # 测试梯度回传
    # ============================================================
    print("\n[4] 测试梯度回传...")

    total_loss.backward()

    # 检查 pred_swivel 的梯度
    assert pred_swivel.grad is not None, "梯度为 None!"
    assert not torch.isnan(pred_swivel.grad).any(), "梯度中检测到 NaN!"
    print(f"  pred_swivel.grad 形状: {pred_swivel.grad.shape}")
    print(f"  pred_swivel.grad 范围: [{pred_swivel.grad.min().item():.4f}, {pred_swivel.grad.max().item():.4f}]")
    print("  ✅ 梯度回传成功，无 NaN")

    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "=" * 60)
    print("所有测试通过! ✅")
    print("=" * 60)
