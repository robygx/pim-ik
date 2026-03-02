"""
PiM-IK: 物理内化 Mamba 逆运动学网络
Mamba 时序特征提取网络与 6D 连续位姿表征

作者: PiM-IK 项目
日期: 2025-02-27

本模块包含:
1. transform_to_9d: 将 4x4 齐次变换矩阵转换为 9D 连续位姿表征
2. PiM_IK_Net: 基于 Mamba 的时序特征提取网络

依赖安装:
    pip install mamba-ssm causal-conv1d

参考文献:
    Zhou et al., "On the Continuity of Rotation Representations", CVPR 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 尝试导入 mamba_ssm，如果失败则提供提示
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("警告: mamba_ssm 未安装。请运行: pip install mamba-ssm causal-conv1d")


# ============================================================================
# 数据预处理模块: 6D 连续位姿表征
# ============================================================================

def transform_to_9d(T_ee: torch.Tensor) -> torch.Tensor:
    """
    将 4x4 齐次变换矩阵转换为 9D 连续位姿表征

    基于 CVPR 2019 论文《On the Continuity of Rotation Representations》，
    使用 6D 连续旋转表征 + 3D 平移，避免四元数/欧拉角的奇异性问题。

    Args:
        T_ee: (B, W, 4, 4) 或 (B, 4, 4) 齐次变换矩阵
              其中 B=Batch, W=时间窗口大小

    Returns:
        x_9d: (B, W, 9) 或 (B, 9) 连续位姿表征
              前 3 维: 平移向量 [x, y, z]
              后 6 维: 6D 旋转表征 (3x3 旋转矩阵的前两列展平)
    """
    # 处理不同维度的输入
    if T_ee.dim() == 3:
        # (B, 4, 4) -> (B, 9)
        translation = T_ee[:, :3, 3]           # (B, 3)
        # 注意: 切片后必须 .contiguous() 确保内存连续，否则 view 会报错
        rotation_6d = T_ee[:, :3, :2].contiguous()  # (B, 3, 2)
        rotation_6d = rotation_6d.view(-1, 6)  # (B, 6)
    elif T_ee.dim() == 4:
        # (B, W, 4, 4) -> (B, W, 9)
        translation = T_ee[:, :, :3, 3]        # (B, W, 3)
        # 注意: 切片后必须 .contiguous() 确保内存连续，否则 view 会报错
        rotation_6d = T_ee[:, :, :3, :2].contiguous()  # (B, W, 3, 2)
        rotation_6d = rotation_6d.view(*rotation_6d.shape[:2], 6)  # (B, W, 6)
    else:
        raise ValueError(f"T_ee 维度错误: 期望 3 或 4，实际 {T_ee.dim()}")

    # 拼接平移和旋转表征
    x_9d = torch.cat([translation, rotation_6d], dim=-1)  # (B, W, 9) 或 (B, 9)

    return x_9d


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """
    将 6D 旋转表征转换回 3x3 旋转矩阵（用于验证或可视化）

    Args:
        rotation_6d: (..., 6) 6D 旋转表征

    Returns:
        rotation_matrix: (..., 3, 3) 旋转矩阵
    """
    # 重塑为 (..., 3, 2)
    shape = rotation_6d.shape[:-1]
    a1 = rotation_6d[..., :3]  # 第一列
    a2 = rotation_6d[..., 3:]  # 第二列

    # Gram-Schmidt 正交化
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    # 堆叠为旋转矩阵
    rotation_matrix = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)

    return rotation_matrix


# ============================================================================
# Mamba 网络模块
# ============================================================================

class MambaBlock(nn.Module):
    """
    单个 Mamba 块，包含 LayerNorm + Mamba + 残差连接
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        """
        Args:
            d_model: 模型维度
            d_state: SSM 状态维度（默认 16）
            d_conv: 局部卷积核大小（默认 4）
            expand: 扩展因子（默认 2）
        """
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm 未安装，请运行: pip install mamba-ssm causal-conv1d")

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, W, d_model) 输入序列

        Returns:
            x: (B, W, d_model) 输出序列
        """
        # Pre-Norm + Mamba + 残差连接
        return x + self.mamba(self.norm(x))


class PiM_IK_Net(nn.Module):
    """
    物理内化 Mamba 逆运动学网络

    使用 Mamba/LSTM/Transformer 架构处理时序末端位姿数据，预测连续臂角 [cos(φ), sin(φ)]

    网络结构:
    1. Stem: 浅层特征映射 (Linear + Conv1d + GELU)
    2. Backbone: 堆叠的 Mamba 块 / 单向 LSTM / Causal Transformer (带残差连接)
    3. Output Head: 意图解码 MLP + L2 归一化

    Args:
        d_model: 模型隐空间维度（默认 256）
        num_layers: 骨干网络堆叠层数（默认 4）
        d_state: Mamba SSM 状态维度（默认 16）
        d_conv: Mamba 卷积核大小（默认 4）
        expand: Mamba 扩展因子（默认 2）
        backbone_type: 骨干网络类型，可选 'mamba'/'lstm'/'transformer'（默认 'mamba'）
        dropout: Dropout 概率（默认 0.1）
    """

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        backbone_type: str = 'mamba',
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.dropout = dropout

        # ================================================================
        # 【抗过拟合】多级 Dropout 模块
        # ================================================================
        # Dropout: 用于 Backbone 和 Head（全连接层）
        self.dropout_layer = nn.Dropout(dropout)
        # Dropout1d: 用于 Stem（Conv1d 后的时序特征）
        self.dropout1d = nn.Dropout1d(dropout)

        # ================================================================
        # 1. Stem: 浅层特征映射
        # ================================================================
        # 将 9D 输入投影到高维隐空间
        self.stem_linear = nn.Linear(9, d_model)

        # Conv1d 进行局部时序平滑
        # 注意: Conv1d 期望 (B, C, W)，需要 permute
        # 使用 padding_mode='replicate' 边缘复制，避免零填充引入高频噪声
        # （零填充相当于机械臂瞬移到原点，不符合连续运动假设）
        self.stem_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,  # 保持序列长度
            padding_mode='replicate'  # 边缘复制：边界帧保持静止，而非瞬移到0
        )

        self.stem_act = nn.GELU()

        # ================================================================
        # 2. Backbone: 根据 backbone_type 选择不同的骨干网络
        # ================================================================
        if backbone_type == 'mamba':
            if not MAMBA_AVAILABLE:
                raise ImportError("backbone_type='mamba' 需要 mamba_ssm，请运行: pip install mamba-ssm causal-conv1d")
            self.mamba_blocks = nn.ModuleList([
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                for _ in range(num_layers)
            ])

        elif backbone_type == 'lstm':
            # 单向 LSTM，确保因果性（严禁使用 bidirectional=True）
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True
            )

        elif backbone_type == 'transformer':
            # 可学习位置编码 (500 足以覆盖所有可能的窗口大小)
            self.pos_embedding = nn.Parameter(torch.zeros(1, 500, d_model))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)

            # Causal Transformer 编码器
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                batch_first=True,
                norm_first=True  # Pre-LN: 训练更稳定，对齐现代 Transformer 标准
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )

        else:
            raise ValueError(f"不支持的 backbone_type: {backbone_type}，请选择 'mamba'/'lstm'/'transformer'")

        # ================================================================
        # 3. Output Head: 意图解码 MLP
        # ================================================================
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(p=dropout),  # 正则化：防止过拟合
            nn.Linear(128, 2)  # 输出 [cos(φ), sin(φ)]
        )

    def forward(self, T_ee: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            T_ee: (B, W, 4, 4) 末端位姿齐次变换矩阵
                  B = Batch size
                  W = 时间窗口大小

        Returns:
            pred_swivel: (B, W, 2) 预测的臂角 [cos(φ), sin(φ)]
                         已归一化为单位向量（每个时间步）

        Note:
            当 W == 1 时，网络作为无时序记忆的 MLP 基线运行（跳过 Conv1d 和 Mamba）。
            这是用于消融实验，验证时序建模的重要性。
        """
        B, W, _, _ = T_ee.shape

        # ================================================================
        # Step 1: 数据预处理 - 4x4 矩阵转 9D 连续表征
        # ================================================================
        x = transform_to_9d(T_ee)  # (B, W, 9)

        # ================================================================
        # Step 2: Stem - 浅层特征映射
        # ================================================================
        # Linear 投影
        x = self.stem_linear(x)  # (B, W, d_model)

        # ================================================================
        # 【消融实验】条件分支：根据窗口大小决定是否使用时序组件
        # ================================================================
        if W == 1:
            # ============================================================
            # W=1: 无时序记忆基线 (消融实验用)
            # 跳过 Conv1d 和 Mamba，直接通过 MLP
            # 这代表一个纯粹的单帧推理网络，无法利用历史信息
            # ============================================================
            x = self.stem_act(x)  # (B, 1, d_model)
        else:
            # ============================================================
            # W>1: 完整时序网络 (Conv1d + Mamba)
            # ============================================================
            # Conv1d 时序平滑 (需要 permute)
            x = x.permute(0, 2, 1)   # (B, d_model, W)
            x = self.stem_conv(x)    # (B, d_model, W)
            x = x.permute(0, 2, 1)   # (B, W, d_model)

            # GELU 激活
            x = self.stem_act(x)     # (B, W, d_model)

            # 【抗过拟合】Stem 级 Dropout（在 channel 维度）
            # 先转为 (B, C, W) 格式应用 Dropout1d，再转回
            x = x.permute(0, 2, 1)   # (B, d_model, W)
            x = self.dropout1d(x)    # (B, d_model, W)
            x = x.permute(0, 2, 1)   # (B, W, d_model)

            # ============================================================
            # Step 3: Backbone - 根据 backbone_type 路由
            # ============================================================
            if self.backbone_type == 'mamba':
                # Mamba 堆叠
                for mamba_block in self.mamba_blocks:
                    x = mamba_block(x)   # (B, W, d_model) with residual
                    # 【抗过拟合】Backbone 级 Dropout（每层 Mamba 后）
                    x = self.dropout_layer(x)

            elif self.backbone_type == 'lstm':
                # 单向 LSTM (因果性保证)
                x, _ = self.lstm(x)  # (B, W, d_model)

            elif self.backbone_type == 'transformer':
                # 添加位置编码
                x = x + self.pos_embedding[:, :W, :]

                # 生成因果掩码 (防止未来信息泄露)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    W, device=x.device
                )  # (W, W)

                # 前向传播
                x = self.transformer(x, mask=causal_mask, is_causal=True)

        # ================================================================
        # Step 4: Output Head - 全时间窗口解码
        # ================================================================
        # 直接对整个时间窗口进行解码，输出形状 (B, W, 2)
        # 这样 PhysicsInformedLoss 可以计算整个窗口的平滑惩罚
        out = self.head(x)  # (B, W, 2)

        # ================================================================
        # Step 5: L2 归一化 - 强制输出为单位向量
        # ================================================================
        # 在最后一个维度归一化，确保每个时间步满足 cos² + sin² = 1
        pred_swivel = F.normalize(out, p=2, dim=-1)  # (B, W, 2)

        return pred_swivel

    def forward_from_9d(self, x_9d: torch.Tensor) -> torch.Tensor:
        """
        直接从 9D 表征进行前向传播（跳过数据预处理）

        用于已预处理的输入或调试

        Args:
            x_9d: (B, W, 9) 9D 连续位姿表征

        Returns:
            pred_swivel: (B, W, 2) 预测的臂角 [cos(φ), sin(φ)]
        """
        # Stem
        x = self.stem_linear(x_9d)
        x = x.permute(0, 2, 1)
        x = self.stem_conv(x)
        x = x.permute(0, 2, 1)
        x = self.stem_act(x)

        # 【抗过拟合】Stem 级 Dropout
        x = x.permute(0, 2, 1)
        x = self.dropout1d(x)
        x = x.permute(0, 2, 1)

        # Backbone
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
            x = self.dropout_layer(x)  # 【抗过拟合】Backbone 级 Dropout

        # Output Head - 全时间窗口解码
        out = self.head(x)  # (B, W, 2)

        # L2 归一化
        pred_swivel = F.normalize(out, p=2, dim=-1)  # (B, W, 2)

        return pred_swivel


# ============================================================================
# 测试模块
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PiM-IK Mamba 网络架构测试")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(42)

    # 测试参数
    B = 4   # Batch size
    W = 30  # 时间窗口大小

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # ================================================================
    # 测试 1: 数据预处理模块
    # ================================================================
    print("\n[1] 测试 transform_to_9d 数据预处理...")

    # 生成随机齐次变换矩阵
    # 为了模拟真实的变换矩阵，我们构建有效的旋转矩阵
    T_ee = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
    T_ee = T_ee.expand(B, W, -1, -1).clone()  # (B, W, 4, 4)

    # 添加随机扰动（模拟真实数据）
    T_ee = T_ee + torch.randn(B, W, 4, 4, device=device) * 0.1

    # 转换为 9D 表征
    x_9d = transform_to_9d(T_ee)

    print(f"  输入 T_ee 形状: {T_ee.shape}")
    print(f"  输出 x_9d 形状: {x_9d.shape}")
    assert x_9d.shape == (B, W, 9), f"形状错误: 期望 {(B, W, 9)}, 实际 {x_9d.shape}"
    print("  ✅ 9D 表征形状正确")

    # ================================================================
    # 测试 2: Mamba 网络前向传播
    # ================================================================
    print("\n[2] 测试 PiM_IK_Net 前向传播...")

    if not MAMBA_AVAILABLE:
        print("  ⚠️ 跳过网络测试: mamba_ssm 未安装")
        print("  请运行: pip install mamba-ssm causal-conv1d")
    else:
        # 实例化网络
        model = PiM_IK_Net(
            d_model=256,
            num_layers=4,
        ).to(device)

        # 统计参数量
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  模型参数量: {num_params:,}")

        # 前向传播
        T_ee_test = torch.randn(B, W, 4, 4, device=device)
        pred_swivel = model(T_ee_test)

        print(f"  输出 pred_swivel 形状: {pred_swivel.shape}")
        assert pred_swivel.shape == (B, 2), f"形状错误: 期望 {(B, 2)}, 实际 {pred_swivel.shape}"
        print("  ✅ 预测臂角形状正确")

        # ================================================================
        # 测试 3: L2 范数验证
        # ================================================================
        print("\n[3] 测试 L2 范数约束...")

        l2_norm = torch.norm(pred_swivel, p=2, dim=-1)
        print(f"  L2 范数: {l2_norm}")

        # 验证所有范数接近 1
        ones = torch.ones(B, device=device)
        assert torch.allclose(l2_norm, ones, atol=1e-5), \
            f"L2 范数不为 1: {l2_norm}"
        print("  ✅ L2 范数验证通过 (所有值 ≈ 1.0)")

        # ================================================================
        # 测试 4: 梯度回传
        # ================================================================
        print("\n[4] 测试梯度回传...")

        T_ee_grad = torch.randn(B, W, 4, 4, device=device, requires_grad=True)
        pred = model(T_ee_grad)
        loss = pred.sum()
        loss.backward()

        assert T_ee_grad.grad is not None, "梯度为 None!"
        assert not torch.isnan(T_ee_grad.grad).any(), "梯度中检测到 NaN!"
        print(f"  T_ee.grad 形状: {T_ee_grad.grad.shape}")
        print("  ✅ 梯度回传成功，无 NaN")

    # ================================================================
    # 完成
    # ================================================================
    print("\n" + "=" * 60)
    print("所有测试通过! ✅")
    print("=" * 60)
