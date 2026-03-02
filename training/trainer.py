"""
PiM-IK: 物理内化 Mamba 逆运动学网络
工业级 DDP 分布式训练脚本

作者: PiM-IK 项目
日期: 2025-02-27

================================================================================
使用说明 (torchrun 启动)
================================================================================

【双卡训练】
torchrun --nproc_per_node=2 trainer.py

【单卡调试】
torchrun --nproc_per_node=1 trainer.py

【四卡训练】
torchrun --nproc_per_node=4 trainer.py

【指定 GPU】
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 trainer.py

================================================================================
依赖安装
================================================================================

pip install wandb transformers  # transformers 用于 get_cosine_schedule_with_warmup
pip install mamba-ssm causal-conv1d  # Mamba 依赖

================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR

import wandb

# 添加 core 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))

# 导入自定义模块
from pim_ik_net import PiM_IK_Net, transform_to_9d
from pim_ik_kinematics import PhysicsInformedLoss


# ============================================================================
# DDP 环境初始化与销毁
# ============================================================================

def setup() -> int:
    """
    初始化 DDP 分布式环境

    Returns:
        local_rank: 当前进程的本地 rank
    """
    # 从环境变量获取 rank 信息（torchrun 自动设置）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 初始化进程组
    dist.init_process_group(
        backend="nccl",  # 使用 NCCL 后端（GPU 通信最优）
        init_method="env://"
    )

    # 设置当前设备
    torch.cuda.set_device(local_rank)

    # 同步所有进程
    dist.barrier()

    if local_rank == 0:
        print(f"[DDP] 初始化完成: world_size={world_size}, local_rank={local_rank}")

    return local_rank


def cleanup():
    """销毁 DDP 分布式环境"""
    dist.destroy_process_group()


def parse_args():
    """
    解析命令行参数

    用于消融实验，支持不同窗口大小的训练

    Returns:
        args: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='PiM-IK 训练脚本 - 支持窗口长度消融实验',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据相关
    parser.add_argument('--data_path', type=str,
                        default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz',
                        help='训练数据路径 (.npz 文件)')
    parser.add_argument('--window_size', type=int, default=30,
                        choices=[1, 15, 30],
                        help='时序窗口大小 (消融实验: 30=完整Mamba, 15=中等记忆, 1=无记忆基线)')
    parser.add_argument('--train_stride', type=int, default=5,
                        help='【抗过拟合】训练集滑动窗口步长，降低相邻样本重叠。验证集自动保持 stride=1 (默认 5)')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='每卡批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减 (L2 正则化)')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='学习率预热轮数')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪最大范数')
    parser.add_argument('--t0', type=int, default=10,
                        help='CosineAnnealingWarmRestarts 首次重启周期 (epochs)')
    parser.add_argument('--t_mult', type=int, default=2,
                        help='CosineAnnealingWarmRestarts 重启周期乘子')

    # 模型相关
    parser.add_argument('--d_model', type=int, default=256,
                        help='Mamba 隐空间维度')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Mamba 堆叠层数')
    parser.add_argument('--backbone', type=str, default='mamba',
                        choices=['mamba', 'lstm', 'transformer'],
                        help='骨干网络类型 (消融实验: mamba/lstm/transformer)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 概率')

    # 保存与日志
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--wandb_project', type=str, default='PiM-IK',
                        help='WandB 项目名称')
    parser.add_argument('--no_wandb', action='store_true',
                        help='禁用 WandB 日志')

    # 损失函数权重 (消融实验)
    parser.add_argument('--w_swivel', type=float, default=1.0,
                        help='拟人先验约束权重 L_swivel')
    parser.add_argument('--w_elbow', type=float, default=1.0,
                        help='三维空间约束权重 L_elbow')
    parser.add_argument('--w_smooth', type=float, default=0.01,
                        help='时序平滑惩罚权重 L_smooth')
    parser.add_argument('--elbow_loss_type', type=str, default='rmse',
                        choices=['mse', 'rmse', 'l1'],
                        help='肘部损失类型: mse(m²量级), rmse(m量级,推荐), l1(m量级)')

    # 【抗过拟合】数据增强
    parser.add_argument('--add_noise', action='store_true',
                        help='【抗过拟合】在训练集 T_ee 上添加物理噪声（平移 2mm，旋转 0.005），打破坐标记忆')

    return parser.parse_args()


# ============================================================================
# 数据集定义
# ============================================================================

class SwivelSequenceDataset(Dataset):
    """
    滑动窗口时序数据集

    全量载入 .npz 数据到 CPU 内存，按滑动窗口切片返回样本。
    训练/验证集划分：前 95% 训练，后 5% 验证（不打乱，防止时序穿越）。

    【抗过拟合】支持步长采样，避免相邻样本过度重叠：
        - 训练集：使用 stride > 1，降低样本相关性（例如 stride=5，相邻样本仅 25 帧重叠）
        - 验证集：保持 stride=1，确保评估的连续性和完整性

    Args:
        npz_path: .npz 数据文件路径
        window_size: 时间窗口大小（默认 30）
        train: True 为训练集，False 为验证集
        stride: 滑动窗口步长（默认 1）。训练集建议使用 3-5，验证集必须使用 1
    """

    def __init__(
        self,
        npz_path: str,
        window_size: int = 30,
        train: bool = True,
        stride: int = 1
    ):
        self.window_size = window_size
        self.train = train
        self.stride = stride

        # ================================================================
        # 全量载入数据到 CPU 内存 (~5GB)
        # ================================================================
        if dist.get_rank() == 0:
            print(f"[Dataset] 正在加载数据: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        # 提取所需数组，一次性转换为 float32
        # 牺牲约 5GB 内存换取极致的 IO 速度
        self.T_ee = data['T_ee'].astype(np.float32)                    # (N, 4, 4)
        self.swivel_angle = data['swivel_angle'].astype(np.float32)    # (N, 2)
        self.joint_positions = data['joint_positions'].astype(np.float32)  # (N, 3, 3) [p_s, p_e, p_w]
        self.L_upper = data['L_upper'].astype(np.float32)              # (N,)
        self.L_lower = data['L_lower'].astype(np.float32)              # (N,)
        self.is_valid = data['is_valid'].astype(np.float32)            # (N,)

        self.total_frames = len(self.T_ee)

        if dist.get_rank() == 0:
            print(f"[Dataset] 总帧数: {self.total_frames:,}")

        # ================================================================
        # 训练/验证集划分（不打乱，防止时序穿越）
        # ================================================================
        train_split = int(self.total_frames * 0.95)

        if train:
            self.start_idx = 0
            self.end_idx = train_split
        else:
            self.start_idx = train_split
            self.end_idx = self.total_frames

        # ================================================================
        # 【抗过拟合】样本数计算：考虑步长
        # ================================================================
        # 可用样本数 = floor(可用帧数 - 窗口大小) / stride + 1
        available_frames = self.end_idx - self.start_idx - window_size
        self.num_samples = available_frames // self.stride + 1

        if dist.get_rank() == 0:
            split_name = "训练集" if train else "验证集"
            stride_info = f", stride={self.stride}" if self.stride > 1 else ""
            print(f"[Dataset] {split_name}: {self.num_samples:,} 样本 (帧 {self.start_idx} - {self.end_idx}{stride_info})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引 (0 到 num_samples-1)

        Returns:
            dict: 包含以下张量的字典
                - T_ee: (W, 4, 4) 末端位姿矩阵
                - gt_swivel: (W, 2) 真实臂角 [cos, sin]
                - p_s: (W, 3) 肩部坐标
                - p_e_gt: (W, 3) 肘部坐标
                - p_w: (W, 3) 腕部坐标
                - L_upper: (W,) 上臂长度
                - L_lower: (W,) 前臂长度
                - is_valid: (W,) 有效性掩码
        """
        # ================================================================
        # 【抗过拟合】按步长计算实际起始帧索引
        # ================================================================
        start_frame = self.start_idx + idx * self.stride
        end_frame = start_frame + self.window_size

        # 提取窗口数据（已在 __init__ 中转换为 float32）
        T_ee = torch.from_numpy(self.T_ee[start_frame:end_frame])
        gt_swivel = torch.from_numpy(self.swivel_angle[start_frame:end_frame])
        joint_pos = torch.from_numpy(self.joint_positions[start_frame:end_frame])
        L_upper = torch.from_numpy(self.L_upper[start_frame:end_frame])
        L_lower = torch.from_numpy(self.L_lower[start_frame:end_frame])
        is_valid = torch.from_numpy(self.is_valid[start_frame:end_frame])

        # 从 joint_positions 提取各关节坐标
        # joint_positions: (W, 3, 3) -> [肩, 肘, 腕]
        p_s = joint_pos[:, 0, :]   # (W, 3) 肩部
        p_e_gt = joint_pos[:, 1, :]  # (W, 3) 肘部
        p_w = joint_pos[:, 2, :]   # (W, 3) 腕部

        return {
            'T_ee': T_ee,           # (W, 4, 4)
            'gt_swivel': gt_swivel,  # (W, 2)
            'p_s': p_s,             # (W, 3)
            'p_e_gt': p_e_gt,       # (W, 3)
            'p_w': p_w,             # (W, 3)
            'L_upper': L_upper,     # (W,)
            'L_lower': L_lower,     # (W,)
            'is_valid': is_valid    # (W,)
        }


def collate_fn(batch: Dict) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数，将样本堆叠为 batch

    Args:
        batch: List of dicts from __getitem__

    Returns:
        dict: 堆叠后的张量字典，增加 batch 维度
    """
    return {
        'T_ee': torch.stack([item['T_ee'] for item in batch], dim=0),
        'gt_swivel': torch.stack([item['gt_swivel'] for item in batch], dim=0),
        'p_s': torch.stack([item['p_s'] for item in batch], dim=0),
        'p_e_gt': torch.stack([item['p_e_gt'] for item in batch], dim=0),
        'p_w': torch.stack([item['p_w'] for item in batch], dim=0),
        'L_upper': torch.stack([item['L_upper'] for item in batch], dim=0),
        'L_lower': torch.stack([item['L_lower'] for item in batch], dim=0),
        'is_valid': torch.stack([item['is_valid'] for item in batch], dim=0),
    }


# ============================================================================
# 训练与验证函数
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: PhysicsInformedLoss,
    device: torch.device,
    local_rank: int,
    epoch: int,
    use_wandb: bool = True,
    add_noise: bool = False
) -> Dict[str, float]:
    """
    训练一个 Epoch

    Args:
        model: DDP 包装的模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        loss_fn: 物理内化损失函数
        device: 设备
        local_rank: 本地 rank
        epoch: 当前 epoch
        use_wandb: 是否使用 WandB 日志
        add_noise: 是否添加物理噪声（抗过拟合）

    Returns:
        avg_metrics: 平均损失指标字典
    """
    model.train()

    total_loss = 0.0
    total_swivel = 0.0
    total_elbow = 0.0
    total_smooth = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # ================================================================
        # 将数据移动到 GPU
        # ================================================================
        T_ee = batch['T_ee'].to(device)           # (B, W, 4, 4)
        gt_swivel = batch['gt_swivel'].to(device)  # (B, W, 2)
        p_s = batch['p_s'].to(device)             # (B, W, 3)
        p_e_gt = batch['p_e_gt'].to(device)       # (B, W, 3)
        p_w = batch['p_w'].to(device)             # (B, W, 3)
        L_upper = batch['L_upper'].to(device)     # (B, W)
        L_lower = batch['L_lower'].to(device)     # (B, W)
        is_valid = batch['is_valid'].to(device)   # (B, W)

        # ================================================================
        # 【抗过拟合】物理噪声数据增强
        # ================================================================
        if add_noise:
            # 平移部分添加高斯噪声 N(0, 0.002^2)，约 2mm 标准差
            T_ee[:, :, :3, 3] += torch.randn_like(T_ee[:, :, :3, 3]) * 0.002

            # 旋转矩阵（3x3 部分）添加极微弱高斯噪声 N(0, 0.005^2)
            # 只扰动前两列，保持旋转矩阵连续性
            T_ee[:, :, :3, :2] += torch.randn_like(T_ee[:, :, :3, :2]) * 0.005

        # ================================================================
        # 前向传播
        # ================================================================
        pred_swivel = model(T_ee)  # (B, W, 2)

        # 计算损失
        loss, loss_dict = loss_fn(
            pred_swivel=pred_swivel,
            gt_swivel=gt_swivel,
            p_s=p_s,
            p_w=p_w,
            p_e_gt=p_e_gt,
            L_upper=L_upper,
            L_lower=L_lower,
            is_valid=is_valid
        )

        # ================================================================
        # 反向传播
        # ================================================================
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪（防止 Mamba 梯度爆炸）
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # 每个 batch 更新学习率

        # ================================================================
        # 累积指标
        # ================================================================
        total_loss += loss_dict['total_loss']
        total_swivel += loss_dict['L_swivel']
        total_elbow += loss_dict['L_elbow']
        total_smooth += loss_dict['L_smooth']
        num_batches += 1

        # WandB 日志（仅 Rank 0，每 100 步记录一次）
        if use_wandb and local_rank == 0 and batch_idx % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                'train/loss_step': loss_dict['total_loss'],
                'train/lr': current_lr,
                'train/step': epoch * len(dataloader) + batch_idx
            })

    # 计算平均值
    avg_metrics = {
        'loss': total_loss / num_batches,
        'swivel': total_swivel / num_batches,
        'elbow': total_elbow / num_batches,
        'smooth': total_smooth / num_batches
    }

    return avg_metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: PhysicsInformedLoss,
    device: torch.device
) -> Dict[str, float]:
    """
    验证集评估

    Args:
        model: DDP 包装的模型
        dataloader: 验证数据加载器
        loss_fn: 物理内化损失函数
        device: 设备

    Returns:
        avg_metrics: 平均损失指标字典
    """
    model.eval()

    total_loss = 0.0
    total_swivel = 0.0
    total_elbow = 0.0
    total_smooth = 0.0
    num_batches = 0

    for batch in dataloader:
        # 移动到 GPU
        T_ee = batch['T_ee'].to(device)
        gt_swivel = batch['gt_swivel'].to(device)
        p_s = batch['p_s'].to(device)
        p_e_gt = batch['p_e_gt'].to(device)
        p_w = batch['p_w'].to(device)
        L_upper = batch['L_upper'].to(device)
        L_lower = batch['L_lower'].to(device)
        is_valid = batch['is_valid'].to(device)

        # 前向传播
        pred_swivel = model(T_ee)

        # 计算损失
        loss, loss_dict = loss_fn(
            pred_swivel=pred_swivel,
            gt_swivel=gt_swivel,
            p_s=p_s,
            p_w=p_w,
            p_e_gt=p_e_gt,
            L_upper=L_upper,
            L_lower=L_lower,
            is_valid=is_valid
        )

        # 累积指标
        total_loss += loss_dict['total_loss']
        total_swivel += loss_dict['L_swivel']
        total_elbow += loss_dict['L_elbow']
        total_smooth += loss_dict['L_smooth']
        num_batches += 1

    # 计算平均值（DDP 同步）
    # 转为 tensor 以便 all_reduce
    avg_loss = torch.tensor(total_loss / num_batches, device=device)
    avg_swivel = torch.tensor(total_swivel / num_batches, device=device)
    avg_elbow = torch.tensor(total_elbow / num_batches, device=device)
    avg_smooth = torch.tensor(total_smooth / num_batches, device=device)

    # 跨卡同步并求平均
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_swivel, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_elbow, op=dist.ReduceOp.SUM)
    dist.all_reduce(avg_smooth, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()

    avg_metrics = {
        'loss': (avg_loss / world_size).item(),
        'swivel': (avg_swivel / world_size).item(),
        'elbow': (avg_elbow / world_size).item(),
        'smooth': (avg_smooth / world_size).item()
    }

    return avg_metrics


# ============================================================================
# DDP 环境初始化与销毁
# ============================================================================


# ============================================================================
# 主训练流程
# ============================================================================

def main():
    # ================================================================
    # 解析命令行参数
    # ================================================================
    args = parse_args()

    # ================================================================
    # 超参数配置 (从命令行参数构建)
    # ================================================================
    CONFIG = {
        # 数据
        'data_path': args.data_path,
        'window_size': args.window_size,
        'train_stride': args.train_stride,  # 【抗过拟合】训练集步长
        'add_noise': args.add_noise,         # 【抗过拟合】物理噪声增强

        # 训练
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'grad_clip': args.grad_clip,
        't0': args.t0,
        't_mult': args.t_mult,

        # 模型
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'backbone': args.backbone,
        'dropout': args.dropout,

        # 损失函数权重
        'w_swivel': args.w_swivel,
        'w_elbow': args.w_elbow,
        'w_smooth': args.w_smooth,
        'elbow_loss_type': args.elbow_loss_type,

        # 保存
        'save_dir': args.save_dir,
        'wandb_project': args.wandb_project,
    }

    # ================================================================
    # DDP 初始化
    # ================================================================
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    # ================================================================
    # WandB 初始化（仅 Rank 0）
    # ================================================================
    # 构建损失权重标识字符串 (用于 WandB 和保存目录命名)
    loss_tag = f"sw{args.w_swivel}_el{args.w_elbow}_sm{args.w_smooth}"
    # 构建骨干网络标识
    backbone_tag = args.backbone
    # 构建层数标识
    layers_tag = f"L{args.num_layers}"

    if local_rank == 0:
        if not args.no_wandb:
            wandb.init(
                project=CONFIG['wandb_project'],
                config=CONFIG,
                name=f"PiM-IK-{backbone_tag}-{layers_tag}-W{args.window_size}-{loss_tag}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print(f"[WandB] 初始化完成: {wandb.run.name}")
        else:
            print("[WandB] 已禁用 (使用 --no_wandb)")

        # 创建带时间戳、骨干网络、层数、窗口大小和损失权重的保存目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(CONFIG['save_dir'], f"{backbone_tag}_{layers_tag}_W{args.window_size}_loss_{loss_tag}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        CONFIG['save_dir'] = save_dir  # 更新配置

        print(f"[Save] 检查点保存目录: {save_dir}")
        print(f"[Config] 窗口大小: W={args.window_size}, 损失权重: swivel={args.w_swivel}, elbow={args.w_elbow}, smooth={args.w_smooth}")

    # ================================================================
    # 数据集与数据加载器
    # ================================================================
    if local_rank == 0:
        print("[Data] 创建数据集...")

    train_dataset = SwivelSequenceDataset(
        npz_path=CONFIG['data_path'],
        window_size=CONFIG['window_size'],
        train=True,
        stride=CONFIG.get('train_stride', 5)  # 训练集使用步长采样
    )

    val_dataset = SwivelSequenceDataset(
        npz_path=CONFIG['data_path'],
        window_size=CONFIG['window_size'],
        train=False,
        stride=1  # 验证集必须保持 stride=1，确保评估连续性
    )

    # 分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True  # DDP 训练建议 drop_last
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    if local_rank == 0:
        print(f"[Data] 训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

    # ================================================================
    # 模型、优化器、损失函数
    # ================================================================
    if local_rank == 0:
        print("[Model] 创建模型...")

    model = PiM_IK_Net(
        d_model=CONFIG['d_model'],
        num_layers=CONFIG['num_layers'],
        backbone_type=CONFIG['backbone'],
        dropout=CONFIG['dropout']
    ).to(device)

    # DDP 包装
    # find_unused_parameters=True 允许部分参数不参与梯度计算 (消融实验 W=1 时需要)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    # 统计参数量
    if local_rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] 可训练参数量: {num_params:,}")

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    # 学习率调度器: 带重启的余弦退火
    warmup_steps = len(train_loader) * CONFIG['warmup_epochs']
    t0_steps = len(train_loader) * CONFIG['t0']

    # 1. 预热调度器 (从极其微小的 lr 线性增加到初始 lr)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_steps
    )

    # 2. 重启余弦调度器
    cosine_restarts_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=t0_steps,
        T_mult=CONFIG['t_mult'],
        eta_min=1e-6
    )

    # 3. 组合调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_restarts_scheduler],
        milestones=[warmup_steps]
    )

    # 损失函数 (使用命令行参数配置权重)
    loss_fn = PhysicsInformedLoss(
        w_swivel=CONFIG['w_swivel'],
        w_elbow=CONFIG['w_elbow'],
        w_smooth=CONFIG['w_smooth'],
        elbow_loss_type=CONFIG['elbow_loss_type']
    ).to(device)

    # ================================================================
    # 训练循环
    # ================================================================
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"[Train] 开始训练: {CONFIG['epochs']} epochs")
        print(f"[Train] 策略: 最佳模型保存 + 每10轮无改善时强制保存")
        print(f"{'='*60}\n")

    best_val_loss = float('inf')
    epochs_without_improvement = 0  # 记录连续无改善的轮数
    SAVE_INTERVAL = 10  # 每10轮无改善时强制保存

    for epoch in range(CONFIG['epochs']):
        # 设置 sampler 的 epoch（确保每个 epoch 的 shuffle 不同）
        train_sampler.set_epoch(epoch)

        # ============================================================
        # 训练
        # ============================================================
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            local_rank=local_rank,
            epoch=epoch,
            use_wandb=not args.no_wandb,
            add_noise=CONFIG['add_noise']  # 【抗过拟合】物理噪声增强
        )

        # ============================================================
        # 验证
        # ============================================================
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device
        )

        # ============================================================
        # 日志记录（仅 Rank 0）
        # ============================================================
        if local_rank == 0:
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"(swivel: {train_metrics['swivel']:.4f}, "
                f"elbow: {train_metrics['elbow']:.4f}, "
                f"smooth: {train_metrics['smooth']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # WandB 日志
            if not args.no_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': train_metrics['loss'],
                    'train/swivel': train_metrics['swivel'],
                    'train/elbow': train_metrics['elbow'],
                    'train/smooth': train_metrics['smooth'],
                    'val/loss': val_metrics['loss'],
                    'val/swivel': val_metrics['swivel'],
                    'val/elbow': val_metrics['elbow'],
                    'val/smooth': val_metrics['smooth'],
                    'lr': current_lr
                })

            # 保存检查点的辅助函数
            def save_checkpoint(save_name, is_best=False):
                save_path = os.path.join(CONFIG['save_dir'], save_name)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': CONFIG,
                    'window_size': args.window_size,
                    'loss_weights': {
                        'w_swivel': args.w_swivel,
                        'w_elbow': args.w_elbow,
                        'w_smooth': args.w_smooth,
                        'elbow_loss_type': args.elbow_loss_type
                    },
                    'args': vars(args)
                }, save_path)
                tag = "最佳模型" if is_best else "检查点"
                print(f"  -> 保存{tag}: {save_path} (val_loss={val_metrics['loss']:.4f})")
                return save_path

            # 判断是否保存
            if val_metrics['loss'] < best_val_loss:
                # 有改善：保存为最佳模型
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
                checkpoint_name = f'best_model_{args.backbone}_L{args.num_layers}_w{args.window_size}_{loss_tag}.pth'
                save_checkpoint(checkpoint_name, is_best=True)
            else:
                # 无改善：计数+1
                epochs_without_improvement += 1

                # 每10轮无改善时强制保存一次
                if epochs_without_improvement >= SAVE_INTERVAL:
                    checkpoint_name = f'checkpoint_epoch{epoch+1}_{args.backbone}_L{args.num_layers}_w{args.window_size}_{loss_tag}.pth'
                    save_checkpoint(checkpoint_name, is_best=False)
                    epochs_without_improvement = 0  # 重置计数器

                    # 如果已经保存了强制检查点且达到最大轮数，可以提前停止
                    if epoch + 1 >= CONFIG['epochs']:
                        print(f"  [Info] 达到最大轮数 {CONFIG['epochs']}，停止训练")

    # ================================================================
    # 训练完成
    # ================================================================
    if local_rank == 0:
        if not args.no_wandb:
            wandb.finish()
        print(f"\n{'='*60}")
        print(f"[Train] 训练完成! 最佳验证损失: {best_val_loss:.4f}")
        print(f"{'='*60}")

    # 清理 DDP
    cleanup()


if __name__ == "__main__":
    main()
