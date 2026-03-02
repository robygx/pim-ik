"""
PiM-IK 模型评测脚本
使用 GRAB 数据集评测训练好的模型

用法:
    CUDA_VISIBLE_DEVICES=4 python evaluate.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

# 导入自定义模块
from pim_ik_net import PiM_IK_Net
from pim_ik_kinematics import DifferentiableKinematicsLayer, PhysicsInformedLoss


# ============================================================================
# 数据集类（复用 trainer.py 的实现）
# ============================================================================

class SwivelSequenceDataset(Dataset):
    """
    滑动窗口时序数据集
    全量载入 .npz 数据到 CPU 内存，按滑动窗口切片返回样本
    """

    def __init__(
        self,
        npz_path: str,
        window_size: int = 30,
        train: bool = True,
        verbose: bool = True
    ):
        self.window_size = window_size
        self.train = train

        # 加载数据
        if verbose:
            print(f"[Dataset] 正在加载数据: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        # 提取所需数组
        self.T_ee = data['T_ee']                    # (N, 4, 4)
        self.swivel_angle = data['swivel_angle']    # (N, 2)
        self.joint_positions = data['joint_positions']  # (N, 3, 3) [p_s, p_e, p_w]
        self.L_upper = data['L_upper']              # (N,)
        self.L_lower = data['L_lower']              # (N,)
        self.is_valid = data['is_valid']            # (N,)

        self.total_frames = len(self.T_ee)

        if verbose:
            print(f"[Dataset] 总帧数: {self.total_frames:,}")

        # 训练/验证集划分（不打乱，防止时序穿越）
        train_split = int(self.total_frames * 0.95)

        if train:
            self.start_idx = 0
            self.end_idx = train_split
        else:
            self.start_idx = train_split
            self.end_idx = self.total_frames

        # 可用样本数
        self.num_samples = self.end_idx - self.start_idx - window_size + 1

        if verbose:
            split_name = "训练集" if train else "测试集"
            print(f"[Dataset] {split_name}: {self.num_samples:,} 样本 (帧 {self.start_idx} - {self.end_idx})")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 计算实际起始帧索引
        start_frame = self.start_idx + idx
        end_frame = start_frame + self.window_size

        # 提取窗口数据
        T_ee = torch.from_numpy(self.T_ee[start_frame:end_frame].astype(np.float32))
        gt_swivel = torch.from_numpy(self.swivel_angle[start_frame:end_frame].astype(np.float32))
        joint_pos = torch.from_numpy(self.joint_positions[start_frame:end_frame].astype(np.float32))
        L_upper = torch.from_numpy(self.L_upper[start_frame:end_frame].astype(np.float32))
        L_lower = torch.from_numpy(self.L_lower[start_frame:end_frame].astype(np.float32))
        is_valid = torch.from_numpy(self.is_valid[start_frame:end_frame].astype(np.float32))

        # 从 joint_positions 提取各关节坐标
        p_s = joint_pos[:, 0, :]   # (W, 3) 肩部
        p_e_gt = joint_pos[:, 1, :]  # (W, 3) 肘部
        p_w = joint_pos[:, 2, :]   # (W, 3) 腕部

        return {
            'T_ee': T_ee,
            'gt_swivel': gt_swivel,
            'p_s': p_s,
            'p_e_gt': p_e_gt,
            'p_w': p_w,
            'L_upper': L_upper,
            'L_lower': L_lower,
            'is_valid': is_valid
        }


def collate_fn(batch):
    """自定义 collate 函数"""
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
# 评测指标计算
# ============================================================================

def compute_angle_error(pred_swivel, gt_swivel, is_valid=None):
    """
    计算角度误差（度）

    Args:
        pred_swivel: (N, 2) 预测臂角 [cos, sin]
        gt_swivel: (N, 2) 真实臂角 [cos, sin]
        is_valid: (N,) 有效性掩码

    Returns:
        angle_errors: (N,) 角度误差（度）
    """
    # 计算余弦相似度
    cos_sim = (pred_swivel * gt_swivel).sum(dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # 转换为角度（弧度 -> 度）
    angle_errors_rad = torch.acos(cos_sim)
    angle_errors_deg = angle_errors_rad * 180.0 / np.pi

    # 应用有效性掩码
    if is_valid is not None:
        angle_errors_deg = angle_errors_deg[is_valid > 0.5]

    return angle_errors_deg


def compute_r2_score(pred, gt, is_valid=None):
    """
    计算 R² 分数

    Args:
        pred: (N, D) 预测值
        gt: (N, D) 真实值
        is_valid: (N,) 有效性掩码

    Returns:
        r2: R² 分数
    """
    if is_valid is not None:
        pred = pred[is_valid > 0.5]
        gt = gt[is_valid > 0.5]

    # 展平
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    # 计算 R²
    ss_res = ((gt_flat - pred_flat) ** 2).sum()
    ss_tot = ((gt_flat - gt_flat.mean()) ** 2).sum()

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    r2 = 1.0 - ss_res / ss_tot
    return r2.item()


def compute_elbow_error(p_e_pred, p_e_gt, is_valid=None):
    """
    计算肘部位置误差（米）

    Args:
        p_e_pred: (N, 3) 预测肘部位置
        p_e_gt: (N, 3) 真实肘部位置
        is_valid: (N,) 有效性掩码

    Returns:
        errors: (N,) 位置误差（米）
    """
    errors = torch.norm(p_e_pred - p_e_gt, dim=-1)

    if is_valid is not None:
        errors = errors[is_valid > 0.5]

    return errors


# ============================================================================
# 主评测函数
# ============================================================================

def evaluate(model, dataloader, device, kinematics_layer, loss_fn):
    """
    在数据集上评测模型

    Returns:
        results: 包含各种指标的字典
    """
    model.eval()

    # 收集所有预测和真实值
    all_pred_swivel = []
    all_gt_swivel = []
    all_p_e_pred = []
    all_p_e_gt = []
    all_is_valid = []
    total_loss = 0.0
    total_swivel = 0.0
    total_elbow = 0.0
    total_smooth = 0.0
    num_batches = 0

    print("[Evaluate] 开始评测...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移动到 GPU
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

            total_loss += loss_dict['total_loss']
            total_swivel += loss_dict['L_swivel']
            total_elbow += loss_dict['L_elbow']
            total_smooth += loss_dict['L_smooth']
            num_batches += 1

            # 计算预测肘部位置
            p_e_pred = kinematics_layer(pred_swivel, p_s, p_w, L_upper, L_lower)

            # 收集结果（转换到 CPU）
            all_pred_swivel.append(pred_swivel.cpu())
            all_gt_swivel.append(gt_swivel.cpu())
            all_p_e_pred.append(p_e_pred.cpu())
            all_p_e_gt.append(p_e_gt.cpu())
            all_is_valid.append(is_valid.cpu())

            # 进度显示
            if (batch_idx + 1) % 50 == 0:
                print(f"[Evaluate] 已处理: {batch_idx + 1}/{len(dataloader)} batches")

    # 合并所有批次
    all_pred_swivel = torch.cat(all_pred_swivel, dim=0)  # (Total, W, 2)
    all_gt_swivel = torch.cat(all_gt_swivel, dim=0)
    all_p_e_pred = torch.cat(all_p_e_pred, dim=0)  # (Total, W, 3)
    all_p_e_gt = torch.cat(all_p_e_gt, dim=0)
    all_is_valid = torch.cat(all_is_valid, dim=0)  # (Total, W)

    # 展平时间维度
    B, W, C = all_pred_swivel.shape
    pred_flat = all_pred_swivel.reshape(-1, C)  # (Total*W, 2)
    gt_flat = all_gt_swivel.reshape(-1, C)
    p_e_pred_flat = all_p_e_pred.reshape(-1, 3)  # (Total*W, 3)
    p_e_gt_flat = all_p_e_gt.reshape(-1, 3)
    is_valid_flat = all_is_valid.reshape(-1)  # (Total*W,)

    # 计算指标
    angle_errors = compute_angle_error(pred_flat, gt_flat, is_valid_flat)
    elbow_errors = compute_elbow_error(p_e_pred_flat, p_e_gt_flat, is_valid_flat)
    r2_swivel = compute_r2_score(pred_flat, gt_flat, is_valid_flat)
    r2_elbow = compute_r2_score(p_e_pred_flat, p_e_gt_flat, is_valid_flat)

    # 统计量
    angle_errors_np = angle_errors.cpu().numpy()
    elbow_errors_np = elbow_errors.cpu().numpy()

    results = {
        # 角度误差（度）
        'angle_mean': float(np.mean(angle_errors_np)),
        'angle_median': float(np.median(angle_errors_np)),
        'angle_std': float(np.std(angle_errors_np)),
        'angle_p90': float(np.percentile(angle_errors_np, 90)),
        'angle_p95': float(np.percentile(angle_errors_np, 95)),
        'angle_p99': float(np.percentile(angle_errors_np, 99)),

        # 肘部位置误差（毫米）
        'elbow_mean_mm': float(np.mean(elbow_errors_np) * 1000),
        'elbow_median_mm': float(np.median(elbow_errors_np) * 1000),
        'elbow_std_mm': float(np.std(elbow_errors_np) * 1000),
        'elbow_p90_mm': float(np.percentile(elbow_errors_np, 90) * 1000),
        'elbow_p95_mm': float(np.percentile(elbow_errors_np, 95) * 1000),
        'elbow_p99_mm': float(np.percentile(elbow_errors_np, 99) * 1000),

        # R² 分数
        'r2_swivel': r2_swivel,
        'r2_elbow': r2_elbow,

        # 损失
        'total_loss': total_loss / num_batches,
        'l_swivel': total_swivel / num_batches,
        'l_elbow': total_elbow / num_batches,
        'l_smooth': total_smooth / num_batches,

        # 样本数
        'num_samples': len(angle_errors_np),
    }

    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 配置
    MODEL_PATH = './checkpoints/best_model.pth'
    DATA_PATH = '/data0/wwb_data/ygx_data/data_ygx_pose+dof/GRAB_training_data_with_swivel.npz'
    WINDOW_SIZE = 30
    BATCH_SIZE = 512
    D_MODEL = 256
    NUM_LAYERS = 4
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("PiM-IK 模型评测 - GRAB 数据集")
    print("=" * 60)
    print()

    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] 模型文件不存在: {MODEL_PATH}")
        print("请先运行训练脚本生成模型")
        sys.exit(1)

    # 检查数据文件
    if not os.path.exists(DATA_PATH):
        print(f"[Error] 数据文件不存在: {DATA_PATH}")
        sys.exit(1)

    # 加载模型
    print(f"[Model] 创建模型 (d_model={D_MODEL}, num_layers={NUM_LAYERS})...")
    model = PiM_IK_Net(d_model=D_MODEL, num_layers=NUM_LAYERS).to(DEVICE)

    # 加载权重
    print(f"[Model] 加载权重: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # 处理 DDP 包装的权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 移除 'module.' 前缀（DDP 包装）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

        # 打印训练信息
        if 'epoch' in checkpoint:
            print(f"[Model] 训练轮数: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"[Model] 验证损失: {checkpoint['val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)

    print("[Model] 模型加载完成")
    print()

    # 创建运动学层和损失函数
    kinematics_layer = DifferentiableKinematicsLayer().to(DEVICE)
    loss_fn = PhysicsInformedLoss(w_swivel=1.0, w_elbow=1.0, w_smooth=0.1).to(DEVICE)

    # 加载数据集
    print(f"[Data] 加载测试集...")
    dataset = SwivelSequenceDataset(
        npz_path=DATA_PATH,
        window_size=WINDOW_SIZE,
        train=False,  # 使用测试集（后 5%）
        verbose=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    print(f"[Data] 批次数: {len(dataloader)}, Batch size: {BATCH_SIZE}")
    print()

    # 运行评测
    print("=" * 60)
    print("开始评测...")
    print("=" * 60)
    print()

    results = evaluate(model, dataloader, DEVICE, kinematics_layer, loss_fn)

    # 打印结果
    print()
    print("=" * 60)
    print("评测结果")
    print("=" * 60)
    print()

    print("角度误差 (Swivel Angle Error):")
    print(f"  平均: {results['angle_mean']:.2f}°")
    print(f"  中位数: {results['angle_median']:.2f}°")
    print(f"  标准差: {results['angle_std']:.2f}°")
    print(f"  90分位: {results['angle_p90']:.2f}°")
    print(f"  95分位: {results['angle_p95']:.2f}°")
    print(f"  99分位: {results['angle_p99']:.2f}°")
    print()

    print("肘部位置误差 (Elbow Position Error):")
    print(f"  平均: {results['elbow_mean_mm']:.1f} mm")
    print(f"  中位数: {results['elbow_median_mm']:.1f} mm")
    print(f"  标准差: {results['elbow_std_mm']:.1f} mm")
    print(f"  90分位: {results['elbow_p90_mm']:.1f} mm")
    print(f"  95分位: {results['elbow_p95_mm']:.1f} mm")
    print(f"  99分位: {results['elbow_p99_mm']:.1f} mm")
    print()

    print("R² 分数:")
    print(f"  Swivel Angle: {results['r2_swivel']:.4f}")
    print(f"  Elbow Position: {results['r2_elbow']:.4f}")
    print()

    print("损失分解:")
    print(f"  L_swivel: {results['l_swivel']:.6f}")
    print(f"  L_elbow: {results['l_elbow']:.6f}")
    print(f"  L_smooth: {results['l_smooth']:.6f}")
    print(f"  Total: {results['total_loss']:.6f}")
    print()

    print(f"有效样本数: {results['num_samples']:,}")
    print()

    # 保存结果
    output_path = './evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print(f"评测完成！结果已保存至: {output_path}")
    print("=" * 60)

    # 简单评估
    print()
    print("[评估] ", end="")

    # 判断标准
    if results['angle_mean'] < 5:
        print("✓ 角度误差优秀 (< 5°)", end="")
    elif results['angle_mean'] < 10:
        print("△ 角度误差良好 (< 10°)", end="")
    else:
        print("✗ 角度误差较大 (> 10°)", end="")

    if results['elbow_mean_mm'] < 30:
        print(" | ✓ 肘部误差优秀 (< 30mm)")
    elif results['elbow_mean_mm'] < 50:
        print(" | △ 肘部误差良好 (< 50mm)")
    else:
        print(" | ✗ 肘部误差较大 (> 50mm)")


if __name__ == '__main__':
    main()
