#!/usr/bin/env python3
"""
VR 工作空间到机器人数据集空间的实时坐标映射工具类

功能：
1. 读取 VR 和数据集的边界统计 JSON
2. 计算鲁棒中心、活动范围和缩放比例
3. 实时映射 VR 控制器位姿到数据集舒适区
4. 自动安全钳制，防止 OOD 越限
"""

import json
import os
from typing import Dict, Tuple, Union

import numpy as np


class WorkspaceRetargeter:
    """
    VR 工作空间到机器人数据集空间的实时坐标映射器

    该类将 VR 控制器的位置映射到训练数据集的舒适工作空间内，
    防止逆运动学求解器遇到 Out-of-Distribution (OOD) 输入。

    映射流程：
        1. 提取 VR 平移向量
        2. 减去 VR 中心 → 相对位移
        3. 乘以缩放比例 → 缩放
        4. 加上数据集中心 → 绝对位置
        5. 安全钳制 → 确保在舒适区内
    """

    def __init__(
        self,
        vr_json_path: str,
        dataset_json_path: str,
        uniform_scale: bool = True
    ):
        """
        初始化工作空间映射器

        Args:
            vr_json_path: VR 工作空间边界 JSON 文件路径
            dataset_json_path: 数据集工作空间边界 JSON 文件路径
            uniform_scale: 是否使用统一缩放系数（取三轴最小值，防止几何变形）
        """
        # 读取 VR 统计数据
        vr_stats = self._load_json(vr_json_path)
        # 读取数据集统计数据
        dataset_stats = self._load_json(dataset_json_path)

        # 提取鲁棒边界（处理两种不同的 JSON 格式）
        self.vr_min, self.vr_max = self._extract_robust_bounds(vr_stats, source='vr')
        self.dataset_min, self.dataset_max = self._extract_robust_bounds(dataset_stats, source='dataset')

        # 计算鲁棒中心点: (robust_max + robust_min) / 2
        self.vr_center = (self.vr_max + self.vr_min) / 2.0
        self.dataset_center = (self.dataset_max + self.dataset_min) / 2.0

        # 计算活动范围: robust_max - robust_min
        vr_range = self.vr_max - self.vr_min
        dataset_range = self.dataset_max - self.dataset_min

        # 计算缩放比例: dataset_range / vr_range
        scale_per_axis = dataset_range / vr_range

        if uniform_scale:
            # 使用统一缩放（取最小值），防止动作几何变形
            self.scale = np.full(3, scale_per_axis.min())
            self.scale_type = 'uniform'
        else:
            # 各轴独立缩放
            self.scale = scale_per_axis
            self.scale_type = 'anisotropic'

        # 打印初始化信息
        self._print_init_info(vr_range, dataset_range, scale_per_axis, uniform_scale)

    def _load_json(self, path: str) -> Dict:
        """加载 JSON 文件"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_robust_bounds(self, stats: Dict, source: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从统计字典中提取鲁棒边界（处理两种不同的 JSON 格式）

        VR 格式:
            {"robust_bounds": {"x_1pct": 0.022, "x_99pct": 0.609, ...}}

        Dataset 格式:
            {"statistics": {"robust_bounds": {"x": {"p1": -0.145, "p99": 0.262}, ...}}}

        Args:
            stats: 统计字典
            source: 'vr' 或 'dataset'，指示数据来源

        Returns:
            min_bounds: (3,) 各轴 1% 分位数
            max_bounds: (3,) 各轴 99% 分位数
        """
        if source == 'vr':
            # VR 格式: robust_bounds -> x_1pct, x_99pct
            rb = stats['robust_bounds']
            min_bounds = np.array([
                rb['x_1pct'],
                rb['y_1pct'],
                rb['z_1pct']
            ])
            max_bounds = np.array([
                rb['x_99pct'],
                rb['y_99pct'],
                rb['z_99pct']
            ])
        else:
            # Dataset 格式: statistics -> robust_bounds -> x -> p1, p99
            rb = stats['statistics']['robust_bounds']
            min_bounds = np.array([
                rb['x']['p1'],
                rb['y']['p1'],
                rb['z']['p1']
            ])
            max_bounds = np.array([
                rb['x']['p99'],
                rb['y']['p99'],
                rb['z']['p99']
            ])

        return min_bounds, max_bounds

    def _print_init_info(
        self,
        vr_range: np.ndarray,
        dataset_range: np.ndarray,
        scale_per_axis: np.ndarray,
        uniform_scale: bool
    ):
        """打印初始化信息"""
        print("\n" + "=" * 70)
        print("WorkspaceRetargeter 初始化完成".center(70))
        print("=" * 70)

        print(f"\n缩放模式: {'统一缩放 (防止变形)' if uniform_scale else '各轴独立缩放'}")

        print("\n鲁棒边界 (1% - 99%):")
        print(f"{'轴':<6} {'VR 下界':<12} {'VR 上界':<12} {'数据集下界':<12} {'数据集上界':<12}")
        print("-" * 70)
        axes = ['X', 'Y', 'Z']
        for i, axis in enumerate(axes):
            print(f"{axis:<6} {self.vr_min[i]:<12.4f} {self.vr_max[i]:<12.4f} "
                  f"{self.dataset_min[i]:<12.4f} {self.dataset_max[i]:<12.4f}")

        print("\n活动范围:")
        print(f"{'轴':<6} {'VR 范围':<12} {'数据集范围':<12} {'原始缩放比':<12} {'应用缩放比':<12}")
        print("-" * 70)
        for i, axis in enumerate(axes):
            print(f"{axis:<6} {vr_range[i]:<12.4f} {dataset_range[i]:<12.4f} "
                  f"{scale_per_axis[i]:<12.4f} {self.scale[i]:<12.4f}")

        print("\n中心点:")
        print(f"  VR 中心:     ({self.vr_center[0]:.4f}, {self.vr_center[1]:.4f}, {self.vr_center[2]:.4f})")
        print(f"  数据集中心: ({self.dataset_center[0]:.4f}, {self.dataset_center[1]:.4f}, {self.dataset_center[2]:.4f})")
        print(f"  中心偏移:   ({self.dataset_center[0] - self.vr_center[0]:.4f}, "
              f"{self.dataset_center[1] - self.vr_center[1]:.4f}, "
              f"{self.dataset_center[2] - self.vr_center[2]:.4f})")

        print("\n" + "=" * 70)

    def map_pose(self, T_vr: Union[np.ndarray, list]) -> np.ndarray:
        """
        将 VR 位姿映射到数据集工作空间

        映射流程：
            1. 提取 VR 平移向量: p_vr = T_vr[:3, 3]
            2. 减去 VR 中心: p_rel = p_vr - vr_center
            3. 应用缩放: p_scaled = p_rel * scale
            4. 加上数据集中心: p_mapped = p_scaled + dataset_center
            5. 安全钳制: p_clipped = clip(p_mapped, min, max)
            6. 保持旋转不变，返回新矩阵

        Args:
            T_vr: VR 输入的 4x4 齐次变换矩阵 (numpy array or list)

        Returns:
            T_mapped: 映射后的 4x4 齐次变换矩阵
        """
        # 转换为 numpy array
        T_vr = np.asarray(T_vr, dtype=np.float64)

        # 确保形状正确
        if T_vr.shape != (4, 4):
            raise ValueError(f"输入矩阵形状错误，期望 (4, 4)，实际 {T_vr.shape}")

        # 1. 提取 VR 平移向量
        p_vr = T_vr[:3, 3].copy()

        # 2. 减去 VR 中心，转为相对位移
        p_rel = p_vr - self.vr_center

        # 3. 应用缩放比例
        p_scaled = p_rel * self.scale

        # 4. 加上数据集中心
        p_mapped = p_scaled + self.dataset_center

        # 5. 绝对安全钳制（确保不会超出数据集舒适区）
        p_clipped = np.clip(p_mapped, self.dataset_min, self.dataset_max)

        # 6. 构建输出矩阵（保持旋转矩阵完全不变）
        T_mapped = T_vr.copy()
        T_mapped[:3, 3] = p_clipped

        return T_mapped

    def get_mapping_info(self) -> Dict:
        """
        获取映射参数信息（用于调试或显示）

        Returns:
            包含映射参数的字典
        """
        return {
            'vr_center': self.vr_center.tolist(),
            'dataset_center': self.dataset_center.tolist(),
            'vr_min': self.vr_min.tolist(),
            'vr_max': self.vr_max.tolist(),
            'dataset_min': self.dataset_min.tolist(),
            'dataset_max': self.dataset_max.tolist(),
            'scale': self.scale.tolist(),
            'scale_type': self.scale_type
        }


# ============================================
# 测试代码
# ============================================

def test_retargeter():
    """测试映射器功能"""
    import argparse

    parser = argparse.ArgumentParser(description='测试 WorkspaceRetargeter')
    parser.add_argument('--vr-json', default='/home/ygx/VR_data_any/vr_workspace_limits.json')
    parser.add_argument('--dataset-json', default='/home/ygx/dataset_workspace_limits.json')
    parser.add_argument('--uniform-scale', type=bool, default=True)
    args = parser.parse_args()

    # 初始化映射器
    retargeter = WorkspaceRetargeter(
        vr_json_path=args.vr_json,
        dataset_json_path=args.dataset_json,
        uniform_scale=args.uniform_scale
    )

    # 测试几个关键位置
    print("\n测试映射:")
    print("-" * 70)

    test_cases = [
        ("VR 中心点", np.array([0.3055, 0.2275, 0.1445])),
        ("VR 最大范围", np.array([0.7004, 0.5952, 0.7272])),
        ("VR 最小范围", np.array([-0.1113, -0.1223, -0.2899])),
    ]

    for name, p_vr in test_cases:
        # 构造齐次变换矩阵
        T_vr = np.eye(4)
        T_vr[:3, 3] = p_vr

        # 映射
        T_mapped = retargeter.map_pose(T_vr)
        p_mapped = T_mapped[:3, 3]

        # 检查是否钳制
        is_clipped = np.any(p_mapped != p_vr)  # 简化判断

        print(f"{name}:")
        print(f"  输入: ({p_vr[0]:.4f}, {p_vr[1]:.4f}, {p_vr[2]:.4f})")
        print(f"  输出: ({p_mapped[0]:.4f}, {p_mapped[1]:.4f}, {p_mapped[2]:.4f})")
        print(f"  钳制: {'是' if is_clipped else '否'}")
        print()


if __name__ == '__main__':
    test_retargeter()
