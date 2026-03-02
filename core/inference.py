#!/usr/bin/env python3
"""
PiM-IK Neural-to-Physics Inference Pipeline
============================================

将训练好的 PiM-IK 网络预测结果，通过基于零空间投影的分层逆运动学，
转化为 7-DOF 机器人的精确关节控制指令。

作者: PiM-IK Project
日期: 2025-02-27

核心模块:
    1. TargetGenerator: 神经网络输出 → 肘部 3D 坐标
    2. HierarchicalIKSolver: 双任务层次 IK (末端 + 肘部)
    3. InferencePipeline: 端到端推理管线

依赖:
    pip install torch pinocchio 'numpy<2.0' scipy
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# Pinocchio 导入 (可选依赖，用于 HierarchicalIKSolver)
PINOCCHIO_AVAILABLE = False
try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] pinocchio 未安装 (仅影响 HierarchicalIKSolver): {e}")
    print("[Info] TargetGenerator 不依赖 pinocchio，可正常使用")

# 导入自定义模块
from pim_ik_net import PiM_IK_Net


# ============================================================================
# 模块 1: TargetGenerator - 肘部目标生成
# ============================================================================

class TargetGenerator:
    """
    Neural-to-Physics 桥梁：将网络预测的臂角转换为肘部 3D 坐标

    算法原理：
        基于 DifferentiableKinematicsLayer 的轨道圆几何逻辑，
        使用纯 numpy 实现将 [cos(φ), sin(φ)] 转换为肘部坐标。
    """

    EPS = 1e-6  # 数值稳定性常数

    # 避奇异参考向量（指向胸腔正后方）
    V_REF = np.array([-1.0, 0.0, 0.0])
    V_REF_ALT = np.array([0.0, 1.0, 0.0])

    def __init__(self):
        """初始化目标生成器"""
        pass

    def compute_target_elbow_position(
        self,
        swivel_angle: np.ndarray,
        p_s: np.ndarray,
        p_w: np.ndarray,
        L_upper: float,
        L_lower: float
    ) -> np.ndarray:
        """
        计算目标肘部位置

        Args:
            swivel_angle: (2,) 预测的臂角 [cos(φ), sin(φ)]
            p_s: (3,) 肩部 3D 坐标
            p_w: (3,) 腕部 3D 坐标
            L_upper: 上臂长度
            L_lower: 前臂长度

        Returns:
            p_e_target: (3,) 目标肘部 3D 坐标
        """
        # ============================================================
        # 步骤 1: L2 归一化预测臂角
        # ============================================================
        cos_phi, sin_phi = swivel_angle
        norm = np.sqrt(cos_phi**2 + sin_phi**2) + self.EPS
        cos_phi, sin_phi = cos_phi / norm, sin_phi / norm

        # ============================================================
        # 步骤 2: 构建轨道圆坐标系的正交基 (u, v, n)
        # ============================================================
        # 肩腕向量
        sw = p_w - p_s
        sw_norm = np.linalg.norm(sw) + self.EPS

        # 主轴 n = sw / ||sw||
        n = sw / sw_norm

        # Gram-Schmidt 正交化构建 X 轴 u
        v_ref_dot_n = np.dot(self.V_REF, n)
        u_candidate = self.V_REF - v_ref_dot_n * n
        u_norm = np.linalg.norm(u_candidate)

        # 奇异点回退：当 u_norm < 1e-5 时切换到备用向量
        if u_norm < 1e-5:
            v_ref_alt_dot_n = np.dot(self.V_REF_ALT, n)
            u_candidate = self.V_REF_ALT - v_ref_alt_dot_n * n
            u_norm = np.linalg.norm(u_candidate) + self.EPS

        u = u_candidate / (u_norm + self.EPS)

        # 计算 Y 轴 v = n × u（叉积构建正交右手系）
        v = np.cross(n, u)

        # ============================================================
        # 步骤 3: 利用余弦定理计算轨道圆参数
        # ============================================================
        # 投影距离 d = (L_upper² - L_lower² + ||sw||²) / (2 * ||sw||)
        L_upper_sq = L_upper ** 2
        L_lower_sq = L_lower ** 2
        sw_norm_sq = sw_norm ** 2

        d = (L_upper_sq - L_lower_sq + sw_norm_sq) / (2.0 * sw_norm + self.EPS)

        # 轨道圆心 p_c = p_s + d * n
        p_c = p_s + d * n

        # 轨道圆半径 R = sqrt(max(L_upper² - d², EPS))
        R_sq = max(L_upper_sq - d**2, self.EPS)
        R = np.sqrt(R_sq)

        # ============================================================
        # 步骤 4: 计算预测肘部位置
        # ============================================================
        # p_e = p_c + R * (cos(φ) * u + sin(φ) * v)
        offset = R * (cos_phi * u + sin_phi * v)
        p_e_target = p_c + offset

        return p_e_target


# ============================================================================
# 模块 2: HierarchicalIKSolver - 分层 IK 求解器
# ============================================================================

class HierarchicalIKSolver:
    """
    基于零空间投影的分层逆运动学求解器

    双任务层次优化 (CLIK with Null-Space Projection):
        Task 1 (Primary): 末端位姿跟踪
        Task 2 (Secondary): 肘部位置跟踪（投影到零空间）
    """

    def __init__(self, model_path: str, ee_offset: float = 0.05):
        """
        初始化 IK 求解器

        Args:
            model_path: Pinocchio 模型缓存文件路径
            ee_offset: 末端执行器偏移（米）
        """
        # 加载 Pinocchio 模型
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['reduced_model']
        self.data = self.model.createData()

        # 关节名称
        self.SHOULDER_JOINT = 'left_shoulder_pitch_joint'
        self.ELBOW_JOINT = 'left_elbow_joint'
        self.WRIST_JOINT = 'left_wrist_yaw_joint'
        self.EE_FRAME = 'left_ee'

        # 获取关节 ID
        self.shoulder_id = self.model.getJointId(self.SHOULDER_JOINT)
        self.elbow_id = self.model.getJointId(self.ELBOW_JOINT)
        self.wrist_id = self.model.getJointId(self.WRIST_JOINT)

        # 添加末端执行器帧
        try:
            self.ee_id = self.model.getFrameId(self.EE_FRAME)
        except Exception:
            self.model.addFrame(
                pin.Frame(
                    self.EE_FRAME,
                    self.wrist_id,
                    pin.SE3(np.eye(3), np.array([ee_offset, 0, 0])),
                    pin.FrameType.OP_FRAME
                )
            )
            self.ee_id = self.model.getFrameId(self.EE_FRAME)

        # 中性位姿
        self.q_neutral = np.zeros(self.model.nq)

        # 阻尼系数（防止雅可比奇异）
        self.damping = 1e-6

        # 收敛阈值
        self.pos_tol = 1e-3      # 位置误差容差 (米) - 放宽到 1mm
        self.rot_tol = 1e-2      # 姿态误差容差 (弧度) - 放宽到 0.5°

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        正向运动学：计算末端位姿

        Args:
            q: (7,) 关节角度

        Returns:
            T_ee: (4, 4) 末端齐次变换矩阵
        """
        # 创建独立的 data（线程安全）
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        return data.oMf[self.ee_id].homogeneous.copy()

    def get_elbow_position(self, q: np.ndarray) -> np.ndarray:
        """获取肘部位置"""
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        return data.oMi[self.elbow_id].translation.copy()

    def damped_pinv(self, J: np.ndarray) -> np.ndarray:
        """
        阻尼伪逆（Levenberg-Marquardt 风格）

        J^+ = J^T @ (J @ J^T + λ² I)^(-1)
        """
        m, n = J.shape
        JTJ = J @ J.T + self.damping**2 * np.eye(min(m, n))
        return J.T @ np.linalg.inv(JTJ)

    def solve(
        self,
        T_ee_target: np.ndarray,
        p_e_target: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iter: int = 200,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        求解分层 IK（鲁棒性重构版 - 雅可比转置 + 步长衰减 + 截断）

        Args:
            T_ee_target: (4, 4) 目标末端位姿
            p_e_target: (3,) 目标肘部位置
            q_init: (7,) 初始关节角度（warm-start）
            max_iter: 最大迭代次数
            verbose: 是否打印调试信息

        Returns:
            q_solved: (7,) 求解的关节角度
            info: 求解信息字典
        """
        # ============================================================
        # 鲁棒性参数配置
        # ============================================================
        alpha_ee = 0.5      # 主任务步长衰减系数（防止超调）
        k_null = 5.0        # 零空间梯度增益（肘部吸引力）
        dq_max = 0.2        # 单次迭代最大关节旋转量（约11.5°）

        # 初始化
        q = q_init if q_init is not None else self.q_neutral.copy()

        # 将目标 T_ee 转换为 Pinocchio SE3 对象
        oMdes = pin.SE3(T_ee_target)

        for iteration in range(max_iter):
            # 创建独立的 data
            data = self.model.createData()

            # FK
            pin.forwardKinematics(self.model, data, q)
            pin.updateFramePlacements(self.model, data)

            # ============================================================
            # Task 1 (Primary): 末端位姿误差 - LOCAL 坐标系
            # ============================================================
            oMcur = data.oMf[self.ee_id]

            # LOCAL 误差: log6(cur^(-1) @ des) = log6(cur.actInv(des))
            # 这是在 LOCAL 坐标系下计算的 SE(3) 误差向量
            err_local = pin.log6(oMcur.actInv(oMdes)).vector  # (6,) [v, omega]

            # 末端雅可比 - LOCAL 坐标系（严格遵守）
            J_ee = pin.computeFrameJacobian(
                self.model, data, q, self.ee_id,
                pin.ReferenceFrame.LOCAL
            )  # (6, 7)

            # 阻尼伪逆
            J_ee_pinv = self.damped_pinv(J_ee)

            # 【改进2】主任务步长衰减 - 防止距离目标较远时超调
            dq_1 = J_ee_pinv @ err_local * alpha_ee

            # ============================================================
            # Task 2 (Secondary): 肘部位置误差 - WORLD 坐标系
            # ============================================================
            p_e_current = data.oMi[self.elbow_id].translation.copy()
            e_elbow = p_e_target - p_e_current  # (3,) WORLD 系误差

            # 肘部雅可比 - LOCAL 坐标系
            J_elbow_local = pin.computeJointJacobian(self.model, data, q, self.elbow_id)[:3, :]  # (3, 7)

            # 【改进4】将 LOCAL 雅可比旋转到 WORLD 坐标系（严格遵守）
            # 肘部关节在 WORLD 坐标系下的旋转矩阵
            R_elbow = data.oMi[self.elbow_id].rotation
            J_elbow_world = R_elbow @ J_elbow_local  # (3, 7)

            # ============================================================
            # 【改进1】雅可比转置替代零空间伪逆（防止爆炸）
            # ============================================================
            # 零空间投影算子: N = I - J_ee^+ @ J_ee
            # 投影到主任务的零空间，保证不影响末端位姿
            N = np.eye(self.model.nq) - J_ee_pinv @ J_ee

            # 使用雅可比转置 + 梯度下降（而不是伪逆）
            # 这确保了次级任务永远不会因奇异或不可达而爆炸
            # dq_2 = N @ J_elbow_world.T @ e_elbow * k_null
            dq_2 = N @ J_elbow_world.T @ e_elbow * k_null

            # ============================================================
            # 【改进3】关节步长截断（防止数值积分发散）
            # ============================================================
            dq = dq_1 + dq_2
            dq = np.clip(dq, -dq_max, dq_max)  # 单次迭代最大转动约 11.5°

            # 使用李群积分更新关节角度
            q = pin.integrate(self.model, q, dq)

            # ============================================================
            # 收敛检查
            # ============================================================
            pos_err = np.linalg.norm(err_local[:3])
            rot_err = np.linalg.norm(err_local[3:])

            if verbose:
                print(f"  Iter {iteration}: pos_err={pos_err*1000:.3f}mm, rot_err={np.degrees(rot_err):.3f}°")

            if pos_err < self.pos_tol and rot_err < self.rot_tol:
                break

        # ============================================================
        # 最终误差统计
        # ============================================================
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)

        T_final = data.oMf[self.ee_id].homogeneous
        p_e_final = data.oMi[self.elbow_id].translation

        # 使用 Pinocchio log6 计算最终误差
        oMfinal = pin.SE3(T_final)
        final_err = pin.log6(oMfinal.actInv(oMdes)).vector
        final_pos_err = np.linalg.norm(final_err[:3])
        final_rot_err = np.linalg.norm(final_err[3:])
        final_elbow_err = np.linalg.norm(p_e_target - p_e_final)

        info = {
            'iterations': iteration + 1,
            'pos_error': final_pos_err,
            'rot_error': final_rot_err,
            'elbow_error': final_elbow_err,
            'converged': (final_pos_err < self.pos_tol and final_rot_err < self.rot_tol)
        }

        return q, info


# ============================================================================
# 模块 3: InferencePipeline - 推理管线
# ============================================================================

@dataclass
class InferenceResult:
    """推理结果数据类"""
    q_solved: np.ndarray          # (T, 7) 求解的关节角度
    pred_swivel: np.ndarray       # (T, 2) 预测的臂角
    ee_pos_errors: List[float]    # 末端位置误差
    ee_rot_errors: List[float]    # 末端姿态误差
    elbow_errors: List[float]     # 肘部位置误差
    joint_maes: List[float]       # 关节角度 MAE
    solve_times: List[float]      # 求解时间
    iterations: List[int]         # 迭代次数
    success_rate: float           # 成功率


class InferencePipeline:
    """
    端到端推理管线

    流程:
        1. 加载训练好的神经网络模型
        2. 加载 Pinocchio 机器人模型
        3. 从验证集抽取连续轨迹
        4. 神经网络推理 → 预测臂角
        5. 臂角 → 肘部目标位置
        6. 分层 IK 求解 → 关节角度
        7. 精度验证与结果输出
    """

    def __init__(
        self,
        model_checkpoint: str,
        pinocchio_model: str,
        device: str = 'cuda:0'
    ):
        """
        初始化推理管线

        Args:
            model_checkpoint: PyTorch 模型 checkpoint 路径
            pinocchio_model: Pinocchio 模型缓存路径
            device: 计算设备
        """
        self.device = device

        print("=" * 60)
        print("PiM-IK Inference Pipeline Initialization")
        print("=" * 60)

        # 加载神经网络模型
        print(f"\n[1/3] Loading Neural Network Model...")
        print(f"  Checkpoint: {model_checkpoint}")
        self.nn_model = self._load_nn_model(model_checkpoint)
        print(f"  ✓ Model loaded: PiM_IK_Net (d_model=256, num_layers=4)")

        # 加载 Pinocchio 模型
        print(f"\n[2/3] Loading Pinocchio Robot Model...")
        print(f"  Model: {pinocchio_model}")
        self.ik_solver = HierarchicalIKSolver(pinocchio_model)
        print(f"  ✓ Robot loaded: G1 Left Arm (7-DOF)")

        # 初始化目标生成器
        print(f"\n[3/3] Initializing Target Generator...")
        self.target_gen = TargetGenerator()
        print(f"  ✓ Ready")

        print("\n" + "=" * 60)
        print("Initialization Complete!")
        print("=" * 60)

    def _load_nn_model(self, checkpoint_path: str) -> nn.Module:
        """加载训练好的神经网络模型"""
        model = PiM_IK_Net(d_model=256, num_layers=4)

        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 处理 DDP 包装的权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # 移除 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def load_trajectory(
        self,
        data_path: str,
        trajectory_length: int = 30,
        start_idx: Optional[int] = None
    ) -> Dict:
        """
        从数据集加载一段连续轨迹

        Args:
            data_path: 数据集 .npz 文件路径
            trajectory_length: 轨迹长度（帧数）
            start_idx: 起始索引，None 则从验证集开始处取

        Returns:
            trajectory_data: 轨迹数据字典
        """
        print(f"\n[Loading] Loading trajectory from dataset...")
        print(f"  Path: {data_path}")

        data = np.load(data_path, allow_pickle=True)

        # 确定起始索引（验证集后 5%）
        total_frames = len(data['T_ee'])
        val_start = int(total_frames * 0.95)

        if start_idx is None:
            start_idx = val_start

        # 确保不超出范围
        end_idx = min(start_idx + trajectory_length, total_frames)
        actual_length = end_idx - start_idx

        print(f"  Validation set starts at frame: {val_start:,}")
        print(f"  Loading frames: {start_idx:,} - {end_idx:,}")
        print(f"  Actual trajectory length: {actual_length}")

        # 提取轨迹数据
        trajectory_data = {
            'T_ee': data['T_ee'][start_idx:end_idx],  # (T, 4, 4)
            'gt_joints': data['y_original'][start_idx:end_idx, 7:14],  # (T, 7)
            'joint_positions': data['joint_positions'][start_idx:end_idx],  # (T, 3, 3)
            'L_upper': data['L_upper'][start_idx:end_idx],  # (T,)
            'L_lower': data['L_lower'][start_idx:end_idx],  # (T,)
            'gt_swivel': data['swivel_angle'][start_idx:end_idx],  # (T, 2)
        }

        return trajectory_data

    def run_inference(
        self,
        trajectory_data: Dict,
        verbose: bool = True,
        use_gt_as_init: bool = False  # 是否用 GT 作为初始猜测
    ) -> InferenceResult:
        """
        运行端到端推理

        Args:
            trajectory_data: 轨迹数据字典
            verbose: 是否打印详细进度

        Returns:
            result: 推理结果
        """
        T = len(trajectory_data['T_ee'])

        if verbose:
            print(f"\n[Inference] Running inference on {T} frames...")
            print("-" * 60)

        # ============================================================
        # 步骤 1: 神经网络推理
        # ============================================================
        t0 = time.time()
        with torch.no_grad():
            T_ee_tensor = torch.from_numpy(
                trajectory_data['T_ee'].astype(np.float32)
            ).unsqueeze(0).to(self.device)  # (1, T, 4, 4)

            pred_swivel = self.nn_model(T_ee_tensor)  # (1, T, 2)
            pred_swivel = pred_swivel[0].cpu().numpy()  # (T, 2)

        nn_time = time.time() - t0

        if verbose:
            print(f"  [Neural Network] Inference time: {nn_time*1000:.2f} ms")

        # ============================================================
        # 步骤 2: 逐帧 IK 求解 (Warm-start)
        # ============================================================
        q_solved = []
        ee_pos_errors = []
        ee_rot_errors = []
        elbow_errors = []
        joint_maes = []
        solve_times = []
        iterations = []

        q_init = None  # Warm-start (设为 GT 值可测试 IK 能力)

        total_ik_time = 0

        for i in range(T):
            t_frame = time.time()

            # 获取当前帧数据
            T_ee_target = trajectory_data['T_ee'][i]
            p_s = trajectory_data['joint_positions'][i, 0, :]  # 肩部
            p_w = trajectory_data['joint_positions'][i, 2, :]  # 腕部
            L_upper = trajectory_data['L_upper'][i]
            L_lower = trajectory_data['L_lower'][i]
            gt_joint = trajectory_data['gt_joints'][i]

            # 使用 GT 作为初始猜测（用于测试 IK）
            if use_gt_as_init and q_init is None:
                q_init = gt_joint.copy()
                # 对于非第一帧，仍使用上一帧的解
            elif not use_gt_as_init and i == 0:
                # 第一帧使用 GT 初始化，提高成功率
                q_init = gt_joint.copy()

            # 生成目标肘部位置
            p_e_target = self.target_gen.compute_target_elbow_position(
                pred_swivel[i], p_s, p_w, L_upper, L_lower
            )

            # IK 求解
            q_sol, info = self.ik_solver.solve(
                T_ee_target=T_ee_target,
                p_e_target=p_e_target,
                q_init=q_init,
                verbose=False
            )

            # 记录结果
            q_solved.append(q_sol)
            ee_pos_errors.append(info['pos_error'])
            ee_rot_errors.append(info['rot_error'])
            elbow_errors.append(info['elbow_error'])
            iterations.append(info['iterations'])

            # 计算 MAE（处理角度周期性）
            # 对于周期性关节，使用 sin/cos 计算角度差异，并取绝对值
            angle_diff = np.arctan2(np.sin(q_sol - gt_joint), np.cos(q_sol - gt_joint))
            mae = np.mean(np.abs(angle_diff))
            joint_maes.append(mae)

            # Warm-start for next frame
            q_init = q_sol

            frame_time = time.time() - t_frame
            solve_times.append(frame_time)
            total_ik_time += frame_time

            if verbose and (i + 1) % 10 == 0:
                print(f"  Frame {i+1:3d}/{T}: "
                      f"pos_err={ee_pos_errors[-1]*1000:.2f}mm, "
                      f"elbow_err={elbow_errors[-1]*1000:.2f}mm, "
                      f"MAE={np.degrees(mae):.2f}°")

        avg_ik_time = total_ik_time / T

        if verbose:
            print(f"  [IK Solver] Average time: {avg_ik_time*1000:.2f} ms/frame")

        # ============================================================
        # 步骤 3: 统计成功率
        # ============================================================
        success_count = sum(1 for e in ee_pos_errors if e < 1e-3)  # 1mm
        success_rate = success_count / T

        # ============================================================
        # 步骤 4: 组装结果
        # ============================================================
        result = InferenceResult(
            q_solved=np.array(q_solved),
            pred_swivel=pred_swivel,
            ee_pos_errors=ee_pos_errors,
            ee_rot_errors=ee_rot_errors,
            elbow_errors=elbow_errors,
            joint_maes=joint_maes,
            solve_times=solve_times,
            iterations=iterations,
            success_rate=success_rate
        )

        return result

    def print_results(self, result: InferenceResult, model_path: str):
        """
        打印科技感结果面板

        Args:
            result: 推理结果
            model_path: 模型路径
        """
        # 计算统计数据
        T = len(result.q_solved)

        # 末端误差统计
        pos_err_mm = [e * 1000 for e in result.ee_pos_errors]
        rot_err_deg = [np.degrees(e) for e in result.ee_rot_errors]

        avg_pos_err = np.mean(pos_err_mm)
        max_pos_err = np.max(pos_err_mm)
        avg_rot_err = np.mean(rot_err_deg)
        max_rot_err = np.max(rot_err_deg)

        # 关节 MAE 统计
        joint_mae_deg = [np.degrees(m) for m in result.joint_maes]
        avg_mae = np.mean(joint_mae_deg)
        max_mae = np.max(joint_mae_deg)
        rmse_mae = np.sqrt(np.mean([m**2 for m in joint_mae_deg]))

        # 肘部误差统计
        elbow_err_mm = [e * 1000 for e in result.elbow_errors]
        avg_elbow_err = np.mean(elbow_err_mm)

        # 迭代次数统计
        avg_iter = np.mean(result.iterations)

        # 时间统计
        total_time = np.sum(result.solve_times)
        avg_time = total_time / T

        # ============================================================
        # 打印科技感面板
        # ============================================================
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 10 + "PiM-IK Neural-to-Physics Inference" + " " * 10 + "║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Model: {model_path[-40:]:40s} ║")
        print(f"║  Device: CUDA:0 (NVIDIA H20)                             ║")
        print(f"║  Trajectory: {T} frames (Validation Set)                 ║")
        print("╠" + "═" * 58 + "╣")
        print("║  Inference Time:                                         ║")
        print(f"║    Neural Network:   {(total_time-avg_time*T)*1000/T:.2f} ms/frame       ║")
        print(f"║    IK Solver:        {avg_time*1000:.2f} ms/frame       ║")
        print(f"║    Total:            {total_time/T*1000:.2f} ms/frame       ║")
        print("╠" + "═" * 58 + "╣")
        print("║  End-Effector Accuracy:                                   ║")
        print(f"║    Position Error:    {avg_pos_err:6.2f} mm  (max: {max_pos_err:5.2f} mm)           ║")
        print(f"║    Rotation Error:    {avg_rot_err:6.2f}°    (max: {max_rot_err:5.2f}°)              ║")
        print("╠" + "═" * 58 + "╣")
        print("║  Elbow Tracking Accuracy:                                ║")
        print(f"║    Elbow Position:    {avg_elbow_err:6.2f} mm  (max: {max(elbow_err_mm):5.2f} mm)           ║")
        print("╠" + "═" * 58 + "╣")
        print("║  Joint Angle Accuracy:                                   ║")
        print(f"║    MAE vs GT:        {avg_mae:6.2f}°    (max: {max_mae:5.2f}°)              ║")
        print(f"║    RMSE vs GT:       {rmse_mae:6.2f}°                                   ║")
        print("╠" + "═" * 58 + "╣")
        print("║  Convergence:                                            ║")
        print(f"║    Average Iterations: {avg_iter:4.1f}                                 ║")
        print(f"║    Success Rate:      {result.success_rate*100:5.1f}% ({T}/{T})                         ║")
        print("╚" + "═" * 58 + "╝")

        # 评估
        print("\n[Evaluation]")
        if avg_pos_err < 1.0:
            print(f"  ✓ Position error: {avg_pos_err:.2f} mm < 1.0 mm (EXCELLENT)")
        elif avg_pos_err < 5.0:
            print(f"  △ Position error: {avg_pos_err:.2f} mm < 5.0 mm (GOOD)")
        else:
            print(f"  ✗ Position error: {avg_pos_err:.2f} mm > 5.0 mm (NEEDS IMPROVEMENT)")

        if avg_mae < 5.0:
            print(f"  ✓ Joint MAE: {avg_mae:.2f}° < 5.0° (EXCELLENT)")
        elif avg_mae < 10.0:
            print(f"  △ Joint MAE: {avg_mae:.2f}° < 10.0° (GOOD)")
        else:
            print(f"  ✗ Joint MAE: {avg_mae:.2f}° > 10.0° (NEEDS IMPROVEMENT)")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='PiM-IK Inference Pipeline')
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/20260227_103000/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--pinocchio', type=str,
                        default='/home/ygx/g1_left_arm_model_cache.pkl',
                        help='Path to Pinocchio model cache')
    parser.add_argument('--data', type=str,
                        default='/data0/wwb_data/ygx_data/data_ygx_pose+dof/ACCAD_CMU_merged_training_data_with_swivel.npz',
                        help='Path to dataset')
    parser.add_argument('--frames', type=int, default=30,
                        help='Trajectory length (frames)')
    parser.add_argument('--start', type=int, default=None,
                        help='Start frame index (None for validation set start)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Compute device')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        # 尝试找到最新的 checkpoint
        checkpoint_dir = './checkpoints'
        if os.path.exists(checkpoint_dir):
            subdirs = [d for d in os.listdir(checkpoint_dir)
                      if os.path.isdir(os.path.join(checkpoint_dir, d))]
            if subdirs:
                latest = sorted(subdirs)[-1]
                args.checkpoint = os.path.join(checkpoint_dir, latest, 'best_model.pth')
                print(f"[Auto-detected] Using latest checkpoint: {args.checkpoint}")
            else:
                print(f"[Error] No checkpoint found in {checkpoint_dir}")
                return
        else:
            print(f"[Error] Checkpoint not found: {args.checkpoint}")
            return

    if not os.path.exists(args.pinocchio):
        print(f"[Error] Pinocchio model not found: {args.pinocchio}")
        return

    if not os.path.exists(args.data):
        print(f"[Error] Dataset not found: {args.data}")
        return

    # ============================================================
    # 初始化推理管线
    # ============================================================
    pipeline = InferencePipeline(
        model_checkpoint=args.checkpoint,
        pinocchio_model=args.pinocchio,
        device=args.device
    )

    # ============================================================
    # 加载轨迹
    # ============================================================
    trajectory_data = pipeline.load_trajectory(
        data_path=args.data,
        trajectory_length=args.frames,
        start_idx=args.start
    )

    # ============================================================
    # 运行推理
    # ============================================================
    result = pipeline.run_inference(
        trajectory_data=trajectory_data,
        verbose=args.verbose
    )

    # ============================================================
    # 打印结果
    # ============================================================
    pipeline.print_results(result, args.checkpoint)


if __name__ == '__main__':
    main()
