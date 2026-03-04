#!/usr/bin/env python3
"""
集成 Pinocchio IK 到 PiM-IK 系统

使用 Pinocchio FK + Scipy 优化实现可靠的 IK 求解
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ygx/pim-ik')

import pinocchio as pin
from scipy.optimize import minimize


class PinocchioIKSolver:
    """
    使用 Pinocchio FK + Scipy 优化的 IK 求解器

    这是数值 IK，不是解析 IK，但速度快（~10-15ms）且精度高（<1mm）
    """

    def __init__(self, model_path='/home/ygx/g1_29_model_cache.pkl'):
        """加载 Pinocchio 模型"""
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['reduced_model']
        self.data = self.model.createData()

        # 左臂关节索引
        self.left_arm_indices = [1, 2, 3, 4, 5, 6, 7]  # [sp, sr, sy, el, wr, wp, wy]

        # 关节名称映射
        self.joint_names = {
            1: 'left_shoulder_pitch_joint',
            2: 'left_shoulder_roll_joint',
            3: 'left_shoulder_yaw_joint',
            4: 'left_elbow_joint',
            5: 'left_wrist_roll_joint',
            6: 'left_wrist_pitch_joint',
            7: 'left_wrist_yaw_joint',
        }

        # 关节限位 (从 G1KinematicsConfig)
        self.joint_limits = {
            1: (-np.pi/2, np.pi/2),      # shoulder_pitch
            2: (-0.3, np.pi/2),           # shoulder_roll
            3: (-np.pi/2, np.pi/2),       # shoulder_yaw
            4: (0, np.pi*2/3),            # elbow
            5: (-np.pi, np.pi),           # wrist_roll
            6: (-np.pi/2, np.pi/2),       # wrist_pitch
            7: (-np.pi, np.pi),           # wrist_yaw
        }

        print("[PinocchioIKSolver] 模型加载成功")

    def solve(
        self,
        T_ee_target: np.ndarray,
        p_e_target: np.ndarray = None,
        swivel_angle: np.ndarray = None,
        p_shoulder: np.ndarray = None,
        q_init: np.ndarray = None,
        max_iter: int = 200,
        verbose: bool = False,
        **kwargs
    ) -> tuple:
        """
        求解逆运动学

        Args:
            T_ee_target: (4, 4) 目标末端位姿
            p_e_target: (可选) 肘部目标位置（用于 swivel 约束）
            swivel_angle: (可选) [cos, sin] swivel angle（用于 swivel 约束）
            p_shoulder: (可选) 肩部位置
            q_init: 初始关节角度
            max_iter: 最大迭代次数
            verbose: 是否打印调试信息

        Returns:
            q_solved: (14,) 完整关节角度（包括右臂）
            info: 求解信息字典
        """
        # 提取目标位置和姿态
        P_target = T_ee_target[:3, 3]
        R_target = T_ee_target[:3, :3]

        # 初始猜测
        if q_init is None or len(q_init) != self.model.nq:
            q_init_full = np.zeros(self.model.nq)
        else:
            q_init_full = q_init.copy()

        # 只优化左臂 7 个关节
        q_left_init = q_init_full[self.left_arm_indices].copy()

        def objective(q_left):
            """目标函数"""
            q_full = q_init_full.copy()
            q_full[self.left_arm_indices] = q_left
            pin.forwardKinematics(self.model, self.data, q_full)
            pin.updateFramePlacements(self.model, self.data)
            T = self.data.oMi[7]

            # 位置误差
            pos_error = np.sum((P_target - T.translation)**2)

            # 旋转误差
            R_diff = R_target.T @ T.rotation
            rot_error = 1 - np.trace(R_diff) / 2

            error = pos_error + 0.1 * rot_error

            # 可选: 添加 swivel angle 约束
            if swivel_angle is not None and p_shoulder is not None:
                # 计算 swivel 约束
                arm_angle = np.arctan2(swivel_angle[1], swivel_angle[0])
                # 使用 FK 计算肘部位置，然后计算实际 arm angle
                p_elbow = self.data.oMi[4].translation  # elbow joint
                d_vec = P_target - p_shoulder
                n = d_vec / np.linalg.norm(d_vec)
                v_ref = np.array([-1.0, 0.0, 0.0])
                u = v_ref - np.dot(v_ref, n) * n
                u_norm = np.linalg.norm(u)
                if u_norm < 1e-9:
                    v_ref_alt = np.array([0.0, 1.0, 0.0])
                    u = v_ref_alt - np.dot(v_ref_alt, n) * n
                    u_norm = np.linalg.norm(u)
                u = u / u_norm
                v = np.cross(n, u)
                elbow_vec = p_elbow - p_shoulder
                actual_arm_angle = np.arctan2(np.dot(elbow_vec, v), np.dot(elbow_vec, u))
                swivel_error = (actual_arm_angle - arm_angle)**2
                error += 0.01 * swivel_error

            return error

        # 关节限位
        bounds = [self.joint_limits[i] for i in self.left_arm_indices]

        # 优化
        import time
        t0 = time.time()

        result = minimize(
            objective,
            q_left_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-9}
        )

        t1 = time.time()
        solve_time = (t1 - t0) * 1000

        # 组装结果
        q_solved = q_init_full.copy()
        q_solved[self.left_arm_indices] = result.x

        # 计算误差
        pin.forwardKinematics(self.model, self.data, q_solved)
        pin.updateFramePlacements(self.model, self.data)
        T_solved = self.data.oMi[7]

        pos_error = np.linalg.norm(P_target - T_solved.translation) * 1000
        R_diff = R_target.T @ T_solved.rotation
        rot_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        rot_error_deg = np.degrees(rot_error)

        info = {
            'method': 'pinocchio_numerical',
            'success': result.success,
            'iterations': result.nit,
            'converged': result.success,
            'pos_error_mm': pos_error,
            'rot_error_deg': rot_error_deg,
            'ik_latency_ms': solve_time,
        }

        if verbose:
            print(f"  [PinocchioIK] Time: {solve_time:.2f}ms, "
                  f"PosErr: {pos_error:.3f}mm, RotErr: {rot_error_deg:.3f}°")

        return q_solved, info

    def compute_fk(self, q: np.ndarray) -> np.ndarray:
        """
        计算正向运动学

        Args:
            q: 关节角度

        Returns:
            T_ee: (4, 4) 末端位姿
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T_ee = self.data.oMi[7].homogeneous.copy()
        return T_ee


def test_pinocchio_ik_solver():
    """测试 PinocchioIKSolver"""
    print("="*80)
    print("PinocchioIKSolver 测试")
    print("="*80)

    solver = PinocchioIKSolver()

    # 测试用例
    q_test = np.zeros(solver.model.nq)
    q_test[1] = 0.3   # shoulder_pitch
    q_test[2] = 0.2   # shoulder_roll
    q_test[3] = -0.1  # shoulder_yaw
    q_test[4] = 0.8   # elbow
    q_test[5] = 0.1   # wrist_roll
    q_test[6] = -0.1  # wrist_pitch
    q_test[7] = 0.2   # wrist_yaw

    print(f"\n原始关节角度 (左臂，度): {np.degrees(q_test[solver.left_arm_indices]).round(1)}")

    # 计算目标位姿
    T_target = solver.compute_fk(q_test)
    P_target = T_target[:3, 3]

    print(f"目标位置 (mm): {P_target*1000}")

    # 求解 IK
    print("\n求解 IK...")
    q_solved, info = solver.solve(
        T_ee_target=T_target,
        q_init=np.zeros(solver.model.nq),
        verbose=True
    )

    print(f"\n求解关节角度 (左臂，度): {np.degrees(q_solved[solver.left_arm_indices]).round(1)}")
    print(f"求解时间: {info['ik_latency_ms']:.3f} ms")
    print(f"位置误差: {info['pos_error_mm']:.3f} mm")
    print(f"旋转误差: {info['rot_error_deg']:.3f}°")

    if info['pos_error_mm'] < 1.0 and info['rot_error_deg'] < 1.0:
        print("\n✓ 测试通过!")
    else:
        print("\n✗ 测试失败!")


if __name__ == '__main__':
    test_pinocchio_ik_solver()
