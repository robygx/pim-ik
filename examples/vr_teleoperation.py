#!/usr/bin/env python3
"""
PiM-IK VR 接入示例
==================

本脚本展示如何将 PiM-IK 接入 VR 控制系统。
VR 系统传入末端位姿，输出 7-DOF 关节角度。

使用场景:
    VR 手柄追踪 → 末端位姿 → PiM-IK → 机械臂关节控制

作者: PiM-IK Project
日期: 2025-02-27
"""

import numpy as np
import torch
from typing import Optional
from pathlib import Path


class VR IKController:
    """
    VR IK 控制器 - 封装 PiM-IK 推理管线

    提供简化的单帧推理接口，方便 VR 系统接入。
    """

    def __init__(
        self,
        model_checkpoint: str = './checkpoints/20260227_111614/best_model.pth',
        pinocchio_model: str = '/home/ygx/g1_left_arm_model_cache.pkl',
        device: str = 'cuda:0'
    ):
        """
        初始化 IK 控制器

        Args:
            model_checkpoint: PyTorch 模型权重路径
            pinocchio_model: Pinocchio 机器人模型路径
            device: 计算设备 ('cuda:0' 或 'cpu')
        """
        from inference import InferencePipeline

        print("[VR IK Controller] 初始化中...")
        self.pipeline = InferencePipeline(
            model_checkpoint=model_checkpoint,
            pinocchio_model=pinocchio_model,
            device=device
        )
        print("[VR IK Controller] 初始化完成!")

        # 缓存上一帧结果用于 warm-start
        self._q_init = None

    def compute_ik(
        self,
        T_ee: np.ndarray,
        shoulder_pos: np.ndarray,
        wrist_pos: np.ndarray,
        upper_arm_length: float,
        forearm_length: float
    ) -> np.ndarray:
        """
        单帧 IK 求解 (主要接口)

        Args:
            T_ee: (4, 4) 末端位姿齐次矩阵 (WORLD 坐标系)
            shoulder_pos: (3,) 肩部位置
            wrist_pos: (3,) 腕部位置
            upper_arm_length: 上臂长度 (米)
            forearm_length: 前臂长度 (米)

        Returns:
            joint_angles: (7,) 关节角度 (弧度)
        """
        from inference import TargetGenerator

        # 确保输入是 numpy 数组
        T_ee = np.asarray(T_ee, dtype=np.float32)
        shoulder_pos = np.asarray(shoulder_pos, dtype=np.float32)
        wrist_pos = np.asarray(wrist_pos, dtype=np.float32)

        # ================================================================
        # 步骤 1: 神经网络推理 (预测臂角)
        # ================================================================
        with torch.no_grad():
            T_ee_tensor = torch.from_numpy(T_ee).unsqueeze(0).unsqueeze(0)
            T_ee_tensor = T_ee_tensor.to(self.pipeline.device)
            pred_swivel = self.pipeline.nn_model(T_ee_tensor)  # (1, 1, 2)
            pred_swivel = pred_swivel[0, 0].cpu().numpy()  # (2,)

        # ================================================================
        # 步骤 2: 肘部目标位置计算
        # ================================================================
        target_gen = TargetGenerator()
        p_e_target = target_gen.compute_target_elbow_position(
            swivel_angle=pred_swivel,
            p_s=shoulder_pos,
            p_w=wrist_pos,
            L_upper=upper_arm_length,
            L_lower=forearm_length
        )

        # ================================================================
        # 步骤 3: 分层 IK 求解
        # ================================================================
        q_sol, info = self.pipeline.ik_solver.solve(
            T_ee_target=T_ee,
            p_e_target=p_e_target,
            q_init=self._q_init,  # warm-start
            verbose=False
        )

        # 更新 warm-start
        self._q_init = q_sol

        return q_sol

    def reset(self):
        """重置 warm-start 状态"""
        self._q_init = None

    def get_joint_names(self) -> list:
        """获取关节名称列表"""
        return [
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'left_wrist_yaw_joint',
            'left_wrist_roll_joint',
            'left_wrist_pitch_joint',
        ]


# ============================================================================
# VR 系统接入示例
# ============================================================================

def example_vr_integration():
    """
    VR 系统接入示例

    模拟 VR 手柄传入末端位姿，输出机械臂关节角度。
    """
    print("=" * 60)
    print("PiM-IK VR 接入示例")
    print("=" * 60)

    # ================================================================
    # 初始化 IK 控制器
    # ================================================================
    controller = VR IKController(
        model_checkpoint='./checkpoints/20260227_111614/best_model.pth',
        pinocchio_model='/home/ygx/g1_left_arm_model_cache.pkl',
        device='cuda:0'
    )

    # ================================================================
    # 模拟 VR 输入
    # ================================================================
    print("\n[模拟 VR 输入]")

    # 示例 1: 单位矩阵 (机械臂指向正前方)
    T_ee_1 = np.eye(4, dtype=np.float32)
    T_ee_1[:3, 3] = [0.3, 0.0, 0.3]  # 末端位置 (x, y, z)

    # 示例 2: 旋转 45 度
    T_ee_2 = np.eye(4, dtype=np.float32)
    theta = np.pi / 4
    T_ee_2[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])
    T_ee_2[:3, 3] = [0.3, 0.1, 0.4]

    # 机械臂参数 (G1 机器人)
    shoulder_pos = np.array([0.0, 0.0, 0.0])
    wrist_pos = np.array([0.3, 0.0, 0.3])
    upper_arm_length = 0.25  # 米
    forearm_length = 0.25    # 米

    # ================================================================
    # 求解 IK
    # ================================================================
    print("\n[求解 IK]")

    for i, T_ee in enumerate([T_ee_1, T_ee_2], 1):
        print(f"\n--- 帧数 {i} ---")
        print(f"末端位姿:\n{T_ee}")

        joint_angles = controller.compute_ik(
            T_ee=T_ee,
            shoulder_pos=shoulder_pos,
            wrist_pos=T_ee[:3, 3],  # 简化: 使用末端位置代替腕部
            upper_arm_length=upper_arm_length,
            forearm_length=forearm_length
        )

        print(f"\n关节角度 (弧度):")
        for name, angle in zip(controller.get_joint_names(), joint_angles):
            print(f"  {name:30s}: {angle:8.4f} rad ({np.degrees(angle):6.2f}°)")

    print("\n" + "=" * 60)
    print("VR 接入示例完成!")
    print("=" * 60)


# ============================================================================
# ROS 接入示例 (可选)
# ============================================================================

class ROSBridge:
    """
    ROS 接入示例 (伪代码)

    如果需要接入 ROS，可以参考以下代码结构。
    """

    def __init__(self):
        import rospy
        from sensor_msgs.msg import JointState
        from geometry_msgs.msg import PoseStamped

        # 初始化 IK 控制器
        self.ik_controller = VR IKController()

        # ROS 发布者
        self.joint_pub = rospy.Publisher('/joint_commands', JointState, queue_size=10)

        # ROS 订阅者
        rospy.Subscriber('/vr/end_effector_pose', PoseStamped, self.pose_callback)

    def pose_callback(self, msg):
        """
        VR 位姿回调

        Args:
            msg: geometry_msgs/PoseStamped
                - pose.position: 末端位置
                - pose.orientation: 末端姿态 (四元数)
        """
        import tf.transformations as tf

        # 转换四元数到旋转矩阵
        q = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        R = tf.quaternion_matrix(q)[:3, :3]

        # 构造齐次变换矩阵
        T_ee = np.eye(4)
        T_ee[:3, :3] = R
        T_ee[:3, 3] = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]

        # 求解 IK
        joint_angles = self.ik_controller.compute_ik(
            T_ee=T_ee,
            shoulder_pos=np.array([0.0, 0.0, 0.0]),
            wrist_pos=T_ee[:3, 3],
            upper_arm_length=0.25,
            forearm_length=0.25
        )

        # 发布关节角度
        joint_msg = JointState()
        joint_msg.name = self.ik_controller.get_joint_names()
        joint_msg.position = joint_angles.tolist()
        self.joint_pub.publish(joint_msg)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='PiM-IK VR 接入示例')
    parser.add_argument('--model', type=str,
                        default='./checkpoints/20260227_111614/best_model.pth',
                        help='模型检查点路径')
    parser.add_argument('--pinocchio', type=str,
                        default='/home/ygx/g1_left_arm_model_cache.pkl',
                        help='Pinocchio 模型路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备')

    args = parser.parse_args()

    # 运行示例
    example_vr_integration()


if __name__ == '__main__':
    main()
