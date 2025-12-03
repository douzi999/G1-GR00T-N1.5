#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 机器人控制类
包含7自由度手臂的状态实时异步获取及动作写入功能
支持 motion 模式
"""

import numpy as np
import threading
import time
from enum import IntEnum
from typing import Optional, Tuple, Dict, Any

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

# 话题定义
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

# 机器人配置
G1_Num_Motors = 35

class MotorState:
    """电机状态类"""
    def __init__(self):
        self.q = None      # 关节角度
        self.dq = None     # 关节角速度

class G1LowState:
    """G1机器人状态类"""
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_Num_Motors)]

class DataBuffer:
    """线程安全的数据缓冲区"""
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def get_data(self):
        with self.lock:
            return self.data

    def set_data(self, data):
        with self.lock:
            self.data = data

class G1Controller:
    """G1 机器人控制器"""
    
    def __init__(self, control_freq: float = 250.0, motion_mode: bool = False):
        """
        初始化G1 机器人控制器
        
        Args:
            control_freq: 控制频率 (Hz)
            motion_mode: 是否启用 motion 模式
        """
        print("初始化 G1 机器人控制器...")
        self._speed_gradual_max = False
        
        # 控制参数
        self.control_freq = control_freq
        self.control_dt = 1.0 / control_freq
        self.motion_mode = motion_mode  # motion 模式
        
        # 手臂控制参数
        self.arm_velocity_limit = 20.0
        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5
        
        # 目标值
        self.arm_q_target = np.zeros(14)  # 双臂14个关节
        
        # 状态缓冲区
        self.g1_state_buffer = DataBuffer()
        
        # 初始化DDS通信
        self._init_dds_state()
        
        # 初始化控制线程
        self._init_control_threads()
        
        print(f"G1 机器人控制器初始化完成! Motion模式: {motion_mode}")

    def _init_dds_state(self):
        """初始化DDS通信"""
        ChannelFactoryInitialize(0)
        self._init_arm_dds_state()

    def _init_arm_dds_state(self):
        # G1机器人通信 - 根据 motion 模式选择话题
        if self.motion_mode:
            # motion 模式使用 arm_sdk 话题
            self.g1_cmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
            print("使用 motion 模式 (rt/arm_sdk 话题)")
        else:
            # 正常模式使用 lowcmd 话题
            self.g1_cmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
            print("使用正常模式 (rt/lowcmd 话题)")
            
        self.g1_cmd_publisher.Init()
        self.g1_state_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.g1_state_subscriber.Init()
        
        # 启动状态订阅线程
        self._start_arm_state_subscription()
        
        # CRC校验
        self.crc = CRC()
        
        # 初始化G1控制消息
        self.g1_msg = unitree_hg_msg_dds__LowCmd_()
        self.g1_msg.mode_pr = 0
        self.g1_msg.mode_machine = self._get_mode_machine()
        
        # 锁定其他关节
        print("Lock all joints...\n")
        self._lock_other_joints()

    def _start_arm_state_subscription(self):
        """G1状态订阅线程"""
        self.g1_subscribe_thread = threading.Thread(target=self._subscribe_g1_state)
        self.g1_subscribe_thread.daemon = True
        self.g1_subscribe_thread.start()
        # 等待状态数据
        while not self.g1_state_buffer.get_data():
            time.sleep(0.01)
            print("[G1Controller] 等待G1状态数据...")
        self.all_motor_q = self.get_current_motor_q()
        print(f"当前机器人关节位置:\n{self.all_motor_q} \n")

    def _init_control_threads(self):
        """启动控制线程"""
        # G1控制线程
        self.g1_control_thread = threading.Thread(target=self._control_loop)
        self.arm_control_lock = threading.Lock()
        self.g1_control_thread.daemon = True
        self.g1_control_thread.start()

    def _subscribe_g1_state(self):
        """订阅G1机器人状态"""
        while True:
            msg = self.g1_state_subscriber.Read()
            if msg is not None:
                g1_state = G1LowState()
                for id in range(G1_Num_Motors):
                    g1_state.motor_state[id].q = msg.motor_state[id].q
                    g1_state.motor_state[id].dq = msg.motor_state[id].dq
                self.g1_state_buffer.set_data(g1_state)
            time.sleep(0.002)

    def _control_loop(self):
        """G1控制循环"""
        while True:
            start_time = time.time()
            
            # motion 模式特殊处理
            if self.motion_mode:
                self.g1_msg.motor_cmd[G1JointIndex.kNotUsedJoint0].q = 1.0
            
            with self.arm_control_lock:
                arm_q_target = self.arm_q_target.copy()
            
            # 速度限制
            clipped_arm_q = self.clip_arm_q_target(
                target_q=arm_q_target, 
                velocity_limit=self.arm_velocity_limit
            )
            
            # 设置电机命令
            for idx, joint_id in enumerate(G1JointArmIndex):
                self.g1_msg.motor_cmd[joint_id].q = clipped_arm_q[idx]
                self.g1_msg.motor_cmd[joint_id].dq = 0
                self.g1_msg.motor_cmd[joint_id].tau = 0
                
            # 发送命令
            self.g1_msg.crc = self.crc.Crc(self.g1_msg)
            self.g1_cmd_publisher.Write(self.g1_msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))
                
            # 控制频率
            elapsed = time.time() - start_time
            sleep_time = max(0, self.control_dt - elapsed)
            time.sleep(sleep_time)

    def clip_arm_q_target(self, target_q, velocity_limit=20.0):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _lock_other_joints(self):
        """锁定非手臂关节"""
        arm_indices = set(member.value for member in G1JointArmIndex)
        for id in G1JointIndex:
            self.g1_msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._is_wrist_motor(id):
                    self.g1_msg.motor_cmd[id].kp = self.kp_wrist
                    self.g1_msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.g1_msg.motor_cmd[id].kp = self.kp_low
                    self.g1_msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._is_weak_motor(id):
                    self.g1_msg.motor_cmd[id].kp = self.kp_low
                    self.g1_msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.g1_msg.motor_cmd[id].kp = self.kp_high
                    self.g1_msg.motor_cmd[id].kd = self.kd_high
            self.g1_msg.motor_cmd[id].q  = self.all_motor_q[id]
        print("Lock OK!\n")

    def _get_mode_machine(self):
        """获取当前模式"""
        return self.g1_state_subscriber.Read().mode_machine

    # 公共接口方法
    def set_arm_targets(self, left_arm_q: np.ndarray, right_arm_q: np.ndarray):
        """
        设置双臂目标位置

        Args:
            left_arm_q: 左臂关节角度 (7维)
            right_arm_q: 右臂关节角度 (7维)
        """
        if left_arm_q.shape[0] != 7 or right_arm_q.shape[0] != 7:
            raise ValueError("手臂关节角度必须为7维")

        with self.arm_control_lock:
            self.arm_q_target[:7] = left_arm_q
            self.arm_q_target[7:] = right_arm_q

    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.g1_state_buffer.get_data().motor_state[id].q for id in G1JointArmIndex])

    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.g1_state_buffer.get_data().motor_state[id].dq for id in G1JointArmIndex])
    
    def get_current_motor_q(self) -> np.ndarray:
        """获取所有关节位置"""
        g1_state = self.g1_state_buffer.get_data()
        if g1_state is None:
            return np.zeros(G1_Num_Motors)
        
        all_positions = np.zeros(G1_Num_Motors)
        for i in range(G1_Num_Motors):
            all_positions[i] = g1_state.motor_state[i].q
        return all_positions

    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        print("[G1Controller] ctrl_dual_arm_go_home start...")
        with self.arm_control_lock:
            self.arm_q_target = np.zeros(14)
            
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        
        while True:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                # motion 模式特殊处理
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.g1_msg.motor_cmd[G1JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                print("[G1Controller] both arms have reached the home position.")
                break
            time.sleep(0.05)

    def set_velocity_limit(self, velocity_limit: float):
        """设置速度限制"""
        self.arm_velocity_limit = velocity_limit

    def get_robot_state(self) -> Dict[str, Any]:
        """获取机器人手臂状态"""
        arm_q = self.get_current_dual_arm_q()
        arm_dq = self.get_current_dual_arm_dq()
        
        return {
            'left_arm_q': arm_q[:7],
            'right_arm_q': arm_q[7:],
            'left_arm_dq': arm_dq[:7],
            'right_arm_dq': arm_dq[7:],
            'timestamp': time.time()
        }

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def get_config(self) -> Dict[str, Any]:
        """获取控制器配置"""
        return {
            'motion_mode': self.motion_mode,
            'control_freq': self.control_freq,
            'velocity_limit': self.arm_velocity_limit
        }

    def _is_weak_motor(self, joint_id) -> bool:
        """判断是否为弱电机"""
        weak_motors = [
            G1JointIndex.kLeftAnklePitch.value,
            G1JointIndex.kRightAnklePitch.value,
            G1JointIndex.kLeftShoulderPitch.value,
            G1JointIndex.kLeftShoulderRoll.value,
            G1JointIndex.kLeftShoulderYaw.value,
            G1JointIndex.kLeftElbow.value,
            G1JointIndex.kRightShoulderPitch.value,
            G1JointIndex.kRightShoulderRoll.value,
            G1JointIndex.kRightShoulderYaw.value,
            G1JointIndex.kRightElbow.value,
        ]
        return joint_id.value in weak_motors

    def _is_wrist_motor(self, joint_id) -> bool:
        """判断是否为腕部电机"""
        wrist_motors = [
            G1JointIndex.kLeftWristRoll.value,
            G1JointIndex.kLeftWristPitch.value,
            G1JointIndex.kLeftWristYaw.value,
            G1JointIndex.kRightWristRoll.value,
            G1JointIndex.kRightWristPitch.value,
            G1JointIndex.kRightWristYaw.value,
        ]
        return joint_id.value in wrist_motors

# 关节索引定义
class G1JointArmIndex(IntEnum):
    """G1手臂关节索引"""
    # 左臂
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21
    
    # 右臂
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1JointIndex(IntEnum):
    """G1所有关节索引"""
    # 左腿
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5
    
    # 右腿
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11
    
    # 腰部
    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14
    
    # 左臂
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristYaw = 21
    
    # 右臂
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # 未使用
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="G1控制器测试")
    parser.add_argument("--motion", action="store_true", help="启用 motion 模式")
    args = parser.parse_args()
    
    # 创建控制器
    controller = G1Controller(motion_mode=args.motion)
    
    try:
        # 获取当前状态
        state = controller.get_robot_state()
        print("当前机器人状态:")
        print(f"左臂关节角度: {state['left_arm_q']}")
        print(f"右臂关节角度: {state['right_arm_q']}")
        
        # 设置目标位置
        left_arm_target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        right_arm_target = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7])
        
        controller.set_arm_targets(left_arm_target, right_arm_target)
        
        # 运行一段时间
        time.sleep(5.0)
        
        # 回到初始位置
        controller.ctrl_dual_arm_go_home()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        print("程序结束")