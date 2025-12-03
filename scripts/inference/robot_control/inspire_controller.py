#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
因时（Inspire）灵巧手控制类 - 适配带触觉版本
"""

import numpy as np
import threading
import time
from enum import IntEnum
from typing import Optional, Tuple, Dict, Any

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

# 话题定义
kTopicInspireCtrlLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
kTopicInspireStateRight = "rt/inspire_hand/state/r"

# 机器人配置
Inspire_Num_Motors = 6

class MotorState:
    """电机状态类"""
    def __init__(self):
        self.q = None      # 关节角度
        self.dq = None     # 关节角速度

class InspireHandState:
    """因时手状态类"""
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Inspire_Num_Motors * 2)]  # 左右手

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

class InspireController:
    """因时灵巧手控制器 - 适配带触觉版本"""
    
    def __init__(self, hand_freq: float = 100.0):
        """
        初始化因时灵巧手控制器
        
        Args:
            hand_freq: 手部控制频率 (Hz)
        """
        print("初始化因时灵巧手控制器（带触觉版本）...")
        
        # 控制参数
        self.hand_freq = hand_freq
        self.hand_dt = 1.0 / hand_freq
        
        # 目标值 - 使用归一化值 [0, 1]
        self.left_hand_target  = np.full(Inspire_Num_Motors, 1.0)  # 1.0 = 全开
        self.right_hand_target = np.full(Inspire_Num_Motors, 1.0)
        
        # 状态缓冲区
        self.left_hand_state_buffer = DataBuffer()
        self.right_hand_state_buffer = DataBuffer()
        
        # 初始化DDS通信
        self._init_dds_state()
        
        # 初始化控制线程
        self._init_control_threads()
        
        print("因时灵巧手控制器（带触觉版本）初始化完成!")

    def _init_dds_state(self):
        """初始化DDS通信 - 修改为带触觉版本的通信方式"""
        # ChannelFactoryInitialize(0) # 当机械臂和灵巧手同时使用时，需要注释掉
        self._init_hand_dds_state()

    def _init_hand_dds_state(self):
        """初始化手部DDS通信 - 完全重写"""
        # 初始化左右手命令发布器
        self.left_hand_cmd_publisher = ChannelPublisher(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self.left_hand_cmd_publisher.Init()
        self.right_hand_cmd_publisher = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self.right_hand_cmd_publisher.Init()

        # 初始化左右手状态订阅器
        self.left_hand_state_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self.left_hand_state_subscriber.Init()
        self.right_hand_state_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self.right_hand_state_subscriber.Init()

        # 启动状态订阅线程
        self._start_hand_state_subscription()

    
    
    
    def _start_hand_state_subscription(self):
        """手部状态订阅线程 - 修改为分开订阅左右手"""
        self.hand_subscribe_thread = threading.Thread(target=self._subscribe_hand_state)
        self.hand_subscribe_thread.daemon = True
        self.hand_subscribe_thread.start()
        
        # 等待状态数据
        wait_count = 0
        while True:
            left_data = self.left_hand_state_buffer.get_data()
            right_data = self.right_hand_state_buffer.get_data()
    
            # 修复：使用明确的判断条件
            if left_data is not None and right_data is not None:
                break
        
            time.sleep(0.01)
            wait_count += 1
            if wait_count % 100 == 0:  # 每1秒打印一次
                print(f"[InspireController] 等待因时手状态数据... (已等待 {wait_count/100:.1f}秒)")
            if wait_count > 500:  # 5秒超时
                print("[InspireController] 警告: 等待手部状态数据超时")
                break
        
        left_hand_q, right_hand_q = self.get_current_dual_hand_q()
        print(f"当前灵巧手关节位置 - 左手: {left_hand_q}, 右手: {right_hand_q}")

    def _subscribe_hand_state(self):
        """订阅因时手状态 - 修改为inspire_dds格式"""
        while True:
            # 左手状态
            left_msg = self.left_hand_state_subscriber.Read()
            if left_msg is not None:
                if hasattr(left_msg, 'angle_act') and len(left_msg.angle_act) == Inspire_Num_Motors:
                    # 转换为归一化值 [0, 1]
                    left_state = np.array([angle / 1000.0 for angle in left_msg.angle_act])
                    self.left_hand_state_buffer.set_data(left_state)
            
            # 右手状态
            right_msg = self.right_hand_state_subscriber.Read()
            if right_msg is not None:
                if hasattr(right_msg, 'angle_act') and len(right_msg.angle_act) == Inspire_Num_Motors:
                    # 转换为归一化值 [0, 1]
                    right_state = np.array([angle / 1000.0 for angle in right_msg.angle_act])
                    self.right_hand_state_buffer.set_data(right_state)
            
            time.sleep(0.002)

    def _init_control_threads(self):
        """启动控制线程"""
        self._start_hand_threads()

    def _start_hand_threads(self):
        """启动手部控制线程"""
        self.hand_control_thread = threading.Thread(target=self._hand_control_loop)
        self.hand_control_lock = threading.Lock()
        self.hand_control_thread.daemon = True
        self.hand_control_thread.start()

    def _hand_control_loop(self):
        """因时手控制循环 - 修改为inspire_dds格式"""
        while True:
            start_time = time.time()
            
            with self.hand_control_lock:
                left_target = self.left_hand_target.copy()
                right_target = self.right_hand_target.copy()
            
            # 发送左手命令 - 修改为inspire_dds格式
            left_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
            left_cmd_msg.angle_set = [int(np.clip(val * 1000, 0, 1000)) for val in left_target]
            left_cmd_msg.mode = 0b0001  # Mode 1: Angle control
            self.left_hand_cmd_publisher.Write(left_cmd_msg)
            
            # 发送右手命令
            right_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
            right_cmd_msg.angle_set = [int(np.clip(val * 1000, 0, 1000)) for val in right_target]
            right_cmd_msg.mode = 0b0001  # Mode 1: Angle control
            self.right_hand_cmd_publisher.Write(right_cmd_msg)
            
            # 控制频率
            elapsed = time.time() - start_time
            sleep_time = max(0, self.hand_dt - elapsed)
            time.sleep(sleep_time)

    # 公共接口方法 - 保持不变
    def set_hand_targets(self, left_hand_q: np.ndarray, right_hand_q: np.ndarray):
        """
        设置双手目标位置
        
        Args:
            left_hand_q: 左手关节角度 (6维，归一化值 [0, 1])
            right_hand_q: 右手关节角度 (6维，归一化值 [0, 1])
        """
        if left_hand_q.shape[0] != 6 or right_hand_q.shape[0] != 6:
            raise ValueError("手部关节角度必须为6维")
        
        # 确保值在 [0, 1] 范围内
        left_hand_q = np.clip(left_hand_q, 0.0, 1.0)
        right_hand_q = np.clip(right_hand_q, 0.0, 1.0)
        
        with self.hand_control_lock:
            self.left_hand_target = left_hand_q.copy()
            self.right_hand_target = right_hand_q.copy()

    def get_current_dual_hand_q(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取双手关节位置 - 修改为从分开的buffer获取"""
        left_state = self.left_hand_state_buffer.get_data()
        right_state = self.right_hand_state_buffer.get_data()
        
        if left_state is None:
            left_positions = np.zeros(6)
        else:
            left_positions = left_state.copy()
            
        if right_state is None:
            right_positions = np.zeros(6)
        else:
            right_positions = right_state.copy()
        
        return left_positions, right_positions

    def ctrl_dual_hand_go_home(self):
        '''移动双手到初始位置（全开）'''
        print("[InspireController] ctrl_dual_hand_go_home start...")
        with self.hand_control_lock:
            self.left_hand_target = np.full(Inspire_Num_Motors, 1.0)  # 1.0 = 全开
            self.right_hand_target = np.full(Inspire_Num_Motors, 1.0)
        
        # 等待到达目标位置
        tolerance = 0.05
        while True:
            left_hand_q, right_hand_q = self.get_current_dual_hand_q()
            
            if (np.all(np.abs(left_hand_q - 1.0) < tolerance) and 
                np.all(np.abs(right_hand_q - 1.0) < tolerance)):
                print("[InspireController] 灵巧手已到达初始位置（全开）")
                break
            time.sleep(0.05)

    def get_hand_state(self) -> Dict[str, Any]:
        """获取手部状态"""
        left_hand_q, right_hand_q = self.get_current_dual_hand_q()
        
        return {
            'left_hand_q': left_hand_q,
            'right_hand_q': right_hand_q,
            'timestamp': time.time()
        }
    

class InspireLeftHandIndex(IntEnum):
    """因时左手关节索引 - 带触觉版本"""
    kLeftHandPinky = 0
    kLeftHandRing = 1
    kLeftHandMiddle = 2
    kLeftHandIndex = 3
    kLeftHandThumbBend = 4
    kLeftHandThumbRotation = 5

class InspireRightHandIndex(IntEnum):
    """因时右手关节索引 - 带触觉版本"""
    kRightHandPinky = 0
    kRightHandRing = 1
    kRightHandMiddle = 2
    kRightHandIndex = 3
    kRightHandThumbBend = 4
    kRightHandThumbRotation = 5


# 使用示例
if __name__ == "__main__":
    # 创建控制器
    controller = InspireController()
    
    try:
        # 获取当前状态
        state = controller.get_hand_state()
        print("当前手部状态:")
        print(f"左手关节角度: {state['left_hand_q']}")
        print(f"右手关节角度: {state['right_hand_q']}")
        
        # 设置目标位置（0.5 = 半开）
        left_hand_target = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        right_hand_target = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
        
        controller.set_hand_targets(left_hand_target, right_hand_target)
        
        # 运行一段时间
        time.sleep(5.0)
        
        # 回到初始位置（全开）
        controller.ctrl_dual_hand_go_home()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        print("程序结束")