#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强脑(Brainco)灵巧手控制类
"""

import numpy as np
import threading
import time
from enum import IntEnum
from typing import Optional, Tuple, Dict, Any

# 强脑手SDK导入
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

# 话题定义
kTopicBraincoCtrlLeft = "rt/brainco/left/cmd"
kTopicBraincoCtrlRight = "rt/brainco/right/cmd"
kTopicBraincoStateLeft = "rt/brainco/left/state"
kTopicBraincoStateRight = "rt/brainco/right/state"

# 机器人配置 - 强脑手6个关节
Brainco_Num_Motors = 6

class MotorState:
    """电机状态类"""
    def __init__(self):
        self.q = None      # 关节角度
        self.dq = None     # 关节角速度

class BraincoHandState:
    """强脑手状态类"""
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(Brainco_Num_Motors * 2)]  # 左右手

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

class BraincoController:
    """强脑灵巧手控制器"""
    
    def __init__(self, hand_freq: float = 100.0, simulation_mode: bool = False):
        """
        初始化强脑灵巧手控制器
        
        Args:
            hand_freq: 手部控制频率 (Hz)
            simulation_mode: 是否为仿真模式
        """
        print("初始化强脑灵巧手控制器...")
        
        # 控制参数
        self.hand_freq = hand_freq
        self.hand_dt = 1.0 / hand_freq
        self.simulation_mode = simulation_mode
        
        # 目标值 - 使用实际角度值 (弧度)
        self.left_hand_target = np.zeros(Brainco_Num_Motors)
        self.right_hand_target = np.zeros(Brainco_Num_Motors)
        
        # 状态缓冲区
        self.left_hand_state_buffer = DataBuffer()
        self.right_hand_state_buffer = DataBuffer()
        
        # 初始化DDS通信
        self._init_dds_state()
        
        # 初始化控制线程
        self._init_control_threads()
        
        print("强脑灵巧手控制器初始化完成!")

    def _init_dds_state(self):
        """初始化DDS通信"""
        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)
            
        self._init_hand_dds_state()

    def _init_hand_dds_state(self):
        """初始化手部DDS通信"""
        # 初始化左右手命令发布器
        self.left_hand_cmd_publisher = ChannelPublisher(kTopicBraincoCtrlLeft, MotorCmds_)
        self.left_hand_cmd_publisher.Init()
        self.right_hand_cmd_publisher = ChannelPublisher(kTopicBraincoCtrlRight, MotorCmds_)
        self.right_hand_cmd_publisher.Init()

        # 初始化左右手状态订阅器
        self.left_hand_state_subscriber = ChannelSubscriber(kTopicBraincoStateLeft, MotorStates_)
        self.left_hand_state_subscriber.Init()
        self.right_hand_state_subscriber = ChannelSubscriber(kTopicBraincoStateRight, MotorStates_)
        self.right_hand_state_subscriber.Init()

        # 初始化命令消息
        self._init_command_messages()
        
        # 启动状态订阅线程
        self._start_hand_state_subscription()

    def _init_command_messages(self):
        """初始化命令消息"""
        # 左手命令消息
        self.left_hand_msg = MotorCmds_()
        self.left_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(Brainco_Num_Motors)]
        
        # 右手命令消息
        self.right_hand_msg = MotorCmds_()
        self.right_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(Brainco_Num_Motors)]
        
        # 初始化默认参数
        for i in range(Brainco_Num_Motors):
            # 左手
            self.left_hand_msg.cmds[i].q = 0.0
            self.left_hand_msg.cmds[i].dq = 1.0  # 默认角速度
            self.left_hand_msg.cmds[i].kp = 100.0  # 默认刚度
            self.left_hand_msg.cmds[i].kd = 5.0    # 默认阻尼
            self.left_hand_msg.cmds[i].tau = 0.0   # 默认力矩
            
            # 右手
            self.right_hand_msg.cmds[i].q = 0.0
            self.right_hand_msg.cmds[i].dq = 1.0
            self.right_hand_msg.cmds[i].kp = 100.0
            self.right_hand_msg.cmds[i].kd = 5.0
            self.right_hand_msg.cmds[i].tau = 0.0

    def _start_hand_state_subscription(self):
        """手部状态订阅线程"""
        self.hand_subscribe_thread = threading.Thread(target=self._subscribe_hand_state)
        self.hand_subscribe_thread.daemon = True
        self.hand_subscribe_thread.start()
        
        # 等待状态数据
        wait_count = 0
        while True:
            left_data = self.left_hand_state_buffer.get_data()
            right_data = self.right_hand_state_buffer.get_data()
    
            if left_data is not None and right_data is not None:
                break
        
            time.sleep(0.01)
            wait_count += 1
            if wait_count % 100 == 0:  # 每1秒打印一次
                print(f"[BraincoController] 等待强脑手状态数据... (已等待 {wait_count/100:.1f}秒)")
            if wait_count > 500:  # 5秒超时
                print("[BraincoController] 警告: 等待手部状态数据超时")
                break
        
        left_hand_q, right_hand_q = self.get_current_dual_hand_q()
        print(f"当前强脑手关节位置 - 左手: {left_hand_q}, 右手: {right_hand_q}")

    def _subscribe_hand_state(self):
        """订阅强脑手状态"""
        while True:
            # 左手状态
            left_msg = self.left_hand_state_subscriber.Read()
            if left_msg is not None:
                if hasattr(left_msg, 'states') and len(left_msg.states) >= Brainco_Num_Motors:
                    left_state = np.zeros(Brainco_Num_Motors)
                    for idx, joint_id in enumerate(BraincoLeftHandIndex):
                        if joint_id < len(left_msg.states):
                            left_state[idx] = left_msg.states[joint_id].q
                    self.left_hand_state_buffer.set_data(left_state)
            
            # 右手状态
            right_msg = self.right_hand_state_subscriber.Read()
            if right_msg is not None:
                if hasattr(right_msg, 'states') and len(right_msg.states) >= Brainco_Num_Motors:
                    right_state = np.zeros(Brainco_Num_Motors)
                    for idx, joint_id in enumerate(BraincoRightHandIndex):
                        if joint_id < len(right_msg.states):
                            right_state[idx] = right_msg.states[joint_id].q
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
        """强脑手控制循环"""
        while True:
            start_time = time.time()
            
            with self.hand_control_lock:
                left_target = self.left_hand_target.copy()
                right_target = self.right_hand_target.copy()
            
            # 发送左手命令
            for idx, joint_id in enumerate(BraincoLeftHandIndex):
                if joint_id < len(self.left_hand_msg.cmds):
                    self.left_hand_msg.cmds[joint_id].q = left_target[idx]
            self.left_hand_cmd_publisher.Write(self.left_hand_msg)
            
            # 发送右手命令
            for idx, joint_id in enumerate(BraincoRightHandIndex):
                if joint_id < len(self.right_hand_msg.cmds):
                    self.right_hand_msg.cmds[joint_id].q = right_target[idx]
            self.right_hand_cmd_publisher.Write(self.right_hand_msg)
            
            # 控制频率
            elapsed = time.time() - start_time
            sleep_time = max(0, self.hand_dt - elapsed)
            time.sleep(sleep_time)

    def set_hand_targets(self, left_hand_q: np.ndarray, right_hand_q: np.ndarray):
        """
        设置双手目标位置
        
        Args:
            left_hand_q: 左手关节角度 (6维，实际角度值)
            right_hand_q: 右手关节角度 (6维，实际角度值)
        """
        if left_hand_q.shape[0] != Brainco_Num_Motors or right_hand_q.shape[0] != Brainco_Num_Motors:
            raise ValueError(f"手部关节角度必须为{Brainco_Num_Motors}维")
        
        # 强脑手使用实际角度值，不需要归一化
        # 可以根据需要添加角度限制
        left_hand_q = np.clip(left_hand_q, -3.14, 3.14)  # 限制在±π范围内
        right_hand_q = np.clip(right_hand_q, -3.14, 3.14)
        
        with self.hand_control_lock:
            self.left_hand_target = left_hand_q.copy()
            self.right_hand_target = right_hand_q.copy()

    def get_current_dual_hand_q(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取双手关节位置"""
        left_state = self.left_hand_state_buffer.get_data()
        right_state = self.right_hand_state_buffer.get_data()
        
        if left_state is None:
            left_positions = np.zeros(Brainco_Num_Motors)
        else:
            left_positions = left_state.copy()
            
        if right_state is None:
            right_positions = np.zeros(Brainco_Num_Motors)
        else:
            right_positions = right_state.copy()
        
        return left_positions, right_positions

    def ctrl_dual_hand_go_home(self):
        '''移动双手到初始位置（全开）'''
        print("[BraincoController] ctrl_dual_hand_go_home start...")
        
        # 强脑手的初始位置（根据实际配置调整）
        home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        with self.hand_control_lock:
            self.left_hand_target = home_position.copy()
            self.right_hand_target = home_position.copy()
        
        # 等待到达目标位置
        tolerance = 0.05
        while True:
            left_hand_q, right_hand_q = self.get_current_dual_hand_q()
            
            if (np.all(np.abs(left_hand_q - home_position) < tolerance) and 
                np.all(np.abs(right_hand_q - home_position) < tolerance)):
                print("[BraincoController] 强脑手已到达初始位置")
                break
            time.sleep(0.05)

    def set_hand_gains(self, kp: float = 100.0, kd: float = 5.0):
        """
        设置手部控制增益
        
        Args:
            kp: 位置刚度增益
            kd: 阻尼增益
        """
        for i in range(Brainco_Num_Motors):
            self.left_hand_msg.cmds[i].kp = kp
            self.left_hand_msg.cmds[i].kd = kd
            self.right_hand_msg.cmds[i].kp = kp
            self.right_hand_msg.cmds[i].kd = kd

    def get_hand_state(self) -> Dict[str, Any]:
        """获取手部状态"""
        left_hand_q, right_hand_q = self.get_current_dual_hand_q()
        
        return {
            'left_hand_q': left_hand_q,
            'right_hand_q': right_hand_q,
            'timestamp': time.time()
        }


class BraincoLeftHandIndex(IntEnum):
    """强脑左手关节索引"""
    kLeftHandThumb = 0
    kLeftHandThumbAux = 1
    kLeftHandIndex = 2
    kLeftHandMiddle = 3
    kLeftHandRing = 4
    kLeftHandPinky = 5

class BraincoRightHandIndex(IntEnum):
    """强脑右手关节索引"""
    kRightHandThumb = 0
    kRightHandThumbAux = 1
    kRightHandIndex = 2
    kRightHandMiddle = 3
    kRightHandRing = 4
    kRightHandPinky = 5


# 使用示例
if __name__ == "__main__":
    # 创建控制器
    controller = BraincoController(simulation_mode=False)
    
    try:
        # 获取当前状态
        state = controller.get_hand_state()
        print("当前手部状态:")
        print(f"左手关节角度: {state['left_hand_q']}")
        print(f"右手关节角度: {state['right_hand_q']}")
        
        # 设置目标位置（实际角度值）
        left_hand_target = np.array([0.5, 0.3, 0.2, 0.1, 0.0, -0.1])
        right_hand_target = np.array([0.3, 0.2, 0.1, 0.0, -0.1, -0.2])
        
        controller.set_hand_targets(left_hand_target, right_hand_target)
        
        # 运行一段时间
        time.sleep(5.0)
        
        # 回到初始位置
        controller.ctrl_dual_hand_go_home()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        print("程序结束")