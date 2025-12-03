#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 + 强脑(Brainco)灵巧手组合控制类
包含7自由度手臂+6自由度强脑手的状态实时异步获取及动作写入功能
"""

import numpy as np
import threading
import time
from enum import IntEnum
from typing import Optional, Tuple, Dict, Any

from .g1_controller import G1Controller, G1JointArmIndex, G1JointIndex
from .brainco_controller import BraincoController, BraincoLeftHandIndex, BraincoRightHandIndex

class G1BraincoController:
    """G1 + 强脑灵巧手组合控制器"""
    
    def __init__(self, control_freq: float = 250.0, hand_freq: float = 100.0, simulation_mode: bool = False):
        """
        初始化G1 + 强脑灵巧手组合控制器
        
        Args:
            control_freq: 控制频率 (Hz)
            hand_freq: 手部控制频率 (Hz)
            simulation_mode: 是否为仿真模式
        """
        print("初始化 G1 + 强脑灵巧手组合控制器...")
        
        # 创建子控制器
        self.g1_controller = G1Controller(control_freq)
        self.brainco_controller = BraincoController(hand_freq, simulation_mode)
        
        print("G1 + 强脑灵巧手组合控制器初始化完成!")

    # G1 控制器方法代理 - 保持不变
    def set_arm_targets(self, left_arm_q: np.ndarray, right_arm_q: np.ndarray):
        """设置双臂目标位置"""
        self.g1_controller.set_arm_targets(left_arm_q, right_arm_q)

    def get_current_dual_arm_q(self):
        """获取双臂关节位置"""
        return self.g1_controller.get_current_dual_arm_q()

    def get_current_dual_arm_dq(self):
        """获取双臂关节速度"""
        return self.g1_controller.get_current_dual_arm_dq()

    def get_current_motor_q(self) -> np.ndarray:
        """获取所有关节位置"""
        return self.g1_controller.get_current_motor_q()

    def ctrl_dual_arm_go_home(self):
        """双臂回到初始位置"""
        self.g1_controller.ctrl_dual_arm_go_home()

    def set_velocity_limit(self, velocity_limit: float):
        """设置速度限制"""
        self.g1_controller.set_velocity_limit(velocity_limit)

    # 强脑手控制器方法代理 - 接口保持不变，内部实现不同
    def set_hand_targets(self, left_hand_q: np.ndarray, right_hand_q: np.ndarray):
        """设置双手目标位置"""
        self.brainco_controller.set_hand_targets(left_hand_q, right_hand_q)

    def get_current_dual_hand_q(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取双手关节位置"""
        return self.brainco_controller.get_current_dual_hand_q()

    def ctrl_dual_hand_go_home(self):
        """双手回到初始位置"""
        self.brainco_controller.ctrl_dual_hand_go_home()

    # 强脑手特有方法
    def set_hand_gains(self, kp: float = 100.0, kd: float = 5.0):
        """
        设置手部控制增益
        
        Args:
            kp: 位置刚度增益
            kd: 阻尼增益
        """
        self.brainco_controller.set_hand_gains(kp, kd)

    # 组合方法 - 保持不变
    def ctrl_go_home(self):
        """回到初始位置"""
        print("[G1BraincoController] 回到初始位置...")
        
        # 手臂回到零位
        self.ctrl_dual_arm_go_home()
        
        # 手部回到零位
        self.ctrl_dual_hand_go_home()

    def get_robot_state(self) -> Dict[str, Any]:
        """获取完整机器人状态"""
        # 获取G1状态
        g1_state = self.g1_controller.get_robot_state()
        
        # 获取手部状态
        hand_state = self.brainco_controller.get_hand_state()
        
        # 合并状态
        return {
            **g1_state,
            **hand_state
        }


# 使用示例
if __name__ == "__main__":
    # 创建控制器
    controller = G1BraincoController(simulation_mode=False)
    
    try:
        # 获取当前状态
        state = controller.get_robot_state()
        print("当前机器人状态:")
        print(f"左臂关节角度: {state['left_arm_q']}")
        print(f"右臂关节角度: {state['right_arm_q']}")
        print(f"左手关节角度: {state['left_hand_q']}")
        print(f"右手关节角度: {state['right_hand_q']}")
        
        # 设置手部控制增益
        controller.set_hand_gains(kp=150.0, kd=8.0)
        
        # 设置目标位置
        # 注意：强脑手使用实际角度值（弧度），不是归一化值
        left_arm_target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        right_arm_target = np.array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7])
        
        # 强脑手目标位置（实际角度值）
        left_hand_target = np.array([0.5, 0.3, 0.2, 0.1, 0.0, -0.1])  # 弧度
        right_hand_target = np.array([0.3, 0.2, 0.1, 0.0, -0.1, -0.2])  # 弧度
        
        controller.set_arm_targets(left_arm_target, right_arm_target)
        controller.set_hand_targets(left_hand_target, right_hand_target)
        
        # 运行一段时间
        time.sleep(5.0)
        
        # 回到初始位置
        controller.ctrl_go_home()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        print("程序结束")