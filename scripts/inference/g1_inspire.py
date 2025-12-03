#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1 + 因时灵巧手推理执行器 - 适配双摄像头数据集版本
只使用 cam_left_high 和 cam_right_high 两个摄像头
"""

import re
import time
import threading
import numpy as np
import json_numpy
from multiprocessing import shared_memory
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import sys
import os
import requests
import cv2
import termios

from pyarrow import Array

from image_server.image_client import ImageClient
from robot_control.g1_controller import G1Controller
from robot_control.inspire_controller import InspireController
from model_client import ModelClient, ModelClientConfig
import pyarrow.parquet as pq

@dataclass
class InferenceConfig:
    """推理执行器配置 - 双摄像头版本"""

    # 模型服务配置
    model_host: str = "192.168.110.42"
    model_port: int = 6666
    api_token: Optional[str] = None
    timeout: float = 5.0
    use_http: bool = True

    # 图像服务配置
    image_server_host: str = "192.168.123.164"
    image_server_port: int = 5555
    
    # 分辨率 (服务端期望的)
    image_height: int = 480
    image_width: int = 640
    
    use_shared_memory: bool = True
    
    # 2个摄像头的共享内存名称
    cam_left_high_shm_name: str = "tv_image_cam_left_high"
    cam_right_high_shm_name: str = "tv_image_cam_right_high"

    # 控制配置
    control_freq: float = 30.0
    action_horizon: int = 16
    motion_mode: bool = False  # 默认关闭 motion 模式

    # 任务描述
    task_description: str = "pick the red cube on the table."

    test_mode: bool = False

class G1InferenceExecutor:
    """G1 + 因时灵巧手推理执行器 - 双摄像头版本"""
    
    def __init__(self, config: InferenceConfig):
        """
        初始化推理执行器

        Args:
            config: 推理执行器配置
        """
        self.config = config
        self.dt = 1.0 / config.control_freq
        # 默认不执行
        self.is_executing = True

        # 控制状态
        self.is_running = False
        self.control_thread = None
        self.control_lock = threading.Lock()

        # 键盘监控状态
        self.keyboard_monitoring = False
        self.keyboard_thread = None
        self.should_exit = False
        self.should_start_inference = False

        # 初始化模型客户端
        print("初始化模型客户端...")
        model_client_config = ModelClientConfig(
            host=config.model_host,
            port=config.model_port,
            api_token=config.api_token,
            use_http=config.use_http,
            timeout=config.timeout
        )
        self.model_client = ModelClient(model_client_config)
        
        # 初始化机器人控制器
        print("初始化G1机器人控制器...")
        self.g1_controller = G1Controller(
            control_freq=250.0,
            motion_mode=config.motion_mode
        )
        self.inspire_controller = InspireController(hand_freq=100.0)
        
        if config.test_mode:
            print("测试模式：跳过图像和机器人初始化")
            # 测试模式下的初始化
            self.image_clients = {}
            self.camera_shm = {}
            self.camera_arrays = {}
            self.image_lock = None
        else:
            # 正常模式下的完整初始化
            # 初始化共享内存
            self._init_shared_memory()

            # 初始化图像客户端 - 2个摄像头
            self._init_image_clients()

            # 图像缓存
            self.image_lock = threading.Lock()

        print("G1推理执行器初始化完成!")

    def _init_shared_memory(self):
        """初始化共享内存 - 双摄像头版本"""
        print("初始化共享内存...")
        
        # 为2个摄像头创建共享内存
        cam_configs = {
            "cam_left_high": self.config.cam_left_high_shm_name,
            "cam_right_high": self.config.cam_right_high_shm_name,
        }
        
        self.camera_shm = {}
        self.camera_arrays = {}
        
        # 使用服务端期望的分辨率
        img_shape = (self.config.image_height, self.config.image_width, 3)
        
        for cam_name, shm_name in cam_configs.items():
            try:
                self.camera_shm[cam_name] = shared_memory.SharedMemory(
                    create=True,
                    size=int(np.prod(img_shape) * np.uint8().itemsize),
                    name=shm_name
                )
                self.camera_arrays[cam_name] = np.ndarray(
                    img_shape, dtype=np.uint8, buffer=self.camera_shm[cam_name].buf
                )
                print(f"创建共享内存成功: {shm_name}, 大小: {img_shape}")
            except Exception as e:
                print(f"创建共享内存 {cam_name} 失败: {e}")
                self.camera_shm[cam_name] = None
                self.camera_arrays[cam_name] = None

    def _init_image_clients(self):
        """初始化图像客户端 - 2个摄像头版本"""
        print("初始化图像客户端...")
        
        # 为2个摄像头创建不同的图像客户端
        self.image_clients = {}
        
        # 左眼摄像头
        self.image_clients["left_high"] = ImageClient(
            tv_img_shape=(self.config.image_height, self.config.image_width, 3),
            tv_img_shm_name=self.config.cam_left_high_shm_name,
            image_show=False,
            server_address=self.config.image_server_host,
            port=self.config.image_server_port,
            Unit_Test=False
        )
        
        # 右眼摄像头
        self.image_clients["right_high"] = ImageClient(
            tv_img_shape=(self.config.image_height, self.config.image_width, 3),
            tv_img_shm_name=self.config.cam_right_high_shm_name,
            image_show=False,
            server_address=self.config.image_server_host,
            port=self.config.image_server_port,
            Unit_Test=False
        )
        
        # 启动所有图像客户端的接收线程
        for cam_name, client in self.image_clients.items():
            thread = threading.Thread(target=client.receive_process)
            thread.daemon = True
            thread.start()
            print(f"启动 {cam_name} 图像客户端线程")
        
        # 等待图像客户端启动
        time.sleep(2.0)
        print("所有图像客户端初始化完成")

    def _get_camera_images(self) -> Dict[str, np.ndarray]:
        """获取2个摄像头的图像"""
        if self.config.test_mode:
            # 测试模式下返回2个随机图像
            target_shape = (self.config.image_height, self.config.image_width, 3)
            return {
                "cam_left_high": np.random.randint(0, 256, target_shape, dtype=np.uint8),
                "cam_right_high": np.random.randint(0, 256, target_shape, dtype=np.uint8),
            }

        # 从共享内存读取2个摄像头的图像
        images = {}
        for cam_name in ["cam_left_high", "cam_right_high"]:
            if (cam_name in self.camera_arrays and 
                self.camera_arrays[cam_name] is not None):
                
                # 直接从共享内存读取图像
                images[cam_name] = self.camera_arrays[cam_name].copy()
                    
            else:
                # 如果没有图像，返回随机图像作为占位符
                img_shape = (self.config.image_height, self.config.image_width, 3)
                images[cam_name] = np.random.randint(0, 256, img_shape, dtype=np.uint8)
                print(f"警告: 摄像头 {cam_name} 无图像数据，使用随机图像替代")
    
        return images

    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        """获取机器人状态"""
        if self.config.test_mode:
            # 测试模式下返回模拟的机器人状态
            state = {
                "state.left_arm": np.random.rand(1, 7),  # 左臂7个关节
                "state.right_arm": np.random.rand(1, 7), # 右臂7个关节
                "state.left_hand": np.random.rand(1, 6), # 左手6个关节
                "state.right_hand": np.random.rand(1, 6), # 右手6个关节
                "state.left_leg": np.zeros((1, 6)),
                "state.right_leg": np.zeros((1, 6)),
                "state.waist": np.zeros((1, 3)),
            }
            return state

        # 获取关节位置
        arm_positions = self.g1_controller.get_current_dual_arm_q()
        left_hand_pos, right_hand_pos = self.inspire_controller.get_current_dual_hand_q()

        # 构建状态字典
        state = {
            "state.left_arm": arm_positions[:7].reshape(1, -1),
            "state.right_arm": arm_positions[7:].reshape(1, -1),
            "state.left_hand": left_hand_pos.reshape(1, -1),
            "state.right_hand": right_hand_pos.reshape(1, -1),
            "state.left_leg": np.zeros((1, 6)),
            "state.right_leg": np.zeros((1, 6)),
            "state.waist": np.zeros((1, 3)),
        }
        return state

    def _get_observation(self) -> Dict[str, Any]:
        """获取完整观察 - 双摄像头版本"""
        # 获取2个摄像头的图像
        camera_images = self._get_camera_images()
        
        # 获取机器人状态
        robot_state = self._get_robot_state()
        
        # 构建观察字典 - 只包含2个摄像头
        observation = {
            "video.cam_left_high": camera_images["cam_left_high"].reshape(1, *camera_images["cam_left_high"].shape),
            "video.cam_right_high": camera_images["cam_right_high"].reshape(1, *camera_images["cam_right_high"].shape),
            **robot_state,
            "annotation.human.action.task_description": [self.config.task_description]
        }
        return observation

    def _get_action_from_model(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """从模型获取动作"""
        # 启用numpy数组的JSON序列化支持
        json_numpy.patch()

        # 使用配置的地址
        model_url = f"http://{self.config.model_host}:{self.config.model_port}/act"
        
        try:
            response = requests.post(
                model_url,
                json={"observation": obs},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                print("模型服务响应成功")
                return response.json()
            else:
                print(f"模型服务响应错误: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            print(f"请求模型服务失败: {e}")
            return {}

    def _execute_action(self, action: Dict[str, np.ndarray]):
        """执行动作"""
        if not action and self.is_executing:
            print("警告: 收到空动作，跳过执行")
            return
        
        try:
            # 提取动作数据
            left_arm_action = action.get("action.left_arm", np.zeros((self.config.action_horizon, 7)))
            right_arm_action = action.get("action.right_arm", np.zeros((self.config.action_horizon, 7)))
            left_hand_action = action.get("action.left_hand", np.zeros((self.config.action_horizon, 6)))
            right_hand_action = action.get("action.right_hand", np.zeros((self.config.action_horizon, 6)))
            
            print(f"执行动作序列 - 右臂动作形状: {right_arm_action.shape}")
            
            # 执行动作序列
            for i in range(len(right_arm_action)):
                # 设置目标位置
                self.g1_controller.set_arm_targets(
                    left_arm_action[i],
                    right_arm_action[i]
                )
                self.inspire_controller.set_hand_targets(
                    left_hand_action[i],
                    right_hand_action[i]
                )
                
                # 等待一个控制周期
                time.sleep(self.dt)
                
        except Exception as e:
            print(f"执行动作时发生错误: {e}")

    def _control_loop(self):
        """控制循环"""
        print("开始控制循环...")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # 获取观察
                observation = self._get_observation()
                
                # 从模型获取动作
                action = self._get_action_from_model(observation)
                
                # 执行动作
                self._execute_action(action)
                
                # 控制频率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"控制循环中发生错误: {e}")
                time.sleep(0.1)  # 错误时短暂等待

    def start(self):
        """开始推理执行"""
        if self.is_running:
            print("推理执行器已在运行")
            return
        
        print("启动G1推理执行器...")
        self.is_running = True
        
        # 启动控制线程
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        print("G1推理执行器已启动")

    def stop(self):
        """停止推理执行"""
        if not self.is_running:
            print("推理执行器未在运行")
            return
        
        print("停止G1推理执行器...")
        self.is_running = False
        
        # 等待控制线程结束
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        
        # 机器人回到初始位置
        print("机器人回到初始位置...")
        self.g1_controller.ctrl_dual_arm_go_home()
        self.inspire_controller.ctrl_dual_hand_go_home()
        print("G1推理执行器已停止")

    def get_robot_status(self) -> Dict[str, Any]:
        """获取机器人状态"""
        # 获取G1状态
        g1_state = self.g1_controller.get_robot_state()

        # 获取手部状态
        hand_state = self.inspire_controller.get_hand_state()

        # 合并状态
        return {
            **g1_state,
            **hand_state
        }

    def set_task_description(self, task_description: str):
        """设置任务描述"""
        self.config.task_description = task_description
        print(f"任务描述已更新: {task_description}")

    def cleanup(self):
        """清理资源"""
        print("清理资源...")

        # 停止推理执行
        self.stop()
        
        # 只在非测试模式下清理硬件相关资源
        if not self.config.test_mode:
            # 停止所有图像客户端
            if hasattr(self, 'image_clients'):
                for cam_name, client in self.image_clients.items():
                    client.running = False
                    print(f"停止 {cam_name} 图像客户端")

            # 清理2个摄像头的共享内存
            if hasattr(self, 'camera_shm'):
                for cam_name, shm in self.camera_shm.items():
                    if shm is not None:
                        try:
                            shm.close()
                            shm.unlink()
                            print(f"共享内存 {cam_name} 已清理")
                        except Exception as e:
                            print(f"清理共享内存 {cam_name} 时出错: {e}")

        print("资源清理完成")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="G1推理执行器 - 双摄像头版本")
    parser.add_argument("--host", default="192.168.110.42", help="模型服务主机")
    parser.add_argument("--port", type=int, default=6666, help="模型服务端口")
    parser.add_argument("--image-host", default="192.168.123.164", help="图像服务主机")
    parser.add_argument("--image-port", type=int, default=5555, help="图像服务端口")
    parser.add_argument("--task", default="pick the red cube on the table.", help="任务描述")

    parser.add_argument("--motion", action="store_true", help="启用 motion 模式")

    args = parser.parse_args()
    
    # 创建配置
    config = InferenceConfig(
        model_host=args.host,
        model_port=args.port,
        image_server_host=args.image_host,
        image_server_port=args.image_port,
        task_description=args.task,
        motion_mode=args.motion
    )
    
    # 创建推理执行器
    with G1InferenceExecutor(config) as executor:
        user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
        if user_input.lower() == 'r':
            executor.g1_controller.speed_gradual_max()
            try:
                # 连续推理
                executor.start()
                print("按 Ctrl+C 停止推理执行")

                while True:
                    time.sleep(1.0)
                        
            except KeyboardInterrupt:
                print("\n用户中断，停止推理执行")
            except Exception as e:
                print(f"发生错误: {e}")
            finally:
                executor.stop()