#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型客户端类
负责与模型服务通信，获取动作预测
"""

import sys
sys.path.append("/home/xc/Isaac-GR00T")   


import time
import json_numpy
import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass

from gr00t.eval.robot import RobotInferenceClient


@dataclass
class ModelClientConfig:
    """模型客户端配置"""
    
    # 服务配置
    host: str = "localhost"
    port: int = 6666
    api_token: Optional[str] = None
    use_http: bool = False
    
    # 超时配置
    timeout: float = 5.0


class ModelClient:
    """模型客户端类"""
    
    def __init__(self, config: ModelClientConfig):
        """
        初始化模型客户端
        
        Args:
            config: 模型客户端配置
        """
        self.config = config
        
        # 初始化客户端
        if config.use_http:
            self._init_http_client()
        else:
            self._init_zmq_client()
        
        print("模型客户端初始化完成!")

    def _init_zmq_client(self):
        """初始化ZMQ客户端"""
        print("初始化ZMQ模型客户端...")
        self.client = RobotInferenceClient(
            host=self.config.host,
            port=self.config.port,
            api_token=self.config.api_token
        )

    def _init_http_client(self):
        """初始化HTTP客户端"""
        print("初始化HTTP模型客户端...")
        json_numpy.patch()
        self.client = None  # HTTP客户端不需要预初始化

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        从模型获取动作
        
        Args:
            observation: 观察数据
            
        Returns:
            动作字典
        """
        if self.config.use_http:
            return self._get_action_http(observation)
        else:
            return self._get_action_zmq(observation)

    def _get_action_zmq(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """通过ZMQ获取动作"""
        start_time = time.time()
        
        try:
            action = self.client.get_action(observation)
            inference_time = time.time() - start_time
            print(f"ZMQ模型推理时间: {inference_time:.3f}秒")
            return action
        except Exception as e:
            print(f"ZMQ模型推理错误: {e}")
            return {}

    def _get_action_http(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """通过HTTP获取动作"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"http://{self.config.host}:{self.config.port}/act",
                json={"observation": observation},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                action = response.json()
                inference_time = time.time() - start_time
                print(f"HTTP模型推理时间: {inference_time:.3f}秒")
                return action
            else:
                print(f"HTTP请求错误: {response.status_code} - {response.text}")
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"HTTP请求异常: {e}")
            return {}

    def test_connection(self) -> bool:
        """
        测试连接
        
        Returns:
            连接是否成功
        """
        try:
            if self.config.use_http:
                # HTTP连接测试
                response = requests.get(
                    f"http://{self.config.host}:{self.config.port}/health",
                    timeout=self.config.timeout
                )
                return response.status_code == 200
            else:
                # ZMQ连接测试
                if hasattr(self.client, 'get_modality_config'):
                    self.client.get_modality_config()
                    return True
                return False
        except Exception as e:
            print(f"连接测试失败: {e}")
            return False

    def get_modality_config(self) -> Dict[str, Any]:
        """
        获取模态配置
        
        Returns:
            模态配置字典
        """
        if self.config.use_http:
            try:
                response = requests.get(
                    f"http://{self.config.host}:{self.config.port}/modality_config",
                    timeout=self.config.timeout
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"获取模态配置失败: {response.status_code}")
                    return {}
            except Exception as e:
                print(f"获取模态配置异常: {e}")
                return {}
        else:
            try:
                return self.client.get_modality_config()
            except Exception as e:
                print(f"获取模态配置异常: {e}")
                return {}

    def cleanup(self):
        """清理资源"""
        print("清理模型客户端资源...")
        # 这里可以添加必要的清理逻辑
        print("模型客户端资源清理完成")


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型客户端测试")
    parser.add_argument("--host", default="localhost", help="模型服务主机")
    parser.add_argument("--port", type=int, default=5555, help="模型服务端口")
    parser.add_argument("--http", action="store_true", help="使用HTTP客户端")
    parser.add_argument("--test", action="store_true", help="测试连接")
    
    args = parser.parse_args()
    
    # 创建配置
    config = ModelClientConfig(
        host=args.host,
        port=args.port,
        use_http=args.http
    )
    
    # 创建模型客户端
    client = ModelClient(config)
    
    try:
        if args.test:
            # 测试连接
            if client.test_connection():
                print("连接测试成功!")
            else:
                print("连接测试失败!")
        else:
            # 测试获取模态配置
            modality_config = client.get_modality_config()
            print(f"模态配置: {list(modality_config.keys())}")
            
    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        client.cleanup()
