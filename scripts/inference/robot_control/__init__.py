# """
# 机器人控制模块
# 包含G1 + 因时灵巧手的控制类和推理执行器
# """

# from .g1_inspire_controller import G1InspireController
# __all__ = ['G1InspireController']

"""
机器人控制模块
包含G1 + 灵巧手的控制类和推理执行器
支持因时手和强脑手两种版本
"""

from .g1_inspire_controller import G1InspireController
from .g1_brainco_controller import G1BraincoController

__all__ = [
    'G1InspireController', 
    'G1BraincoController',
]