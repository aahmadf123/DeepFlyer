"""
DeepFlyer RL Environment Package

Main exports:
- DeepFlyerEnv: Primary environment for hoop navigation training
- make_deepflyer_env: Convenience function to create environments
"""

import logging
from typing import Optional, Dict, Any, Union

import gymnasium as gym

# Main environment (recommended)
from .px4_env import DeepFlyerEnv, make_deepflyer_env

# Optional base components (for advanced users - may not be available)
try:
    from .vision_processor import create_yolo11_processor
    VISION_AVAILABLE = True
except Exception:
    VISION_AVAILABLE = False
    
try:
    from .safety_layer import SafetyLayer
    SAFETY_AVAILABLE = True
except Exception:
    SAFETY_AVAILABLE = False

# ROS components (optional) — be resilient to any import errors in ROS stack
ROS_AVAILABLE = False
try:
    from .ros_env import RosEnv
    ROS_AVAILABLE = True
except Exception:
    RosEnv = None

logger = logging.getLogger(__name__)

# Convenience function at package level
def make_env(env: Optional[Union[str, None]] = None, **kwargs):
    """
    Create an environment by ID or return DeepFlyer env by default.

    Behaviors:
    - If env is a Gym/Gymnasium ID (e.g., 'CartPole-v1'), returns gym.make(env, **kwargs)
    - If env is a ROS-style hint like 'ros:Dummy', returns DeepFlyerEnv(**kwargs)
    - If env is None or starts with 'DeepFlyer', returns DeepFlyerEnv(**kwargs)
    """
    if isinstance(env, str):
        if env.lower().startswith('deepflyer'):
            return make_deepflyer_env(**kwargs)
        return gym.make(env, **kwargs)
    # Default: DeepFlyer environment
    return make_deepflyer_env(**kwargs)

__all__ = [
    # Main environment API
    'DeepFlyerEnv',
    'make_deepflyer_env', 
    'make_env',
    
    # Status flags
    'ROS_AVAILABLE',
    'VISION_AVAILABLE', 
    'SAFETY_AVAILABLE'
]

# Conditionally add optional components to __all__
if ROS_AVAILABLE:
    __all__.append('RosEnv')
if VISION_AVAILABLE:
    __all__.append('create_yolo11_processor')
if SAFETY_AVAILABLE:
    __all__.append('SafetyLayer')
