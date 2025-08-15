"""
DeepFlyer RL Environment Package

Main exports:
- DeepFlyerEnv: Primary environment for hoop navigation training
- make_deepflyer_env: Convenience function to create environments
"""

import logging
from typing import Optional, Dict, Any

# Main environment (recommended)
from .px4_env import DeepFlyerEnv, make_deepflyer_env

# Base components (for advanced users)
from .ros_env import RosEnv, ROS_AVAILABLE
from .vision_processor import create_yolo11_processor
from .safety_layer import SafetyLayer

logger = logging.getLogger(__name__)

# Convenience function at package level
def make_env(**kwargs):
    """
    Create a DeepFlyer environment with default settings.
    
    Usage:
        from rl_agent.env import make_env
        env = make_env(enable_safety=True)
    """
    return make_deepflyer_env(**kwargs)

__all__ = [
    # Main environment API
    'DeepFlyerEnv',
    'make_deepflyer_env', 
    'make_env',
    
    # Base components
    'RosEnv',
    'create_yolo11_processor',
    'SafetyLayer',
    'ROS_AVAILABLE'
]
