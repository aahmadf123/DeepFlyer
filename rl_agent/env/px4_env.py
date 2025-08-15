"""
DeepFlyer Hoop Navigation Environment
Main RL environment for drone hoop navigation training
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
import logging

from .ros_env import RosEnv
from .safety_layer import BeginnerSafetyLayer  
from .vision_processor import create_yolo11_processor
from ..rewards import HoopRewardFunction, HoopRewardConfig
from ..config import DeepFlyerConfig

logger = logging.getLogger(__name__)


class DeepFlyerEnv(RosEnv):
    """
    Main RL environment for DeepFlyer hoop navigation.
    
    This is the primary environment class for training RL agents.
    Combines ZED camera, YOLO detection, safety layer, and hoop navigation.
    """
    
    def __init__(self, 
                 config_dict: Optional[Dict[str, Any]] = None,
                 enable_safety: bool = True,
                 **kwargs):
        """
        Initialize DeepFlyer hoop navigation environment
        
        Args:
            config_dict: Configuration overrides
            enable_safety: Enable safety constraints
            **kwargs: Additional arguments for RosEnv
        """
        # Load configuration
        self.config = DeepFlyerConfig()
        if config_dict:
            self.config.update(config_dict)
        
        # Initialize reward function
        self.reward_function = HoopRewardFunction()
        
        # Initialize safety layer
        self.safety_layer = BeginnerSafetyLayer() if enable_safety else None
        
        # Set defaults for ZED camera usage
        kwargs.setdefault('use_zed', True)
        kwargs.setdefault('namespace', 'deepflyer')
        
        # Initialize base environment
        super().__init__(**kwargs)
        
        # Hoop navigation state
        self.current_hoop = 0
        self.hoops_passed = 0
        self.episode_start_time = 0.0
        
        logger.info("DeepFlyer hoop navigation environment initialized")
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        # Reset reward function state
        self.reward_function.reset_episode()
        
        # Reset navigation state
        self.current_hoop = 0
        self.hoops_passed = 0
        self.episode_start_time = 0.0
        
        # Reset base environment
        obs, info = super().reset(seed=seed, options=options)
        
        return obs, info
    
    def step(self, action):
        """Execute one environment step"""
        # Apply safety constraints if enabled
        if self.safety_layer:
            action = self.safety_layer.constrain_action(action, self.get_state())
        
        # Execute step in base environment
        obs, _, terminated, truncated, info = super().step(action)
        
        # Calculate custom reward
        state = self._extract_state_for_reward(obs, info)
        reward = self.reward_function.compute_reward(state, action, info=info)
        
        # Add reward components to info
        info['reward_components'] = self.reward_function.component_values
        
        return obs, reward, terminated, truncated, info
    
    def _extract_state_for_reward(self, obs, info):
        """Extract state dict for reward function from observation"""
        # This extracts the relevant state information from the observation
        # for the reward function (vision features, position, etc.)
        return {
            'hoop_visible': info.get('hoop_visible', 0),
            'hoop_x_center_norm': info.get('hoop_x_center', 0.0),
            'hoop_y_center_norm': info.get('hoop_y_center', 0.0), 
            'hoop_distance_norm': info.get('hoop_distance', 1.0),
            'collision': info.get('collision', False),
            'speed': np.linalg.norm(obs.get('linear_velocity', [0, 0, 0])),
            'all_systems_normal': not info.get('collision', False),
            'hoop_passages_completed': self.hoops_passed,
            'flight_phase': info.get('flight_phase', 'TAKEOFF')
        }


def make_deepflyer_env(**kwargs):
    """
    Convenience function to create DeepFlyer environment
    
    Usage:
        env = make_deepflyer_env(enable_safety=True)
    """
    return DeepFlyerEnv(**kwargs)


__all__ = ['DeepFlyerEnv', 'make_deepflyer_env'] 