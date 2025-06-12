"""
Reward functions for RL supervisor.

This module defines reward functions for the RL supervisor.
"""

import numpy as np
from typing import Dict, Any, Tuple


class RewardFunction:
    """Reward function for RL supervisor."""
    
    def __init__(
        self,
        cross_track_weight: float = 1.0,
        heading_weight: float = 0.5,
        control_effort_weight: float = 0.1
    ):
        """
        Initialize reward function.
        
        Args:
            cross_track_weight: Weight for cross-track error
            heading_weight: Weight for heading error
            control_effort_weight: Weight for control effort
        """
        self.cross_track_weight = cross_track_weight
        self.heading_weight = heading_weight
        self.control_effort_weight = control_effort_weight
        
        # Previous control values for computing changes
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
    
    def compute_reward(
        self,
        cross_track_error: float,
        heading_error: float,
        linear_vel: float,
        angular_vel: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward based on errors and control effort.
        
        Args:
            cross_track_error: Perpendicular distance from the path
            heading_error: Difference between current heading and path direction
            linear_vel: Linear velocity command
            angular_vel: Angular velocity command
            
        Returns:
            reward: Total reward
            components: Dictionary of reward components
        """
        # Compute control effort (changes in control)
        linear_change = abs(linear_vel - self.prev_linear_vel)
        angular_change = abs(angular_vel - self.prev_angular_vel)
        control_effort = linear_change + angular_change
        
        # Update previous control values
        self.prev_linear_vel = linear_vel
        self.prev_angular_vel = angular_vel
        
        # Compute reward components
        cross_track_reward = -abs(cross_track_error) * self.cross_track_weight
        heading_reward = -abs(heading_error) * self.heading_weight
        control_effort_reward = -control_effort * self.control_effort_weight
        
        # Compute total reward
        reward = cross_track_reward + heading_reward + control_effort_reward
        
        # Return reward and components
        components = {
            "cross_track_reward": cross_track_reward,
            "heading_reward": heading_reward,
            "control_effort_reward": control_effort_reward,
            "total_reward": reward
        }
        
        return reward, components
    
    def reset(self):
        """Reset reward function state."""
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0 