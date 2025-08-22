#!/usr/bin/env python3
"""
DeepFlyer Configuration Class
Replacement for deleted config.py that works with the new config system
"""

from typing import Dict, Any
from .config_loader import load_config, get_p3o_config

class DeepFlyerConfig:
    """
    DeepFlyer configuration class that loads from YAML/JSON config files
    Replacement for deleted config.py that maintains API compatibility
    """
    
    def __init__(self):
        # Load all configuration from files
        self._config = load_config()
        self._p3o_config = get_p3o_config()
        
        # Build configuration dictionaries in expected format
        self._build_config_dicts()
    
    def _build_config_dicts(self):
        """Build configuration dictionaries in the expected format"""
        
        # Observation and Action configuration
        control_config = self._config.get('control', {})
        self.OBSERVATION_CONFIG = {
            'dimension': control_config.get('obs_dim', 8),
            'features': ['hoop_x', 'hoop_y', 'hoop_visible', 'hoop_distance', 'vel_x', 'vel_y', 'vel_z', 'yaw_rate']
        }
        
        self.ACTION_CONFIG = {
            'dimension': control_config.get('action_dim', 4),
            'components': ['vel_x', 'vel_y', 'vel_z', 'yaw_rate'],
            'max_velocity': control_config.get('max_velocity', 2.0),
            'max_yaw_rate': control_config.get('max_yaw_rate', 1.0)
        }
        
        # P3O configuration
        self.P3O_CONFIG = {
            'learning_rate': self._p3o_config.learning_rate,
            'gamma': self._p3o_config.gamma,
            'clip_epsilon': self._p3o_config.clip_ratio,  # Note: clip_ratio maps to clip_epsilon
            'entropy_coef': self._p3o_config.entropy_coef,
            'batch_size': self._p3o_config.batch_size,
            'num_epochs': self._p3o_config.num_epochs,
            'gae_lambda': self._p3o_config.gae_lambda,
            'value_loss_coef': self._p3o_config.value_loss_coef,
            'max_grad_norm': self._p3o_config.max_grad_norm,
            'procrastination_factor': self._p3o_config.procrastination_factor,
            'action_noise': self._p3o_config.action_noise
        }
        
        # Reward configuration  
        reward_config = self._config.get('reward', {})
        self.REWARD_CONFIG = {
            'hoop_visible_reward': reward_config.get('hoop_visible_reward', 2.0),
            'horizontal_alignment_scale': reward_config.get('horizontal_alignment_scale', 10.0),
            'vertical_alignment_scale': reward_config.get('vertical_alignment_scale', 10.0),
            'perfect_alignment_bonus': reward_config.get('perfect_alignment_bonus', 20.0),
            'approach_reward_scale': reward_config.get('approach_reward_scale', 15.0),
            'proximity_bonus_threshold': reward_config.get('proximity_bonus_threshold', 0.3),
            'proximity_bonus': reward_config.get('proximity_bonus', 30.0),
            'hoop_passage_reward': reward_config.get('hoop_passage_reward', 100.0),
            'clean_passage_bonus': reward_config.get('clean_passage_bonus', 50.0),
            'forward_progress_scale': reward_config.get('forward_progress_scale', 5.0),
            'smooth_control_scale': reward_config.get('smooth_control_scale', 2.0),
            'hover_penalty': reward_config.get('hover_penalty', -1.0),
            'collision_penalty': reward_config.get('collision_penalty', -50.0),
            'out_of_bounds_penalty': reward_config.get('out_of_bounds_penalty', -30.0),
            'excessive_yaw_penalty': reward_config.get('excessive_yaw_penalty', -5.0),
            'lost_visual_penalty': reward_config.get('lost_visual_penalty', -3.0),
            'time_penalty_per_step': reward_config.get('time_penalty_per_step', -0.1)
        }
        
        # Safety configuration
        safety_config = self._config.get('safety', {})
        self.SAFETY_CONFIG = {
            'min_altitude': safety_config.get('min_altitude', 0.3),
            'max_altitude': safety_config.get('max_altitude', 3.0),
            'geofence_radius': safety_config.get('geofence_radius', 5.0),
            'max_acceleration': safety_config.get('max_acceleration', 5.0),
            'emergency_land_altitude': safety_config.get('emergency_land_altitude', 0.1)
        }
        
        # Camera configuration
        camera_config = self._config.get('camera', {})
        self.CAMERA_CONFIG = {
            'resolution': camera_config.get('resolution', 'HD720'),
            'fps': camera_config.get('fps', 30),
            'depth_mode': camera_config.get('depth_mode', 'NEURAL'),
            'fov': camera_config.get('fov', 90.0),
            'max_detection_range': camera_config.get('max_detection_range', 10.0),
            'min_depth': camera_config.get('min_depth', 0.3),
            'max_depth': camera_config.get('max_depth', 10.0)
        }
        
        # Course dimensions (for compatibility with AWS DeepRacer-style configs)
        self.COURSE_DIMENSIONS = {
            'hoop_diameter': 1.2,  # meters
            'hoop_thickness': 0.1,  # meters
            'course_length': 10.0,  # meters
            'course_width': 8.0,    # meters
            'flight_ceiling': 3.0   # meters
        }

def get_course_layout():
    """
    Get course layout configuration (for compatibility)
    Returns basic hoop racing course layout
    """
    return {
        'course_type': 'hoop_racing',
        'num_hoops': 3,
        'hoop_positions': [
            {'x': 2.0, 'y': 0.0, 'z': 1.5},
            {'x': 5.0, 'y': 1.0, 'z': 1.8}, 
            {'x': 8.0, 'y': -0.5, 'z': 1.2}
        ],
        'start_position': {'x': 0.0, 'y': 0.0, 'z': 1.0},
        'finish_position': {'x': 10.0, 'y': 0.0, 'z': 1.0}
    }
