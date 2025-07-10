"""
Configuration for the drone RL platform with hoop navigation

This module contains all configuration classes for:
- P3O reinforcement learning algorithm
- Course layout and hoop positioning  
- Vision processing and camera settings
- Action space and control parameters
- Reward function settings
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DeepFlyerConfig:
    """Main configuration class for DeepFlyer drone RL platform"""
    
    # Course Configuration
    
    # Course Layout
    COURSE_DIMENSIONS = {
        'length': 2.1,  # meters
        'width': 1.6,   # meters  
        'height': 1.5,  # meters
        'safety_buffer': 0.2  # meters from walls
    }
    
    # Hoop Configuration
    HOOP_CONFIG = {
        'num_hoops': 5,           # Fixed 5-hoop circuit
        'diameter': 0.8,          # meters
        'num_laps': 3,            # Total laps to complete
        'flight_altitude': 0.8,   # meters above ground
        'detection_distance': 5.0 # maximum detection range
    }
    
    # RL Agent Configuration
    
    # P3O Hyperparameters
    P3O_CONFIG = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'batch_size': 64,
        'buffer_size': 100000,
        'n_updates': 10,
        'tau': 0.005,
        'alpha': 0.2,                    # Blend factor for P3O
        'procrastination_factor': 0.95,  # P3O procrastination factor
        'hidden_dims': [256, 256]
    }
    
    # State Space Configuration (12-dimensional)
    OBSERVATION_CONFIG = {
        'dimension': 12,
        'components': {
            'direction_to_hoop': 3,      # [0-2] Normalized x,y,z direction vector
            'current_velocity': 2,       # [3-4] Forward, lateral velocity
            'navigation_metrics': 2,     # [5-6] Distance to target, velocity alignment
            'vision_features': 3,        # [7-9] Hoop alignment, visual distance, visibility
            'course_progress': 2         # [10-11] Lap progress, overall completion
        }
    }
    
    # Action Space Configuration (3-dimensional)
    ACTION_CONFIG = {
        'dimension': 3,
        'components': {
            'lateral_cmd': {'range': [-1, 1], 'max_speed': 0.8},      # m/s
            'vertical_cmd': {'range': [-1, 1], 'max_speed': 0.4},     # m/s
            'speed_cmd': {'range': [-1, 1], 'base_speed': 0.6}        # m/s
        },
        'safety_limits': {
            'max_velocity': 2.0,        # m/s
            'max_acceleration': 1.0     # m/s²
        }
    }
    
    # Vision System Configuration
    
    VISION_CONFIG = {
        # YOLO11 Settings  
        'yolo_model': 'weights/best.pt',  # DeepFlyer custom-trained hoop detection model
        'confidence_threshold': 0.3,
        'nms_threshold': 0.5,
        'target_classes': ['hoop'],       # Custom-trained class for hoop detection
        'processing_frequency': 30,       # Hz
        
        # ZED Mini Settings
        'camera_config': {
            'resolution': 'HD720',       # 1280x720
            'fps': 60,
            'depth_mode': 'PERFORMANCE',
            'coordinate_system': 'RIGHT_HANDED_Z_UP'
        },
        
        # Vision Features for RL
        'features': {
            'hoop_alignment_range': [-1.0, 1.0],    # -1=left, +1=right, 0=centered
            'distance_range': [0.5, 5.0],           # meters
            'detection_stability_window': 10         # frames for stability calculation
        }
    }
    
    # PX4-ROS-COM Configuration
    
    PX4_CONFIG = {
        # ROS2 Topics
        'input_topics': {
            'vehicle_local_position': '/fmu/out/vehicle_local_position',
            'vehicle_attitude': '/fmu/out/vehicle_attitude', 
            'vehicle_status': '/fmu/out/vehicle_status_v1',
            'vehicle_control_mode': '/fmu/out/vehicle_control_mode',
            'battery_status': '/fmu/out/battery_status'
        },
        
        'output_topics': {
            'trajectory_setpoint': '/fmu/in/trajectory_setpoint',
            'offboard_control_mode': '/fmu/in/offboard_control_mode',
            'vehicle_command': '/fmu/in/vehicle_command'
        },
        
        # Control Configuration
        'control_frequency': 20,  # Hz
        'offboard_mode_frequency': 2,  # Hz
        
        # Safety Parameters
        'safety': {
            'max_tilt_angle': 30,        # degrees
            'emergency_land_height': 0.3, # meters
            'failsafe_timeout': 2.0,     # seconds
            'geofence_enabled': True
        }
    }
    
    # Reward Function Configuration
    
    REWARD_CONFIG = {
        # Positive Rewards
        'hoop_approach_reward': 10.0,
        'hoop_passage_reward': 50.0,
        'hoop_center_bonus': 20.0,
        'visual_alignment_reward': 5.0,
        'forward_progress_reward': 3.0,
        'speed_efficiency_bonus': 2.0,
        'lap_completion_bonus': 100.0,
        'course_completion_bonus': 500.0,
        'smooth_flight_bonus': 1.0,
        'precision_bonus': 15.0,
        
        # Penalties
        'wrong_direction_penalty': -2.0,
        'hoop_miss_penalty': -25.0,
        'collision_penalty': -100.0,
        'slow_progress_penalty': -1.0,
        'erratic_flight_penalty': -3.0,
        
        # Non-Tunable Safety Penalties
        'boundary_violation_penalty': -200.0,
        'emergency_landing_penalty': -500.0,
        
        # Reward Shaping
        'normalization_ranges': {
            'distance_decay_factor': 2.0,
            'alignment_tolerance': 0.2,
            'speed_optimal_range': [0.4, 0.8]
        }
    }
    
    # Training Configuration
    
    TRAINING_CONFIG = {
        # Episode Parameters
        'max_episodes': 1000,
        'max_steps_per_episode': 500,
        'evaluation_frequency': 50,
        'early_stopping_patience': 100,
        'success_threshold': 0.8,
        
        # Training Parameters
        'learning_frequency': 20,     # Hz
        'model_save_frequency': 100,  # episodes
        'checkpoint_dir': './models',
        'log_dir': './logs',
        
        # Performance Metrics
        'metrics': [
            'hoop_completion_rate',
            'average_lap_time', 
            'collision_rate',
            'path_efficiency',
            'training_stability'
        ]
    }
    
    # Hardware Configuration
    
    HARDWARE_CONFIG = {
        'drone_frame': 'Holybro S500',
        'flight_controller': 'Pixhawk 6C',
        'compute_platform': 'Raspberry Pi 4B',
        'camera': 'ZED Mini',
        'emergency_stop': True,
        
        # Communication
        'px4_connection': {
            'protocol': 'PX4-ROS-COM',
            'transport': 'UDP',
            'port': 14540
        }
    }


# Convenience functions for easy access
def get_p3o_config() -> Dict[str, Any]:
    """Get P3O algorithm configuration"""
    return DeepFlyerConfig.P3O_CONFIG.copy()


def get_observation_config() -> Dict[str, Any]:
    """Get observation space configuration"""
    return DeepFlyerConfig.OBSERVATION_CONFIG.copy()


def get_action_config() -> Dict[str, Any]:
    """Get action space configuration"""
    return DeepFlyerConfig.ACTION_CONFIG.copy()


def get_vision_config() -> Dict[str, Any]:
    """Get vision system configuration"""
    return DeepFlyerConfig.VISION_CONFIG.copy()


def get_px4_config() -> Dict[str, Any]:
    """Get PX4-ROS-COM configuration"""
    return DeepFlyerConfig.PX4_CONFIG.copy()


def get_reward_config() -> Dict[str, Any]:
    """Get reward function configuration"""
    return DeepFlyerConfig.REWARD_CONFIG.copy()


def get_course_layout(spawn_position: Tuple[float, float, float]) -> List[Dict[str, Any]]:
    """
    Generate hoop positions for the standard 5-hoop course
    
    Args:
        spawn_position: (x, y, z) drone spawn location
        
    Returns:
        List of hoop configurations
    """
    hoops = []
    altitude = DeepFlyerConfig.HOOP_CONFIG['flight_altitude']
    
    # Standard 5-hoop circuit layout
    hoop_positions = [
        (spawn_position[0] + 0.5, spawn_position[1] - 0.5, altitude),  # Hoop 1
        (spawn_position[0] + 1.0, spawn_position[1] - 0.5, altitude),  # Hoop 2  
        (spawn_position[0] + 1.5, spawn_position[1] + 0.0, altitude),  # Hoop 3
        (spawn_position[0] + 1.0, spawn_position[1] + 0.5, altitude),  # Hoop 4
        (spawn_position[0] + 0.5, spawn_position[1] + 0.0, altitude)   # Hoop 5
    ]
    
    for i, pos in enumerate(hoop_positions):
        hoops.append({
            'id': i + 1,
            'position': pos,
            'diameter': DeepFlyerConfig.HOOP_CONFIG['diameter'],
            'sequence': i + 1
        })
    
    return hoops


# Export main config class
__all__ = ['DeepFlyerConfig', 'get_p3o_config', 'get_observation_config', 
           'get_action_config', 'get_vision_config', 'get_px4_config', 
           'get_reward_config', 'get_course_layout'] 