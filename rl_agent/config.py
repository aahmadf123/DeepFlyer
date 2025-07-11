"""
Configuration for the drone RL platform with hoop navigation

This module contains all configuration classes for:
- P3O reinforcement learning algorithm
- Course layout and hoop positioning  
- Vision processing and camera settings
- Action space and control parameters
- Reward function settings
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


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
        'learning_rate': float(os.getenv('DEEPFLYER_LEARNING_RATE', '3e-4')),
        'gamma': float(os.getenv('DEEPFLYER_GAMMA', '0.99')),
        'clip_ratio': float(os.getenv('DEEPFLYER_CLIP_RATIO', '0.2')),
        'entropy_coef': float(os.getenv('DEEPFLYER_ENTROPY_COEF', '0.01')),
        'batch_size': int(os.getenv('DEEPFLYER_BATCH_SIZE', '64')),
        'buffer_size': int(os.getenv('DEEPFLYER_BUFFER_SIZE', '100000')),
        'n_updates': int(os.getenv('DEEPFLYER_N_UPDATES', '10')),
        'tau': float(os.getenv('DEEPFLYER_TAU', '0.005')),
        'alpha': float(os.getenv('DEEPFLYER_ALPHA', '0.2')),
        'procrastination_factor': float(os.getenv('DEEPFLYER_PROCRASTINATION_FACTOR', '0.95')),
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
            'max_velocity': float(os.getenv('DEEPFLYER_MAX_VELOCITY', '2.0')),
            'max_acceleration': float(os.getenv('DEEPFLYER_MAX_ACCELERATION', '1.0'))
        }
    }
    
    # Vision System Configuration
    
    VISION_CONFIG = {
        # YOLO11 Settings  
        'yolo_model': os.getenv('DEEPFLYER_YOLO_MODEL_PATH', str(Path(__file__).parent.parent / 'weights' / 'best.pt')),
        'confidence_threshold': float(os.getenv('DEEPFLYER_CONFIDENCE_THRESHOLD', '0.3')),
        'nms_threshold': float(os.getenv('DEEPFLYER_NMS_THRESHOLD', '0.5')),
        'target_classes': ['hoop'],
        'processing_frequency': int(os.getenv('DEEPFLYER_VISION_FREQ', '30')),
        
        # ZED Mini Settings
        'camera_config': {
            'resolution': os.getenv('DEEPFLYER_CAMERA_RESOLUTION', 'HD720'),
            'fps': int(os.getenv('DEEPFLYER_CAMERA_FPS', '60')),
            'depth_mode': os.getenv('DEEPFLYER_DEPTH_MODE', 'PERFORMANCE'),
            'coordinate_system': 'RIGHT_HANDED_Z_UP'
        },
        
        # Vision Features for RL
        'features': {
            'hoop_alignment_range': [-1.0, 1.0],
            'distance_range': [0.5, 5.0],
            'detection_stability_window': int(os.getenv('DEEPFLYER_DETECTION_WINDOW', '10'))
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
        'control_frequency': int(os.getenv('DEEPFLYER_CONTROL_FREQ', '20')),
        'offboard_mode_frequency': int(os.getenv('DEEPFLYER_OFFBOARD_FREQ', '2')),
        
        # Connection Settings
        'connection': {
            'host': os.getenv('PX4_HOST', 'localhost'),
            'port': int(os.getenv('PX4_PORT', '14540')),
            'timeout': float(os.getenv('PX4_TIMEOUT', '5.0')),
            'retry_attempts': int(os.getenv('PX4_RETRY_ATTEMPTS', '3')),
            'retry_delay': float(os.getenv('PX4_RETRY_DELAY', '1.0'))
        },
        
        # Safety Parameters
        'safety': {
            'max_tilt_angle': float(os.getenv('DEEPFLYER_MAX_TILT', '30')),
            'emergency_land_height': float(os.getenv('DEEPFLYER_EMERGENCY_HEIGHT', '0.3')),
            'failsafe_timeout': float(os.getenv('DEEPFLYER_FAILSAFE_TIMEOUT', '2.0')),
            'geofence_enabled': os.getenv('DEEPFLYER_GEOFENCE_ENABLED', 'True').lower() == 'true'
        }
    }
    
    # Reward Function Configuration
    
    REWARD_CONFIG = {
        # Positive Rewards
        'hoop_approach_reward': float(os.getenv('REWARD_HOOP_APPROACH', '10.0')),
        'hoop_passage_reward': float(os.getenv('REWARD_HOOP_PASSAGE', '50.0')),
        'hoop_center_bonus': float(os.getenv('REWARD_HOOP_CENTER', '20.0')),
        'visual_alignment_reward': float(os.getenv('REWARD_VISUAL_ALIGNMENT', '5.0')),
        'forward_progress_reward': float(os.getenv('REWARD_FORWARD_PROGRESS', '3.0')),
        'speed_efficiency_bonus': float(os.getenv('REWARD_SPEED_EFFICIENCY', '2.0')),
        'lap_completion_bonus': float(os.getenv('REWARD_LAP_COMPLETION', '100.0')),
        'course_completion_bonus': float(os.getenv('REWARD_COURSE_COMPLETION', '500.0')),
        'smooth_flight_bonus': float(os.getenv('REWARD_SMOOTH_FLIGHT', '1.0')),
        'precision_bonus': float(os.getenv('REWARD_PRECISION', '15.0')),
        
        # Penalties
        'wrong_direction_penalty': float(os.getenv('PENALTY_WRONG_DIRECTION', '-2.0')),
        'hoop_miss_penalty': float(os.getenv('PENALTY_HOOP_MISS', '-25.0')),
        'collision_penalty': float(os.getenv('PENALTY_COLLISION', '-100.0')),
        'slow_progress_penalty': float(os.getenv('PENALTY_SLOW_PROGRESS', '-1.0')),
        'erratic_flight_penalty': float(os.getenv('PENALTY_ERRATIC_FLIGHT', '-3.0')),
        
        # Non-Tunable Safety Penalties
        'boundary_violation_penalty': -200.0,
        'emergency_landing_penalty': -500.0,
        
        # Reward Shaping
        'normalization_ranges': {
            'distance_decay_factor': float(os.getenv('REWARD_DISTANCE_DECAY', '2.0')),
            'alignment_tolerance': float(os.getenv('REWARD_ALIGNMENT_TOLERANCE', '0.2')),
            'speed_optimal_range': [0.4, 0.8]
        }
    }
    
    # Training Configuration
    
    TRAINING_CONFIG = {
        # Episode Parameters
        'max_episodes': int(os.getenv('DEEPFLYER_MAX_EPISODES', '1000')),
        'max_steps_per_episode': int(os.getenv('DEEPFLYER_MAX_STEPS', '500')),
        'evaluation_frequency': int(os.getenv('DEEPFLYER_EVAL_FREQ', '50')),
        'early_stopping_patience': int(os.getenv('DEEPFLYER_EARLY_STOP_PATIENCE', '100')),
        'success_threshold': float(os.getenv('DEEPFLYER_SUCCESS_THRESHOLD', '0.8')),
        
        # Training Parameters
        'learning_frequency': int(os.getenv('DEEPFLYER_LEARNING_FREQ', '20')),
        'model_save_frequency': int(os.getenv('DEEPFLYER_SAVE_FREQ', '100')),
        'checkpoint_dir': os.getenv('DEEPFLYER_CHECKPOINT_DIR', str(Path.home() / 'deepflyer' / 'models')),
        'log_dir': os.getenv('DEEPFLYER_LOG_DIR', str(Path.home() / 'deepflyer' / 'logs')),
        
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
        'drone_frame': os.getenv('DEEPFLYER_DRONE_FRAME', 'Holybro S500'),
        'flight_controller': os.getenv('DEEPFLYER_FLIGHT_CONTROLLER', 'Pixhawk 6C'),
        'compute_platform': os.getenv('DEEPFLYER_COMPUTE_PLATFORM', 'Raspberry Pi 4B'),
        'camera': os.getenv('DEEPFLYER_CAMERA', 'ZED Mini'),
        'emergency_stop': os.getenv('DEEPFLYER_EMERGENCY_STOP', 'True').lower() == 'true',
        
        # Communication
        'px4_connection': {
            'protocol': 'PX4-ROS-COM',
            'transport': 'UDP',
            'port': int(os.getenv('PX4_PORT', '14540'))
        }
    }


def validate_config():
    """Validate critical configuration values for safety and functionality"""
    config = DeepFlyerConfig()
    
    # Safety validation
    assert config.ACTION_CONFIG['safety_limits']['max_velocity'] > 0, "Max velocity must be positive"
    assert config.ACTION_CONFIG['safety_limits']['max_acceleration'] > 0, "Max acceleration must be positive"
    assert config.PX4_CONFIG['safety']['max_tilt_angle'] <= 45, "Max tilt angle should not exceed 45 degrees"
    assert config.PX4_CONFIG['safety']['emergency_land_height'] >= 0.1, "Emergency land height too low"
    assert config.PX4_CONFIG['safety']['failsafe_timeout'] > 0, "Failsafe timeout must be positive"
    
    # Path validation
    yolo_path = Path(config.VISION_CONFIG['yolo_model'])
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO model not found at {yolo_path}")
    
    # Create necessary directories
    Path(config.TRAINING_CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config.TRAINING_CONFIG['log_dir']).mkdir(parents=True, exist_ok=True)
    
    return True


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


def get_course_layout(spawn_position: Tuple[float, float, float], course_type: str = 'standard') -> List[Dict[str, Any]]:
    """
    Generate hoop positions for various course layouts
    
    Args:
        spawn_position: (x, y, z) drone spawn location
        course_type: Type of course ('standard', 'figure_eight', 'linear', 'oval')
        
    Returns:
        List of hoop configurations
    """
    hoops = []
    altitude = DeepFlyerConfig.HOOP_CONFIG['flight_altitude']
    diameter = DeepFlyerConfig.HOOP_CONFIG['diameter']
    num_hoops = DeepFlyerConfig.HOOP_CONFIG['num_hoops']
    
    x0, y0, z0 = spawn_position
    
    if course_type == 'standard':
        # Rectangular circuit
        hoop_positions = [
            (x0 + 0.5, y0 - 0.5, altitude),  # Hoop 1
            (x0 + 1.0, y0 - 0.5, altitude),  # Hoop 2  
            (x0 + 1.5, y0 + 0.0, altitude),  # Hoop 3
            (x0 + 1.0, y0 + 0.5, altitude),  # Hoop 4
            (x0 + 0.5, y0 + 0.0, altitude)   # Hoop 5
        ]
    elif course_type == 'figure_eight':
        # Figure-8 pattern
        hoop_positions = [
            (x0 + 0.3, y0 + 0.0, altitude),
            (x0 + 0.6, y0 + 0.3, altitude),
            (x0 + 0.9, y0 + 0.0, altitude),
            (x0 + 0.6, y0 - 0.3, altitude),
            (x0 + 0.3, y0 + 0.0, altitude)
        ]
    elif course_type == 'linear':
        # Straight line course
        hoop_positions = [(x0 + i * 0.4, y0, altitude) for i in range(1, num_hoops + 1)]
    elif course_type == 'oval':
        # Oval track
        angles = np.linspace(0, 2 * np.pi, num_hoops, endpoint=False)
        radius_x, radius_y = 0.8, 0.5
        hoop_positions = [(x0 + radius_x * np.cos(angle), y0 + radius_y * np.sin(angle), altitude) 
                         for angle in angles]
    else:
        raise ValueError(f"Unknown course type: {course_type}")
    
    for i, pos in enumerate(hoop_positions):
        hoops.append({
            'id': i + 1,
            'position': pos,
            'diameter': diameter,
            'sequence': i + 1,
            'course_type': course_type
        })
    
    return hoops


# Export main config class
__all__ = ['DeepFlyerConfig', 'get_p3o_config', 'get_observation_config', 
           'get_action_config', 'get_vision_config', 'get_px4_config', 
           'get_reward_config', 'get_course_layout', 'validate_config'] 