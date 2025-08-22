#!/usr/bin/env python3
"""
Unified DeepFlyer Configuration Management
Provides single source of truth for all configurations in DeepRacer-style architecture
Loads configuration from YAML and JSON files in a clear hierarchy:
1. CLI arguments (highest priority)
2. config/p3o_config.json (if exists)  
3. config/student_tuning.json (student defaults)
4. config/training_config.yaml (base defaults)
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class P3OConfig:
    """P3O Algorithm Configuration"""
    learning_rate: float = 0.0003
    clip_ratio: float = 0.2
    procrastination_factor: float = 0.95
    batch_size: int = 64
    num_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    action_noise: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

@dataclass
class TrainingConfig:
    """Training Configuration"""
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    save_frequency: int = 50
    log_frequency: int = 10
    eval_frequency: int = 100
    eval_episodes: int = 5
    early_stopping_patience: int = 200
    target_reward: float = 500.0
    resume_training: bool = False
    
@dataclass
class SafetyConfig:
    """Safety Configuration"""
    enable_geofence: bool = True
    enable_collision_avoidance: bool = True
    enable_attitude_limits: bool = True
    enable_velocity_ramping: bool = True
    max_acceleration: float = 0.5
    emergency_stop_on_violation: bool = True
    
@dataclass
class CameraConfig:
    """Camera Configuration"""
    resolution: str = "HD720"
    fps: int = 30
    depth_mode: str = "NEURAL"
    use_zed_sdk: bool = True
    use_ros_topics: bool = False
    namespace: str = "zed_mini"
    
@dataclass
class RewardConfig:
    """Reward Configuration"""
    hoop_visible_reward: float = 2.0
    alignment_scale: float = 10.0
    proximity_bonus: float = 30.0
    passage_reward: float = 100.0
    collision_penalty: float = -50.0
    time_penalty: float = -0.1

class UnifiedConfigManager:
    """Unified configuration manager for all DeepFlyer settings"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self._base_config = {}
        self._load_base_config()
        
    def _load_base_config(self):
        """Load base configuration from YAML file"""
        yaml_path = self.config_dir / "training_config.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                self._base_config = yaml.safe_load(f)
                logger.info(f"Loaded base config from {yaml_path}")
        else:
            logger.warning(f"Base config file not found: {yaml_path}")
            self._base_config = {}
    
    def _load_student_config(self) -> Dict[str, Any]:
        """Load student tuning configuration"""
        student_path = self.config_dir / "student_tuning.json"
        if student_path.exists():
            with open(student_path, 'r') as f:
                student_config = json.load(f)
                logger.info(f"Loaded student config from {student_path}")
                return student_config
        return {}
    
    def _load_user_overrides(self) -> Dict[str, Any]:
        """Load user override configuration"""
        user_path = self.config_dir / "p3o_config.json"
        if user_path.exists():
            with open(user_path, 'r') as f:
                user_config = json.load(f)
                logger.info(f"Loaded user overrides from {user_path}")
                return user_config
        return {}
    
    def get_unified_config(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get complete unified configuration with all overrides applied"""
        config = self._base_config.copy()
        
        # Apply student tuning defaults
        student_config = self._load_student_config()
        if "p3o_hyperparameters" in student_config:
            p3o_defaults = {}
            for key, value in student_config["p3o_hyperparameters"].items():
                if isinstance(value, dict) and "default" in value:
                    p3o_defaults[key] = value["default"]
            config.setdefault("p3o", {}).update(p3o_defaults)
        
        # Apply user overrides
        user_overrides = self._load_user_overrides()
        self._deep_update(config, user_overrides)
        
        # Apply runtime overrides
        if overrides:
            self._deep_update(config, overrides)
        
        return config
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update of nested dictionaries"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_p3o_config(self, overrides: Optional[Dict[str, Any]] = None) -> P3OConfig:
        """Get P3O configuration"""
        config = self.get_unified_config(overrides)
        p3o_params = config.get("p3o", {})
        
        # Filter valid parameters
        valid_params = {}
        p3o_config = P3OConfig()
        for key, value in p3o_params.items():
            if hasattr(p3o_config, key):
                valid_params[key] = value
        
        return P3OConfig(**valid_params)
    
    def get_training_config(self, overrides: Optional[Dict[str, Any]] = None) -> TrainingConfig:
        """Get training configuration"""
        config = self.get_unified_config(overrides)
        training_params = config.get("training", {})
        
        valid_params = {}
        training_config = TrainingConfig()
        for key, value in training_params.items():
            if hasattr(training_config, key):
                valid_params[key] = value
                
        return TrainingConfig(**valid_params)
    
    def get_safety_config(self, overrides: Optional[Dict[str, Any]] = None) -> SafetyConfig:
        """Get safety configuration"""
        config = self.get_unified_config(overrides)
        safety_params = config.get("safety", {})
        
        valid_params = {}
        safety_config = SafetyConfig()
        for key, value in safety_params.items():
            if hasattr(safety_config, key):
                valid_params[key] = value
                
        return SafetyConfig(**valid_params)
    
    def get_camera_config(self, overrides: Optional[Dict[str, Any]] = None) -> CameraConfig:
        """Get camera configuration"""
        config = self.get_unified_config(overrides)
        camera_params = config.get("camera", {})
        
        valid_params = {}
        camera_config = CameraConfig()
        for key, value in camera_params.items():
            if hasattr(camera_config, key):
                valid_params[key] = value
                
        return CameraConfig(**valid_params)
    
    def get_reward_config(self, overrides: Optional[Dict[str, Any]] = None) -> RewardConfig:
        """Get reward configuration"""
        config = self.get_unified_config(overrides)
        reward_params = config.get("reward", {})
        
        valid_params = {}
        reward_config = RewardConfig()
        for key, value in reward_params.items():
            if hasattr(reward_config, key):
                valid_params[key] = value
                
        return RewardConfig(**valid_params)
    
    def save_user_config(self, config_updates: Dict[str, Any]):
        """Save user configuration updates to file"""
        user_path = self.config_dir / "p3o_config.json"
        
        # Load existing config
        existing_config = self._load_user_overrides()
        
        # Update with new values
        self._deep_update(existing_config, config_updates)
        
        # Save to file
        with open(user_path, 'w') as f:
            json.dump(existing_config, f, indent=2)
        
        logger.info(f"Saved user config to {user_path}")

# Global config manager instance
_global_config_manager = None

def get_config_manager() -> UnifiedConfigManager:
    """Get global configuration manager instance"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = UnifiedConfigManager()
    return _global_config_manager

# Legacy functions for backwards compatibility
def load_config() -> Dict[str, Any]:
    """Load configuration from files (legacy function)"""
    return get_config_manager().get_unified_config()

def get_p3o_config(overrides: Optional[Dict] = None) -> P3OConfig:
    """Get P3O configuration with overrides (legacy function)"""
    return get_config_manager().get_p3o_config(overrides)
