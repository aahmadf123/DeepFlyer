#!/usr/bin/env python3
"""
DeepFlyer Configuration Loader
Loads configuration from YAML and JSON files in a clear hierarchy:
1. CLI arguments (highest priority)
2. config/p3o_config.json (if exists)  
3. config/student_tuning.json (student defaults)
4. config/training_config.yaml (base defaults)
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

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

def load_config() -> Dict[str, Any]:
    """Load configuration from files"""
    config = {}
    
    # Load base configuration from YAML
    yaml_path = Path("config/training_config.yaml")
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)
    
    # Load student tuning defaults from JSON
    student_path = Path("config/student_tuning.json")
    if student_path.exists():
        with open(student_path, 'r') as f:
            student_config = json.load(f)
            # Apply defaults from student tuning
            if "p3o_hyperparameters" in student_config:
                p3o_defaults = {}
                for key, value in student_config["p3o_hyperparameters"].items():
                    if isinstance(value, dict) and "default" in value:
                        p3o_defaults[key] = value["default"]
                config.setdefault("p3o", {}).update(p3o_defaults)
    
    # Load user overrides from JSON (if exists)
    user_path = Path("config/p3o_config.json")
    if user_path.exists():
        with open(user_path, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    return config

def get_p3o_config(overrides: Optional[Dict] = None) -> P3OConfig:
    """Get P3O configuration with overrides"""
    config = load_config()
    p3o_params = config.get("p3o", {})
    
    if overrides:
        p3o_params.update(overrides)
    
    # Filter out parameters that don't exist in P3OConfig
    valid_params = {}
    p3o_config = P3OConfig()
    for key, value in p3o_params.items():
        if hasattr(p3o_config, key):
            valid_params[key] = value
    
    return P3OConfig(**valid_params)
