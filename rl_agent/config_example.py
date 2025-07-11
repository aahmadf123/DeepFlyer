"""
Example production configuration for DeepFlyer

This file demonstrates how to use environment variables for production deployment.
Set these environment variables before running the application.

For production deployment:
1. Set environment variables in your deployment system
2. Use the validate_config() function to ensure proper configuration
3. Never commit sensitive values to version control

Example environment variables:
"""

import os
from rl_agent.config import DeepFlyerConfig, validate_config

# Example: Set environment variables for production
PRODUCTION_ENV_VARS = {
    # P3O Algorithm
    'DEEPFLYER_LEARNING_RATE': '3e-4',
    'DEEPFLYER_GAMMA': '0.99',
    'DEEPFLYER_BATCH_SIZE': '64',
    
    # Safety Critical Settings
    'DEEPFLYER_MAX_VELOCITY': '2.0',
    'DEEPFLYER_MAX_ACCELERATION': '1.0',
    'DEEPFLYER_MAX_TILT': '30',
    'DEEPFLYER_EMERGENCY_HEIGHT': '0.3',
    'DEEPFLYER_FAILSAFE_TIMEOUT': '2.0',
    'DEEPFLYER_GEOFENCE_ENABLED': 'true',
    
    # Vision System
    'DEEPFLYER_YOLO_MODEL_PATH': '/opt/deepflyer/weights/best.pt',
    'DEEPFLYER_CONFIDENCE_THRESHOLD': '0.3',
    'DEEPFLYER_VISION_FREQ': '30',
    
    # PX4 Connection
    'PX4_HOST': 'localhost',
    'PX4_PORT': '14540',
    'PX4_TIMEOUT': '5.0',
    'PX4_RETRY_ATTEMPTS': '3',
    
    # File Paths (use absolute paths in production)
    'DEEPFLYER_CHECKPOINT_DIR': '/opt/deepflyer/models',
    'DEEPFLYER_LOG_DIR': '/opt/deepflyer/logs',
    
    # Reward Tuning
    'REWARD_HOOP_APPROACH': '10.0',
    'REWARD_HOOP_PASSAGE': '50.0',
    'PENALTY_COLLISION': '-100.0',
}

def setup_production_config():
    """
    Example function to set up production configuration
    Call this before initializing DeepFlyerConfig in production
    """
    for key, value in PRODUCTION_ENV_VARS.items():
        os.environ[key] = value
    
    # Validate configuration after setting environment variables
    try:
        validate_config()
        print("Configuration validated successfully")
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

def get_production_config():
    """Get a validated production configuration"""
    if setup_production_config():
        return DeepFlyerConfig()
    else:
        raise RuntimeError("Failed to set up production configuration")

if __name__ == "__main__":
    # Example usage
    config = get_production_config()
    print("Production configuration loaded successfully")
    print(f"Checkpoint directory: {config.TRAINING_CONFIG['checkpoint_dir']}")
    print(f"Model path: {config.VISION_CONFIG['yolo_model']}") 