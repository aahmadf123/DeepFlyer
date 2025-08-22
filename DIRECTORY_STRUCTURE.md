# DeepFlyer Directory Structure Guide

## Core Directories

### `config/` - Configuration Files
- `training_config.yaml` - Base training parameters  
- `student_tuning.json` - Student-adjustable hyperparameters
- `p3o_config.json` - User overrides (auto-created)

### `rl_agent/` - RL Algorithm Implementation
- `algorithms/p3o.py` - P3O algorithm
- `direct_control_agent.py` - Main agent class
- `config_loader.py` - Configuration loading
- `depth_processor.py` - YOLO11 + depth processing
- `rewards/rewards.py` - Reward functions
- `env/safety_layer.py` - Safety constraints

### `nodes/` - ROS2 Nodes  
- `rl_agent_node.py` - P3O RL agent with complete training functionality
- `vision_processor_node.py` - Camera processing
- `reward_calculator_node.py` - Reward computation
- `course_manager_node.py` - Episode management
- `px4_interface_node.py` - PX4 drone control interface

### `scripts/` - Training & Utilities
- `train_p3o.py` - Main training script
- `hyperopt_runner.py` - Hyperparameter optimization
- `cleanup_codebase.py` - This cleanup script

### `trained_models/` - Model Outputs
- `yolo/best.pt` - Trained YOLO weights
- `p3o/` - P3O model checkpoints

### `datasets/` - Training Data
- `yolo_models/` - Pre-trained YOLO models
- `training_data/` - Dataset files

## Removed Files (Redundant/Broken)

### Broken Files:
- `rl_agent/config_example.py` - Had syntax errors
- `rl_agent/env/px4_env.py` - Incomplete implementation

### Duplicates Removed:
- `rl_agent/env/vision_processor.py` → Use `rl_agent/depth_processor.py`
- `rl_agent/direct_control_node.py` → Use `nodes/rl_agent_node.py`
- `rl_agent/px4_training_node.py` → Use `nodes/rl_agent_node.py`
- `nodes/p3o_agent_node.py` → Use `nodes/rl_agent_node.py` (more complete)
- `rl_agent/env/flight_phases.py` → Use `rl_agent/trajectory.py` (consolidated)
- `rl_agent/env/mvp_trajectory.py` → Use `rl_agent/trajectory.py` (production version)
- `rl_agent/env/px4_env_extensions.py` → Untracked file, removed
- `rl_agent/config.py` → Recreated to work with YAML/JSON config system

## Clear Separation of Concerns

- **Training**: `scripts/train_p3o.py`
- **ROS Integration**: `nodes/` directory
- **Algorithm**: `rl_agent/algorithms/`
- **Configuration**: `config/` directory
- **Vision**: `rl_agent/depth_processor.py` + `nodes/vision_processor_node.py`

No more confusion about which file does what!
