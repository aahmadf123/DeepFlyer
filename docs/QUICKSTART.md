# DeepFlyer Quick Start Guide

Get up and running with DeepFlyer, the drone-based reinforcement learning platform that extends AWS DeepRacer principles to 3D flight.

## Prerequisites

### Hardware Requirements
- **Drone**: HolyBro X500 V2 or compatible quadcopter
- **Flight Controller**: Pixhawk 6C with PX4 firmware
- **Companion Computer**: Raspberry Pi 4B (4GB+ RAM recommended)
- **Camera**: ZED Mini stereo camera
- **Safety Equipment**: Emergency stop system, safety net/cage

### Software Requirements
- Ubuntu 22.04 LTS
- ROS2 Humble
- Python 3.10+
- CUDA-capable GPU (recommended for training)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/DeepFlyer.git
cd DeepFlyer
```

### 2. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install ROS2 dependencies
sudo apt update
sudo apt install ros-humble-desktop ros-humble-px4-msgs

# Install ZED SDK (if using ZED camera)
# Download from https://www.stereolabs.com/developers/release/
```

### 3. Build the Package
```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Build
colcon build --packages-select deepflyer
source install/setup.bash
```

## Quick Training (Simulation)

Start with simulated training to familiarize yourself with the system:

### 1. Basic Training
```bash
# Run basic P3O training with default settings
python scripts/train_p3o.py --episodes 100 --reward_preset beginner
```

### 2. Monitor Training
```bash
# In another terminal, watch training progress
tensorboard --logdir logs/
```

### 3. Test Environment
```bash
# Validate your environment setup
python scripts/validate_environment.py
```

## Hardware Setup

### 1. Camera Calibration
```bash
# Launch ZED camera calibration
ros2 launch deepflyer zed_calibration.launch.py
```

### 2. Flight Controller Setup
```bash
# Test PX4 connection
ros2 launch deepflyer px4_connection_test.launch.py
```

### 3. Safety Systems
```bash
# Verify emergency stop
ros2 run deepflyer emergency_stop_test
```

## Real Flight Training

**⚠️ SAFETY FIRST**: Always test in a safe, enclosed area with emergency stop ready.

### 1. Pre-flight Checks
```bash
# Run comprehensive system check
python scripts/preflight_check.py
```

### 2. Launch Full System
```bash
# Start all nodes for real flight
ros2 launch deepflyer deepflyer_training.launch.py
```

### 3. Begin Training
```bash
# Start training with hardware
python scripts/train_p3o.py --use_hardware --episodes 50 --reward_preset intermediate
```

## Reward Function Customization

DeepFlyer follows AWS DeepRacer patterns for easy reward tuning:

### 1. Available Presets
- `beginner`: Forgiving, focuses on basic navigation
- `intermediate`: Balanced performance and precision  
- `advanced`: Demanding, optimizes for speed and accuracy
- `speed_focused`: Emphasizes fast completion
- `precision_focused`: Rewards accurate hoop passage

### 2. Custom Reward Function
Edit `rl_agent/rewards/rewards.py`:
```python
def reward_function(params):
    """
    Custom reward function (AWS DeepRacer style)
    
    params contains:
    - hoop_detected: bool
    - hoop_center_x, hoop_center_y: float [-1, 1]
    - hoop_distance: float [0, 1]
    - vx_norm, vy_norm, vz_norm: float [-1, 1]
    - yaw_rate_norm: float [-1, 1]
    - collision, out_of_bounds, hoop_passed: bool
    """
    reward = 0.0
    
    # Your custom logic here
    if params['hoop_detected']:
        reward += 10.0
        
    if params['hoop_passed']:
        reward += 100.0
        
    return reward
```

## Configuration

### 1. Training Parameters
Edit `config/training_config.yaml`:
```yaml
p3o:
  learning_rate: 0.0003
  batch_size: 64
  procrastination_factor: 0.95

training:
  max_episodes: 1000
  max_steps_per_episode: 500
```

### 2. Environment Variables
```bash
# Customize key parameters
export DEEPFLYER_LEARNING_RATE=0.001
export DEEPFLYER_MAX_VELOCITY=2.0
export REWARD_HOOP_PASSAGE=100.0
```

## Monitoring and Visualization

### 1. Live Training Dashboard
```bash
# Start web dashboard (if available)
python api/ml_interface.py
# Access at http://localhost:8080
```

### 2. ROS2 Visualization
```bash
# Launch RViz for 3D visualization
ros2 launch deepflyer visualization.launch.py
```

### 3. Performance Metrics
```bash
# View training statistics
python scripts/analyze_training.py --log_dir logs/
```

## Course Configuration

### 1. Standard Course
5-hoop rectangular circuit (default)

### 2. Custom Courses
Edit `rl_agent/config.py`:
```python
# Generate custom course layout
hoops = get_course_layout(
    spawn_position=(0, 0, 0.8),
    course_type='figure_eight'  # 'standard', 'linear', 'oval'
)
```

## Troubleshooting

### Common Issues

1. **"ZED camera not detected"**
   ```bash
   # Check USB connection and permissions
   lsusb | grep -i stereolabs
   sudo usermod -a -G video $USER
   ```

2. **"PX4 connection failed"** 
   ```bash
   # Verify PX4 is running
   ros2 topic list | grep fmu
   ```

3. **"YOLO model not found"**
   ```bash
   # Download pre-trained model
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt -O weights/yolo11n.pt
   ```

4. **"Training not converging"**
   - Try `beginner` reward preset
   - Reduce learning rate
   - Increase episode length
   - Check observation normalization

### Getting Help

- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Report bugs on GitHub Issues
- **Community**: Join our Discord/Slack for support
- **Email**: contact@deepflyer.ai

## Next Steps

1. **Complete the Tutorial**: Work through `docs/TUTORIAL.md`
2. **Explore Advanced Features**: Read `docs/ADVANCED_USAGE.md`  
3. **Customize Your Platform**: See `docs/CUSTOMIZATION.md`
4. **Deploy to Competition**: Follow `docs/COMPETITION_GUIDE.md`

## Safety Reminders

- ⚠️ Always use emergency stop system
- ⚠️ Fly only in designated areas
- ⚠️ Test thoroughly in simulation first
- ⚠️ Monitor battery levels constantly
- ⚠️ Have safety observer present
- ⚠️ Follow local aviation regulations

---

**Happy Flying! 🚁**

For more information, visit our [documentation](docs/) or [website](https://deepflyer.ai).
