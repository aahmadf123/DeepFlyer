# DeepFlyer Integration Guide

This document provides detailed instructions for integrating DeepFlyer's reinforcement learning environment with ROS/MAVROS systems. It covers required dependencies, ROS topic structure, and step-by-step integration instructions.

## Table of Contents
1. [System Overview](#system-overview)
2. [Dependencies](#dependencies)
3. [ROS Topic Structure](#ros-topic-structure)
4. [Integration Steps](#integration-steps)
5. [Configuration Options](#configuration-options)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

## System Overview

DeepFlyer is a reinforcement learning platform for drones consisting of:

### Core Components

| Component | Description | File |
|-----------|-------------|------|
| **RosEnv** | Base environment with ROS2 integration | `rl_agent/env/ros_env.py` |
| **MAVROSEnv** | Specialized environment for PX4/MAVROS | `rl_agent/env/mavros_env.py` |
| **SafetyLayer** | Collision prevention and safety monitoring | `rl_agent/env/safety_layer.py` |
| **ZED Integration** | Interface for ZED Mini stereo camera | `rl_agent/env/zed_integration.py` |
| **Reward Functions** | Modular reward system for RL | `rl_agent/rewards.py` |
| **Mock ROS** | Simulated ROS for development without ROS | `rl_agent/env/mock_ros.py` |

### Environment Modes

- **Explorer Mode** (`MAVROSExplorerEnv`): Simplified for beginners (ages 11-22)
- **Researcher Mode** (`MAVROSResearcherEnv`): Full features for advanced users

## Dependencies

### Python Dependencies
```
gymnasium>=0.26.0
numpy>=1.20.0
matplotlib>=3.5.0
pyyaml>=6.0
opencv-python>=4.5.0
```

### Optional Dependencies
- **ROS2** (Humble or newer): For real hardware integration
- **MAVROS**: For PX4 flight controller interaction
- **ZED SDK**: For direct ZED camera access (without ROS)
- **scipy**: For quaternion/rotation operations

## ROS Topic Structure

DeepFlyer subscribes to and publishes on the following ROS topics:

### MAVROS Topics

| Topic | Message Type | Direction | Purpose |
|-------|--------------|-----------|---------|
| `/mavros/state` | `mavros_msgs/State` | Subscribe | Flight controller state |
| `/mavros/local_position/pose` | `geometry_msgs/PoseStamped` | Subscribe | Position data |
| `/mavros/imu/data` | `sensor_msgs/Imu` | Subscribe | IMU measurements |
| `/mavros/setpoint_velocity/cmd_vel_unstamped` | `geometry_msgs/Twist` | Publish | Velocity commands |

### ZED Camera Topics

| Topic | Message Type | Direction | Purpose |
|-------|--------------|-----------|---------|
| `/zed_mini/zed_node/rgb/image_rect_color` | `sensor_msgs/Image` | Subscribe | RGB image |
| `/zed_mini/zed_node/depth/depth_registered` | `sensor_msgs/Image` | Subscribe | Depth image |

### Standard Topics (Non-MAVROS Mode)

| Topic | Message Type | Direction | Purpose |
|-------|--------------|-----------|---------|
| `/{namespace}/pose` | `geometry_msgs/PoseStamped` | Subscribe | Position data |
| `/{namespace}/odom` | `nav_msgs/Odometry` | Subscribe | Velocity data |
| `/{namespace}/imu` | `sensor_msgs/Imu` | Subscribe | IMU measurements |
| `/{namespace}/camera/front/image_raw` | `sensor_msgs/Image` | Subscribe | Front camera |
| `/{namespace}/camera/down/image_raw` | `sensor_msgs/Image` | Subscribe | Down camera |
| `/{namespace}/collision` | `std_msgs/Bool` | Subscribe | Collision flag |
| `/{namespace}/obstacle_distance` | `std_msgs/Float32` | Subscribe | Distance to obstacle |
| `/{namespace}/battery_level` | `std_msgs/Float32` | Subscribe | Battery level |
| `/{namespace}/cmd_vel` | `geometry_msgs/Twist` | Publish | Velocity commands |
| `/{namespace}/reset` | `std_msgs/Bool` | Publish | Reset simulation |

### MAVROS Services

| Service | Type | Purpose |
|---------|------|---------|
| `/mavros/cmd/arming` | `mavros_msgs/CommandBool` | Arm/disarm drone |
| `/mavros/set_mode` | `mavros_msgs/SetMode` | Change flight mode |

## Integration Steps

### 1. Install the Package

```bash
# Clone the repository (replace with actual repo URL)
git clone https://github.com/username/deepflyer.git
cd deepflyer

# Install in development mode
pip install -e .
```

### 2. Configure ROS Environment

Ensure ROS2 and MAVROS are properly installed and configured on your system:

```bash
# Install ROS2 dependencies
sudo apt install ros-humble-mavros ros-humble-mavros-extras
```

If using the ZED camera, ensure the ROS2 ZED wrapper is installed:

```bash
# Install ZED ROS2 wrapper
sudo apt install ros-humble-zed-ros2-wrapper
```

### 3. Configure Coordinate Frames

DeepFlyer uses ENU (East-North-Up) coordinate frame by default. If your system uses a different frame, you'll need to configure the appropriate transformations.

### 4. Integration Options

#### Option A: Using with Real Hardware

1. Start ROS2 and MAVROS:
   ```bash
   # Terminal 1: Start ROS2 core
   ros2 launch mavros px4.launch fcu_url:="serial:///dev/ttyACM0:57600"

   # Terminal 2 (if using ZED): Start ZED camera
   ros2 launch zed_wrapper zed_mini.launch.py
   ```

2. Run your RL agent with the real environment:
   ```python
   from rl_agent.env.mavros_env import MAVROSExplorerEnv

   # Create environment with real hardware
   env = MAVROSExplorerEnv(
       namespace="deepflyer",
       use_zed=True,
       auto_arm=False  # For safety
   )

   # Use environment with your RL algorithm
   obs, info = env.reset()
   # ... rest of your RL code
   ```

#### Option B: Using with Simulation

DeepFlyer can work with Gazebo or any other simulator that publishes the required ROS topics.

1. Start your simulator:
   ```bash
   # Example with PX4 SITL and Gazebo
   cd ~/PX4-Autopilot
   make px4_sitl gazebo
   ```

2. In another terminal, run MAVROS:
   ```bash
   ros2 launch mavros px4.launch fcu_url:="udp://:14540@localhost:14557"
   ```

3. Run your RL code as in Option A.

#### Option C: Development without ROS

For development without ROS, DeepFlyer provides a mock implementation:

```python
from rl_agent.env.mavros_env import MAVROSExplorerEnv

# The environment will automatically use mock implementation
env = MAVROSExplorerEnv()

# Use environment as normal
obs, info = env.reset()
```

### 5. Key Integration Points

1. **Namespace**: Set the correct namespace to match your ROS setup
2. **Topic Names**: Ensure topics match what your system is publishing
3. **QoS Settings**: Adjust if message dropping occurs
4. **Safety Boundaries**: Configure to match your physical environment

## Configuration Options

### Environment Parameters

| Parameter | Description | Default | Note |
|-----------|-------------|---------|------|
| `namespace` | ROS namespace | "deepflyer" | Must match your topic structure |
| `use_zed` | Use ZED camera | True | Set to False if using standard cameras |
| `use_mavros` | Use MAVROS topics | True | Automatically falls back to standard topics if MAVROS not available |
| `auto_arm` | Auto arm on reset | False | Set to True with caution |
| `auto_offboard` | Auto OFFBOARD mode | False | Set to True with caution |
| `safety_boundaries` | Position and velocity limits | See code | Adjust for your drone |
| `enable_safety_layer` | Use safety features | True | Strongly recommended |
| `goal_position` | Target position | [5.0, 5.0, 1.5] | For navigation tasks |

### Reward Function Configuration

The reward function can be customized by adjusting weights:

```python
custom_weights = {
    'reach_target': 1.0,
    'avoid_crashes': 1.5,
    'fly_steady': 0.5,
    'minimize_time': 0.1,
}

env = MAVROSEnv(
    custom_reward_weights=custom_weights
)
```

Or by creating a custom reward function:

```python
from rl_agent.rewards import create_default_reward_function

reward_fn = create_default_reward_function()
reward_fn.add_component("follow_trajectory", weight=1.0, 
                        parameters={'trajectory': [
                            np.array([0.0, 0.0, 1.5]),
                            np.array([5.0, 5.0, 1.5]),
                        ]})

env = MAVROSEnv(reward_function=reward_fn)
```

## Testing

### Basic Functionality Test

```bash
# Test Explorer mode environment
python scripts/test_mavros_env.py --mode explorer

# Test Researcher mode environment
python scripts/test_mavros_env.py --mode researcher

# Test MAVROS-specific functions
python scripts/test_mavros_env.py --mode mavros
```

### Safety and Reward Test

```bash
# Test safety layer and reward functions
python scripts/test_safety_rewards.py
```

This will produce visualization plots showing how the safety layer prevents unsafe actions and how different reward configurations affect behavior.

## Troubleshooting

### Common Issues

1. **Missing ROS Topics**:
   - Check that all required topics are published with `ros2 topic list`
   - Verify topic names match what your code expects
   - Use `ros2 topic echo <topic>` to check message content

2. **Safety Layer Overrides**:
   - If the drone won't move as expected, check if safety layer is restricting movement
   - Check logs for "Safety violation detected!" messages

3. **Mock Mode Detection**:
   - If using real hardware but environment falls back to mock mode:
   - Ensure ROS environment variables are set correctly
   - Check Python environment has rclpy and required packages

4. **ZED Camera Integration**:
   - If depth data is not being processed correctly:
   - Check topic format with `ros2 topic echo /zed_mini/zed_node/depth/depth_registered`
   - Verify the camera is calibrated properly

### Diagnostic Tools

```python
# Print environment info
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"MAVROS available: {env.use_mavros}")

# Get safety status
if env.enable_safety_layer:
    print(f"Safety status: {env.safety_layer.get_status()}")
    
# Check if connected to real flight controller
print(f"Connected to FC: {env.is_connected()}")
```

## Coordinate Systems

DeepFlyer uses the following coordinate system:
- X: Forward
- Y: Left
- Z: Up

If your system uses a different convention, you'll need to transform coordinates appropriately.

## Contact

For integration issues or questions, please contact the DeepFlyer team. 