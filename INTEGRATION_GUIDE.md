# DeepFlyer Integration Guide

This comprehensive guide explains how to integrate the DeepFlyer reinforcement learning platform with your drone system. It's designed to provide both high-level architectural understanding and detailed implementation instructions.

## Table of Contents
1. [Introduction to DeepFlyer](#introduction-to-deepflyer)
2. [System Architecture](#system-architecture)
3. [Core Components Explained](#core-components-explained)
4. [ROS and MAVROS Fundamentals](#ros-and-mavros-fundamentals)
5. [ZED Camera Integration](#zed-camera-integration)
6. [Dependencies and Setup](#dependencies-and-setup)
7. [Topic Structure and Message Flow](#topic-structure-and-message-flow)
8. [Integration Scenarios](#integration-scenarios)
9. [Configuration Reference](#configuration-reference)
10. [Testing and Validation](#testing-and-validation)
11. [Troubleshooting](#troubleshooting)
12. [Appendix: Code Examples](#appendix-code-examples)

## Introduction to DeepFlyer

DeepFlyer is an educational reinforcement learning (RL) platform for drones, designed to provide a flexible and safe environment for learning and research. The platform enables:

- Training RL agents to control drones in both simulated and real environments
- Collecting and processing sensor data from cameras and flight controllers
- Implementing and testing various reward functions for different tasks
- Enforcing safety constraints to prevent damage to drones
- Supporting users from beginner to advanced skill levels

What sets DeepFlyer apart is its flexible architecture that works in three modes:
1. **Full hardware mode**: Connects to actual drones with MAVROS and real sensors
2. **Simulation mode**: Works with Gazebo or other simulators through ROS
3. **Mock mode**: Functions without any ROS installation, using simulated data

### The Explorer and Researcher Paradigm

DeepFlyer implements two distinct user experience levels:

- **Explorer Mode**: Designed for beginners (ages 11-22) with simplified interfaces, restricted flight envelopes, and strong safety constraints. Perfect for educational settings.
  
- **Researcher Mode**: Aimed at advanced users with full feature access, customizable parameters, and flexible constraints. Ideal for university research and development.

## System Architecture

DeepFlyer follows a modular architecture built around ROS (Robot Operating System) and Gymnasium (formerly OpenAI Gym) interfaces. Here's a high-level overview of the system:

```
┌─────────────────────────────────────────────────────────────────┐
│                      DeepFlyer Architecture                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
  ┌───────────────────┐  ┌─────────────────┐  ┌──────────────┐
  │    RL Agents      │  │  Environments   │  │  Simulation/  │
  │                   │  │                 │  │  Hardware     │
  │ - P3O algorithm    │  │ - RosEnv       │  │               │
  │ - Custom agents   │  │ - MAVROSEnv    │  │ - Gazebo      │
  │                   │  │ - Explorer     │  │ - Real Drone   │
  └────────┬──────────┘  │ - Researcher   │  │ - Mock System │
           │             └────────┬────────┘  └───────┬──────┘
           │                      │                   │
           ▼                      ▼                   ▼
  ┌───────────────────┐  ┌─────────────────┐  ┌──────────────┐
  │  Reward System    │  │  Safety Layer   │  │   Sensors     │
  │                   │  │                 │  │               │
  │ - Reach target    │  │ - Geofencing    │  │ - ZED Camera  │
  │ - Avoid crashes   │  │ - Collision     │  │ - IMU         │
  │ - Fly steady      │  │   prevention    │  │ - Position    │
  └───────────────────┘  └─────────────────┘  └──────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │   ROS/MAVROS    │
                        │   Interface     │
                        └─────────────────┘
```

### Data Flow Diagram

This diagram illustrates how data flows through the system:

```
┌─────────────┐    Commands     ┌───────────┐    Actions     ┌──────────┐
│ RL Algorithm │ ───────────────► Environment│ ───────────────► Drone/Sim │
└──────┬──────┘                 └─────┬─────┘                └────┬─────┘
       │                              │                           │
       │                              │                           │
       │                        Safety Layer                      │
       │                     (action modification)                │
       │                              │                           │
       │                              │                           │
       │         Observations         │         Sensor Data       │
       ◄──────────────────────────────┤◄──────────────────────────┘
                                      │
                                      │
                              ┌──────────────┐
                              │ Reward System │
                              └──────────────┘
```

## Core Components Explained

### 1. RosEnv (`rl_agent/env/ros_env.py`)

This is the base environment class that handles ROS interaction. It:

- Implements the Gymnasium interface (reset, step, etc.)
- Manages ROS topic subscriptions and publishers
- Handles observation space and action space definitions
- Processes sensor data into a format suitable for RL algorithms
- Provides thread-safe state management
- Falls back to mock implementations when ROS is unavailable

The `RosEnv` class inherits from `gym.Env` and implements the standard Gymnasium methods:

```python
def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
    # Reset the environment state
    # Return initial observation and info

def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
    # Process the action and advance the simulation
    # Return (observation, reward, terminated, truncated, info)
```

### 2. MAVROSEnv (`rl_agent/env/mavros_env.py`)

This specialized environment extends `RosEnv` to work specifically with PX4 flight controllers via MAVROS. It:

- Adds PX4-specific flight modes and commands
- Integrates with the safety layer
- Connects with the reward system
- Provides utility methods for arming, mode changes, etc.
- Comes in Explorer and Researcher variants

Key methods include:

```python
def arm(self) -> bool:
    # Arm the drone

def disarm(self) -> bool:
    # Disarm the drone

def set_mode(self, mode: str) -> bool:
    # Set flight mode (OFFBOARD, POSCTL, etc.)

def takeoff(self, target_altitude: float) -> bool:
    # Initiate automatic takeoff sequence

def land(self) -> bool:
    # Initiate landing sequence
```

### 3. SafetyLayer (`rl_agent/env/safety_layer.py`)

The safety layer is critical for preventing drone damage. It:

- Defines spatial boundaries (geofencing)
- Implements collision prevention based on depth data
- Provides velocity ramping for smooth control
- Monitors the drone state for safety violations
- Can trigger emergency stop procedures

The main classes are:

- **SafetyBounds**: Defines limits for position, velocity, and angles
- **SafetyMonitor**: Continuously checks for violations
- **VelocityRamper**: Smooths transitions between velocity commands
- **SafetyLayer**: Combines all safety features
- **BeginnerSafetyLayer**: More restrictive version for beginners

### 4. Reward Functions (`rl_agent/rewards.py`)

The reward system provides feedback signals for RL algorithms. It:

- Defines various reward components (reaching targets, avoiding crashes, etc.)
- Allows combining multiple rewards with different weights
- Supports customization for different tasks
- Handles reward normalization and statistics

The main components are:

- **RewardRegistry**: Collection of available reward functions
- **RewardFunction**: Combines multiple rewards with weights
- **RewardComponent**: Individual reward calculation functions

### 5. ZED Integration (`rl_agent/env/zed_integration.py`)

This module handles the ZED Mini stereo camera, which provides:

- RGB images for visual processing
- Depth data for obstacle detection
- Position tracking capabilities

The two main interfaces are:

- **ROSZEDInterface**: Works with standard ZED ROS topics
- **DirectZEDInterface**: Uses the ZED SDK directly without ROS

### 6. Mock ROS (`rl_agent/env/mock_ros.py`)

This simulates ROS functionality for development without ROS, providing:

- Mock message types and topics
- Simulated sensor data generation
- Fake drone dynamics for testing
- Mock MAVROS services

## ROS and MAVROS Fundamentals

### What is ROS?

ROS (Robot Operating System) is a middleware framework for robotics, providing:

- **Node-based architecture**: Programs run as "nodes" that communicate through messages
- **Topics**: Named buses for nodes to publish/subscribe to messages
- **Services**: Request/response interactions between nodes
- **Actions**: For longer-running tasks with feedback
- **Parameter Server**: For storing configuration values

In the context of DeepFlyer, ROS enables communication between:
- The drone's flight controller
- Sensors like cameras and IMUs
- The reinforcement learning environment
- Visualization and debugging tools

### What is MAVROS?

MAVROS is a ROS package that bridges ROS and MAVLink-based flight controllers like Pixhawk running PX4 or ArduPilot:

- **MAVLink**: A lightweight messaging protocol for communicating with drones
- **MAVROS**: Translates between ROS topics/services and MAVLink messages

MAVROS provides ROS topics for:
- Sending velocity/position commands to the drone
- Reading drone state (position, orientation, etc.)
- Accessing sensor data
- Changing flight modes
- Arming/disarming the drone

### Key ROS Concepts for Integration

1. **Namespaces**: ROS uses namespaces to organize topics. DeepFlyer uses the namespace parameter (default: "deepflyer") to prefix standard topics.

2. **Quality of Service (QoS)**: ROS2 uses QoS profiles to define reliability and history policies for topics:
   - **Reliable**: Guarantees delivery but may cause delays
   - **Best Effort**: Faster but may drop messages
   - DeepFlyer configures appropriate QoS for different types of data

3. **Coordinate Frames**: ROS uses TF2 for coordinate transformations:
   - DeepFlyer uses ENU (East-North-Up) as its default frame
   - Conversion may be needed if your system uses a different frame

## ZED Camera Integration

The ZED Mini is a stereo camera that provides:

- RGB images at up to 1080p resolution
- Depth maps with 15m range
- Visual-inertial odometry for position tracking

DeepFlyer integrates with the ZED camera in two ways:

1. **Via ROS topics** (preferred): Using the ZED ROS wrapper
2. **Directly via SDK**: For systems without ROS

### ZED ROS Topics

When using the ZED with ROS, DeepFlyer subscribes to:

- `/zed_mini/zed_node/rgb/image_rect_color`: Color images
- `/zed_mini/zed_node/depth/depth_registered`: Depth maps

The data is used for:
- Visual feedback in observations
- Obstacle detection via depth processing
- Position tracking (if MAVROS is unavailable)

## Dependencies and Setup

### Python Dependencies

```
gymnasium>=0.26.0      # RL environment interface
numpy>=1.20.0          # Numerical operations
matplotlib>=3.5.0      # Visualization
opencv-python>=4.5.0   # Image processing
pyyaml>=6.0            # Configuration
```

### ROS2 Dependencies (Optional)

For full functionality with real hardware:

```bash
# Install ROS2 Humble (or newer)
sudo apt install ros-humble-desktop

# Install MAVROS
sudo apt install ros-humble-mavros ros-humble-mavros-extras

# Install ZED ROS2 wrapper
sudo apt install ros-humble-zed-ros2-wrapper
```

### ZED SDK (Optional)

For direct ZED camera access without ROS:

1. Download the ZED SDK from https://www.stereolabs.com/developers/release/
2. Install following the instructions for your platform
3. Install the Python API: `pip install pyzed`

### Installing DeepFlyer

```bash
# Clone the repository
git clone https://github.com/username/deepflyer.git
cd deepflyer

# Install in development mode
pip install -e .
```

## Topic Structure and Message Flow

### MAVROS Topics in Detail

| Topic | Message Type | Fields | Purpose |
|-------|--------------|--------|---------|
| `/mavros/state` | `State` | connected, armed, guided, mode | Overall drone state |
| `/mavros/local_position/pose` | `PoseStamped` | position.{x,y,z}, orientation.{x,y,z,w} | Position and orientation |
| `/mavros/imu/data` | `Imu` | linear_acceleration, angular_velocity | IMU measurements |
| `/mavros/setpoint_velocity/cmd_vel_unstamped` | `Twist` | linear.{x,y,z}, angular.{x,y,z} | Velocity commands |

### Message Flow Example: Velocity Command

1. **RL Agent** outputs an action: `[0.5, 0.0, 0.2, 0.1]` (forward, no lateral, up, slight yaw)
2. **Environment** processes action via `_process_action()` into linear and angular velocity
3. **Safety Layer** checks if the command is safe and modifies if needed
4. **ROS Interface** creates a `Twist` message with the velocity components
5. **MAVROS** converts the ROS message to MAVLink and sends to flight controller
6. **Flight Controller** executes the command on the motors

### Message Flow Example: State Update

1. **Flight Controller** sends MAVLink message with position update
2. **MAVROS** converts to ROS `PoseStamped` message on `/mavros/local_position/pose`
3. **ROS Interface** receives message via subscription callback
4. **State Container** updates position data in thread-safe manner
5. **Environment** accesses updated state when computing observations
6. **RL Agent** receives the new state as part of observation

## Integration Scenarios

### Scenario 1: Integration with PX4 Simulation

This scenario uses Gazebo with PX4 Software-In-The-Loop (SITL):

1. **Start PX4 SITL with Gazebo**:
   ```bash
   cd ~/PX4-Autopilot
   make px4_sitl gazebo
   ```

2. **Start MAVROS**:
   ```bash
   # In a new terminal
   source /opt/ros/humble/setup.bash
   ros2 launch mavros px4.launch fcu_url:="udp://:14540@localhost:14557"
   ```

3. **Create and use the environment**:
   ```python
   import numpy as np
   from rl_agent.env.mavros_env import MAVROSExplorerEnv
   
   # Create environment with default settings
   env = MAVROSExplorerEnv()
   
   # Reset the environment
   obs, info = env.reset()
   
   # Run a simple control loop
   for _ in range(100):
       # Take a simple action (move forward)
       action = np.array([0.5, 0.0, 0.0, 0.0])
       
       # Step the environment
       obs, reward, terminated, truncated, info = env.step(action)
       
       # Print some information
       print(f"Position: {obs['position']}, Reward: {reward}")
       
       if terminated or truncated:
           break
   
   # Clean up
   env.close()
   ```

### Scenario 2: Integration with Real Drone

For a real drone with Pixhawk and ZED Mini:

1. **Start MAVROS**:
   ```bash
   # Replace with your serial port and baud rate
   ros2 launch mavros px4.launch fcu_url:="serial:///dev/ttyACM0:921600"
   ```

2. **Start ZED Camera**:
   ```bash
   ros2 launch zed_wrapper zed_mini.launch.py
   ```

3. **Create and use the environment with safety emphasis**:
   ```python
   import numpy as np
   from rl_agent.env.mavros_env import MAVROSExplorerEnv
   import time
   
   # Create environment with cautious settings
   env = MAVROSExplorerEnv(
       auto_arm=False,            # Don't arm automatically
       auto_offboard=False,       # Don't set OFFBOARD automatically
       step_duration=0.1,         # Slower control (10Hz)
       use_zed=True,              # Use ZED camera
   )
   
   # Reset the environment
   obs, info = env.reset()
   
   # Manually prepare for flight (only when ready)
   input("Press Enter to arm and set OFFBOARD mode...")
   
   # Arm the drone
   if env.arm():
       print("Arming successful")
   else:
       print("Arming failed")
       env.close()
       exit(1)
   
   # Wait for arming to complete
   time.sleep(1.0)
   
   # Set to OFFBOARD mode
   if env.set_offboard_mode():
       print("OFFBOARD mode set")
   else:
       print("Failed to set OFFBOARD mode")
       env.disarm()
       env.close()
       exit(1)
   
   # Wait for mode change to take effect
   time.sleep(1.0)
   
   # Execute a simple takeoff and hover
   try:
       # Move up slowly
       for _ in range(20):
           action = np.array([0.0, 0.0, 0.2, 0.0])  # Up at 0.2 m/s
           obs, reward, terminated, truncated, info = env.step(action)
           print(f"Altitude: {obs['position'][2]:.2f}m")
       
       # Hover for 5 seconds
       for _ in range(50):  # 50 steps at 10Hz = 5 seconds
           action = np.array([0.0, 0.0, 0.0, 0.0])  # Hover
           obs, reward, terminated, truncated, info = env.step(action)
           print(f"Position: {obs['position']}")
           
           if terminated or truncated:
               break
       
       # Land
       print("Landing...")
       env.land()
       
   except KeyboardInterrupt:
       print("Emergency landing...")
       env.land()
   
   finally:
       # Make sure to close properly
       env.close()
   ```

### Scenario 3: Development without ROS

For development on systems without ROS:

```python
import numpy as np
from rl_agent.env.mavros_env import MAVROSResearcherEnv
import matplotlib.pyplot as plt

# Create environment with researcher settings
# Will automatically use mock implementation
env = MAVROSResearcherEnv(
    with_noise=True,           # Add noise to observations
    noise_level=0.02,          # 2% noise
)

# Reset with random goal
obs, info = env.reset()

# Storage for visualization
positions = []
rewards = []

# Run a simple episode
for i in range(200):
    # Implement a simple "move toward goal" policy
    goal_relative = obs['goal_relative']
    distance = np.linalg.norm(goal_relative)
    
    if distance > 0.1:
        # Normalize direction vector
        direction = goal_relative / distance
        
        # Scale by distance (faster when far, slower when close)
        speed = min(0.8, distance / 2.0)
        
        # Create action: scaled direction vector + zero yaw
        action = np.array([
            direction[0] * speed,
            direction[1] * speed,
            direction[2] * speed,
            0.0
        ])
    else:
        # At goal, just hover
        action = np.zeros(4)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Store for visualization
    positions.append(obs['position'].copy())
    rewards.append(reward)
    
    if terminated or truncated:
        print(f"Episode ended after {i+1} steps")
        break

# Plot results
positions = np.array(positions)
plt.figure(figsize=(12, 8))

# Position plot
plt.subplot(2, 1, 1)
plt.plot(positions[:, 0], label='X')
plt.plot(positions[:, 1], label='Y')
plt.plot(positions[:, 2], label='Z')
plt.legend()
plt.title('Drone Position')
plt.xlabel('Step')
plt.ylabel('Position (m)')

# Reward plot
plt.subplot(2, 1, 2)
plt.plot(rewards)
plt.title('Rewards')
plt.xlabel('Step')
plt.ylabel('Reward')

plt.tight_layout()
plt.savefig('simulation_results.png')
plt.show()

# Clean up
env.close()
```

## Configuration Reference

### Environment Parameters Explained

| Parameter | Type | Description | Default | 
|-----------|------|-------------|---------|
| `namespace` | str | ROS namespace for topics | "deepflyer" |
| `observation_config` | Dict[str, bool] | Which observations to include | See below |
| `action_mode` | str | "continuous" or "discrete" | "continuous" |
| `max_episode_steps` | int | Maximum steps per episode | 500 |
| `step_duration` | float | Duration of each step in seconds | 0.05 |
| `timeout` | float | Timeout for sensor data in seconds | 5.0 |
| `goal_position` | List[float] | Target position [x, y, z] | [5.0, 5.0, 1.5] |
| `target_altitude` | float | Default altitude for takeoff | None |
| `camera_resolution` | Tuple[int, int] | Resolution for camera images | (84, 84) |
| `cross_track_weight` | float | Weight for cross-track error (path following) | 1.0 |
| `heading_weight` | float | Weight for heading error | 0.1 |
| `use_zed` | bool | Whether to use ZED camera | True |
| `auto_arm` | bool | Automatically arm on reset | False |
| `auto_offboard` | bool | Auto set OFFBOARD mode | False |
| `safety_boundaries` | Dict[str, float] | Position and velocity limits | See SafetyBounds |
| `enable_safety_layer` | bool | Whether to enable safety layer | True |
| `reward_function` | RewardFunction | Custom reward function | Default reward |

#### Default Observation Config
```python
{
    'position': True,           # [x, y, z] position
    'orientation': True,        # Quaternion [x, y, z, w]
    'linear_velocity': True,    # [vx, vy, vz] velocity
    'angular_velocity': True,   # [wx, wy, wz] angular velocity
    'front_camera': True,       # RGB image from front camera
    'down_camera': True,        # RGB image from down camera (if not using ZED)
    'collision': True,          # Collision flag
    'obstacle_distance': True,  # Distance to nearest obstacle
    'goal_relative': True,      # Vector to goal position
}
```

### Safety Boundaries

The SafetyBounds class defines limits for safe operation:

```python
@dataclass
class SafetyBounds:
    x_min: float = -10.0        # Minimum X position
    x_max: float = 10.0         # Maximum X position
    y_min: float = -10.0        # Minimum Y position
    y_max: float = 10.0         # Maximum Y position
    z_min: float = 0.1          # Minimum height (m)
    z_max: float = 5.0          # Maximum height (m)
    
    vel_max_xy: float = 2.0     # Maximum horizontal velocity (m/s)
    vel_max_z_up: float = 1.0   # Maximum ascent velocity (m/s)
    vel_max_z_down: float = 0.5 # Maximum descent velocity (m/s)
    
    max_tilt_angle: float = 30.0  # Maximum tilt angle in degrees
    min_distance_to_obstacle: float = 0.5  # Min distance to obstacles (m)
```

To customize these boundaries:

```python
safety_boundaries = {
    'x_min': -5.0, 'x_max': 5.0,
    'y_min': -5.0, 'y_max': 5.0,
    'z_min': 0.3,  'z_max': 3.0,
    'vel_max_xy': 1.0,
    'min_distance_to_obstacle': 1.0,
}

env = MAVROSEnv(safety_boundaries=safety_boundaries)
```

## Testing and Validation

### 1. Basic Functionality Test

This tests the core environment functionality:

```bash
# Test Explorer mode environment
python scripts/test_mavros_env.py --mode explorer

# Test Researcher mode environment
python scripts/test_mavros_env.py --mode researcher

# Test MAVROS-specific functions
python scripts/test_mavros_env.py --mode mavros
```

### 2. Safety Layer Test

This tests how the safety layer prevents unsafe actions:

```bash
# Test safety layer functionality
python scripts/test_safety_rewards.py --test safety
```

The test:
- Creates potentially unsafe actions (e.g., trying to fly too high)
- Shows how the safety layer modifies these actions
- Visualizes the original vs. modified commands
- Tests emergency stop functionality

### 3. Reward Function Test

This tests different reward configurations:

```bash
# Test reward functions
python scripts/test_safety_rewards.py --test rewards
```

The test:
- Compares different reward weightings
- Shows how rewards affect behavior
- Visualizes reward components and total rewards

### 4. Integration Validation Checklist

When integrating with your own system, verify:

1. **ROS Communication**:
   - All required topics are present: `ros2 topic list | grep <namespace>`
   - Message formats match expectations: `ros2 interface show <message_type>`
   - QoS settings are compatible: Check for dropped messages

2. **Flight Controller**:
   - MAVROS can connect to the flight controller: Check `connected` status
   - Arming commands work: Test `env.arm()` and verify `armed` status
   - Mode changes work: Test `env.set_mode("OFFBOARD")` and verify mode

3. **Sensor Integration**:
   - Camera images are received: Check image dimensions and content
   - IMU data is processed: Verify acceleration and angular velocity values
   - Position tracking works: Check position values match expectations

4. **Safety System**:
   - Boundaries are respected: Try to fly beyond limits and verify prevention
   - Collision avoidance works: Approach obstacles and verify safe behavior
   - Emergency stop functions: Trigger emergency and verify zero velocity

## Troubleshooting

### Common Issues and Solutions

#### 1. ROS Topic Issues

**Problem**: Missing or mismatched topics  
**Diagnostic**: `ros2 topic list | grep <expected_topic>`  
**Solutions**:
- Check namespace settings match your ROS setup
- Verify MAVROS is running: `ros2 node list | grep mavros`
- For ZED camera: `ros2 node list | grep zed`
- Adjust topic names in code if your setup uses different conventions

#### 2. MAVROS Connection Issues

**Problem**: MAVROS cannot connect to flight controller  
**Diagnostic**: Check `/mavros/state` topic: `ros2 topic echo /mavros/state`  
**Solutions**:
- Check physical connection (USB cable, telemetry radio)
- Verify correct port and baud rate in MAVROS launch
- Check flight controller firmware is compatible (PX4 recommended)
- Try increasing connection timeout

#### 3. Safety Layer Restrictions

**Problem**: Drone won't move as commanded  
**Diagnostic**: Enable debug logging and check for safety violation messages  
**Solutions**:
- Check if actions are being limited by safety bounds
- Adjust safety boundaries to match your environment
- Temporarily disable safety layer for testing: `enable_safety_layer=False`
- Check `intervention_count` in info dict to see if safety layer is active

#### 4. Mock Mode Detection Issues

**Problem**: Environment using mock implementation despite ROS being available  
**Diagnostic**: Check log for "ROS2 not available, using mock objects" message  
**Solutions**:
- Ensure ROS environment variables are set: `echo $ROS_DISTRO`
- Verify Python environment can import rclpy: `python -c "import rclpy; print('OK')"`
- Check for ROS package installation issues
- Reinstall ROS Python packages: `pip install -e /opt/ros/<distro>/lib/python3.x/site-packages`

#### 5. ZED Camera Issues

**Problem**: ZED camera data not being processed correctly  
**Diagnostic**: Check topic data: `ros2 topic echo /zed_mini/zed_node/rgb/image_rect_color/camera_info`  
**Solutions**:
- Verify ZED camera is connected and powered
- Check ZED SDK installation: `zed_camera_tool`
- Ensure ZED ROS wrapper is running: `ros2 node list | grep zed`
- Try using direct SDK interface instead of ROS: `use_zed_ros=False`

### Logging and Diagnostics

#### Enabling Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Create environment with detailed logging
env = MAVROSEnv()
```

#### Diagnostic Information

```python
# Get environment information
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"MAVROS available: {env.use_mavros}")

# Check current state
state = env.node.state.get_snapshot()
print(f"Current position: {state['position']}")
print(f"Armed status: {state['armed']}")
print(f"Flight mode: {state['flight_mode']}")

# Check safety status
if env.enable_safety_layer:
    safety_status = env.safety_layer.get_status()
    print(f"Safety status: {safety_status}")
    print(f"Safety violations: {safety_status['violations']}")
    print(f"Intervention count: {safety_status['interventions']}")
```

## Appendix: Code Examples

### Example 1: Simple Hover Controller

```python
import numpy as np
import time
from rl_agent.env.mavros_env import MAVROSEnv

# Create environment
env = MAVROSEnv(
    auto_arm=False,
    auto_offboard=False,
)

# Reset environment
obs, info = env.reset()

# Manual preparation
print("Arming drone...")
env.arm()
time.sleep(1.0)

print("Setting OFFBOARD mode...")
env.set_offboard_mode()
time.sleep(1.0)

# Takeoff to 1.5m
print("Taking off...")
current_altitude = obs['position'][2]
target_altitude = 1.5

while current_altitude < target_altitude - 0.1:
    # Simple proportional control for takeoff
    altitude_error = target_altitude - current_altitude
    vz = min(0.5, altitude_error * 0.5)  # P controller with limit
    
    # Action: [vx, vy, vz, yaw_rate]
    action = np.array([0.0, 0.0, vz, 0.0])
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    current_altitude = obs['position'][2]
    print(f"Altitude: {current_altitude:.2f}m")
    
    if terminated or truncated:
        break

# Hover for 10 seconds
print("Hovering...")
hover_start = time.time()
while time.time() - hover_start < 10.0:
    # Hover action
    action = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Position: {obs['position']}")
    
    if terminated or truncated:
        break

# Land
print("Landing...")
env.land()

# Clean up
env.close()
```

### Example 2: Custom Reward Function

```python
import numpy as np
from rl_agent.env.mavros_env import MAVROSResearcherEnv
from rl_agent.rewards import create_cross_track_and_heading_reward

# Create a custom two-term reward function
reward_fn = create_cross_track_and_heading_reward(
    cross_track_weight=1.0,
    heading_weight=0.1,
    max_error=2.0,
    max_heading_error=np.pi,
    trajectory=trajectory,
)

# Create environment with custom reward function
env = MAVROSResearcherEnv(
    reward_function=reward_fn,
    goal_position=trajectory[-1],  # Last point in trajectory
)

# Reset environment and add trajectory index to state
obs, info = env.reset()

# Use environment as usual...
# The reward function will now include trajectory following
```

### Example 3: Safety Layer Configuration

```python
import numpy as np
from rl_agent.env.mavros_env import MAVROSEnv
from rl_agent.env.safety_layer import SafetyBounds

# Create custom safety bounds
safety_bounds = SafetyBounds(
    # Position limits
    x_min=-3.0, x_max=3.0,
    y_min=-3.0, y_max=3.0,
    z_min=0.5,  z_max=2.0,
    
    # Velocity limits
    vel_max_xy=1.0,       # Max horizontal speed 1 m/s
    vel_max_z_up=0.5,     # Max ascent speed 0.5 m/s
    vel_max_z_down=0.3,   # Max descent speed 0.3 m/s
    
    # Attitude limits
    max_tilt_angle=15.0,  # Max tilt 15 degrees
    
    # Obstacle avoidance
    min_distance_to_obstacle=1.0,  # Stay 1m from obstacles
)

# Create environment with custom safety bounds
env = MAVROSEnv(
    enable_safety_layer=True,
    safety_boundaries={
        'x_min': -3.0, 'x_max': 3.0,
        'y_min': -3.0, 'y_max': 3.0,
        'z_min': 0.5,  'z_max': 2.0,
        'vel_max_xy': 1.0,
        'vel_max_z_up': 0.5,
        'vel_max_z_down': 0.3,
        'max_tilt_angle': 15.0,
        'min_distance_to_obstacle': 1.0,
    }
)

# Use environment as usual...
# The safety layer will enforce these custom bounds
``` 