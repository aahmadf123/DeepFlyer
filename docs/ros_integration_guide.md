# ROS2/Gazebo Integration Guide for DeepFlyer

This guide explains how the RL agent integrates with ROS2 and Gazebo simulation for the DeepFlyer project.

## Overview

The integration consists of three main components:

1. **ROS2 Environment (`rl_agent/env/ros_env.py`)** - Gymnasium-compatible environment that communicates with Gazebo
2. **Utilities (`rl_agent/env/ros_utils.py`)** - Message conversion, safety monitoring, and PX4 interface
3. **Reward Integration** - Seamless integration with the reward function registry

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Gazebo World   │────►│   ROS2 Topics    │────►│    RosEnv       │
│  (Uma's work)   │     │                  │     │  (RL Agent)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         ▲                                                 │
         │                                                 │
         └─────────────────────────────────────────────────┘
                        Control Commands
```

## ROS2 Topics

The RL agent expects the following ROS2 topics from the Gazebo simulation:

### Input Topics (Subscribed by RL Agent)

| Topic | Message Type | Description | Rate |
|-------|-------------|-------------|------|
| `/deepflyer/pose` | `geometry_msgs/PoseStamped` | Drone position and orientation | 50 Hz |
| `/deepflyer/odom` | `nav_msgs/Odometry` | Velocity information | 100 Hz |
| `/deepflyer/imu` | `sensor_msgs/Imu` | IMU data (accelerations, angular rates) | 100 Hz |
| `/deepflyer/camera/front/image_raw` | `sensor_msgs/Image` | Front camera RGB image | 15 Hz |
| `/deepflyer/camera/down/image_raw` | `sensor_msgs/Image` | Downward camera RGB image | 10 Hz |
| `/deepflyer/collision` | `std_msgs/Bool` | Collision detection flag | On event |
| `/deepflyer/obstacle_distance` | `std_msgs/Float32` | Distance to nearest obstacle | 20 Hz |
| `/deepflyer/battery_level` | `std_msgs/Float32` | Battery level (0-1) | 1 Hz |

### Output Topics (Published by RL Agent)

| Topic | Message Type | Description | Rate |
|-------|-------------|-------------|------|
| `/deepflyer/cmd_vel` | `geometry_msgs/Twist` | Velocity commands | 20 Hz |
| `/deepflyer/reset` | `std_msgs/Bool` | Reset simulation | On demand |

## Environment Parameters

The ROS environment accepts these parameters from documentation:

```python
env = RosEnv(
    namespace="deepflyer",              # ROS2 namespace
    action_mode="continuous",           # or "discrete"
    max_episode_steps=500,             # 25 seconds at 20Hz
    step_duration=0.05,                # 20Hz control
    goal_position=[5.0, 5.0, 1.5],     # Target position
    target_altitude=1.5,               # For altitude hold tasks
    camera_resolution=(84, 84),        # Downsampled for RL
)
```

## Gazebo World Requirements

For Uma's Gazebo worlds, ensure:

1. **Coordinate System**: Use ENU (East-North-Up) or provide transforms
2. **Collision Detection**: Publish to `/deepflyer/collision` on contact
3. **Reset Service**: Handle `/deepflyer/reset` to reset drone position
4. **Physics Rate**: At least 1000 Hz for stable simulation

## State Space

The environment provides these observations:

```python
observation = {
    'position': [x, y, z],              # meters
    'orientation': [x, y, z, w],        # quaternion
    'linear_velocity': [vx, vy, vz],    # m/s
    'angular_velocity': [wx, wy, wz],   # rad/s
    'linear_acceleration': [ax, ay, az], # m/s²
    'front_camera': (84, 84, 3),        # RGB image
    'down_camera': (84, 84, 3),         # RGB image
    'collision': [0 or 1],              # boolean flag
    'obstacle_distance': [distance],     # meters
    'goal_relative': [dx, dy, dz],      # relative to goal
}
```

## Action Space

### Continuous Mode
- Action: `[vx, vy, vz, wz]` normalized to [-1, 1]
- Mapped to: 
  - Linear velocity: ±1.5 m/s (x, y), ±1.0 m/s (z)
  - Angular velocity: ±π/2 rad/s (yaw)

### Discrete Mode
- 9 actions: hover, forward, backward, left, right, up, down, rotate left, rotate right

## Safety Features

The environment includes safety monitoring:

```python
safety_monitor = SafetyMonitor(
    max_velocity=2.0,         # m/s
    max_acceleration=5.0,     # m/s²
    min_altitude=0.3,         # m
    max_altitude=2.8,         # m
    geofence_bounds=(-5, -5, 15, 15)  # x_min, y_min, x_max, y_max
)
```

## PX4/MAVROS Integration

For PX4 SITL integration:

1. **Launch PX4 SITL**: 
   ```bash
   make px4_sitl gazebo
   ```

2. **Launch MAVROS**:
   ```bash
   ros2 launch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557
   ```

3. **Topic Remapping**: Map MAVROS topics to expected topics:
   ```yaml
   remappings:
     - /mavros/local_position/pose -> /deepflyer/pose
     - /mavros/local_position/odom -> /deepflyer/odom
     - /mavros/imu/data -> /deepflyer/imu
     - /mavros/setpoint_velocity/cmd_vel -> /deepflyer/cmd_vel
   ```

## Reward Function Integration

The environment automatically integrates with registered reward functions:

```python
# Using preset reward
env = make_env("ros:deepflyer", reward_function="reach_target")

# Using multi-objective reward
env = make_env(
    "ros:deepflyer", 
    reward_function="multi_objective",
    reward_weights={
        'reach': 1.0,
        'collision': 2.0,
        'energy': 0.5,
        'speed': 0.3
    }
)
```

## Domain Randomization

Support for sim-to-real transfer:

```python
env = RosEnv(
    sensor_noise_config={
        'position_noise': 0.02,      # meters
        'velocity_noise': 0.05,      # m/s
        'imu_noise': 0.02,          # m/s²
        'camera_noise': 5.0,        # intensity
    },
    external_force_range=(0.0, 0.1),  # Newton
)
```

## API Endpoints Integration

For Jay's backend integration:

1. **Environment Creation**: Use the factory with configurations from API
2. **Training Loop**: Integrate with `scripts/train.py`
3. **Metrics**: Environment provides episode metrics in info dict

## Testing Without Gazebo

For development without Gazebo running:

```python
# Falls back to CartPole if ROS not available
env = make_env("ros:deepflyer")

# Or use CartPole with drone rewards for testing
env = make_cartpole_with_reward("reach_target")
```

## Common Issues and Solutions

### Issue: ROS2 not found
**Solution**: Install ROS2 Humble and source setup script:
```bash
source /opt/ros/humble/setup.bash
```

### Issue: Topics not publishing
**Solution**: Check namespace and ensure Gazebo plugins are loaded

### Issue: Slow simulation
**Solution**: Reduce camera resolution or publishing rates

## Example Integration

```python
from rl_agent.env import make_drone_env
from rl_agent.models import PPOAgent

# Create environment
env = make_drone_env(
    namespace="deepflyer",
    reward_function="reach_target",
    goal_position=[8.0, 8.0, 1.5]
)

# Create agent
agent = PPOAgent(env)

# Train
agent.train(total_timesteps=100000)
```

## Next Steps for Integration

1. **Uma**: Ensure Gazebo publishes all required topics with correct message types
2. **Jay**: Use environment factory in API endpoints with user configurations
3. **All**: Test integration with simple hover task first

For questions or issues, refer to the test files in `tests/test_env.py` for examples. 