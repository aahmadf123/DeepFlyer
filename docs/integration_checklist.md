# DeepFlyer Integration Checklist

## For Uma (Simulation/CAD)

### Required Gazebo Components

1. **Drone Model (URDF/SDF)**
   - [ ] Quadcopter with PX4 flight controller plugin
   - [ ] Front-facing camera sensor (640x480 @ 15 Hz)
   - [ ] Downward camera sensor (640x480 @ 10 Hz)
   - [ ] IMU sensor plugin
   - [ ] Collision detection plugin

2. **World Files (3 Maps)**
   - [ ] Map 1: Simple corridor (10x10x3m)
   - [ ] Map 2: Multi-path with dynamic gates
   - [ ] Map 3: Multi-level with platforms

3. **ROS2 Publishers**
   ```yaml
   publishers:
     - /deepflyer/pose            # PoseStamped @ 50Hz
     - /deepflyer/odom            # Odometry @ 100Hz
     - /deepflyer/imu             # Imu @ 100Hz
     - /deepflyer/camera/front/image_raw  # Image @ 15Hz
     - /deepflyer/camera/down/image_raw   # Image @ 10Hz
     - /deepflyer/collision       # Bool (on collision)
     - /deepflyer/obstacle_distance  # Float32 @ 20Hz
     - /deepflyer/battery_level   # Float32 @ 1Hz
   ```

4. **ROS2 Subscribers**
   ```yaml
   subscribers:
     - /deepflyer/cmd_vel  # Twist (velocity commands)
     - /deepflyer/reset    # Bool (reset simulation)
   ```

5. **Launch File Example**
   ```bash
   ros2 launch deepflyer_gazebo world1.launch.py
   ```

## For Jay (UI/Backend)

### API Integration Points

1. **Environment Configuration Endpoint**
   ```python
   @app.post("/api/environment/create")
   async def create_environment(config: EnvironmentConfig):
       env = make_drone_env(
           namespace=config.namespace,
           reward_function=config.reward_function,
           goal_position=config.goal_position,
           **config.dict()
       )
       return {"env_id": env_id}
   ```

2. **Training Configuration**
   ```python
   # Use existing TrainingConfig schema
   config = TrainingConfig(
       algorithm="PPO",
       environment="ros:deepflyer",  # ROS environment
       reward_function="reach_target",
       hyperparameters={...}
   )
   ```

3. **Available Reward Functions**
   - `reach_target` - Navigate to goal
   - `collision_avoidance` - Avoid obstacles
   - `save_energy` - Minimize throttle
   - `fly_steady` - Maintain altitude
   - `fly_smoothly` - Smooth motion
   - `be_fast` - Quick completion
   - `multi_objective` - Weighted combination

4. **Metrics from Environment**
   ```python
   info = {
       'episode_step': int,
       'episode_reward': float,
       'distance_to_goal': float,
       'collision': bool,
       'position': [x, y, z],
       'velocity': [vx, vy, vz],
       'metrics': {
           'total_distance': float,
           'collision_count': int,
           'energy_used': float,
           'max_velocity': float,
           'min_obstacle_distance': float
       }
   }
   ```

## Testing Steps

1. **Without ROS/Gazebo** (Ahmad can test now)
   ```bash
   pytest tests/test_env.py
   pytest tests/test_rewards.py
   ```

2. **With Gazebo Running** (After Uma's setup)
   ```bash
   # Terminal 1: Launch Gazebo
   ros2 launch deepflyer_gazebo world1.launch.py
   
   # Terminal 2: Test environment
   python scripts/test_ros_env.py
   ```

3. **Full Integration** (All together)
   ```bash
   # Terminal 1: Gazebo
   # Terminal 2: Backend API
   # Terminal 3: Frontend
   # Terminal 4: Run training
   python scripts/train.py --env ros:deepflyer
   ```

## Quick Start Commands

```bash
# Install ROS2 dependencies (Ubuntu)
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-mavros ros-humble-mavros-extras

# Install Python dependencies
pip install rclpy cv-bridge tf-transformations

# Source ROS2
source /opt/ros/humble/setup.bash

# Test basic functionality
python -c "from rl_agent.env import make_env; env = make_env('CartPole-v1')"
```

## Key Files to Review

- **Ahmad's Work**:
  - `rl_agent/env/ros_env.py` - ROS environment
  - `rl_agent/env/ros_utils.py` - Utilities
  - `docs/ros_integration_guide.md` - Full guide

- **Integration Points**:
  - `api/endpoints.py` - Add ROS env support
  - `scripts/train.py` - Already supports any env

## Contact Points

- **ROS Topics**: Uma defines, Ahmad subscribes
- **Reward Functions**: Ahmad provides, Jay exposes in UI
- **Training Loop**: Ahmad's code, Jay triggers via API 