# DeepFlyer: Educational Drone Reinforcement Learning Platform

## Table of Contents
- [Overview](#overview)
- [DeepRacer vs DeepFlyer Analogy](#deepracer-vs-deepflyer-analogy)
- [Lab Setup and Constraints](#lab-setup-and-constraints)
- [Flight Path Design](#flight-path-design)
- [Action Space Design](#action-space-design)
- [Reward Function](#reward-function)
- [Technical Implementation](#technical-implementation)
- [Student Experience](#student-experience)
- [Physical Setup Guide](#physical-setup-guide)
- [Troubleshooting](#troubleshooting)

## Overview

DeepFlyer is an educational drone platform that teaches reinforcement learning concepts through autonomous drone navigation. Inspired by AWS DeepRacer, students learn RL by training drones to navigate obstacle courses without needing to write code.

### Key Features
- **No Coding Required**: Students modify reward functions through simple parameter adjustments
- **Sim-to-Real**: Train in simulation, deploy on real hardware
- **Safe Learning Environment**: Constrained lab space with safety boundaries
- **Visual Learning**: Watch drones learn to navigate in real-time
- **Educational Focus**: Learn RL concepts through hands-on experimentation

## DeepRacer vs DeepFlyer Analogy

| Aspect | DeepRacer (Car) | DeepFlyer (Drone) |
|--------|----------------|-------------------|
| **Environment** | 2D track on ground | 3D course in lab space |
| **Vehicle** | RC car | Quadcopter drone |
| **Constraints** | Follows track lines | Fixed altitude flight |
| **Speed** | Variable throttle | Fixed forward speed |
| **Control** | Steering angle | Lateral adjustments |
| **Sensors** | Front camera | Depth camera |
| **Obstacles** | Track boundaries | Physical objects |
| **Goal** | Fast lap times | Safe navigation |
| **Learning** | Lane following | Obstacle avoidance |

### Why This Analogy Works

1. **Familiar Concept**: Students understand "stay on track, avoid obstacles"
2. **Progressive Learning**: Start simple, add complexity
3. **Visual Feedback**: See immediate results of reward changes
4. **Real-World Application**: Concepts transfer to autonomous systems
5. **Engaging**: Physical interaction keeps students motivated

## Lab Setup and Constraints

### Physical Space Requirements
```
Lab Dimensions: 2.5m (L) × 2.0m (W) × 1.5m (H)
```

### Space Allocation
```
┌─────────────────────────────────┐
│ Safety Zone (0.2m buffer)       │
│ ┌─────────────────────────────┐ │
│ │ Flight Zone: 2.1m × 1.6m   │ │ 
│ │ Height: 0.5m - 1.2m        │ │
│ │                             │ │
│ │ Obstacle Area: 1.5m × 1.2m │ │
│ │                             │ │
│ └─────────────────────────────┘ │
│ Equipment/Observer Zone         │
└─────────────────────────────────┐
```

### Safety Parameters
- **Minimum Height**: 0.5m (below this triggers emergency landing)
- **Maximum Height**: 1.2m (ceiling boundary with 0.3m safety margin)
- **Wall Clearance**: 0.2m minimum distance from walls
- **Emergency Stop**: Hardware kill switch accessible to instructor
- **Propeller Guards**: Required on all drones
- **Maximum Speed**: 1.0 m/s (adjustable by instructor only)

## Flight Path Design

### Course Layout (Top View)
```
Y-axis (Width: 2.0m)
2.0 ┌─────┬─────┬─────┬─────┬─────┐
    │     │     │     │     │  F  │  F = Finish
1.8 ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │ W2  │     │  W2 = Waypoint 2
1.6 ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │     │     │
1.4 ├─────┼─────┼─────┼─────┼─────┤
    │     │ 📦  │     │ 🪑  │     │  📦🪑 = Obstacles
1.2 ├─────┼─────┼─────┼─────┼─────┤
    │     │     │ W1  │     │     │  W1 = Waypoint 1
1.0 ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │     │     │
0.8 ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │     │     │
0.6 ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │     │     │
0.4 ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │     │     │
0.2 ├─────┼─────┼─────┼─────┼─────┤
    │  S  │     │     │     │     │  S = Start
0.0 └─────┴─────┴─────┴─────┴─────┘
   0.0  0.5  1.0  1.5  2.0  2.5
                X-axis (Length: 2.5m)
```

### 3D Flight Path (Side View)
```
Height (Z-axis)
1.5m ┌─────────────────────────────┐ ← Ceiling
1.4m │ ⚠️  Safety Margin (0.1m)   │
1.3m ├─────────────────────────────┤
1.2m │ 🔴 Maximum Flight Height    │
1.1m │                             │
1.0m │     📦       🪑             │ ← Obstacle Height
0.9m │                             │
0.8m │ S─────●──────●─────●────F   │ ← Fixed Flight Altitude
0.7m │                             │
0.6m │                             │
0.5m │ 🔴 Minimum Flight Height    │
0.4m ├─────────────────────────────┤
0.3m │ ⚠️  Safety Margin (0.1m)   │
0.2m │                             │
0.1m │                             │
0.0m └─────────────────────────────┘ ← Floor
     0    0.5   1.0   1.5   2.0  2.5m
```

### Trajectory Coordinates
```python
# Primary Flight Path (Fixed Altitude: 0.8m)
FLIGHT_PATH = {
    "start": (0.2, 0.2, 0.8),      # Start position
    "waypoint_1": (1.0, 1.0, 0.8), # Navigate around first obstacle
    "waypoint_2": (2.0, 1.6, 0.8), # Navigate around second obstacle  
    "finish": (2.3, 1.8, 0.8),     # Finish position
}

# Obstacle Positions
OBSTACLES = {
    "box": (0.8, 1.2, 0.0, 0.4, 0.4, 1.0),    # x, y, z, width, depth, height
    "chair": (1.8, 1.3, 0.0, 0.3, 0.3, 1.1),  # x, y, z, width, depth, height
}

# Safe Corridor (Virtual Track)
TRACK_WIDTH = 0.4  # meters (corridor width around ideal path)
```

### Course Variations
Students can practice with different course configurations:

**Beginner Course**: Single obstacle, wide corridor
```python
BEGINNER_PATH = {
    "start": (0.2, 1.0, 0.8),
    "waypoint_1": (1.2, 1.0, 0.8),
    "finish": (2.3, 1.0, 0.8),
}
OBSTACLES = {"box": (1.2, 1.3, 0.0, 0.3, 0.3, 0.8)}
```

**Advanced Course**: Multiple obstacles, narrow corridor
```python
ADVANCED_PATH = {
    "start": (0.2, 0.3, 0.8),
    "waypoint_1": (0.8, 1.0, 0.8),
    "waypoint_2": (1.5, 0.5, 0.8),
    "waypoint_3": (2.0, 1.5, 0.8),
    "finish": (2.3, 1.0, 0.8),
}
OBSTACLES = {
    "box1": (0.6, 0.6, 0.0, 0.2, 0.2, 0.9),
    "box2": (1.2, 1.2, 0.0, 0.2, 0.2, 0.9),
    "chair": (1.7, 0.8, 0.0, 0.3, 0.3, 1.1),
}
```

## Action Space Design

### Simplified Control Interface
Unlike traditional drone control with 6DOF, DeepFlyer uses a simplified 2D action space:

```python
action_space = gym.spaces.Box(
    low=np.array([-1.0, -1.0]), 
    high=np.array([1.0, 1.0]), 
    dtype=np.float32
)
```

### Action Interpretation
```python
def process_action(action):
    """
    Convert RL action to drone velocity commands.
    
    Args:
        action[0]: Lateral correction (-1=left, +1=right)
        action[1]: Speed adjustment (-1=slow, +1=fast)
    
    Returns:
        velocity_command: [vx, vy, vz, yaw_rate]
    """
    # Fixed parameters
    BASE_FORWARD_SPEED = 0.5  # m/s
    MAX_LATERAL_SPEED = 0.3   # m/s
    FIXED_ALTITUDE = 0.8      # m
    
    # Calculate velocity components
    lateral_velocity = action[0] * MAX_LATERAL_SPEED
    forward_velocity = BASE_FORWARD_SPEED * (1.0 + 0.5 * action[1])
    
    # Maintain fixed altitude (PID handles this)
    vertical_velocity = 0.0
    yaw_rate = 0.0  # Keep facing forward
    
    return [forward_velocity, lateral_velocity, vertical_velocity, yaw_rate]
```

### What's Fixed vs What RL Controls

**Fixed by System (PID/Flight Controller)**:
- Altitude hold (0.8m)
- Attitude stabilization (roll, pitch, yaw)
- Basic motor control
- Safety boundaries

**Controlled by RL Agent**:
- Lateral movement (left/right dodging)
- Forward speed adjustment
- Path optimization decisions
- Obstacle avoidance timing

## Reward Function

### Student-Facing Interface
```python
def reward_function(params):
    """
    DeepFlyer Reward Function - Train your drone to navigate safely!
    """
    # Read sensor data
    distance_from_path = params['distance_from_path']
    path_width = params['path_width']
    on_path = params['on_path']
    heading_error = params['heading_error']
    altitude_error = params['altitude_error']
    obstacle_distance = params['obstacle_distance']
    
    # Reward values (students can adjust these)
    HIGH_PATH_REWARD = 10.0      # Staying very close to ideal path
    MEDIUM_PATH_REWARD = 5.0     # Staying reasonably close
    LOW_PATH_REWARD = 1.0        # Barely staying on path
    OFF_PATH_PENALTY = 0.001     # Going off the safe corridor
    
    ALTITUDE_REWARD = 3.0        # Correct altitude
    HEADING_REWARD = 2.0         # Facing the right direction
    
    OBSTACLE_NEAR_PENALTY = -5.0 # Getting too close to obstacles
    COLLISION_PENALTY = -50.0    # Hitting something
    COMPLETION_BONUS = 20.0      # Reaching the finish
    
    # Calculate path following reward
    if distance_from_path <= 0.1 * path_width and on_path:
        path_reward = HIGH_PATH_REWARD
    elif distance_from_path <= 0.25 * path_width and on_path:
        path_reward = MEDIUM_PATH_REWARD
    elif distance_from_path <= 0.5 * path_width and on_path:
        path_reward = LOW_PATH_REWARD
    else:
        path_reward = OFF_PATH_PENALTY
    
    # Altitude bonus (within 10cm of target)
    altitude_reward = ALTITUDE_REWARD if abs(altitude_error) < 0.1 else 0.0
    
    # Heading bonus (pointing forward)
    heading_reward = HEADING_REWARD if abs(heading_error) < 0.2 else 0.0
    
    # Obstacle avoidance
    obstacle_reward = 0.0
    if obstacle_distance < 0.3:  # Too close!
        obstacle_reward = OBSTACLE_NEAR_PENALTY
    elif obstacle_distance < 0.15:  # Collision!
        obstacle_reward = COLLISION_PENALTY
    
    # Combine all rewards
    total_reward = path_reward + altitude_reward + heading_reward + obstacle_reward
    
    return float(total_reward)
```

### Reward Engineering Learning
Students learn RL concepts by adjusting reward parameters:

1. **Path Following**: Increase `HIGH_PATH_REWARD` → drone stays closer to center
2. **Risk Taking**: Decrease `OBSTACLE_NEAR_PENALTY` → drone flies closer to obstacles
3. **Speed vs Safety**: Balance completion bonus vs collision penalty
4. **Smooth Flying**: Add penalties for jerky movements

## Technical Implementation

### Hardware Requirements
- **Drone**: Pixhawk 6C flight controller
- **Sensors**: Intel RealSense D435i depth camera
- **Positioning**: OptiTrack motion capture system (optional)
- **Compute**: NVIDIA Jetson Nano or laptop with GPU
- **Safety**: Emergency stop controller

### Software Stack
```
┌─────────────────┐
│   Student UI    │ ← Web interface for reward editing
├─────────────────┤
│  RL Training    │ ← PPO/SAC algorithms
├─────────────────┤
│   DeepFlyer     │ ← Gym environment wrapper
├─────────────────┤
│   ROS2 Bridge   │ ← Communication layer
├─────────────────┤
│     MAVROS      │ ← Flight controller interface
├─────────────────┤
│   Pixhawk 6C    │ ← Hardware flight control
└─────────────────┘
```

### Simulation Environment
- **Physics**: Gazebo with wind simulation
- **Sensors**: Simulated depth camera with noise
- **Obstacles**: Dynamic object spawning
- **Visualization**: 3D trajectory plotting

### State Space (What RL Agent Sees)
```python
observation = {
    'position': [x, y, z],                    # Current position
    'velocity': [vx, vy, vz],                 # Current velocity
    'distance_from_path': float,              # Cross-track error
    'heading_error': float,                   # Orientation error
    'altitude_error': float,                  # Height error
    'depth_image': [64, 64],                  # Obstacle detection
    'obstacle_distance': float,               # Nearest obstacle
    'waypoint_relative': [dx, dy, dz],        # Vector to next waypoint
    'path_progress': float,                   # 0.0 to 1.0 completion
}
```

## Student Experience

### Learning Progression

**Week 1: Introduction**
- Watch demonstration flights
- Understand reward function basics
- Modify simple parameters (path rewards)

**Week 2: Obstacle Avoidance**
- Add obstacles to course
- Adjust obstacle penalties
- Learn risk vs reward balance

**Week 3: Advanced Tuning**
- Multi-objective optimization
- Speed vs safety tradeoffs  
- Course design challenges

**Week 4: Competition**
- Design custom courses
- Optimize for different metrics
- Present findings

### Assessment Metrics
- **Navigation Success**: Completion rate
- **Efficiency**: Time to complete course
- **Safety**: Collision avoidance
- **Understanding**: Reward function explanations

## Physical Setup Guide

### Lab Preparation

**1. Safety Setup**
```
┌─────────────────┐
│ Emergency Stop  │ ← Instructor controlled
├─────────────────┤  
│ Safety Netting  │ ← Around flight area
├─────────────────┤
│ Foam Padding    │ ← On walls/obstacles
├─────────────────┤
│ Fire Extinguisher│ ← LiPo battery safety
└─────────────────┘
```

**2. Equipment Placement**
- Ground station: Outside flight area
- Motion capture cameras: Ceiling mounted
- Emergency stop: Instructor accessible
- Obstacle storage: Quick reconfiguration

**3. Course Setup**
```python
def setup_course(difficulty="beginner"):
    """
    Physical course setup instructions.
    """
    if difficulty == "beginner":
        obstacles = [
            {"type": "foam_block", "position": (1.2, 1.0), "size": "30x30x80cm"}
        ]
    elif difficulty == "intermediate":
        obstacles = [
            {"type": "foam_block", "position": (0.8, 1.2), "size": "20x20x60cm"},
            {"type": "hanging_ball", "position": (1.8, 0.8), "height": "1.0m"}
        ]
    elif difficulty == "advanced":
        obstacles = [
            {"type": "moving_obstacle", "path": "pendulum", "speed": "slow"},
            {"type": "narrow_gate", "position": (1.5, 1.0), "width": "50cm"}
        ]
    
    return obstacles
```

### Calibration Procedure

**1. Flight Space Calibration**
```bash
# Define safe flight boundaries
rosrun deepflyer calibrate_space.py --corners 4 --height_min 0.5 --height_max 1.2
```

**2. Sensor Calibration**
```bash
# Calibrate depth camera
rosrun deepflyer calibrate_depth.py --target_distance 1.0
```

**3. Motion Capture Setup** (if available)
```bash
# Register drone markers
rosrun deepflyer setup_mocap.py --drone_id 1 --marker_config rigid_body.yaml
```

## Troubleshooting

### Common Issues

**Drone Won't Take Off**
- Check battery level (>50%)
- Verify MAVROS connection
- Ensure OFFBOARD mode enabled
- Check safety boundaries

**Erratic Flight Behavior**
- Recalibrate sensors
- Check for interference (WiFi, other drones)
- Verify PID parameters
- Check propeller condition

**RL Training Not Converging**
- Reduce action space bounds
- Simplify reward function
- Increase training episodes
- Check observation normalization

**Depth Camera Issues**
- Clean camera lens
- Check USB connection
- Verify lighting conditions
- Restart camera node

### Safety Protocols

**Emergency Procedures**
1. **Immediate Stop**: Hit emergency stop button
2. **Soft Landing**: Switch to LAND mode
3. **Battery Fire**: Use sand/Class D extinguisher
4. **Injury**: First aid kit, emergency contacts

**Daily Checklist**
- [ ] Battery voltage >11.1V
- [ ] Propellers secure and undamaged
- [ ] Emergency stop functional
- [ ] Flight area clear
- [ ] Backup systems ready

### Performance Optimization

**Training Speed**
- Use GPU acceleration
- Parallel environment instances
- Efficient reward computation
- Memory optimization

**Flight Performance**
- PID tuning for lab conditions
- Sensor fusion optimization
- Latency minimization
- Robust state estimation

## Conclusion

DeepFlyer provides an engaging, hands-on approach to learning reinforcement learning through autonomous drone navigation. By abstracting complex flight control while preserving the essential RL learning objectives, students gain intuitive understanding of AI decision-making in physical systems.

The platform's design emphasizes safety, educational value, and real-world applicability, making it an ideal tool for introducing students to the exciting field of autonomous robotics and artificial intelligence.

---

**For technical support or course development assistance, contact the DeepFlyer development team.** 