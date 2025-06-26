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
- [ZED Mini Camera Integration](#zed-mini-camera-integration)

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

### Hoop Navigation Course with Progressive Difficulty

**Course Overview:**
Based on the teammate's design sketch, DeepFlyer uses a **hoop-flying course** with progressive difficulty across multiple laps. This design better matches real-world drone racing and provides clearer visual targets for the RL agent.

```
Course Layout: Fixed 4-5 Hoop Circuit (Multiple Laps)
Lab Space: 2.5m × 2.0m × 1.5m (existing lab constraints)

Top View Layout:
┌─────────────────────────────────────────┐
│  Safety Buffer Zone (0.2m)              │
│  ┌───────────────────────────────────┐  │
│  │ Flight Zone: 2.1m × 1.6m         │  │
│  │                                   │  │
│  │  🔴 START/FINISH                   │  │
│  │   ↓           ↑                   │  │
│  │  ⭕(1) ──→ ⭕(2) ──→ ⭕(3)          │  │
│  │   ↑           ↓     ↓             │  │
│  │  ⭕(5) ←──── ⭕(4) ←─┘              │  │
│  │   ↑         ↓                     │  │
│  │   └─────────┘                     │  │
│  │  (Circle back for next lap)       │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘

Flight Pattern (Same course, multiple laps):
Lap 1: START → ⭕(1) → ⭕(2) → ⭕(3) → ⭕(4) → ⭕(5) → back to ⭕(1)
Lap 2: ⭕(1) → ⭕(2) → ⭕(3) → ⭕(4) → ⭕(5) → back to ⭕(1)  
Lap 3: ⭕(1) → ⭕(2) → ⭕(3) → ⭕(4) → ⭕(5) → FINISH

Course Progression:
• Total Hoops: 5 fixed hoops (same layout throughout)
• Total Laps: 3 complete circuits  
• Flight Time: 2-3 minutes per episode
• Difficulty: Speed and precision improvement over repeated laps
```

### Hoop Specifications and Setup

**Physical Hoop Requirements:**
```python
class HoopCourse:
    """
    Fixed 5-hoop circuit for multi-lap navigation
    """
    def __init__(self, lab_dimensions):
        self.lab_bounds = lab_dimensions
        self.flight_altitude = 0.8  # meters
        self.num_hoops = 5
        
        # Fixed hoop specifications (same for all hoops)
        self.hoop_diameter = 0.8  # meters - challenging but achievable
        
        # Hoop materials
        self.hoop_config = {
            'material': 'lightweight_foam_pool_noodles',
            'color': 'bright_orange',  # High contrast for vision
            'mounting': 'adjustable_height_stands',
            'safety': 'breakaway_connections'
        }
    
    def generate_hoop_positions(self, spawn_position):
        """Generate fixed 5-hoop circuit relative to drone spawn"""
        # Circuit layout: rectangular path with 5 hoops
        hoops = [
            {
                'id': 'hoop_1',
                'position': (spawn_position[0] + 0.5, spawn_position[1] - 0.5, 0.8),
                'diameter': self.hoop_diameter,
                'sequence': 1
            },
            {
                'id': 'hoop_2', 
                'position': (spawn_position[0] + 1.0, spawn_position[1] - 0.5, 0.8),
                'diameter': self.hoop_diameter,
                'sequence': 2
            },
            {
                'id': 'hoop_3',
                'position': (spawn_position[0] + 1.5, spawn_position[1] + 0.0, 0.8),
                'diameter': self.hoop_diameter,
                'sequence': 3
            },
            {
                'id': 'hoop_4',
                'position': (spawn_position[0] + 1.0, spawn_position[1] + 0.5, 0.8),
                'diameter': self.hoop_diameter,
                'sequence': 4
            },
            {
                'id': 'hoop_5',
                'position': (spawn_position[0] + 0.5, spawn_position[1] + 0.0, 0.8),
                'diameter': self.hoop_diameter,
                'sequence': 5
            }
        ]
        
        return hoops
    
    def get_next_target_hoop(self, current_hoop_index):
        """Get the next hoop in sequence (loops back to hoop 1 after hoop 5)"""
        return (current_hoop_index + 1) % self.num_hoops
        
    def is_lap_complete(self, hoop_sequence):
        """Check if drone has completed a full lap (passed through all 5 hoops)"""
        return len(set(hoop_sequence[-5:])) == 5 if len(hoop_sequence) >= 5 else False
```

### Environment Detection and Setup
```python
def setup_hoop_course():
    """
    Automatically detect environment and set up hoop course.
    Works in Gazebo simulation or real-world deployments.
    """
    # Get drone spawn position from ROS/Gazebo
    spawn_pos = get_drone_spawn_position()  # From /gazebo/model_states or similar
    
    # Detect safe flight boundaries
    flight_bounds = detect_flight_area()    # From depth sensors or pre-configured
    
    # Create hoop course
    course = HoopCourse(flight_bounds)
    hoops = course.generate_hoop_positions(spawn_pos)
    
    return hoops, course

def get_drone_spawn_position():
    """Get initial drone position from Gazebo or real sensors"""
    # In Gazebo: Read from /gazebo/model_states
    # In real world: Read from /mavros/local_position/pose
    return (x, y, z)  # Actual spawn coordinates

def detect_flight_area():
    """Detect available flight space"""
    # In Gazebo: Parse world file or use ray casting
    # In real world: Use depth sensors or manual configuration
    return {
        'length': 2.1,  # Available flight length
        'width': 1.6,   # Available flight width
        'height': 1.2   # Maximum flight height
    }
```

### Course Design Philosophy
DeepFlyer uses a single, fixed 5-hoop circuit that the drone navigates repeatedly:

**Course Parameters:**
- **Path Type**: Fixed 5-hoop circuit with multi-lap navigation
- **Total Hoops**: 5 fixed hoops (same positions throughout)
- **Flight Height**: Consistent 0.8m above spawn level
- **Hoop Size**: Uniform 0.8m diameter for all hoops
- **Lap Structure**: Drone circles through same hoops multiple times

**Why This Course Design:**
1. **Consistency**: Same course layout for every lap - easy to compare performance
2. **Simplicity**: Only need 5 physical hoops in lab setup
3. **Progressive Improvement**: Students see speed and precision gains lap-over-lap
4. **Real-world Relevance**: Similar to actual drone racing circuits
5. **Educational Focus**: Emphasis on RL learning, not course complexity
6. **Practical Setup**: Fits within standard lab space constraints

### Course Layout Concept (Relative Design)
```
Course Layout: Forward Path with Obstacles
(Coordinates relative to spawn point)

     START ──→ OBSTACLE 1 ──→ OBSTACLE 2 ──→ FINISH
       │            │             │           │
   (spawn + 0m)  (spawn + 3m)  (spawn + 6m) (spawn + 9m)
   
Side View:
     📦 = Static obstacle (box/cylinder)
     🚁 = Drone path (fixed altitude)
     
     ┌─────────────────────────────────────┐
  1m │             📦        📦            │
     │                                     │
0.8m │ 🚁 ────────────────────────────→ 🚁 │ ← Fixed flight altitude
     │                                     │
  0m └─────────────────────────────────────┘
     0m        3m        6m        9m
```

### Adaptive Course Configuration
The course adapts to any environment (Gazebo worlds, real labs, different spaces) by using relative positioning:

```python
# Environment-Agnostic Course Design
class AdaptiveCourse:
    def __init__(self, spawn_position, environment_bounds):
        self.spawn_pos = spawn_position  # Where drone starts (from Gazebo/ROS)
        self.bounds = environment_bounds  # Safe flight area detected
        
    def generate_course(self):
        """Generate course relative to spawn position"""
        # Course parameters (relative distances)
        FORWARD_DISTANCE = 9.0      # Total course length
        OBSTACLE_SPACING = 3.0      # Distance between obstacles
        FLIGHT_HEIGHT = 0.8         # Height above spawn level
        
        # Generate waypoints relative to spawn
        waypoints = []
        for i in range(4):  # Start + 2 obstacles + finish
            distance = i * OBSTACLE_SPACING
            waypoint = {
                'x': self.spawn_pos[0] + distance,
                'y': self.spawn_pos[1],  # Straight line
                'z': self.spawn_pos[2] + FLIGHT_HEIGHT
            }
            waypoints.append(waypoint)
        
        return waypoints
    
    def place_obstacles(self):
        """Place obstacles relative to path"""
        obstacles = [
            {
                'position': (self.spawn_pos[0] + 3.0, self.spawn_pos[1] + 1.0, 0),
                'type': 'cylinder',
                'radius': 0.3,
                'height': 1.0
            },
            {
                'position': (self.spawn_pos[0] + 6.0, self.spawn_pos[1] - 1.0, 0),
                'type': 'box',
                'dimensions': [0.4, 0.4, 1.0]
            }
        ]
        return obstacles
```

### Environment Detection and Setup
```python
def setup_environment():
    """
    Automatically detect environment and set up course.
    Works in Gazebo simulation or real-world deployments.
    """
    # Get drone spawn position from ROS/Gazebo
    spawn_pos = get_drone_spawn_position()  # From /gazebo/model_states or similar
    
    # Detect safe flight boundaries
    flight_bounds = detect_flight_area()    # From depth sensors or pre-configured
    
    # Create adaptive course
    course = AdaptiveCourse(spawn_pos, flight_bounds)
    waypoints = course.generate_course()
    obstacles = course.place_obstacles()
    
    return waypoints, obstacles

def get_drone_spawn_position():
    """Get initial drone position from Gazebo or real sensors"""
    # In Gazebo: Read from /gazebo/model_states
    # In real world: Read from /mavros/local_position/pose
    return (x, y, z)  # Actual spawn coordinates

def detect_flight_area():
    """Detect available flight space"""
    # In Gazebo: Parse world file or use ray casting
    # In real world: Use depth sensors or manual configuration
    return {
        'x_min': spawn_x - 1.0,
        'x_max': spawn_x + 10.0,
        'y_min': spawn_y - 2.0, 
        'y_max': spawn_y + 2.0,
        'z_min': 0.5,
        'z_max': 2.0
    }
```

### Standard Course Design (One Configuration)
DeepFlyer uses a single, well-tested course configuration that adapts to any environment:

**Course Parameters:**
- **Path Type**: Straight line with obstacle avoidance
- **Total Distance**: 9 meters forward from spawn
- **Obstacles**: 2 static objects requiring lateral maneuvering
- **Flight Height**: 0.8m above spawn level
- **Corridor Width**: 1.0m (allowing for obstacle avoidance)

**Why One Course Design:**
1. **Consistency**: All students learn on the same challenge
2. **Focus**: Concentrate on RL concepts, not course complexity
3. **Comparison**: Easy to compare different reward functions
4. **Safety**: Well-tested configuration with known safety margins
5. **Simplicity**: Easier to troubleshoot and maintain

## Action Space Design

### Enhanced 3D Control Interface
For hoop navigation, DeepFlyer uses a 3D action space allowing full spatial control:

```python
class HoopNavigationActionProcessor:
    """
    Enhanced action processor for hoop navigation with 3D movement capability
    """
    
    def __init__(self, config):
        # Action space: 3D movement control
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0]), 
    dtype=np.float32
)
        
        # Control parameters
        self.max_horizontal_speed = 0.8    # m/s
        self.max_vertical_speed = 0.4      # m/s
        self.base_forward_speed = 0.6      # m/s
```

### Action Interpretation
```python
def process_action(self, action, current_state, target_hoop):
    """
    Convert 3D RL action to drone velocity commands for hoop navigation
    
    Args:
        action[0]: Lateral movement (-1=left, +1=right)
        action[1]: Vertical movement (-1=down, +1=up)  
        action[2]: Speed adjustment (-1=slow, +1=fast)
    
    Returns:
        velocity_command: [vx, vy, vz] in drone body frame
    """
    # Extract action components
    lateral_command = np.clip(action[0], -1.0, 1.0)
    vertical_command = np.clip(action[1], -1.0, 1.0)  
    speed_command = np.clip(action[2], -1.0, 1.0)
    
    # Calculate velocity components
    lateral_velocity = lateral_command * self.max_horizontal_speed
    vertical_velocity = vertical_command * self.max_vertical_speed
    forward_velocity = self.base_forward_speed * (1.0 + 0.5 * speed_command)
    
    # Apply dynamic speed limits based on proximity to hoop
    target_distance = np.linalg.norm(
        np.array(current_state['position']) - np.array(target_hoop['position']))
    
    if target_distance < 1.0:  # Close to hoop - reduce max speeds for precision
        lateral_velocity *= 0.7
        forward_velocity *= 0.8
    
    velocity_command = [forward_velocity, lateral_velocity, vertical_velocity]
    
    return velocity_command
```

### What's Fixed vs What RL Controls

**Fixed by System (PID/Flight Controller)**:
- Attitude stabilization (roll, pitch, yaw)
- Basic motor control
- Safety boundaries
- Emergency protocols

**Controlled by RL Agent**:
- 3D spatial movement (X, Y, Z velocities)
- Speed adjustment for precision vs efficiency
- Hoop approach trajectories
- Navigation timing and coordination

## Reward Function

### Student-Facing Reward Function
```python
def hoop_navigation_reward_function(params):
    """
    DeepFlyer Hoop Navigation Reward Function
    
    Students can modify these parameters to change drone behavior:
    """
    
    # === STUDENT-CONFIGURABLE PARAMETERS ===
    
    # Hoop Navigation Rewards
    HOOP_APPROACH_REWARD = 10.0      # Getting closer to target hoop
    HOOP_PASSAGE_REWARD = 50.0       # Successfully passing through hoop
    HOOP_CENTER_BONUS = 20.0         # Passing through center of hoop
    VISUAL_ALIGNMENT_REWARD = 5.0    # Keeping hoop centered in camera view
    
    # Speed and Efficiency
    FORWARD_PROGRESS_REWARD = 3.0    # Making progress toward goal
    SPEED_EFFICIENCY_BONUS = 2.0     # Maintaining good speed
    
    # Lap Completion Bonuses
    LAP_COMPLETION_BONUS = 100.0     # Completing a full lap
    COURSE_COMPLETION_BONUS = 500.0  # Completing all 3 laps
    
    # Precision and Style
    SMOOTH_FLIGHT_BONUS = 1.0        # Smooth, non-jerky movements
    PRECISION_BONUS = 15.0           # Passing through smaller hoops
    
    # === PENALTIES (Students can adjust severity) ===
    
    # Navigation Penalties
    WRONG_DIRECTION_PENALTY = -2.0   # Flying away from target
    HOOP_MISS_PENALTY = -25.0        # Flying around instead of through hoop
    COLLISION_PENALTY = -100.0       # Hitting hoop or obstacle
    
    # Efficiency Penalties
    SLOW_PROGRESS_PENALTY = -1.0     # Taking too long
    ERRATIC_FLIGHT_PENALTY = -3.0    # Jerky, unstable flight
    
    # === SAFETY PENALTIES (NOT student-configurable) ===
    BOUNDARY_VIOLATION = -200.0      # Flying outside safe area
    EMERGENCY_LANDING = -500.0       # Emergency stop triggered
    
    # === REWARD CALCULATION ===
    
    total_reward = 0.0
    
    # 1. Hoop Navigation Component
    distance_to_hoop = params['distance_to_target_hoop']
    previous_distance = params.get('previous_distance_to_hoop', distance_to_hoop)
    
    # Reward for approaching hoop
    if distance_to_hoop < previous_distance:
        approach_reward = HOOP_APPROACH_REWARD * (previous_distance - distance_to_hoop)
        total_reward += approach_reward
    
    # Bonus for hoop passage
    if params.get('hoop_passed', False):
        total_reward += HOOP_PASSAGE_REWARD
        
        # Extra bonus for center passage
        passage_accuracy = params.get('passage_accuracy', 0.5)  # 0-1, 1=perfect center
        if passage_accuracy > 0.8:
            total_reward += HOOP_CENTER_BONUS
    
    # 2. Visual Navigation Component (ZED Mini integration)
    hoop_alignment = params.get('hoop_alignment', 0.0)  # -1 to 1, 0=centered
    if abs(hoop_alignment) < 0.2:  # Well centered
        total_reward += VISUAL_ALIGNMENT_REWARD
    
    hoop_visible = params.get('hoop_visible', False)
    if not hoop_visible and distance_to_hoop < 2.0:  # Should see hoop when close
        total_reward += WRONG_DIRECTION_PENALTY
    
    # 3. Flight Efficiency Component
    forward_velocity = params.get('forward_velocity', 0.0)
    if forward_velocity > 0.3:  # Good forward progress
        total_reward += FORWARD_PROGRESS_REWARD
    
    speed = params.get('speed', 0.0)
    if 0.4 < speed < 0.8:  # Optimal speed range
        total_reward += SPEED_EFFICIENCY_BONUS
    
    # 4. Precision Component
    hoop_diameter = params.get('current_hoop_diameter', 1.0)
    precision_factor = max(0.0, (1.1 - hoop_diameter) / 0.6)  # Higher reward for smaller hoops
    if params.get('hoop_passed', False):
        total_reward += PRECISION_BONUS * precision_factor
    
    # 5. Lap and Course Progress
    if params.get('lap_completed', False):
        total_reward += LAP_COMPLETION_BONUS
        
    if params.get('course_completed', False):
        total_reward += COURSE_COMPLETION_BONUS
    
    # 6. Flight Quality Assessment
    acceleration_magnitude = params.get('acceleration_magnitude', 0.0)
    if acceleration_magnitude < 0.5:  # Smooth flight
        total_reward += SMOOTH_FLIGHT_BONUS
    else:  # Jerky flight
        total_reward += ERRATIC_FLIGHT_PENALTY
    
    # 7. Penalties
    if params.get('missed_hoop', False):
        total_reward += HOOP_MISS_PENALTY
        
    if params.get('collision', False):
        total_reward += COLLISION_PENALTY
    
    # Time-based penalty (encourage efficiency)
    episode_time = params.get('episode_time', 0.0)
    if episode_time > 180.0:  # More than 3 minutes
        total_reward += SLOW_PROGRESS_PENALTY
    
    # === SAFETY OVERRIDES (System-managed, not student-configurable) ===
    if params.get('out_of_bounds', False):
        total_reward += BOUNDARY_VIOLATION
        
    if params.get('emergency_stop', False):
        total_reward += EMERGENCY_LANDING
    
    return float(total_reward)

# Example parameter explanation for students
REWARD_PARAMETER_GUIDE = {
    'HOOP_PASSAGE_REWARD': 'Increase this to make the drone prioritize getting through hoops over other behaviors',
    'VISUAL_ALIGNMENT_REWARD': 'Increase this to make the drone better at centering hoops in camera view',
    'PRECISION_BONUS': 'Increase this to reward more precise flight through smaller hoops',
    'SPEED_EFFICIENCY_BONUS': 'Adjust this to balance speed vs accuracy - higher values encourage faster flight',
    'HOOP_MISS_PENALTY': 'Increase magnitude to strongly discourage missing hoops',
    'SMOOTH_FLIGHT_BONUS': 'Increase this to encourage smoother, more stable flight patterns'
}
```

### How Students Learn
Students experiment with different reward values to see how drone behavior changes:

1. **Increase `HOOP_PASSAGE_REWARD`** → Drone prioritizes getting through hoops
2. **Increase `SPEED_EFFICIENCY_BONUS`** → Drone flies faster but may be less precise
3. **Increase `HOOP_CENTER_BONUS`** → Drone aims for center of hoops
4. **Adjust penalties** → Change risk-taking behavior

Students can also completely rewrite the reward function with their own logic.

## Technical Implementation

### Hardware Requirements
- **Drone Frame**: Holybro S500 quadcopter frame
- **Flight Controller**: Pixhawk 6C flight controller
- **Compute Platform**: Raspberry Pi 4B (minimum 4GB RAM)
- **Vision System**: ZED Mini stereo camera
- **Positioning**: OptiTrack motion capture system (optional)
- **Safety**: Emergency stop controller

### Software Stack
```
┌─────────────────┐
│   Student UI    │ ← Web interface for reward editing
├─────────────────┤
│  RL Training    │ ← P3O algorithm
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
class HoopNavigationObservationProcessor:
    """
    Enhanced observation processor for hoop navigation with ZED Mini integration
    """
    
    def __init__(self, config):
        self.course_hoops = config.course_hoops
        self.current_target_hoop = 0
        self.lap_number = 1
        
        # Observation space: 12-dimensional vector
        self.observation_dim = 12
        
    def process_observation(self, px4_data, vision_data, course_state):
        """
        Create RL observation from drone telemetry and vision data
        
        Args:
            px4_data: Flight controller data (position, velocity, etc.)
            vision_data: ZED Mini processed features
            course_state: Current course progress information
            
        Returns:
            observation: 12-element normalized vector for RL agent
        """
        # Extract current position and velocity
        pos = np.array([px4_data.x, px4_data.y, px4_data.z])
        vel = np.array([px4_data.vx, px4_data.vy, px4_data.vz])
        
        # Get current target hoop information
        target_hoop = self.course_hoops[self.current_target_hoop]
        target_pos = np.array(target_hoop['position'])
        
        # Calculate navigation features
        distance_to_hoop = np.linalg.norm(pos - target_pos)
        direction_to_hoop = (target_pos - pos) / max(distance_to_hoop, 0.01)
        
        # Velocity alignment with target direction
        velocity_magnitude = np.linalg.norm(vel)
        velocity_alignment = np.dot(vel, direction_to_hoop) / max(velocity_magnitude, 0.01)
        
        # Vision-based features (from ZED Mini)
        hoop_alignment = vision_data.get('hoop_alignment', 0.0)
        hoop_distance_vision = vision_data.get('hoop_distance', float('inf'))
        hoop_visible = 1.0 if vision_data.get('hoop_detected', False) else 0.0
        hoop_size_ratio = vision_data.get('hoop_area_ratio', 0.0)
        
        # Course progress features  
        progress_in_lap = self.current_target_hoop % 3  # 3 hoops per lap
        lap_progress = (self.lap_number - 1) / 3.0      # 3 total laps
        
        # Construct normalized observation vector
        observation = np.array([
            # Position relative to target (3 dimensions)
            np.clip(direction_to_hoop[0], -1.0, 1.0),          # 0: X direction to hoop
            np.clip(direction_to_hoop[1], -1.0, 1.0),          # 1: Y direction to hoop  
            np.clip((target_pos[2] - pos[2]) / 2.0, -1.0, 1.0), # 2: Z direction to hoop
            
            # Velocity information (2 dimensions)
            np.clip(vel[0] / 2.0, -1.0, 1.0),                  # 3: Forward velocity
            np.clip(vel[1] / 2.0, -1.0, 1.0),                  # 4: Lateral velocity
            
            # Navigation metrics (2 dimensions)
            np.clip(distance_to_hoop / 5.0, 0.0, 1.0),         # 5: Distance to target
            np.clip(velocity_alignment, -1.0, 1.0),            # 6: Velocity alignment
            
            # Vision-based features (3 dimensions)
            np.clip(hoop_alignment, -1.0, 1.0),                # 7: Visual hoop alignment
            np.clip(hoop_distance_vision / 5.0, 0.0, 1.0),     # 8: Visual distance to hoop
            hoop_visible,                                       # 9: Hoop visibility (0 or 1)
            
            # Course progress (2 dimensions)
            progress_in_lap / 2.0,                              # 10: Progress within current lap
            lap_progress                                        # 11: Overall course progress
        ], dtype=np.float32)
        
        return observation
```

## Student Experience

### Simple Learning Approach

Students receive a working DeepFlyer system with a single reward function they can modify. Like AWS DeepRacer, the focus is entirely on understanding how reward changes affect drone behavior.

**What Students Do:**
- Modify reward function parameters (or rewrite entirely if desired)
- Train RL agent with their reward function
- Test drone behavior in simulation and real flight
- Iterate and improve based on results

**What Students Learn:**
- How reward design affects AI behavior
- Trial-and-error optimization
- Real-world AI system constraints
- Reinforcement learning concepts through experimentation

### Assessment Metrics
- **Navigation Success**: Hoop completion rate
- **Efficiency**: Time to complete course
- **Safety**: Collision avoidance
- **Understanding**: Reward function design decisions

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
def setup_physical_course(lab_dimensions):
    """
    Set up the standard DeepFlyer course in any lab space.
    
    Args:
        lab_dimensions: (length, width, height) of available space
    """
    # Ensure minimum space requirements
    min_length, min_width, min_height = 10.0, 4.0, 2.5
    if any(lab_dimensions[i] < [min_length, min_width, min_height][i] for i in range(3)):
        raise ValueError("Lab space too small for DeepFlyer course")
    
    # Calculate course layout relative to lab center
    lab_center = (lab_dimensions[0]/2, lab_dimensions[1]/2, 0)
    
    # Standard obstacle placement (relative to center)
    obstacles = [
        {
            "type": "foam_cylinder",
            "position": (lab_center[0] - 1.5, lab_center[1] + 0.8, 0),
            "dimensions": {"radius": 0.3, "height": 1.0},
            "safety_clearance": 0.5
        },
        {
            "type": "foam_box", 
            "position": (lab_center[0] + 1.5, lab_center[1] - 0.8, 0),
            "dimensions": {"width": 0.4, "depth": 0.4, "height": 1.0},
            "safety_clearance": 0.5
        }
    ]
    
    # Define flight corridor
    flight_path = {
        "start": (lab_center[0] - 4.0, lab_center[1], 0.8),
        "finish": (lab_center[0] + 4.0, lab_center[1], 0.8),
        "corridor_width": 1.0
    }
    
    return obstacles, flight_path
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



## ZED Mini Camera Integration

### Computer Vision System for Hoop Detection

**Why ZED Mini Camera:**
The teammate specified using a **ZED mini camera**, which provides several advantages for educational drone RL:

- **Stereo Vision**: Provides depth information for accurate hoop distance measurement
- **High Frame Rate**: 100 FPS capability for real-time RL feedback
- **Compact Size**: Fits on educational drones without affecting flight dynamics
- **SDK Integration**: Easy ROS2 integration for real-time processing

### Camera Setup and Configuration

```python
class ZEDMiniVisionSystem:
    """
    ZED Mini camera integration for hoop detection and navigation
    """
    
    def __init__(self, config):
        # ZED Mini configuration
        self.camera_config = {
            'resolution': 'HD720',     # 1280x720 for good performance
            'fps': 60,                 # High enough for RL, not too demanding
            'depth_mode': 'PERFORMANCE', # Balance speed vs quality
            'coordinate_system': 'RIGHT_HANDED_Z_UP'
        }
        
        # Vision processing parameters
        self.hoop_detection = {
            'color_range': {
                'lower_orange': [5, 100, 100],   # HSV for orange hoops
                'upper_orange': [25, 255, 255]
            },
            'min_contour_area': 500,
            'max_contour_area': 50000,
            'min_circularity': 0.3  # Hoops from angles may not be perfect circles
        }
        
        # RL integration
        self.rl_features = {
            'hoop_center_pixel': None,      # [u, v] in image coordinates
            'hoop_distance': None,          # Meters from ZED depth
            'hoop_diameter': None,          # Detected diameter in pixels
            'hoop_alignment': None,         # How centered the hoop is
            'next_hoop_visible': False      # Is next target visible?
        }
    
    def process_frame(self, zed_image, zed_depth):
        """
        Process ZED Mini camera frame to extract hoop information for RL
        
        Args:
            zed_image: RGB image from ZED Mini
            zed_depth: Depth map from ZED Mini
            
        Returns:
            vision_features: Processed features for RL agent
        """
        import cv2
        import numpy as np
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(zed_image, cv2.COLOR_BGR2HSV)
        
        # Create mask for orange hoops
        mask = cv2.inRange(hsv, 
                          np.array(self.hoop_detection['color_range']['lower_orange']),
                          np.array(self.hoop_detection['color_range']['upper_orange']))
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and rank contours by area and circularity
        valid_hoops = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.hoop_detection['min_contour_area'] < area < self.hoop_detection['max_contour_area']:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > self.hoop_detection['min_circularity']:
                        # Extract hoop features
                        center, radius = cv2.minEnclosingCircle(contour)
                        center_int = (int(center[0]), int(center[1]))
                        
                        # Get depth at hoop center
                        depth_value = zed_depth[center_int[1], center_int[0]]
                        
                        if depth_value > 0:  # Valid depth measurement
                            valid_hoops.append({
                                'center': center,
                                'radius': radius,
                                'depth': depth_value,
                                'area': area,
                                'circularity': circularity
                            })
        
        # Find closest/largest hoop (current target)
        if valid_hoops:
            # Sort by size (closer/larger hoops are likely current targets)
            target_hoop = max(valid_hoops, key=lambda h: h['area'])
            
            # Calculate RL features
            image_center = (zed_image.shape[1] // 2, zed_image.shape[0] // 2)
            
            self.rl_features = {
                'hoop_center_pixel': target_hoop['center'],
                'hoop_distance': target_hoop['depth'] / 1000.0,  # Convert mm to m
                'hoop_diameter_pixels': target_hoop['radius'] * 2,
                'hoop_alignment': self.calculate_alignment(target_hoop['center'], image_center),
                'next_hoop_visible': len(valid_hoops) > 1,
                'hoop_area_ratio': target_hoop['area'] / (zed_image.shape[0] * zed_image.shape[1])
            }
        else:
            # No hoop detected
            self.rl_features = {
                'hoop_center_pixel': None,
                'hoop_distance': float('inf'),
                'hoop_diameter_pixels': 0,
                'hoop_alignment': 0.0,
                'next_hoop_visible': False,
                'hoop_area_ratio': 0.0
            }
        
        return self.rl_features
    
    def calculate_alignment(self, hoop_center, image_center):
        """
        Calculate how well aligned the drone is with the hoop center
        
        Returns:
            alignment: -1.0 (far left) to 1.0 (far right), 0.0 = centered
        """
        if hoop_center is None:
            return 0.0
        
        horizontal_offset = hoop_center[0] - image_center[0]
        max_offset = image_center[0]  # Half of image width
        
        return np.clip(horizontal_offset / max_offset, -1.0, 1.0)
```

### ROS2 Integration for ZED Mini

```python
class ZEDMiniROS2Node(Node):
    """
    ROS2 node for ZED Mini camera integration with DeepFlyer
    """
    
    def __init__(self):
        super().__init__('zed_mini_deepflyer')
        
        # Initialize ZED Mini
        self.vision_system = ZEDMiniVisionSystem(config)
        
        # Publishers for vision data
        self.vision_features_pub = self.create_publisher(
            VisionFeatures, '/deepflyer/vision_features', 10)
        
        # ZED Mini image subscribers (from ZED ROS2 wrapper)
        self.image_sub = self.create_subscription(
            Image, '/zed_mini/zed_node/rgb/image_rect_color', 
            self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/zed_mini/zed_node/depth/depth_registered', 
            self.depth_callback, 10)
        
        # Synchronize image and depth
        self.latest_image = None
        self.latest_depth = None
        
        # Processing timer
        self.create_timer(0.033, self.process_vision)  # 30 Hz processing
    
    def image_callback(self, msg):
        """Store latest RGB image"""
        self.latest_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
    
    def depth_callback(self, msg):
        """Store latest depth image"""
        self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
    
    def process_vision(self):
        """Process vision data and publish features for RL agent"""
        if self.latest_image is not None and self.latest_depth is not None:
            # Process frame for hoop detection
            features = self.vision_system.process_frame(
                self.latest_image, self.latest_depth)
            
            # Convert to ROS2 message
            msg = VisionFeatures()
            msg.timestamp = self.get_clock().now().to_msg()
            
            if features['hoop_center_pixel'] is not None:
                msg.hoop_detected = True
                msg.hoop_center_u = features['hoop_center_pixel'][0]
                msg.hoop_center_v = features['hoop_center_pixel'][1]
                msg.hoop_distance = features['hoop_distance']
                msg.hoop_alignment = features['hoop_alignment']
                msg.hoop_diameter_pixels = features['hoop_diameter_pixels']
                msg.next_hoop_visible = features['next_hoop_visible']
                msg.hoop_area_ratio = features['hoop_area_ratio']
            else:
                msg.hoop_detected = False
                msg.hoop_distance = float('inf')
                msg.hoop_alignment = 0.0
            
            self.vision_features_pub.publish(msg)
```

## Conclusion

DeepFlyer provides an engaging, hands-on approach to learning reinforcement learning through autonomous drone navigation. By abstracting complex flight control while preserving the essential RL learning objectives, students gain intuitive understanding of AI decision-making in physical systems.

The platform's design emphasizes safety, educational value, and real-world applicability, making it an ideal tool for introducing students to the exciting field of autonomous robotics and artificial intelligence.

---

**For technical support or course development assistance, contact the DeepFlyer development team.** 