# DeepFlyer Integration Guide for Uma
**Simulation, ROS, Gazebo, CAD Integration**

## Overview
This guide tells you WHERE to find things in my RL agent codebase and HOW to integrate your Gazebo simulation, ROS topics, and CAD components with the ML system I've built.

## Key Files You Need to Know About

### ROS Message Definitions (What You Need to Publish/Subscribe)
```
msg/DroneState.msg          # Comprehensive drone state (52 fields)
msg/VisionFeatures.msg      # YOLO11 + ZED vision output (14 fields) 
msg/CourseState.msg         # Course management state
```

### ROS Nodes (What Subscribes/Publishes)
```
nodes/rl_agent_node.py      # Main RL agent ROS2 node
nodes/px4_interface_node.py # PX4-ROS-COM interface
nodes/vision_processor_node.py # YOLO11 vision processing
```

### Configuration Files
```
rl_agent/config.py          # All system configuration (course, P3O, vision, PX4)
rl_agent/mvp_trajectory.py  # Flight phases and trajectory logic
```

## ROS Topic Interface (YOUR IMPLEMENTATION TARGETS)

### Topics You Need to Provide (Uma's Responsibility)

#### 1. PX4-ROS-COM Topics (Standard PX4 Interface)
```bash
# What my PX4 interface node expects from your simulation:
/fmu/out/vehicle_local_position    # PX4 position/velocity data
/fmu/out/vehicle_status            # Flight controller status  
/fmu/out/vehicle_attitude          # Drone orientation
/fmu/out/battery_status            # Battery information

# What my PX4 interface node publishes (your sim should consume):
/fmu/in/trajectory_setpoint        # Velocity commands from RL agent
/fmu/in/offboard_control_mode      # Control mode settings
/fmu/in/vehicle_command            # Arm/disarm commands
```

#### 2. Vision System Topics
```bash
# Your Gazebo camera simulation should publish:
/zed_mini/zed_node/rgb/image_rect_color     # RGB camera feed
/zed_mini/zed_node/depth/depth_registered   # Depth data

# My vision processor will publish (you can use for debugging):
/deepflyer/vision_features          # Processed hoop detections
```

### Topics My System Publishes (You Can Subscribe)

#### 1. RL Agent Outputs
```bash
/deepflyer/rl_action               # 4D actions: [vx, vy, vz, yaw_rate]
/deepflyer/reward_feedback         # Real-time reward breakdown
```

#### 2. Course Management 
```bash
/deepflyer/course_state            # Episode status, hoop progress
```

## Course Configuration (YOUR CAD/GAZEBO WORLD)

### Course Dimensions (from `rl_agent/config.py`)
```python
COURSE_DIMENSIONS = {
    'length': 2.1,     # meters
    'width': 1.6,      # meters  
    'height': 1.5,     # meters
    'safety_buffer': 0.2  # meters from walls
}

HOOP_CONFIG = {
    'num_hoops': 5,           # Fixed 5-hoop circuit
    'diameter': 0.8,          # meters
    'flight_altitude': 0.8,   # meters above ground
}
```

### Hoop Positions (from `rl_agent/config.py`)
```python
# You need to place 5 hoops at these relative positions:
def get_course_layout(spawn_position):
    hoop_positions = [
        (spawn_position[0] + 0.5, spawn_position[1] - 0.5, 0.8),  # Hoop 1
        (spawn_position[0] + 1.0, spawn_position[1] - 0.5, 0.8),  # Hoop 2  
        (spawn_position[0] + 1.5, spawn_position[1] + 0.0, 0.8),  # Hoop 3
        (spawn_position[0] + 1.0, spawn_position[1] + 0.5, 0.8),  # Hoop 4
        (spawn_position[0] + 0.5, spawn_position[1] + 0.0, 0.8)   # Hoop 5
    ]
```

### YOLO11 Hoop Detection (Your Hoop Models)
```python
# From rl_agent/config.py - your hoops should be detectable as:
VISION_CONFIG = {
    'target_classes': ['sports ball', 'frisbee', 'donut'],  # YOLO11 classes
    'confidence_threshold': 0.3,
    'hoop_diameter': 0.8  # meters
}
```

## Message Definitions (You Need to Match These)

### 1. DroneState.msg (What PX4 interface expects)
```
# Key fields your simulation needs to provide:
geometry_msgs/Point position          # Current position (x, y, z)
geometry_msgs/Quaternion orientation  # Orientation quaternion
geometry_msgs/Vector3 linear_velocity # Velocity (vx, vy, vz)
bool armed                           # Is drone armed
string flight_mode                   # Current flight mode
float32 battery_remaining            # Battery level (0.0-1.0)
```

### 2. VisionFeatures.msg (What vision system outputs)
```
# Your camera simulation should enable my vision processor to output:
bool hoop_detected                   # Is hoop visible
float32 hoop_center_x_norm          # Hoop center X [-1,1]
float32 hoop_center_y_norm          # Hoop center Y [-1,1] 
float32 hoop_distance_norm          # Distance [0,1]
float32 detection_confidence        # YOLO confidence [0,1]
```

## Integration Points

### 1. Gazebo World Setup
**File to reference:** `rl_agent/config.py` (lines 15-30)

Create a Gazebo world with:
- Indoor environment (2.1m x 1.6m x 1.5m)
- 5 hoops at specified positions
- ZED Mini camera model
- PX4 SITL integration
- Physics appropriate for 0.8m diameter hoops

### 2. ROS2 Launch Files
**Reference:** `launch/deepflyer_ml.launch.py` and `launch/mvp_system.launch.py`

Your launch files should start:
```bash
# My nodes (already working):
ros2 run deepflyer rl_agent_node
ros2 run deepflyer px4_interface_node  
ros2 run deepflyer vision_processor_node

# Your nodes (need to create):
ros2 run your_gazebo_package simulation_node
ros2 run your_px4_package px4_sitl_node
```

### 3. Camera Integration
**Files:** `rl_agent/depth_processor.py`, `nodes/vision_processor_node.py`

Your Gazebo camera should:
- Publish standard ZED ROS topics
- Provide 1280x720 RGB + depth
- Use ROS2 image transport
- Enable YOLO11 to detect hoops as 'sports ball', 'frisbee', or 'donut'

### 4. PX4 SITL Integration
**Reference:** `nodes/px4_interface_node.py` (lines 1-100)

Your PX4 simulation should:
- Use PX4-ROS-COM (not MAVROS)
- Respond to `/fmu/in/trajectory_setpoint` velocity commands
- Publish standard PX4 topics
- Support OFFBOARD mode
- Handle 20Hz control frequency

## Flight Phase System (For Course Logic)

### MVP Flight Phases (from `rl_agent/mvp_trajectory.py`)
```python
class MVPFlightPhase(Enum):
    TAKEOFF = "takeoff"           # Initial ascent to 0.8m
    SCAN_360 = "scan_360"         # Look for first hoop 
    NAVIGATE_TO_HOOP = "navigate" # Fly toward detected hoop
    HOOP_PASSAGE = "passage"      # Pass through hoop
    SCAN_NEXT_HOOP = "scan_next"  # Look for next hoop
    TRAJECTORY_COMPLETE = "complete"  # All 5 hoops done
```

Your simulation should track which phase the drone is in and update hoop detection accordingly.

## Safety System Integration

### Safety Limits (from `nodes/px4_interface_node.py`)
```python
# Your simulation should enforce these:
max_velocity_xy = 2.0      # m/s
max_velocity_z = 1.5       # m/s  
max_altitude = 5.0         # meters
min_altitude = 0.2         # meters
max_horizontal_distance = 10.0  # meters from takeoff
```

## Testing Your Integration

### 1. Topic Verification
```bash
# Check that your simulation publishes required topics:
ros2 topic list | grep fmu
ros2 topic hz /fmu/out/vehicle_local_position
ros2 topic echo /zed_mini/zed_node/rgb/image_rect_color
```

### 2. Message Compatibility  
```bash
# Verify message types match:
ros2 interface show px4_msgs/msg/VehicleLocalPosition
ros2 interface show sensor_msgs/msg/Image
```

### 3. Integration Test
```bash
# Run my test scripts with your simulation:
python scripts/test_integration.py
python scripts/test_models.py
```

## What I Don't Need You to Build

- RL training code (I handle this)
- Hyperparameter optimization (I handle this) 
- Reward functions (I handle this)
- UI/frontend (Jay handles this)
- Database (Jay handles this)

## What I Do Need You to Build

1. **Gazebo World:** 5-hoop course matching my specifications
2. **PX4 SITL Integration:** Standard PX4-ROS-COM topics
3. **Camera Simulation:** ZED Mini-compatible RGB + depth
4. **Hoop Models:** Detectable by YOLO11 as 'sports ball'/'frisbee'/'donut'
5. **Physics:** Realistic drone dynamics
6. **ROS2 Launch:** Integration with my existing nodes

## Quick Start Checklist

- [ ] Create Gazebo world with 5 hoops at specified positions
- [ ] Set up PX4 SITL with PX4-ROS-COM
- [ ] Add ZED Mini camera model publishing RGB + depth
- [ ] Create launch files that start both our systems
- [ ] Test integration with my rl_agent_node and px4_interface_node
- [ ] Verify hoop detection works with YOLO11

## Questions?

If you need clarification on any part of my RL agent implementation, refer to:
- `rl_agent/config.py` - All configuration parameters
- `nodes/` directory - All ROS node implementations  
- `msg/` directory - All message definitions
- `scripts/test_*.py` - All test scripts

Your job is the simulation environment - mine is the AI that flies through it! 