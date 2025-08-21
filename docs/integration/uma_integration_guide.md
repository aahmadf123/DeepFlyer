# DeepFlyer Simulation Integration Documentation
**For Simulation, ROS, Gazebo, and CAD Development**

## Overview
This document describes the ROS interfaces, message specifications, and system requirements for integrating simulation environments with the DeepFlyer RL training platform.

### Related Documentation
- **📚 [Technical Overview](TEAM_OVERVIEW.md)** - Complete ML/RL system implementation details
- **📋 [Backend Integration](api/JAY_INTEGRATION_GUIDE.md)** - Jay's UI/database interface specifications
- **📖 [System Architecture](INTEGRATION_GUIDE.md)** - High-level integration overview

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

## ROS Topic Interface Specifications

### Expected Simulation Topics

#### 1. PX4-ROS-COM Topics (Standard PX4 Interface)
```bash
# Expected from simulation environment:
/fmu/out/vehicle_local_position    # PX4 position/velocity data
/fmu/out/vehicle_status            # Flight controller status  
/fmu/out/vehicle_attitude          # Drone orientation
/fmu/out/battery_status            # Battery information

# Published by RL system for simulation consumption:
/fmu/in/trajectory_setpoint        # Velocity commands from RL agent
/fmu/in/offboard_control_mode      # Control mode settings
/fmu/in/vehicle_command            # Arm/disarm commands
```

#### 2. Vision System Topics
```bash
# Expected camera simulation topics:
/zed_mini/zed_node/rgb/image_rect_color     # RGB camera feed
/zed_mini/zed_node/depth/depth_registered   # Depth data

# Available from vision processor (for debugging):
/deepflyer/vision_features          # Processed hoop detections
```

### RL System Published Topics

#### 1. RL Agent Outputs
```bash
/deepflyer/rl_action               # 4D actions: [vx, vy, vz, yaw_rate]
/deepflyer/reward_feedback         # Real-time reward breakdown
```

#### 2. Course Management 
```bash
/deepflyer/course_state            # Episode status, hoop progress
```

## Course Configuration Specifications

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

### Required Hoop Positions (from `rl_agent/config.py`)
```python
# Standard 5-hoop circuit layout:
def get_course_layout(spawn_position):
    hoop_positions = [
        (spawn_position[0] + 0.5, spawn_position[1] - 0.5, 0.8),  # Hoop 1
        (spawn_position[0] + 1.0, spawn_position[1] - 0.5, 0.8),  # Hoop 2  
        (spawn_position[0] + 1.5, spawn_position[1] + 0.0, 0.8),  # Hoop 3
        (spawn_position[0] + 1.0, spawn_position[1] + 0.5, 0.8),  # Hoop 4
        (spawn_position[0] + 0.5, spawn_position[1] + 0.0, 0.8)   # Hoop 5
    ]
```

### YOLO11 Detection Requirements (Hoop Model Specifications)
```python
# From rl_agent/config.py - required detection characteristics:
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

### 1. Gazebo World Specifications
**Reference:** `rl_agent/config.py` (lines 15-30)

Required environment features:
- Indoor environment (2.1m x 1.6m x 1.5m)
- 5 hoops at specified positions
- ZED Mini camera model
- PX4 SITL integration
- Physics appropriate for 0.8m diameter hoops

### 2. ROS2 Launch File Structure
**Reference:** `launch/deepflyer_ml.launch.py` and `launch/mvp_system.launch.py`

Complete system launch sequence:
```bash
# RL system nodes (provided):
ros2 run deepflyer rl_agent_node
ros2 run deepflyer px4_interface_node  
ros2 run deepflyer vision_processor_node

# Simulation system nodes (required):
ros2 run simulation_package gazebo_world_node
ros2 run simulation_package px4_sitl_node
```

### 3. Camera Integration Specifications
**Files:** `rl_agent/depth_processor.py`, `nodes/vision_processor_node.py`

Required camera simulation characteristics:
- Standard ZED ROS topic publishing
- 1280x720 RGB + depth output
- ROS2 image transport compatibility
- YOLO11 compatible object detection for 'sports ball', 'frisbee', or 'donut' classes

### 4. PX4 SITL Integration Requirements
**Reference:** `nodes/px4_interface_node.py` (lines 1-100)

Required PX4 simulation features:
- PX4-ROS-COM protocol (not MAVROS)
- `/fmu/in/trajectory_setpoint` velocity command handling
- Standard PX4 topic publishing
- OFFBOARD mode support
- 20Hz control frequency capability

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

Simulation systems can track current flight phase and update hoop detection accordingly.

## Safety System Integration

### Safety Limits (from `nodes/px4_interface_node.py`)
```python
# Required safety constraint enforcement:
max_velocity_xy = 2.0      # m/s
max_velocity_z = 1.5       # m/s  
max_altitude = 5.0         # meters
min_altitude = 0.2         # meters
max_horizontal_distance = 10.0  # meters from takeoff
```

## Integration Verification

### 1. Topic Verification
```bash
# Verify simulation publishes required topics:
ros2 topic list | grep fmu
ros2 topic hz /fmu/out/vehicle_local_position
ros2 topic echo /zed_mini/zed_node/rgb/image_rect_color
```

### 2. Message Compatibility  
```bash
# Verify message types match specifications:
ros2 interface show px4_msgs/msg/VehicleLocalPosition
ros2 interface show sensor_msgs/msg/Image
```

### 3. Integration Testing
```bash
# Available test scripts for integration validation:
python scripts/test_integration.py
python scripts/test_models.py
```

## System Component Responsibilities

### Provided by RL System
- RL training algorithms and optimization
- Hyperparameter tuning and management
- Reward function execution and processing
- Vision processing (YOLO11) and feature extraction

### Integration Requirements
1. **Gazebo World:** 5-hoop course matching specifications
2. **PX4 SITL Integration:** Standard PX4-ROS-COM topic interface
3. **Camera Simulation:** ZED Mini-compatible RGB + depth output
4. **Hoop Models:** YOLO11-detectable as 'sports ball'/'frisbee'/'donut'
5. **Physics Engine:** Realistic drone dynamics simulation
6. **Launch Integration:** ROS2 launch file coordination

## Implementation Checklist

- [ ] Gazebo world implementation with specified 5-hoop layout
- [ ] PX4 SITL setup with PX4-ROS-COM protocol
- [ ] ZED Mini camera model with RGB + depth publishing
- [ ] Launch file coordination for system integration
- [ ] Integration testing with rl_agent_node and px4_interface_node
- [ ] YOLO11 detection verification for hoop models

## Reference Documentation

For detailed implementation specifications, reference:
- `rl_agent/config.py` - Complete configuration parameters
- `nodes/` directory - ROS node interface specifications
- `msg/` directory - Message format definitions
- `scripts/test_*.py` - Integration test utilities

The simulation environment provides the training platform for the autonomous flight AI system. 