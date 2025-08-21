# DeepFlyer Technical Reference

**Project**: Educational drone RL platform - students tune rewards, watch AI learn hoop navigation  
**Algorithm**: P3O (Procrastinated Policy Optimization)  
**Hardware**: Holybro S500 + Pixhawk 6C + Pi 4B + ZED Mini + Emergency stop

> **Note:** This document details the complete ML/RL system implementation. For integration with this system, see:
> - **Backend/UI Integration:** [api/JAY_INTEGRATION_GUIDE.md](api/JAY_INTEGRATION_GUIDE.md)
> - **Simulation Integration:** [UMA_INTEGRATION_GUIDE.md](UMA_INTEGRATION_GUIDE.md)

## Team Responsibilities

### My Role (RL/ML Developer)
- P3O reinforcement learning algorithm
- Hyperparameter optimization (random search with ClearML)  
- Reward function configuration (AWS DeepRacer style)
- ML interface for backend integration
- Real-time training metrics from computer vision to flight decisions

### Jay's Components (UI/Frontend/Database)
- AWS DeepRacer-style user interface implementation
- Backend API and database system integration
- Real-time training dashboard with live metrics display
- Hyperparameter control interface for student configuration
- Student session management and progress tracking systems

### Uma's Components (Simulation/CAD/ROS)
- Gazebo simulation environment with physics engine
- 5-hoop course CAD design and physics implementation
- PX4-ROS-COM integration and message handling system
- Camera simulation (ZED Mini) for computer vision processing
- ROS topic publishing/subscribing for system integration

**System Flow:** Simulation Environment → ML Training System → Educational User Interface

## My Implementation Overview

I handle the complete RL/AI/Vision pipeline - from raw camera input to trained flight policies. This includes computer vision processing, state representation, reward engineering, P3O training, and all the intelligent decision-making that makes the drone learn to navigate autonomously.

## MVP Flight Trajectory

The Minimum Viable Product (MVP) implements a simplified flight path for educational demonstration:

### MVP Flight Sequence
1. **Takeoff** from Point A to target altitude (0.8m)
2. **360-degree yaw rotation** to scan and detect all visible hoops using ZED Mini + YOLO11
3. **Navigate toward** the single detected hoop on one side
4. **Fly through** the hoop center with precision alignment
5. **Turn around** after passing through to return through the same hoop from the other side
6. **Land** at the original Point A

### MVP vs Full System
- **MVP:** Single hoop, round-trip passage, simplified navigation
- **Full:** 5-hoop circuit, 3 laps, complex course navigation

## P3O Algorithm (Procrastinated Policy-based Observer)

P3O is a hybrid RL algorithm combining on-policy and off-policy updates for improved sample efficiency and learning stability:

### Key P3O Components
- **Procrastinated Updates:** Delays on-policy updates to improve gradient estimation and reduce variance
- **Blended Gradient Learning:** Combines gradients from on-policy (recent experience) and off-policy (replay buffer) sources  
- **Entropy Regularization:** Encourages exploration by penalizing premature convergence

### MVP Observation Space (8D)
The agent receives a normalized vector representing the current state:
```python
obs = [
    hoop_x_center_norm,     # Horizontal position of hoop center in camera frame [-1, 1]
    hoop_y_center_norm,     # Vertical position of hoop center [-1, 1]
    hoop_visible,           # Binary (1 if detected, else 0)
    hoop_distance_norm,     # Depth to hoop [0, 1]
    drone_vx_norm,          # Forward velocity [-1, 1]
    drone_vy_norm,          # Lateral velocity [-1, 1]
    drone_vz_norm,          # Vertical velocity [-1, 1]
    yaw_rate_norm           # Yaw rate [-1, 1]
]
```

### MVP Action Space (4D)
The agent outputs a 4-dimensional continuous action vector:
```python
action = [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
# vx_cmd: Forward/backward velocity (m/s)
# vy_cmd: Left/right velocity (m/s)  
# vz_cmd: Up/down velocity (m/s)
# yaw_rate_cmd: Yaw rotation rate (rad/s)
```

## MVP Reward Function (Student Tunable)

The MVP reward function focuses on detection, alignment, and round-trip completion:

### Positive Rewards (Student Tunable)
| Event | Default Points | Range | Description |
|-------|----------------|-------|-------------|
| Hoop Detected | +1 | 1-5 | The hoop is visible in the camera |
| Horizontal Align | +5 | 1-10 | x_center close to image center |
| Vertical Align | +5 | 1-10 | y_center close to image center |
| Depth Closer | +10 | 5-20 | Approaching hoop |
| Hoop Passage | +100 | 50-200 | Passed through the hoop |
| Roundtrip Finish | +200 | 200-300 | Back through same hoop, landed |

### Penalties (Student Tunable)
| Event | Default Points | Range | Description |
|-------|----------------|-------|-------------|
| Collision | -25 | -10 to -100 | Hit an object or wall |
| Missed Hoop | -25 | -10 to -50 | Passed beside the hoop |
| Drift/Lost | -10 | -5 to -25 | Hoop lost from camera |
| Time Penalty | -1 | -0.5 to -2 | Per timestep |

## Student Training Configuration

### Training Time (Student Specified)
Students specify training time in minutes via UI:
```bash
--train_time 60  # 60 minutes
```
Maps to total environment steps: `max_steps = steps_per_second * 60 * train_time_minutes`

### MVP Hyperparameters (Random Search)
| Parameter | Range | Description |
|-----------|-------|-------------|
| learning_rate | 1e-4 to 3e-3 | Step size for optimizer |
| clip_ratio | 0.1 to 0.3 | Controls PPO-style policy update clipping |
| entropy_coef | 1e-3 to 0.1 | Weight for entropy term to encourage exploration |
| batch_size | 64 to 256 | Minibatch size for updates |
| rollout_steps | 512 to 2048 | Environment steps per update |
| num_epochs | 3 to 10 | Epochs per policy update |
| gamma | 0.9 to 0.99 | Discount factor for future rewards |
| gae_lambda | 0.9 to 0.99 | GAE parameter for advantage estimation |

Default values provided, students can adjust via UI for learning optimization.

## ZED Mini Vision Processing

### YOLO11 Hoop Detection
Using YOLO bounding box and depth image for precise hoop center detection:

**Bounding Box Center:**
- `(x_center, y_center)` from YOLO detection (normalized)

**Hoop Distance:**  
- Use `depth_image[y_center, x_center]`
- Median filter over bounding box region

**Center Alignment Detection:**
- Check if `|x_center| < 0.1` and `|y_center| < 0.1`
- Check if depth is continuously decreasing (approaching)

**Edge Detection:**
- Use bounding box width/height as proximity proxy
- Box growth indicates closer approach
- Box shift/disappearance indicates miss or drift

## ROS2 Topics Reference

### Flight Control (PX4-ROS-COM)
| Topic | Type | Direction | Fields | Purpose |
|-------|------|-----------|--------|---------|
| `/fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` | PUBLISH | `command`, `param1-7`, `target_system` | Direct PX4 commands |
| `/fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | PUBLISH | `position`, `velocity`, `acceleration` | Control mode selection |
| `/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | PUBLISH | `position[3]`, `velocity[3]` | Position/velocity targets |
| `/fmu/out/vehicle_local_position` | `px4_msgs/VehicleLocalPosition` | SUBSCRIBE | `x`, `y`, `z`, `vx`, `vy`, `vz` | Current position/velocity |
| `/fmu/out/vehicle_status` | `px4_msgs/VehicleStatus` | SUBSCRIBE | `arming_state`, `nav_state`, `failsafe` | Flight controller status |

### Vision System  
| Topic | Type | Direction | Fields | Purpose |
|-------|------|-----------|--------|---------|
| `/zed_mini/zed_node/rgb/image_rect_color` | `sensor_msgs/Image` | SUBSCRIBE | `height`, `width`, `encoding`, `data[]` | RGB camera feed |
| `/zed_mini/zed_node/depth/depth_registered` | `sensor_msgs/Image` | SUBSCRIBE | `height`, `width`, `data[]` | Depth data (mm) |
| `/deepflyer/vision_features` | Custom `VisionFeatures.msg` | PUBLISH | See below | Processed vision data |

**VisionFeatures.msg Fields:**
- `hoop_detected` (bool)
- `hoop_center_u`, `hoop_center_v` (int32) - pixel coordinates
- `hoop_distance` (float32) - meters
- `hoop_alignment` (float32) - (-1 to 1, 0=centered)  
- `hoop_diameter_pixels` (float32)
- `next_hoop_visible` (bool)
- `hoop_area_ratio` (float32)

### RL Training System
| Topic | Type | Direction | Fields | Purpose |
|-------|------|-----------|--------|---------|
| `/deepflyer/rl_action` | Custom `RLAction.msg` | PUBLISH | `vx_cmd`, `vy_cmd`, `vz_cmd`, `yaw_rate_cmd` | MVP 4D actions (-1 to 1) |
| `/deepflyer/reward_feedback` | Custom `RewardFeedback.msg` | PUBLISH | See below | Reward breakdown |
| `/deepflyer/course_state` | Custom `CourseState.msg` | SUBSCRIBE | See below | Course progress |

**RLAction.msg Fields (MVP 4D Actions):**
- `vx_cmd` (float32) - forward/backward velocity (-1 to 1)
- `vy_cmd` (float32) - left/right velocity (-1 to 1)
- `vz_cmd` (float32) - up/down velocity (-1 to 1)
- `yaw_rate_cmd` (float32) - yaw rotation rate (-1 to 1)

**RewardFeedback.msg Fields:**
- `total_reward` (float32)
- `hoop_progress_reward` (float32)
- `alignment_reward` (float32) 
- `collision_penalty` (float32)
- `episode_time` (float32)
- `lap_completed` (bool)

**CourseState.msg Fields:**
- `target_hoop_detected` (bool)
- `flight_phase` (int32) - 0=takeoff, 1=scan, 2=approach, 3=through, 4=return, 5=land
- `round_trip_progress` (float32) - 0 to 1
- `episode_time` (float32)

## Vision Processing Pipeline (My Implementation)

### YOLO11 Computer Vision System
I'm implementing a robust vision system using YOLO11 for real-time hoop detection. The ZED Mini provides both RGB and depth data, which I process through YOLO11 for reliable object detection in varying conditions.

**YOLO11 Detection Pipeline:**
- Using YOLO11 for robust hoop detection regardless of lighting/angle conditions
- Custom trained model on hoop dataset with data augmentation
- Handles partial occlusions, varying orientations, and lighting changes
- Outputs bounding boxes with confidence scores for each detected hoop
- Real-time inference on Pi 4B using optimized ONNX model
- Direct integration with ZED Mini depth data for 3D positioning

### ZED Mini Integration Strategy
The ZED Mini gives me stereo vision capabilities that I leverage for precise 3D positioning:

**Camera Configuration:**
- **Resolution**: 1280x720 @ 60Hz for smooth real-time processing
- **Depth Mode**: PERFORMANCE (balance of speed vs accuracy)
- **Coordinate System**: RIGHT_HANDED_Z_UP for ROS compatibility

**Depth Processing:**
- Extract accurate distance measurements from stereo depth map
- Filter depth noise using temporal smoothing and statistical outlier removal
- Convert depth values from millimeters to meters for RL state representation
- Cross-reference YOLO bounding boxes with depth data for 3D hoop positions

### Vision Feature Extraction for RL
I'm processing the raw visual data into meaningful features that the RL agent can use:

**Spatial Features:**
- **Hoop Alignment**: -1.0 (far left) → 0.0 (centered) → +1.0 (far right)
- **Distance**: Stereo depth measurement (0.5m to 5.0m effective range)
- **Pixel Coordinates**: Center point in image frame (0-1280, 0-720)
- **Size Indicator**: Hoop area ratio relative to total image area

**Navigation Features:**
- **Target Identification**: Which hoop is the current navigation target
- **Next Hoop Preview**: Whether the next hoop in sequence is visible
- **Multi-Hoop Tracking**: Simultaneous detection of multiple hoops with priority sorting

### Vision Processing Node Architecture
I'm creating a dedicated ROS2 node that handles all vision processing:

```
vision_processor_node.py:
├── ZED Mini data subscription (/zed_mini/zed_node/rgb + /depth)
├── YOLO11 inference pipeline
├── Depth integration and 3D positioning
├── Feature extraction for RL
└── Publishing to /deepflyer/vision_features
```

The node will run at 30Hz to balance processing load with real-time requirements. I'm implementing frame dropping and async processing to maintain consistent performance on the Pi 4B.

## P3O RL System (My Implementation)

### P3O State Processing for MVP
The MVP uses the 8D observation space detailed above, with all values normalized to ensure stable learning. The P3O algorithm processes these features to make flight decisions for the single-hoop round-trip navigation task.

### Action Space Design
I'm using the continuous 4D action space detailed above that gives the drone fine-grained control while remaining intuitive for the MVP round-trip navigation.

**Action Translation Logic:**
- Actions are smoothed to prevent jerky movements that could destabilize the drone
- Speed reduction near hoops for precision - when within 1m of target, max speeds are reduced by 30%
- Emergency bounds checking ensures actions never exceed safety limits
- Actions get converted to PX4-ROS-COM trajectory setpoints with built-in safety margins

### P3O Algorithm Implementation
I'm implementing P3O with specific adaptations for drone navigation challenges:

**Core Hyperparameters:**
- **Learning Rate**: 3e-4 (adaptive scheduling based on training progress)
- **Clip Epsilon**: 0.2 (prevents destructive policy updates)
- **Batch Size**: 64 (balanced for Pi 4B memory constraints)
- **Discount Factor**: 0.99 (values long-term course completion)
- **Entropy Coefficient**: 0.01 (encourages exploration of new flight paths)

**Training Architecture:**
- Experience collection runs at 20Hz during flight episodes
- Policy updates happen every 64 steps to maintain sample efficiency
- Value function and policy networks are separate to prevent interference
- Gradient clipping at 0.5 to prevent training instability from outlier episodes

## MVP Course Setup

### Physical Setup
- **Course Size**: 2.1m × 1.6m flight area 
- **Flight Height**: 0.8m above ground
- **Hoop Count**: 1 hoop for round-trip navigation
- **Hoop Diameter**: 0.8m
- **Navigation**: Takeoff → Scan → Approach → Through → Return → Through → Land

## Reward Engineering (My Implementation)

### Reward Function Architecture
I'm designing a modular reward system that students can tune without breaking the core safety mechanisms. The challenge is making the rewards educational while ensuring safe flight behavior.

**Positive Rewards (Student Tunable):**
| Event | Default Points | Range | Description |
|-------|----------------|-------|-------------|
| Hoop Passage | +100 | 50-200 | Successfully through hoop |
| Approach Target | +10 | 5-20 | Getting closer to target |
| Center Bonus | +20 | 10-40 | Precise center passage |
| Visual Alignment | +5 | 1-10 | Hoop centered in view |
| Round-trip Complete | +200 | 200-300 | Back through same hoop and landed |

**Penalties (Student Tunable):**
| Event | Default Points | Range | Description |
|-------|----------------|-------|-------------|
| Hoop Miss | -25 | -10 to -50 | Flying around hoop |
| Collision | -100 | -50 to -200 | Hitting obstacles |
| Wrong Direction | -2 | -1 to -5 | Flying away from target |
| Time Penalty | -1 | -0.5 to -2 | Taking too long |
| Erratic Flight | -3 | -1 to -10 | Jerky movements |

**Safety Overrides (Non-tunable):**
| Event | Points | Trigger |
|-------|--------|---------|
| Boundary Violation | -200 | Outside flight area |
| Emergency Stop | -500 | Hardware stop pressed |

### Reward Calculation Logic (Detailed Implementation)
I'm implementing a sophisticated shaped reward system that provides continuous, educational feedback while maintaining safety priorities:

**Core Reward Equation:**
```
Total_Reward = Progress_Reward + Alignment_Reward + Speed_Reward - Penalties - Safety_Overrides
```

**1. Progress Reward (Distance-Based Shaping):**
- **Calculation**: `progress_reward = student_weight * exp(-distance_to_target / decay_factor)`
- **Distance decay**: Rewards exponentially increase as drone approaches target hoop
- **Previous distance tracking**: `delta_reward = (prev_distance - current_distance) * approach_multiplier`
- **Completion bonus**: Immediate +50 points when hoop center is within 0.3m radius

**2. Alignment Reward (Continuous Guidance):**
- **Visual alignment**: `alignment_reward = student_weight * (1 - abs(hoop_center_offset))`
- **Hoop center offset**: Normalized pixel distance from image center (-1 to +1)
- **Approach angle**: Additional reward for approaching hoop perpendicularly rather than at angles
- **Round-trip awareness**: Bonus for completing return journey through same hoop

**3. Speed Management Reward:**
- **Adaptive speed**: Rewards faster flight in open areas, slower near targets
- **Speed calculation**: `speed_reward = optimal_speed_ratio * student_weight`
- **Optimal speed ratio**: `min(current_speed / target_speed, target_speed / current_speed)`
- **Target speed varies**: 0.6 m/s in open space, 0.2 m/s within 1m of hoop

**4. Multi-Component Normalization:**
- **Component weighting**: Each reward type normalized to [0, student_max] range
- **Temporal smoothing**: Running average over last 5 timesteps to prevent noise
- **Priority ordering**: Safety > Progress > Alignment > Speed
- **Student tunability**: All positive weights adjustable, ratios maintained

**5. Safety Override Logic:**
- **Boundary violations**: Immediate -200 points, episode termination trigger
- **Collision detection**: -100 points, speed reduction to 10% for 2 seconds
- **Emergency stop**: -500 points, immediate motor shutdown
- **Non-negotiable**: Safety penalties bypass all student tuning parameters

**6. Reward Shaping Techniques:**
- **Potential-based shaping**: Ensures optimal policy convergence
- **Curriculum progression**: Reward weights automatically adjust as success rate improves
- **Exploration bonuses**: Small rewards for visiting unexplored areas of state space
- **Dense feedback**: Reward calculated every 50ms for continuous learning signal

## Training Configuration (Student Tunable)

Like AWS DeepRacer, students can adjust training parameters to optimize learning:

### Episode Parameters (Student Adjustable)
| Parameter | Default | Range | Unit | Educational Purpose |
|-----------|---------|-------|------|---------------------|
| Max Episodes | 1000 | 100-5000 | per session | Learn about training duration vs performance |
| Max Steps per Episode | 500 | 200-1000 | steps (25-50 sec) | Balance exploration vs efficiency |
| Evaluation Frequency | 50 | 10-200 | episodes | Trade-off between feedback and training time |
| Early Stopping Patience | 100 | 50-500 | episodes | Understand convergence and overfitting |
| Success Threshold | 80% | 50-95% | completion rate | Set learning goals and difficulty |

### P3O Hyperparameters (Student Adjustable)
| Parameter | Default | Range | Educational Focus |
|-----------|---------|-------|-------------------|
| Learning Rate | 3e-4 | 1e-5 to 1e-3 | Speed vs stability trade-off |
| Batch Size | 64 | 16-256 | Memory usage vs sample efficiency |
| Clip Epsilon | 0.2 | 0.1-0.5 | Conservative vs aggressive updates |
| Entropy Coefficient | 0.01 | 0.001-0.1 | Exploration vs exploitation |
| Discount Factor | 0.99 | 0.9-0.999 | Short-term vs long-term thinking |

### Advanced Settings (Instructor Override)
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Safety Boundaries | Fixed | Non-negotiable for hardware protection |
| Emergency Limits | Fixed | Required for safe operation |
| Hardware Timeouts | Fixed | Prevent damage from communication failures |

### Safety Limits
| Boundary | Dimension | Action |
|----------|-----------|--------|
| Flight Area | 2.1m × 1.6m × 1.5m | Auto-land if exceeded |
| Max Speed | 0.8 m/s horizontal | Velocity clamping |
| Max Speed | 0.4 m/s vertical | Velocity clamping |
| Emergency Stop | Hardware button | Immediate motor kill |

## Integration Strategy & Data Flow

### My Node Architecture (6 Specialized Nodes)
I'm creating six specialized ROS2 nodes for robust system separation:

**1. Vision Processor Node** (`vision_processor_node.py`)
- **Purpose**: YOLO11-based hoop detection with ZED Mini integration
- **Subscribes to**: `/zed_mini/zed_node/rgb/image_rect_color`, `/zed_mini/zed_node/depth/depth_registered`
- **Publishes to**: `/deepflyer/vision_features` (8D observation components)
- **Function**: Real-time hoop detection, depth processing, alignment calculation
- **Model**: Uses `weights/best.pt` (custom) with `yolo11l.pt` fallback

**2. Main RL Agent Node** (`rl_agent_node.py`)
- **Purpose**: General RL training infrastructure and episode management
- **Subscribes to**: All system state topics for comprehensive learning
- **Publishes to**: Training metrics, episode management signals
- **Function**: Handles training loops, experience replay, model checkpointing
- **Algorithm**: Framework-agnostic RL infrastructure

**3. P3O Agent Node** (`p3o_agent_node.py`)
- **Purpose**: P3O-specific algorithm implementation and control
- **Subscribes to**: `/deepflyer/vision_features`, `/deepflyer/drone_state`, `/deepflyer/course_state` 
- **Publishes to**: `/deepflyer/rl_action` (4D actions: vx, vy, vz, yaw_rate)
- **Function**: 8D→4D neural network inference, P3O learning updates
- **Algorithm**: Procrastinated Policy Optimization with custom hoop navigation

**4. PX4 Interface Node** (`px4_interface_node.py`)  
- **Purpose**: PX4-ROS-COM communication and safety layer
- **Subscribes to**: `/deepflyer/rl_action`, `/fmu/out/vehicle_local_position`, `/fmu/out/vehicle_status`
- **Publishes to**: `/fmu/in/trajectory_setpoint`, `/fmu/in/offboard_control_mode`, `/deepflyer/course_state`
- **Function**: RL action → PX4 commands, safety constraints, emergency handling
- **Safety**: Velocity limits, geofencing, emergency landing protocols

**5. Reward Calculator Node** (`reward_calculator_node.py`)
- **Purpose**: Multi-component reward calculation with student configurability  
- **Subscribes to**: All state topics (vision, drone, course, actions)
- **Publishes to**: `/deepflyer/reward_feedback` (detailed component breakdown)
- **Function**: Student-tunable reward computation, educational feedback generation
- **Configuration**: YAML-based reward parameters that students can modify

**6. Course Manager Node** (`course_manager_node.py`)
- **Purpose**: MVP trajectory state management and episode coordination
- **Subscribes to**: Drone position, vision features, flight status
- **Publishes to**: `/deepflyer/course_state` (MVP flight phases)
- **Function**: Phase tracking (takeoff→scan→approach→through→return→land), episode reset
- **Navigation**: Single-hoop round-trip coordination

### Data Flow Architecture

The complete system pipeline showing all components, data flows, and student interaction points:

```mermaid
graph TD
    %% Hardware Layer
    ZED["ZED Mini Camera<br/>RGB + Depth @ 60Hz"]
    PX4["Pixhawk 6C<br/>Flight Controller"]
    PI["Raspberry Pi 4B<br/>Compute Unit"]
    
    %% Data Sources
    RGB["/zed_mini/zed_node/rgb/image_rect_color<br/>sensor_msgs/Image"]
    DEPTH["/zed_mini/zed_node/depth/depth_registered<br/>sensor_msgs/Image"]
    POS["/fmu/out/vehicle_local_position<br/>px4_msgs/VehicleLocalPosition"]
    STATUS["/fmu/out/vehicle_status<br/>px4_msgs/VehicleStatus"]
    
    %% Processing Nodes
    VISION["Vision Processor Node<br/>vision_processor.py<br/>YOLO11 + Depth Integration"]
    RL["RL Agent Node<br/>p3o_agent.py<br/>P3O Algorithm"]
REWARD["Reward Calculator Node<br/>reward_calculator.py<br/>Multi-Component Rewards"]
COURSE["Course Manager Node<br/>course_manager.py<br/>Navigation Logic"]
    
    %% Processed Data Topics
    VF["/deepflyer/vision_features<br/>Custom VisionFeatures.msg<br/>hoop_detected, distance, alignment"]
    RLA["/deepflyer/rl_action<br/>Custom RLAction.msg<br/>lateral_cmd, vertical_cmd, speed_cmd"]
    RF["/deepflyer/reward_feedback<br/>Custom RewardFeedback.msg<br/>total_reward, breakdown"]
    CS["/deepflyer/course_state<br/>Custom CourseState.msg<br/>target_hoop_id, lap_number"]
    
    %% Control Outputs
    CMD["/fmu/in/trajectory_setpoint<br/>px4_msgs/TrajectorySetpoint"]
    MODE["/fmu/in/offboard_control_mode<br/>px4_msgs/OffboardControlMode"]
    
    %% Student Interface
    UI["Student Web Interface<br/>Reward Tuning<br/>Training Monitoring<br/>Performance Analytics"]
    
    %% Data Flow
    ZED --> RGB
    ZED --> DEPTH
    PX4 --> POS
    PX4 --> STATUS
    
    RGB --> VISION
    DEPTH --> VISION
    VISION --> VF
    
    VF --> RL
    POS --> RL
    STATUS --> RL
    CS --> RL
    RL --> RLA
    RL --> RF
    
    VF --> REWARD
    POS --> REWARD
    RLA --> REWARD
    CS --> REWARD
    REWARD --> RF
    
    POS --> COURSE
    VF --> COURSE
    COURSE --> CS
    
    RLA --> CMD
    RLA --> MODE
    CMD --> PX4
    MODE --> PX4
    
    RF --> UI
    CS --> UI
    VF --> UI
    
    %% Student Tuning
    UI -.-> REWARD
    UI -.-> RL
    
    %% Safety Override
    ESTOP["Emergency Stop<br/>Hardware Button"]
    ESTOP -.-> PX4
    ESTOP -.-> RL
    
    %% Training Loop
    RF -.-> RL
    
    %% Styling
    classDef hardware fill:#ffcccb
    classDef nodes fill:#cce5ff
    classDef topics fill:#d4edda
    classDef control fill:#fff3cd
    classDef interface fill:#e2e3e5
    
    class ZED,PX4,PI,ESTOP hardware
    class VISION,RL,REWARD,COURSE nodes
    class RGB,DEPTH,POS,STATUS,VF,RLA,RF,CS topics
    class CMD,MODE control
    class UI interface
```mermaid

**Key Architecture Benefits:**
- **Modularity**: Each component can be developed and tested independently
- **Student Safety**: All tunable parameters isolated from core flight safety systems  
- **Educational Focus**: Students see immediate impact of reward tuning on AI behavior
- **Real-time Feedback**: 50ms control loop with continuous reward calculation
- **Scalability**: Additional nodes can be added without disrupting core pipeline

**Key Innovation**: Students directly edit reward function code like AWS DeepRacer, watch AI learn from their code changes 