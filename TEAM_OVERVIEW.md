# DeepFlyer Technical Reference

**Project**: Educational drone RL platform - students tune rewards, watch AI learn hoop navigation  
**Algorithm**: P3O (Procrastinated Policy Optimization)  

## ROS2 Topics Reference

### Flight Control
| Topic | Type | Direction | Fields | Purpose |
|-------|------|-----------|--------|---------|
| `/mavros/setpoint_velocity/cmd_vel` | `geometry_msgs/TwistStamped` | PUBLISH | `linear.x/y/z`, `angular.x/y/z` | Velocity commands to flight controller |
| `/mavros/local_position/pose` | `geometry_msgs/PoseStamped` | SUBSCRIBE | `position.x/y/z`, `orientation.x/y/z/w` | Current position/orientation |
| `/mavros/local_position/velocity_local` | `geometry_msgs/TwistStamped` | SUBSCRIBE | `linear.x/y/z`, `angular.x/y/z` | Current velocity feedback |
| `/mavros/state` | `mavros_msgs/State` | SUBSCRIBE | `connected`, `armed`, `guided`, `mode` | Flight controller status |

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
| `/deepflyer/rl_action` | Custom `RLAction.msg` | PUBLISH | `lateral_cmd`, `vertical_cmd`, `speed_cmd` | RL actions (-1 to 1) |
| `/deepflyer/reward_feedback` | Custom `RewardFeedback.msg` | PUBLISH | See below | Reward breakdown |
| `/deepflyer/course_state` | Custom `CourseState.msg` | SUBSCRIBE | See below | Course progress |

**RLAction.msg Fields:**
- `lateral_cmd` (float32) - left/right (-1 to 1)
- `vertical_cmd` (float32) - up/down (-1 to 1)
- `speed_cmd` (float32) - slow/fast (-1 to 1)

**RewardFeedback.msg Fields:**
- `total_reward` (float32)
- `hoop_progress_reward` (float32)
- `alignment_reward` (float32) 
- `collision_penalty` (float32)
- `episode_time` (float32)
- `lap_completed` (bool)

**CourseState.msg Fields:**
- `current_target_hoop_id` (int32)
- `lap_number` (int32)
- `hoops_completed` (int32)
- `course_progress` (float32) - 0 to 1

## Vision Processing

### ZED Mini Camera Settings
- **Resolution**: 1280x720 (HD720)
- **FPS**: 60Hz
- **Depth Mode**: PERFORMANCE 
- **Coordinate System**: RIGHT_HANDED_Z_UP

### Hoop Detection Parameters
- **HSV Color Range**: Hue 5-25, Saturation 100-255, Value 100-255
- **Contour Area**: Min 500px, Max 50,000px
- **Circularity Threshold**: >0.3
- **Distance Measurement**: ZED stereo depth (mm → meters)
- **Detection Priority**: Largest contour = current target

### Vision Output Values
- **Hoop Alignment**: -1.0 (left) → 0.0 (center) → +1.0 (right)
- **Distance Range**: 0.5m to 5.0m effective range
- **Pixel Coordinates**: u,v in image frame (0-1280, 0-720)
- **Area Ratio**: hoop_pixels / total_image_pixels

## P3O RL System

### State Space (12D Vector)
| Index | Component | Range | Description |
|-------|-----------|-------|-------------|
| 0-2 | Direction to hoop | -1 to 1 | Normalized x,y,z direction vector |
| 3-4 | Current velocity | -1 to 1 | Forward, lateral velocity |
| 5-6 | Navigation metrics | 0 to 1 | Distance to target, velocity alignment |
| 7-9 | Vision features | -1 to 1 | Hoop alignment, visual distance, visibility |
| 10-11 | Course progress | 0 to 1 | Lap progress, overall completion |

### Action Space (3D Continuous)
| Action | Range | Control | Max Speed |
|--------|-------|---------|-----------|
| `lateral_cmd` | -1 to 1 | Left/Right | 0.8 m/s |
| `vertical_cmd` | -1 to 1 | Up/Down | 0.4 m/s |
| `speed_cmd` | -1 to 1 | Slow/Fast | 0.6 m/s base |

### P3O Hyperparameters
- **Learning Rate**: 3e-4 (range: 1e-5 to 1e-2)
- **Clip Epsilon**: 0.2 (range: 0.1 to 0.3)
- **Batch Size**: 64 (options: 32, 64, 128, 256)
- **Discount Factor**: 0.99 (range: 0.90 to 0.999)
- **Entropy Coefficient**: 0.01 (range: 0.0 to 0.1)

## Course Layout

### Physical Setup
- **Course Size**: 2.1m × 1.6m flight area
- **Flight Height**: 0.8m above ground
- **Hoop Count**: 5 hoops per circuit
- **Hoop Diameter**: 0.8m
- **Lap Requirement**: 3 complete circuits
- **Navigation**: Hoop 1 → 2 → 3 → 4 → 5 → repeat

## Reward System

### Positive Rewards (Student Tunable)
| Event | Default Points | Range | Description |
|-------|----------------|-------|-------------|
| Hoop Passage | +50 | 25-100 | Successfully through hoop |
| Approach Target | +10 | 5-20 | Getting closer to target |
| Center Bonus | +20 | 10-40 | Precise center passage |
| Visual Alignment | +5 | 1-10 | Hoop centered in view |
| Lap Complete | +100 | 50-200 | Full circuit finished |
| Course Complete | +500 | 200-1000 | All 3 laps done |

### Penalties (Student Tunable)
| Event | Default Points | Range | Description |
|-------|----------------|-------|-------------|
| Hoop Miss | -25 | -10 to -50 | Flying around hoop |
| Collision | -100 | -50 to -200 | Hitting obstacles |
| Wrong Direction | -2 | -1 to -5 | Flying away from target |
| Time Penalty | -1 | -0.5 to -2 | Taking too long |
| Erratic Flight | -3 | -1 to -10 | Jerky movements |

### Safety Overrides (Non-tunable)
| Event | Points | Trigger |
|-------|--------|---------|
| Boundary Violation | -200 | Outside flight area |
| Emergency Stop | -500 | Hardware stop pressed |

## Training Configuration

### Episode Parameters
| Parameter | Value | Unit |
|-----------|-------|------|
| Max Episodes | 1000 | per training session |
| Max Steps | 500 | per episode (25 sec) |
| Evaluation Freq | 50 | episodes |
| Success Criteria | Course completion | or time limit |

### Safety Limits
| Boundary | Dimension | Action |
|----------|-----------|--------|
| Flight Area | 2.1m × 1.6m × 1.5m | Auto-land if exceeded |
| Max Speed | 0.8 m/s horizontal | Velocity clamping |
| Max Speed | 0.4 m/s vertical | Velocity clamping |
| Emergency Stop | Hardware button | Immediate motor kill |

## Development Phases

### Phase 1 (Weeks 1-4): Core Infrastructure
- ROS2 + MAVROS setup
- P3O agent skeleton  
- Gazebo simulation
- Basic rewards

### Phase 2 (Weeks 5-8): Vision Integration
- ZED Mini integration
- Hoop detection pipeline
- Vision-based RL state
- Enhanced rewards

### Phase 3 (Weeks 9-12): Advanced Features  
- Multi-lap system
- Student interface
- Training visualization
- Sim-to-real prep

**Key Innovation**: Students tune reward parameters, watch AI learn - no coding required 