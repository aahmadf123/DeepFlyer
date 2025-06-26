# DeepFlyer Project Overview - Complete Technical Breakdown

## Project Vision
We're building an educational drone RL platform similar to AWS DeepRacer but for 3D aerial navigation. Students will train drones to fly through hoop courses using reinforcement learning without writing any code - just tuning reward parameters and watching the AI learn.

## Hardware Stack
- **Drone Frame**: Holybro S500 quadcopter
- **Flight Controller**: Pixhawk 6C running PX4 firmware
- **Compute**: Raspberry Pi 4B (4GB RAM minimum) 
- **Vision**: ZED Mini stereo camera for depth perception and hoop detection
- **Safety**: Hardware emergency stop, propeller guards, motion capture system for positioning

## ROS2 Communication Architecture

### Core Flight Topics
**Publisher: `/mavros/setpoint_velocity/cmd_vel`**
- Message Type: `geometry_msgs/TwistStamped`
- Fields: `linear.x/y/z` (velocity m/s), `angular.x/y/z` (rad/s)
- Purpose: Send velocity commands from RL agent to flight controller

**Subscriber: `/mavros/local_position/pose`**
- Message Type: `geometry_msgs/PoseStamped` 
- Fields: `position.x/y/z`, `orientation.x/y/z/w` (quaternion)
- Purpose: Get current drone position and orientation

**Subscriber: `/mavros/local_position/velocity_local`**
- Message Type: `geometry_msgs/TwistStamped`
- Fields: `linear.x/y/z`, `angular.x/y/z`
- Purpose: Current velocity feedback for RL state

### Vision System Topics
**Subscriber: `/zed_mini/zed_node/rgb/image_rect_color`**
- Message Type: `sensor_msgs/Image`
- Fields: `height`, `width`, `encoding`, `data[]`
- Purpose: RGB camera feed for hoop detection

**Subscriber: `/zed_mini/zed_node/depth/depth_registered`**
- Message Type: `sensor_msgs/Image` 
- Fields: `height`, `width`, `encoding`, `data[]` (depth values in mm)
- Purpose: Depth information for distance measurement

**Publisher: `/deepflyer/vision_features`**
- Message Type: Custom `VisionFeatures.msg`
- Fields: `hoop_detected` (bool), `hoop_center_u/v` (pixel coords), `hoop_distance` (meters), `hoop_alignment` (-1 to 1), `hoop_diameter_pixels`, `next_hoop_visible` (bool)
- Purpose: Processed vision data for RL agent

### RL Training Topics  
**Publisher: `/deepflyer/rl_action`**
- Message Type: Custom `RLAction.msg`
- Fields: `lateral_cmd`, `vertical_cmd`, `speed_cmd` (all -1 to 1)
- Purpose: RL agent actions to flight controller

**Publisher: `/deepflyer/reward_feedback`**
- Message Type: Custom `RewardFeedback.msg` 
- Fields: `total_reward`, `hoop_progress_reward`, `alignment_reward`, `collision_penalty`, `episode_time`, `lap_completed`
- Purpose: Reward components for training analysis

**Subscriber: `/deepflyer/course_state`**
- Message Type: Custom `CourseState.msg`
- Fields: `current_target_hoop_id`, `lap_number`, `hoops_completed`, `course_progress` (0-1)
- Purpose: Track progress through hoop course

## Vision Processing Pipeline

### Hoop Detection Strategy
- Use ZED Mini stereo vision for real-time hoop detection
- HSV color filtering for orange hoop identification (range: 5-25 hue, 100-255 saturation)
- Contour analysis with circularity filtering (minimum 0.3 circularity threshold)
- Distance measurement using stereo depth data
- Track multiple hoops simultaneously, prioritize by size/proximity

### Vision Features for RL
- **Hoop Alignment**: -1 (far left) to +1 (far right), 0 = centered in camera view
- **Hoop Distance**: Meters from drone to target hoop center  
- **Hoop Visibility**: Boolean flag if target hoop is detected
- **Hoop Size Ratio**: Detected hoop area / total image area
- **Next Hoop Preview**: Whether next hoop in sequence is visible

## RL System Design

### Algorithm: P3O (Procrastinated Policy Optimization)
- Advanced RL algorithm designed specifically for drone navigation
- Combines exploration efficiency with stable learning
- Handles continuous 3D action spaces well
- Updates policy with delayed optimization for better sample efficiency

### State Space (12-dimensional observation vector)
1. **Direction to target hoop** (3D normalized): x, y, z components
2. **Current velocity** (2D): forward and lateral velocity  
3. **Navigation metrics** (2D): distance to target, velocity alignment
4. **Vision features** (3D): hoop alignment, visual distance, visibility flag
5. **Course progress** (2D): progress within current lap, overall completion

### Action Space (3D continuous control)
- **Lateral Movement** (-1 to +1): left/right adjustment
- **Vertical Movement** (-1 to +1): up/down adjustment  
- **Speed Control** (-1 to +1): slow down/speed up

### Action Translation
- Actions get converted to velocity commands with safety limits
- Maximum horizontal speed: 0.8 m/s
- Maximum vertical speed: 0.4 m/s  
- Base forward speed: 0.6 m/s (adjusted by speed action)
- Dynamic speed reduction when close to hoops for precision

## Course Design & Navigation

### Hoop Course Layout
- Fixed 5-hoop circuit that drones navigate repeatedly  
- 3 complete laps required for course completion
- Hoop diameter: 0.8m (challenging but achievable)
- Flight altitude: 0.8m above ground level
- Course fits within 2.1m x 1.6m flight area

### Flight Trajectory Planning
- Sequential hoop navigation: Hoop 1 → 2 → 3 → 4 → 5 → back to 1
- Relative positioning system adapts to any lab environment
- Fixed altitude flight with 3D maneuvering for hoop passage
- Smooth trajectory optimization through reward shaping

## Reward Function Architecture

### Student-Tunable Parameters
- **Hoop Passage Reward**: +50 points for successful hoop traversal
- **Approach Reward**: +10 points for getting closer to target hoop
- **Center Bonus**: +20 points for passing through hoop center
- **Visual Alignment**: +5 points for keeping hoop centered in view
- **Lap Completion**: +100 points for completing full lap
- **Course Completion**: +500 points for finishing all 3 laps

### Penalty System
- **Hoop Miss**: -25 points for flying around instead of through hoop
- **Collision**: -100 points for hitting obstacles
- **Wrong Direction**: -2 points for flying away from target
- **Time Penalty**: -1 point for taking too long
- **Erratic Flight**: -3 points for jerky, unstable movements

### Safety Overrides (Non-tunable)
- **Boundary Violation**: -200 points for leaving safe flight area
- **Emergency Landing**: -500 points for emergency stop activation

## Training & Learning Parameters

### P3O Hyperparameters  
- **Learning Rate**: 3e-4 (tunable 1e-5 to 1e-2)
- **Clip Epsilon**: 0.2 (tunable 0.1 to 0.3)
- **Batch Size**: 64 (options: 32, 64, 128, 256)
- **Discount Factor**: 0.99 (tunable 0.90 to 0.999)
- **Entropy Coefficient**: 0.01 (tunable 0.0 to 0.1)

### Training Episodes
- **Max Episodes**: 1000 per training session
- **Max Steps per Episode**: 500 steps (25 seconds real-time)
- **Episode Success**: Complete course or reach time limit
- **Evaluation Frequency**: Every 50 episodes with 5-episode average

## Implementation Timeline

### Phase 1: Core Infrastructure (Weeks 1-4)
- ROS2 workspace setup with MAVROS integration
- Basic P3O agent skeleton with dummy actions
- Gazebo simulation environment with Holybro S500 model
- Simple reward system (distance-based navigation)

### Phase 2: Vision Integration (Weeks 5-8)  
- ZED Mini camera integration and calibration
- Hoop detection pipeline with OpenCV
- Vision-based state representation for RL
- Enhanced reward functions with visual feedback

### Phase 3: Advanced Features (Weeks 9-12)
- Multi-lap course completion system
- Student-facing parameter tuning interface
- Training visualization and progress tracking
- Sim-to-real transfer preparation

## Safety & Testing Protocol
- All flights within netted 2.5m x 2.0m x 1.5m lab space
- Hardware emergency stop accessible at all times
- Graduated testing: simulation → tethered → free flight
- Automated boundary checking and emergency landing
- Propeller guards mandatory for all real-world testing

This is our complete technical roadmap. The key innovation is making RL accessible to students without coding - they just tune reward parameters and watch the drone learn to navigate. The P3O algorithm handles the complex 3D flight dynamics while the ZED Mini provides the visual intelligence for hoop detection. 