#!/usr/bin/env python3
"""
DeepFlyer PX4-ROS-COM Training Node
Educational drone RL training node using direct PX4 communication
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import torch
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from threading import Lock

# PX4-ROS-COM message types
try:
    from px4_msgs.msg import (
        VehicleLocalPosition, VehicleAttitude, VehicleStatus,
        TrajectorySetpoint, OffboardControlMode, VehicleCommand,
        BatteryStatus, VehicleControlMode
    )
    PX4_MSGS_AVAILABLE = True
except ImportError:
    PX4_MSGS_AVAILABLE = False
    print("PX4 messages not available. Install px4_msgs package.")

from .direct_control_agent import DirectControlAgent
from .config import DeepFlyerConfig, get_course_layout, get_p3o_config
from .env.vision_processor import create_yolo11_processor, VisionFeatures
from .env.safety_layer import SafetyLayer


class DeepFlyerTrainingNode(Node):
    """
    Complete PX4-ROS-COM training node for DeepFlyer educational platform
    
    Integrates:
    - P3O RL agent
    - YOLO11 vision processing  
    - PX4-ROS-COM direct communication
    - Hoop navigation logic
    - Student-configurable rewards
    """
    
    def __init__(self, 
                 node_name: str = "deepflyer_training",
                 spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.8),
                 yolo_model_size: str = "n",
                 enable_vision: bool = True,
                 enable_safety: bool = True,
                 **kwargs):
        """
        Initialize DeepFlyer training node
        
        Args:
            node_name: ROS2 node name
            spawn_position: Initial drone spawn position
            yolo_model_size: YOLO11 model size ('n', 's', 'm', 'l', 'x')
            enable_vision: Enable YOLO11 vision processing
            enable_safety: Enable safety layer
        """
        super().__init__(node_name)
        
        # Load configuration
        self.config = DeepFlyerConfig()
        self.spawn_position = spawn_position
        
        # Generate course layout
        self.course_hoops = get_course_layout(spawn_position)
        self.current_target_hoop = 0
        self.lap_number = 1
        self.hoops_completed = 0
        
        # Initialize RL agent
        self._setup_rl_agent()
        
        # Initialize vision processor if enabled
        self.enable_vision = enable_vision
        if enable_vision:
            self.vision_processor = create_yolo11_processor(
                model_size=yolo_model_size,
                confidence=self.config.VISION_CONFIG['confidence_threshold']
            )
            self.last_vision_features: Optional[VisionFeatures] = None
        else:
            self.vision_processor = None
            self.last_vision_features = None
        
        # Initialize safety layer if enabled
        self.safety_layer = SafetyLayer(self.config) if enable_safety else None
        
        # State tracking
        self.current_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'attitude': np.zeros(4),  # quaternion [w, x, y, z]
            'armed': False,
            'offboard_active': False,
            'battery_remaining': 1.0
        }
        
        # Episode tracking
        self.episode_active = False
        self.episode_start_time = 0.0
        self.episode_step_count = 0
        self.total_reward = 0.0
        
        # Experience tracking for RL
        self.last_observation = None
        self.last_action = None
        self.training_enabled = False
        
        # Thread safety
        self.state_lock = Lock()
        
        # Setup PX4-ROS-COM communication
        if PX4_MSGS_AVAILABLE:
            self._setup_px4_communication()
        else:
            self.get_logger().error("PX4 messages not available. Cannot initialize PX4 communication.")
            
        # Performance tracking
        self.performance_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'total_steps': 0,
            'avg_reward': 0.0,
            'hoop_completion_rate': 0.0
        }
        
        self.get_logger().info(f"DeepFlyer training node initialized with {len(self.course_hoops)} hoops")
    
    def _setup_rl_agent(self):
        """Initialize P3O RL agent with proper spaces"""
        import gymnasium as gym
        
        # Create observation space (12-dimensional)
        obs_config = self.config.OBSERVATION_CONFIG
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_config['dimension'],),
            dtype=np.float32
        )
        
        # Create action space (3-dimensional)
        action_config = self.config.ACTION_CONFIG
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_config['dimension'],),
            dtype=np.float32
        )
        
        # Initialize P3O agent
        p3o_config = get_p3o_config()
        self.agent = DirectControlAgent(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device="auto",
            **p3o_config
        )
        
        self.get_logger().info("P3O agent initialized")
    
    def _setup_px4_communication(self):
        """Setup PX4-ROS-COM publishers and subscribers"""
        
        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers (commands TO PX4)
        px4_config = self.config.PX4_CONFIG
        
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            px4_config['output_topics']['trajectory_setpoint'],
            qos_profile
        )
        
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            px4_config['output_topics']['offboard_control_mode'],
            qos_profile
        )
        
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            px4_config['output_topics']['vehicle_command'],
            qos_profile
        )
        
        # Subscribers (data FROM PX4)
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            px4_config['input_topics']['vehicle_local_position'],
            self._position_callback,
            qos_profile
        )
        
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            px4_config['input_topics']['vehicle_attitude'],
            self._attitude_callback,
            qos_profile
        )
        
        self.status_sub = self.create_subscription(
            VehicleStatus,
            px4_config['input_topics']['vehicle_status'],
            self._status_callback,
            qos_profile
        )
        
        self.battery_sub = self.create_subscription(
            BatteryStatus,
            px4_config['input_topics']['battery_status'],
            self._battery_callback,
            qos_profile
        )
        
        # Control timer
        control_period = 1.0 / self.config.PX4_CONFIG['control_frequency']
        self.control_timer = self.create_timer(control_period, self._control_loop)
        
        # Offboard mode timer
        offboard_period = 1.0 / self.config.PX4_CONFIG['offboard_mode_frequency']
        self.offboard_timer = self.create_timer(offboard_period, self._send_offboard_mode)
        
        self.get_logger().info("PX4-ROS-COM communication setup complete")
    
    def _position_callback(self, msg: VehicleLocalPosition):
        """Process position messages from PX4"""
        with self.state_lock:
            self.current_state['position'] = np.array([msg.x, msg.y, -msg.z])  # Convert NED to ENU
            self.current_state['velocity'] = np.array([msg.vx, msg.vy, -msg.vz])
    
    def _attitude_callback(self, msg: VehicleAttitude):
        """Process attitude messages from PX4"""
        with self.state_lock:
            self.current_state['attitude'] = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
    
    def _status_callback(self, msg: VehicleStatus):
        """Process status messages from PX4"""
        with self.state_lock:
            self.current_state['armed'] = (msg.arming_state == 2)  # ARMING_STATE_ARMED
            self.current_state['offboard_active'] = (msg.nav_state == 14)  # NAVIGATION_STATE_OFFBOARD
    
    def _battery_callback(self, msg: BatteryStatus):
        """Process battery messages from PX4"""
        with self.state_lock:
            self.current_state['battery_remaining'] = msg.remaining
    
    def _send_offboard_mode(self):
        """Send offboard control mode message"""
        if hasattr(self, 'offboard_mode_pub'):
            msg = OffboardControlMode()
            msg.timestamp = int(self._get_px4_timestamp())
            msg.position = True
            msg.velocity = True
            msg.acceleration = False
            msg.attitude = False
            msg.body_rate = False
            
            self.offboard_mode_pub.publish(msg)
    
    def _control_loop(self):
        """Main RL control loop"""
        if not self.episode_active:
            return
        
        try:
            # Process vision if enabled
            if self.enable_vision:
                self._process_vision_frame()
            
            # Get current observation
            observation = self._get_observation()
            
            # Get action from RL agent
            action, _ = self.agent.predict(observation, deterministic=not self.training_enabled)
            
            # Apply safety constraints if enabled
            if self.safety_layer:
                action = self.safety_layer.apply_safety_constraints(action, self.current_state)
            
            # Convert RL action to PX4 command and send
            self._execute_action(action)
            
            # Handle RL training if enabled
            if self.training_enabled:
                self._handle_rl_step(observation, action)
            
            # Update hoop navigation logic
            self._update_hoop_navigation()
            
            # Check episode termination
            self._check_episode_termination()
            
            # Update tracking
            self.episode_step_count += 1
            self.performance_stats['total_steps'] += 1
            
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
    
    def _process_vision_frame(self):
        """Process current camera frame with YOLO11"""
        try:
            # Get camera images (placeholder - would be implemented with actual camera)
            rgb_image, depth_image = self._get_camera_images()
            
            if rgb_image is not None and depth_image is not None:
                # Process with YOLO11
                self.last_vision_features = self.vision_processor.process_frame(rgb_image, depth_image)
            
        except Exception as e:
            self.get_logger().warning(f"Vision processing failed: {e}")
            self.last_vision_features = None
    
    def _get_camera_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get RGB and depth images from ZED Mini camera
        TODO: Implement actual ZED Mini interface
        """
        # Placeholder - would be replaced with actual ZED Mini integration
        return None, None
    
    def _get_observation(self) -> np.ndarray:
        """Get current RL observation (12-dimensional)"""
        
        with self.state_lock:
            current_pos = self.current_state['position'].copy()
            current_vel = self.current_state['velocity'].copy()
        
        # Get current target hoop
        if self.current_target_hoop < len(self.course_hoops):
            target_hoop = self.course_hoops[self.current_target_hoop]
            target_pos = np.array(target_hoop['position'])
        else:
            target_pos = current_pos  # Fallback
        
        # Calculate direction to hoop (normalized)
        direction_to_hoop = target_pos - current_pos
        distance_to_hoop = np.linalg.norm(direction_to_hoop)
        direction_to_hoop = direction_to_hoop / max(distance_to_hoop, 0.01)  # Normalize
        
        # Velocity alignment with target direction
        velocity_magnitude = np.linalg.norm(current_vel)
        velocity_alignment = np.dot(current_vel, direction_to_hoop) / max(velocity_magnitude, 0.01)
        
        # Vision features
        if self.last_vision_features and self.last_vision_features.primary_hoop:
            hoop_alignment = self.last_vision_features.hoop_alignment
            visual_distance = min(self.last_vision_features.primary_hoop.distance, 5.0) / 5.0
            hoop_visible = 1.0
        else:
            hoop_alignment = 0.0
            visual_distance = 1.0  # Max normalized distance
            hoop_visible = 0.0
        
        # Course progress
        progress_in_lap = self.current_target_hoop / max(len(self.course_hoops), 1)
        overall_progress = (self.lap_number - 1) / self.config.HOOP_CONFIG['num_laps']
        
        # Construct 12-dimensional observation
        observation = np.array([
            # Direction to hoop (3 dimensions)
            np.clip(direction_to_hoop[0], -1.0, 1.0),  # 0: X direction
            np.clip(direction_to_hoop[1], -1.0, 1.0),  # 1: Y direction  
            np.clip(direction_to_hoop[2], -1.0, 1.0),  # 2: Z direction
            
            # Current velocity (2 dimensions)
            np.clip(current_vel[0] / 2.0, -1.0, 1.0), # 3: Forward velocity
            np.clip(current_vel[1] / 2.0, -1.0, 1.0), # 4: Lateral velocity
            
            # Navigation metrics (2 dimensions)
            np.clip(distance_to_hoop / 5.0, 0.0, 1.0), # 5: Distance to target
            np.clip(velocity_alignment, -1.0, 1.0),     # 6: Velocity alignment
            
            # Vision features (3 dimensions)
            np.clip(hoop_alignment, -1.0, 1.0),        # 7: Visual alignment
            visual_distance,                            # 8: Visual distance
            hoop_visible,                              # 9: Hoop visibility
            
            # Course progress (2 dimensions)
            progress_in_lap,                           # 10: Lap progress
            overall_progress                           # 11: Overall progress
        ], dtype=np.float32)
        
        return observation
    
    def _execute_action(self, action: np.ndarray):
        """Convert RL action to PX4 trajectory setpoint and send"""
        
        # Extract action components
        lateral_cmd = np.clip(action[0], -1.0, 1.0)
        vertical_cmd = np.clip(action[1], -1.0, 1.0)  
        speed_cmd = np.clip(action[2], -1.0, 1.0)
        
        # Convert to velocity commands
        action_config = self.config.ACTION_CONFIG
        
        lateral_velocity = lateral_cmd * action_config['components']['lateral_cmd']['max_speed']
        vertical_velocity = vertical_cmd * action_config['components']['vertical_cmd']['max_speed']
        forward_velocity = action_config['components']['speed_cmd']['base_speed'] * (1.0 + 0.5 * speed_cmd)
        
        # Create trajectory setpoint message
        msg = TrajectorySetpoint()
        msg.timestamp = self._get_px4_timestamp()
        
        # Position control (only altitude)
        msg.position[0] = float('nan')  # Let velocity control handle x
        msg.position[1] = float('nan')  # Let velocity control handle y
        msg.position[2] = -self.config.HOOP_CONFIG['flight_altitude']  # NED coordinate
        
        # Velocity control
        msg.velocity[0] = forward_velocity   # Forward (North in NED)
        msg.velocity[1] = lateral_velocity   # Right (East in NED)
        msg.velocity[2] = -vertical_velocity # Down (negative in NED)
        
        # Acceleration (let PX4 handle)
        msg.acceleration[0] = float('nan')
        msg.acceleration[1] = float('nan')
        msg.acceleration[2] = float('nan')
        
        # Yaw control (face forward)
        msg.yaw = 0.0
        msg.yawspeed = 0.0
        
        # Send command
        if hasattr(self, 'trajectory_pub'):
            self.trajectory_pub.publish(msg)
    
    def _handle_rl_step(self, observation: np.ndarray, action: np.ndarray):
        """Handle RL training step"""
        
        if self.last_observation is not None and self.last_action is not None:
            # Calculate reward
            reward = self._calculate_reward(action)
            self.total_reward += reward
            
            # Add experience to agent
            done = not self.episode_active
            self.agent.add_to_buffer(
                self.last_observation,
                self.last_action, 
                observation,
                reward,
                done
            )
            
            # Trigger learning periodically
            if self.episode_step_count % 10 == 0:  # Learn every 10 steps
                metrics = self.agent.learn()
                if metrics:
                    self.get_logger().info(f"Learning metrics: {metrics}")
        
        # Update for next step
        self.last_observation = observation.copy()
        self.last_action = action.copy()
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward using student-configurable reward function"""
        
        try:
            from .rewards.rewards import reward_function
            
            # Prepare parameters for reward function
            params = self._prepare_reward_params(action)
            
            # Calculate reward
            reward = reward_function(params)
            
            return float(reward)
            
        except Exception as e:
            self.get_logger().warning(f"Error in reward calculation: {e}")
            return 0.0
    
    def _prepare_reward_params(self, action: np.ndarray) -> Dict[str, Any]:
        """Prepare parameters for the reward function"""
        
        # Get current state
        with self.state_lock:
            current_pos = self.current_state['position'].copy()
            current_vel = self.current_state['velocity'].copy()
        
        # Get target hoop
        if self.current_target_hoop < len(self.course_hoops):
            target_hoop = self.course_hoops[self.current_target_hoop]
            target_pos = np.array(target_hoop['position'])
            distance_to_hoop = np.linalg.norm(current_pos - target_pos)
        else:
            distance_to_hoop = 0.0
        
        # Vision data
        hoop_detected = self.last_vision_features is not None and self.last_vision_features.primary_hoop is not None
        hoop_distance = self.last_vision_features.primary_hoop.distance if hoop_detected else float('inf')
        hoop_alignment = self.last_vision_features.hoop_alignment if hoop_detected else 0.0
        
        # Check for hoop passage (simple distance-based check)
        hoop_passed = distance_to_hoop < (self.config.HOOP_CONFIG['diameter'] / 2)
        
        # Other parameters
        making_progress = np.dot(current_vel, target_pos - current_pos) > 0 if distance_to_hoop > 0.1 else False
        
        return {
            'hoop_detected': hoop_detected,
            'hoop_distance': hoop_distance,
            'hoop_alignment': hoop_alignment,
            'approaching_hoop': distance_to_hoop < 2.0 and making_progress,
            'hoop_passed': hoop_passed,
            'center_passage': hoop_passed and abs(hoop_alignment) < 0.1,
            'making_progress': making_progress,
            'lap_completed': False,  # Will be updated by hoop navigation logic
            'course_completed': False,  # Will be updated by hoop navigation logic
            'missed_hoop': False,  # Will be updated by hoop tracking logic
            'collision': False,  # Will be updated by safety layer
            'slow_progress': self.episode_step_count > self.config.TRAINING_CONFIG['max_steps_per_episode'] * 0.8,
            'out_of_bounds': self._check_boundaries()
        }
    
    def _check_boundaries(self) -> bool:
        """Check if drone is outside safe flight boundaries"""
        with self.state_lock:
            pos = self.current_state['position']
        
        bounds = self.config.COURSE_DIMENSIONS
        buffer = bounds['safety_buffer']
        
        return (pos[0] < -buffer or pos[0] > bounds['length'] + buffer or
                pos[1] < -buffer or pos[1] > bounds['width'] + buffer or
                pos[2] < 0.2 or pos[2] > bounds['height'])
    
    def _update_hoop_navigation(self):
        """Update hoop navigation logic"""
        
        if self.current_target_hoop >= len(self.course_hoops):
            return
        
        # Get current target hoop
        target_hoop = self.course_hoops[self.current_target_hoop]
        target_pos = np.array(target_hoop['position'])
        
        with self.state_lock:
            current_pos = self.current_state['position'].copy()
        
        # Check if drone passed through hoop
        distance_to_hoop = np.linalg.norm(current_pos - target_pos)
        hoop_radius = self.config.HOOP_CONFIG['diameter'] / 2
        
        if distance_to_hoop < hoop_radius:
            # Hoop completed!
            self.hoops_completed += 1
            self.current_target_hoop += 1
            
            self.get_logger().info(f"Hoop {target_hoop['id']} completed! Total: {self.hoops_completed}")
            
            # Check if lap completed
            if self.current_target_hoop >= len(self.course_hoops):
                self.lap_number += 1
                self.current_target_hoop = 0
                self.get_logger().info(f"Lap {self.lap_number - 1} completed!")
    
    def _check_episode_termination(self):
        """Check if episode should terminate"""
        
        # Success: completed all laps
        total_hoops_needed = len(self.course_hoops) * self.config.HOOP_CONFIG['num_laps']
        if self.hoops_completed >= total_hoops_needed:
            self._end_episode(success=True)
            return
        
        # Failure conditions
        if self._check_boundaries():
            self.get_logger().warning("Episode terminated: out of bounds")
            self._end_episode(success=False)
            return
            
        if self.current_state['battery_remaining'] < 0.2:
            self.get_logger().warning("Episode terminated: low battery")
            self._end_episode(success=False)
            return
        
        # Truncation: maximum episode length
        if self.episode_step_count >= self.config.TRAINING_CONFIG['max_steps_per_episode']:
            self.get_logger().info("Episode terminated: max steps reached")
            self._end_episode(success=False)
            return
    
    def _end_episode(self, success: bool):
        """End current episode"""
        
        if not self.episode_active:
            return
        
        self.episode_active = False
        episode_time = time.time() - self.episode_start_time
        
        # Update performance stats
        self.performance_stats['total_episodes'] += 1
        if success:
            self.performance_stats['successful_episodes'] += 1
        
        # Update averages
        total_episodes = self.performance_stats['total_episodes']
        self.performance_stats['avg_reward'] = (
            (self.performance_stats['avg_reward'] * (total_episodes - 1) + self.total_reward) / total_episodes
        )
        
        completion_rate = self.hoops_completed / (len(self.course_hoops) * self.config.HOOP_CONFIG['num_laps'])
        self.performance_stats['hoop_completion_rate'] = (
            (self.performance_stats['hoop_completion_rate'] * (total_episodes - 1) + completion_rate) / total_episodes
        )
        
        self.get_logger().info(
            f"Episode ended: {'SUCCESS' if success else 'FAILURE'} | "
            f"Time: {episode_time:.1f}s | Steps: {self.episode_step_count} | "
            f"Hoops: {self.hoops_completed} | Reward: {self.total_reward:.1f}"
        )
    
    def _get_px4_timestamp(self) -> int:
        """Get current timestamp in PX4 format (microseconds)"""
        return int(time.time() * 1e6)
    
    # Public interface methods
    
    def start_episode(self):
        """Start a new training episode"""
        self.episode_active = True
        self.episode_start_time = time.time()
        self.episode_step_count = 0
        self.total_reward = 0.0
        self.current_target_hoop = 0
        self.lap_number = 1
        self.hoops_completed = 0
        
        # Reset RL tracking
        self.last_observation = None
        self.last_action = None
        
        self.get_logger().info("New episode started")
    
    def stop_episode(self):
        """Stop current episode"""
        if self.episode_active:
            self._end_episode(success=False)
    
    def enable_training(self, enabled: bool):
        """Enable or disable RL training"""
        self.training_enabled = enabled
        self.get_logger().info(f"Training {'enabled' if enabled else 'disabled'}")
    
    def save_model(self, filepath: str):
        """Save RL agent model"""
        state_dict = self.agent.get_model_state()
        torch.save(state_dict, filepath)
        self.get_logger().info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load RL agent model"""
        state_dict = torch.load(filepath)
        self.agent.load_model_state(state_dict)
        self.get_logger().info(f"Model loaded from {filepath}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()


def main(args=None):
    """Run the DeepFlyer training node"""
    rclpy.init(args=args)
    
    try:
        node = DeepFlyerTrainingNode()
        
        # Enable training
        node.enable_training(True)
        
        # Start first episode
        node.start_episode()
        
        # Run the node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 