"""
DeepFlyer PX4-ROS-COM Environment with YOLO11 Vision Processing
Educational drone RL environment for hoop navigation using direct PX4 communication
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
from gymnasium import spaces
import logging
import time
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
    logging.warning("PX4 messages not available. Install px4_msgs package.")

from .vision_processor import YOLO11VisionProcessor, VisionFeatures, create_yolo11_processor
from .safety_layer import SafetyLayer
from ..config import DeepFlyerConfig, get_course_layout
from ..logger import Logger

logger = logging.getLogger(__name__)


class DeepFlyerHoopNavigationEnv(Node, gym.Env):
    """
    Educational drone RL environment for hoop navigation using PX4-ROS-COM
    Replaces MAVROS with direct PX4 communication for better performance
    """
    
    def __init__(self, 
                 node_name: str = "deepflyer_env",
                 spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.8),
                 yolo_model_size: str = "n",
                 confidence_threshold: float = 0.3,
                 use_custom_hoop_model: bool = False,
                 custom_model_path: Optional[str] = None,
                 enable_safety_layer: bool = True,
                 **kwargs):
        """
        Initialize DeepFlyer hoop navigation environment
        
        Args:
            node_name: ROS2 node name
            spawn_position: Initial drone position (x, y, z)
            yolo_model_size: YOLO11 model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Minimum confidence for hoop detections
            use_custom_hoop_model: Whether to use custom-trained hoop model
            custom_model_path: Path to custom-trained YOLO11 model
            enable_safety_layer: Enable safety constraints
            **kwargs: Additional arguments
        """
        
        # Initialize ROS2 node
        super().__init__(node_name)
        
        # Load configuration
        self.config = DeepFlyerConfig()
        self.spawn_position = spawn_position
        
        # Generate course layout
        self.course_hoops = get_course_layout(spawn_position)
        self.current_target_hoop = 0
        self.lap_number = 1
        self.hoops_completed = 0
        
        # Initialize YOLO11 vision processor
        self.vision_processor = create_yolo11_processor(
            model_size=yolo_model_size,
            confidence=confidence_threshold
        )
        
        # Setup custom hoop model if specified
        if use_custom_hoop_model and custom_model_path:
            try:
                self.vision_processor.setup_custom_hoop_detection(custom_model_path)
                self.get_logger().info(f"Using custom hoop model: {custom_model_path}")
            except Exception as e:
                self.get_logger().warning(f"Failed to load custom model, using default: {e}")
        
        # Initialize safety layer
        self.safety_layer = SafetyLayer(self.config) if enable_safety_layer else None
        
        # State variables
        self.current_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'attitude': np.zeros(4),  # quaternion [w, x, y, z]
            'armed': False,
            'offboard_active': False,
            'battery_remaining': 1.0
        }
        
        # Vision data
        self.last_vision_features: Optional[VisionFeatures] = None
        
        # Episode tracking
        self.episode_start_time = 0.0
        self.episode_step_count = 0
        self.total_reward = 0.0
        
        # Thread safety
        self.state_lock = Lock()
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Setup PX4-ROS-COM communication
        if PX4_MSGS_AVAILABLE:
            self._setup_px4_communication()
        else:
            self.get_logger().error("PX4 messages not available. Cannot initialize PX4 communication.")
            
        # Performance monitoring
        self.performance_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'average_episode_time': 0.0,
            'hoop_completion_rate': 0.0
        }
        
        self.get_logger().info(f"DeepFlyer environment initialized with {len(self.course_hoops)} hoops")
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        
        # Observation space: 12-dimensional as per documentation
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.config.OBSERVATION_CONFIG['dimension'],), 
            dtype=np.float32
        )
        
        # Action space: 3-dimensional velocity commands
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.config.ACTION_CONFIG['dimension'],), 
            dtype=np.float32
        )
    
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
    
    def _control_loop(self):
        """Main control loop - placeholder for action execution"""
        # This will be called by the step() method when actions are received
        pass
    
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
    
    def _get_px4_timestamp(self) -> int:
        """Get current timestamp in PX4 format (microseconds)"""
        return int(time.time() * 1e6)
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode"""
        
        # Reset episode tracking
        self.current_target_hoop = 0
        self.lap_number = 1
        self.hoops_completed = 0
        self.episode_start_time = time.time()
        self.episode_step_count = 0
        self.total_reward = 0.0
        
        # Reset vision features
        self.last_vision_features = None
        
        # Update performance stats
        self.performance_stats['total_episodes'] += 1
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'course_hoops': self.course_hoops,
            'current_target_hoop': self.current_target_hoop,
            'lap_number': self.lap_number,
            'episode_start_time': self.episode_start_time
        }
        
        self.get_logger().info(f"Episode reset - Target hoop: {self.current_target_hoop + 1}")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        
        self.episode_step_count += 1
        
        # Process vision frame
        self._process_vision_frame()
        
        # Apply safety constraints if enabled
        if self.safety_layer:
            action = self.safety_layer.apply_safety_constraints(action, self.current_state)
        
        # Convert RL action to PX4 command and send
        self._execute_action(action)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.total_reward += reward
        
        # Check episode termination
        terminated, truncated = self._check_episode_end()
        
        # Prepare info dict
        info = {
            'current_target_hoop': self.current_target_hoop,
            'lap_number': self.lap_number,
            'hoops_completed': self.hoops_completed,
            'episode_time': time.time() - self.episode_start_time,
            'total_reward': self.total_reward,
            'vision_processing_time': self.last_vision_features.processing_time_ms if self.last_vision_features else 0.0
        }
        
        # Update episode completion stats
        if terminated and self.hoops_completed >= len(self.course_hoops) * self.config.HOOP_CONFIG['num_laps']:
            self.performance_stats['successful_episodes'] += 1
        
        return observation, reward, terminated, truncated, info
    
    def _process_vision_frame(self):
        """Process current camera frame with YOLO11"""
        try:
            # Get camera images (this would be implemented based on your camera interface)
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
        progress_in_lap = self.current_target_hoop / len(self.course_hoops)
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
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward using the student-configurable reward function"""
        
        # Import the student reward function
        try:
            from ..rewards.rewards import reward_function
            
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
            'lap_completed': False,  # Will be updated by episode logic
            'course_completed': False,  # Will be updated by episode logic
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
    
    def _check_episode_end(self) -> Tuple[bool, bool]:
        """Check if episode should terminate"""
        
        # Termination conditions
        terminated = False
        truncated = False
        
        # Success: completed all laps
        total_hoops_needed = len(self.course_hoops) * self.config.HOOP_CONFIG['num_laps']
        if self.hoops_completed >= total_hoops_needed:
            terminated = True
        
        # Failure conditions
        if self._check_boundaries():
            terminated = True
            
        if self.current_state['battery_remaining'] < 0.2:
            terminated = True
        
        # Truncation: maximum episode length
        if self.episode_step_count >= self.config.TRAINING_CONFIG['max_steps_per_episode']:
            truncated = True
        
        return terminated, truncated
    
    def render(self, mode: str = "rgb_array") -> Optional[np.ndarray]:
        """Render the environment (placeholder)"""
        # TODO: Implement visualization
        pass
    
    def close(self):
        """Clean up environment"""
        self.get_logger().info("Closing DeepFlyer environment")
        self.destroy_node()


# Convenience function for creating the environment
def create_deepflyer_env(**kwargs) -> DeepFlyerHoopNavigationEnv:
    """
    Create DeepFlyer hoop navigation environment
    
    Args:
        **kwargs: Arguments for DeepFlyerHoopNavigationEnv
        
    Returns:
        Configured DeepFlyerHoopNavigationEnv
    """
    return DeepFlyerHoopNavigationEnv(**kwargs) 