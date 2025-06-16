# Enhanced ROS2 environment
# See ros_env.py for base implementation
# Additional features can be added here

"""Enhanced ROS2 Gym environment with full reward integration and safety features."""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable, Union
import logging
from threading import Thread, Lock
import time

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, Imu, CompressedImage
    from geometry_msgs.msg import Twist, PoseStamped, TwistStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32, Bool, Header
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object  # Dummy for type hints

from .ros_env import RosEnvNode, DroneState, ROS_AVAILABLE
from .ros_utils import (
    StateProcessor, SafetyMonitor, ImageProcessor,
    PX4Interface, CoordinateTransform
)
from ..registry import RewardRegistry
from ..rewards import multi_objective_reward

logger = logging.getLogger(__name__)


class RosEnvV2(gym.Env):
    """
    Enhanced production-ready ROS2-based Gym environment for drone RL.
    
    Features:
    - Full integration with reward function registry
    - Safety monitoring and limits
    - PX4/MAVROS support
    - Domain randomization support
    - Multiple reward function support
    """
    
    def __init__(
        self,
        namespace: str = "deepflyer",
        reward_function: Optional[Union[str, Callable]] = "follow_trajectory",
        cross_track_weight: float = 1.0,
        heading_weight: float = 0.1,
        observation_config: Optional[Dict[str, bool]] = None,
        action_mode: str = "continuous",
        max_episode_steps: int = 500,
        step_duration: float = 0.05,
        timeout: float = 5.0,
        goal_position: Optional[List[float]] = None,
        target_altitude: Optional[float] = None,
        camera_resolution: Tuple[int, int] = (84, 84),
        enable_safety_monitor: bool = True,
        safety_config: Optional[Dict[str, float]] = None,
        domain_randomization: Optional[Dict[str, Any]] = None,
        external_force_range: Tuple[float, float] = (0.0, 0.0),
        sensor_noise_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize enhanced ROS2 drone environment.
        
        Args:
            namespace: ROS2 namespace for topics
            reward_function: Reward function ID from registry or callable
            cross_track_weight: Weight for cross-track error (path following)
            heading_weight: Weight for heading error
            observation_config: Dict specifying which observations to include
            action_mode: "continuous" or "discrete" action space
            max_episode_steps: Maximum steps per episode
            step_duration: Duration of each environment step
            timeout: Timeout for waiting for sensor data
            goal_position: Goal position [x, y, z] for navigation tasks
            target_altitude: Target altitude for altitude hold tasks
            camera_resolution: Resolution to downsample camera images to
            enable_safety_monitor: Whether to enable safety limits
            safety_config: Configuration for safety monitor
            domain_randomization: Domain randomization configuration
            external_force_range: Range for random external forces (min, max)
            sensor_noise_config: Noise levels for sensors
        """
        super().__init__()
        
        if not ROS_AVAILABLE:
            raise RuntimeError(
                "ROS2 is not available. Please install ROS2 and required packages."
            )
        
        # Configuration
        self.namespace = namespace
        self.action_mode = action_mode
        self.max_episode_steps = max_episode_steps
        self.step_duration = step_duration
        self.timeout = timeout
        self.camera_resolution = camera_resolution
        self.external_force_range = external_force_range
        
        # Setup reward function
        self._setup_reward_function(reward_function, cross_track_weight, heading_weight)
        
        # Task configuration  
        self.goal_position = np.array(goal_position) if goal_position else np.array([5.0, 5.0, 1.5])
        self.target_altitude = target_altitude if target_altitude is not None else 1.5
        
        # Observation configuration
        self.observation_config = observation_config or {
            'position': True,
            'orientation': True,
            'linear_velocity': True,
            'angular_velocity': True,
            'linear_acceleration': True,
            'front_camera': True,
            'down_camera': True,
            'collision': True,
            'obstacle_distance': True,
            'goal_relative': True,
            'external_force': False,  # Only if using adaptive disturbance reward
        }
        
        # Initialize components
        self.state_processor = StateProcessor()
        self.image_processor = ImageProcessor()
        
        # Safety monitor
        self.enable_safety_monitor = enable_safety_monitor
        if enable_safety_monitor:
            safety_config = safety_config or {}
            self.safety_monitor = SafetyMonitor(**safety_config)
        
        # Domain randomization
        self.domain_randomization = domain_randomization or {}
        self.sensor_noise_config = sensor_noise_config or {
            'position_noise': 0.0,
            'velocity_noise': 0.0,
            'imu_noise': 0.0,
            'camera_noise': 0.0,
        }
        
        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()
        
        self.node = RosEnvNode(namespace)
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        
        # Spin in background thread
        self.ros_thread = Thread(target=self._spin_ros, daemon=True)
        self.ros_thread.start()
        
        # Wait for initial sensor data
        self._wait_for_sensors()
        
        # Define spaces
        self._define_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.external_force = np.zeros(3)
        
        # Metrics tracking
        self.episode_metrics = {
            'total_distance': 0.0,
            'collision_count': 0,
            'energy_used': 0.0,
            'max_velocity': 0.0,
            'min_obstacle_distance': float('inf'),
        }
        
        logger.info(f"RosEnvV2 initialized with namespace: {namespace}, reward: {reward_function}")
    
    def _setup_reward_function(
        self,
        reward_function: Union[str, Callable],
        cross_track_weight: float = 1.0,
        heading_weight: float = 0.1,
    ):
        """
        Set up reward function.
        
        Args:
            reward_function: Reward function ID or callable
            cross_track_weight: Weight for cross-track error (path following)
            heading_weight: Weight for heading error
        """
        if isinstance(reward_function, str):
            # Use registry to get function by ID
            self.reward_fn = RewardRegistry.get_fn(reward_function)
        else:
            # Use provided callable
            self.reward_fn = reward_function
            
        # Store weights for reward components
        self.cross_track_weight = cross_track_weight
        self.heading_weight = heading_weight
    
    def _spin_ros(self):
        """Spin ROS2 executor in background thread."""
        try:
            self.executor.spin()
        except Exception as e:
            logger.error(f"ROS2 executor error: {e}")
    
    def _wait_for_sensors(self):
        """Wait for initial sensor data with timeout."""
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            state = self.node.state.get_snapshot()
            if state['position'] is not None and state['linear_velocity'] is not None:
                logger.info("Initial sensor data received")
                return
            time.sleep(0.1)
        
        logger.warning("Timeout waiting for sensor data, using defaults")
    
    def _define_spaces(self):
        """Define observation and action spaces based on configuration."""
        # Build observation space
        obs_dict = {}
        
        if self.observation_config.get('position', True):
            obs_dict['position'] = gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32
            )
        
        if self.observation_config.get('orientation', True):
            obs_dict['orientation'] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )
        
        if self.observation_config.get('linear_velocity', True):
            obs_dict['linear_velocity'] = gym.spaces.Box(
                low=-5.0, high=5.0, shape=(3,), dtype=np.float32
            )
        
        if self.observation_config.get('angular_velocity', True):
            obs_dict['angular_velocity'] = gym.spaces.Box(
                low=-3.14, high=3.14, shape=(3,), dtype=np.float32
            )
        
        if self.observation_config.get('linear_acceleration', True):
            obs_dict['linear_acceleration'] = gym.spaces.Box(
                low=-10.0, high=10.0, shape=(3,), dtype=np.float32
            )
        
        if self.observation_config.get('front_camera', True):
            obs_dict['front_camera'] = gym.spaces.Box(
                low=0, high=255,
                shape=(*self.camera_resolution, 3),
                dtype=np.uint8
            )
        
        if self.observation_config.get('down_camera', True):
            obs_dict['down_camera'] = gym.spaces.Box(
                low=0, high=255,
                shape=(*self.camera_resolution, 3),
                dtype=np.uint8
            )
        
        if self.observation_config.get('collision', True):
            obs_dict['collision'] = gym.spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
        
        if self.observation_config.get('obstacle_distance', True):
            obs_dict['obstacle_distance'] = gym.spaces.Box(
                low=0.0, high=10.0, shape=(1,), dtype=np.float32
            )
        
        if self.observation_config.get('goal_relative', True):
            obs_dict['goal_relative'] = gym.spaces.Box(
                low=-20.0, high=20.0, shape=(3,), dtype=np.float32
            )
        
        if self.observation_config.get('external_force', False):
            obs_dict['external_force'] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )
        
        self.observation_space = gym.spaces.Dict(obs_dict)
        
        # Define action space
        if self.action_mode == "continuous":
            # Continuous: [vx, vy, vz, wz] normalized to [-1, 1]
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )
        else:
            # Discrete: 9 actions (8 directions + hover)
            self.action_space = gym.spaces.Discrete(9)
        
        logger.info(f"Observation space keys: {list(obs_dict.keys())}")
        logger.info(f"Action space: {self.action_space}")
    
    def _apply_sensor_noise(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply sensor noise for domain randomization."""
        noisy_obs = observation.copy()
        
        # Position noise
        if 'position' in noisy_obs and self.sensor_noise_config['position_noise'] > 0:
            noise = np.random.normal(0, self.sensor_noise_config['position_noise'], 3)
            noisy_obs['position'] += noise.astype(np.float32)
        
        # Velocity noise
        if 'linear_velocity' in noisy_obs and self.sensor_noise_config['velocity_noise'] > 0:
            noise = np.random.normal(0, self.sensor_noise_config['velocity_noise'], 3)
            noisy_obs['linear_velocity'] += noise.astype(np.float32)
        
        # IMU noise
        if 'linear_acceleration' in noisy_obs and self.sensor_noise_config['imu_noise'] > 0:
            noise = np.random.normal(0, self.sensor_noise_config['imu_noise'], 3)
            noisy_obs['linear_acceleration'] += noise.astype(np.float32)
        
        # Camera noise
        if self.sensor_noise_config['camera_noise'] > 0:
            for cam_key in ['front_camera', 'down_camera']:
                if cam_key in noisy_obs:
                    noise = np.random.normal(
                        0, self.sensor_noise_config['camera_noise'],
                        noisy_obs[cam_key].shape
                    )
                    noisy_obs[cam_key] = np.clip(
                        noisy_obs[cam_key].astype(np.float32) + noise,
                        0, 255
                    ).astype(np.uint8)
        
        return noisy_obs
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation from ROS2 state with noise."""
        state = self.node.state.get_snapshot()
        obs = {}
        
        if self.observation_config.get('position', True):
            obs['position'] = state['position'].astype(np.float32)
        
        if self.observation_config.get('orientation', True):
            obs['orientation'] = state['orientation'].astype(np.float32)
        
        if self.observation_config.get('linear_velocity', True):
            obs['linear_velocity'] = state['linear_velocity'].astype(np.float32)
        
        if self.observation_config.get('angular_velocity', True):
            obs['angular_velocity'] = state['angular_velocity'].astype(np.float32)
        
        if self.observation_config.get('linear_acceleration', True):
            obs['linear_acceleration'] = state['linear_acceleration'].astype(np.float32)
        
        if self.observation_config.get('front_camera', True):
            img = state['front_camera_image']
            if img is not None:
                img_resized = self.image_processor.resize_image(img, self.camera_resolution)
                obs['front_camera'] = img_resized.astype(np.uint8)
            else:
                obs['front_camera'] = np.zeros((*self.camera_resolution, 3), dtype=np.uint8)
        
        if self.observation_config.get('down_camera', True):
            img = state['down_camera_image']
            if img is not None:
                img_resized = self.image_processor.resize_image(img, self.camera_resolution)
                obs['down_camera'] = img_resized.astype(np.uint8)
            else:
                obs['down_camera'] = np.zeros((*self.camera_resolution, 3), dtype=np.uint8)
        
        if self.observation_config.get('collision', True):
            obs['collision'] = np.array([float(state['collision_flag'])], dtype=np.float32)
        
        if self.observation_config.get('obstacle_distance', True):
            obs['obstacle_distance'] = np.array([state['distance_to_obstacle']], dtype=np.float32)
        
        if self.observation_config.get('goal_relative', True):
            obs['goal_relative'] = (self.goal_position - state['position']).astype(np.float32)
        
        if self.observation_config.get('external_force', False):
            obs['external_force'] = self.external_force.astype(np.float32)
        
        # Apply sensor noise if configured
        obs = self._apply_sensor_noise(obs)
        
        return obs
    
    def _process_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process action into linear and angular velocity commands."""
        if self.action_mode == "continuous":
            # Scale from [-1, 1] to actual velocity limits
            linear = np.zeros(3)
            linear[0] = action[0] * 1.5  # vx: ±1.5 m/s
            linear[1] = action[1] * 1.5  # vy: ±1.5 m/s
            linear[2] = action[2] * 1.0  # vz: ±1.0 m/s
            
            angular = np.zeros(3)
            angular[2] = action[3] * (np.pi / 2)  # wz: ±π/2 rad/s
        else:
            # Discrete actions
            action_map = {
                0: ([0, 0, 0], [0, 0, 0]),      # hover
                1: ([0.5, 0, 0], [0, 0, 0]),    # forward
                2: ([-0.5, 0, 0], [0, 0, 0]),   # backward
                3: ([0, 0.5, 0], [0, 0, 0]),    # left
                4: ([0, -0.5, 0], [0, 0, 0]),   # right
                5: ([0, 0, 0.3], [0, 0, 0]),    # up
                6: ([0, 0, -0.3], [0, 0, 0]),   # down
                7: ([0, 0, 0], [0, 0, 0.5]),    # rotate left
                8: ([0, 0, 0], [0, 0, -0.5]),   # rotate right
            }
            linear, angular = action_map[int(action)]
            linear = np.array(linear)
            angular = np.array(angular)
        
        return linear, angular
    
    def _update_external_forces(self):
        """Update external forces for domain randomization."""
        if self.external_force_range[1] > 0:
            # Random force in random direction
            force_magnitude = np.random.uniform(*self.external_force_range)
            force_direction = np.random.randn(3)
            force_direction = force_direction / (np.linalg.norm(force_direction) + 1e-8)
            self.external_force = force_direction * force_magnitude
        else:
            self.external_force = np.zeros(3)
    
    def _update_metrics(self, state: Dict[str, Any], action: np.ndarray):
        """Update episode metrics for logging."""
        # Distance traveled
        if hasattr(self, '_last_position'):
            distance = np.linalg.norm(state['position'] - self._last_position)
            self.episode_metrics['total_distance'] += distance
        self._last_position = state['position'].copy()
        
        # Collision count
        if state['collision_flag']:
            self.episode_metrics['collision_count'] += 1
        
        # Energy estimation (simplified)
        velocity_magnitude = np.linalg.norm(state['linear_velocity'])
        self.episode_metrics['energy_used'] += velocity_magnitude * self.step_duration
        
        # Max velocity
        self.episode_metrics['max_velocity'] = max(
            self.episode_metrics['max_velocity'],
            velocity_magnitude
        )
        
        # Min obstacle distance
        self.episode_metrics['min_obstacle_distance'] = min(
            self.episode_metrics['min_obstacle_distance'],
            state['distance_to_obstacle']
        )
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Send reset command to simulation
        self.node.send_reset()
        
        # Wait for reset to complete
        time.sleep(0.5)
        self._wait_for_sensors()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self.external_force = np.zeros(3)
        
        # Reset metrics
        self.episode_metrics = {
            'total_distance': 0.0,
            'collision_count': 0,
            'energy_used': 0.0,
            'max_velocity': 0.0,
            'min_obstacle_distance': float('inf'),
        }
        
        # Reset state processor
        self.state_processor = StateProcessor()
        
        # Update configuration from options
        if options:
            if 'goal_position' in options:
                self.goal_position = np.array(options['goal_position'])
            if 'target_altitude' in options:
                self.target_altitude = options['target_altitude']
            if 'cross_track_weight' in options:
                self.cross_track_weight = options['cross_track_weight']
            if 'heading_weight' in options:
                self.heading_weight = options['heading_weight']
        
        # Apply domain randomization
        if self.domain_randomization:
            self._apply_domain_randomization()
        
        observation = self._get_observation()
        info = {
            'episode_step': self.current_step,
            'goal_position': self.goal_position.tolist(),
            'target_altitude': self.target_altitude,
            'cross_track_weight': self.cross_track_weight,
            'heading_weight': self.heading_weight,
        }
        
        return observation, info
    
    def _apply_domain_randomization(self):
        """Apply domain randomization settings."""
        # Randomize sensor noise levels
        if 'sensor_noise' in self.domain_randomization:
            noise_range = self.domain_randomization['sensor_noise']
            for key in self.sensor_noise_config:
                self.sensor_noise_config[key] = np.random.uniform(*noise_range[key])
        
        # Randomize external force range
        if 'external_force' in self.domain_randomization:
            force_range = self.domain_randomization['external_force']
            self.external_force_range = (
                np.random.uniform(*force_range[0]),
                np.random.uniform(*force_range[1])
            )
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Process action
        linear_vel, angular_vel = self._process_action(action)
        
        # Apply safety limits if enabled
        if self.enable_safety_monitor:
            state_snapshot = self.node.state.get_snapshot()
            command = np.concatenate([linear_vel, angular_vel])
            limited_command = self.safety_monitor.apply_safety_limits(
                command, state_snapshot
            )
            linear_vel = limited_command[:3]
            angular_vel = limited_command[3:]
        
        # Update external forces
        self._update_external_forces()
        
        # Apply external forces to command (simplified)
        linear_vel += self.external_force * 0.1
        
        # Send command
        self.node.send_velocity_command(linear_vel, angular_vel)
        
        # Wait for step duration
        time.sleep(self.step_duration)
        
        # Get new observation
        observation = self._get_observation()
        state = self.node.state.get_snapshot()
        
        # Process state for reward function
        reward_state = self.state_processor.process_state(state)
        reward_state['goal'] = self.goal_position
        reward_state['target_altitude'] = self.target_altitude
        reward_state['external_force'] = self.external_force
        reward_state['max_lin_jerk'] = 0.5  # From documentation
        reward_state['max_ang_jerk'] = 0.5
        reward_state['max_altitude_error'] = 1.0
        
        # Process action for reward function
        reward_action = self.state_processor.process_action(action, self.action_mode)
        
        # Calculate reward using registered function
        reward = float(self.reward_fn(reward_state, reward_action))
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if state['collision_flag']:
            terminated = True
        
        # Check if goal reached (for navigation tasks)
        distance_to_goal = np.linalg.norm(state['position'] - self.goal_position)
        if distance_to_goal < 0.2:  # within 20cm
            terminated = True
        
        # Check safety violations
        if self.enable_safety_monitor:
            pos_safe, pos_msg = self.safety_monitor.check_geofence(state['position'])
            alt_safe, alt_msg = self.safety_monitor.check_altitude_limits(state['position'][2])
            if not pos_safe or not alt_safe:
                terminated = True
                reward -= 5.0  # safety violation penalty
        
        # Check truncation
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # Update metrics
        self._update_metrics(state, action)
        self.episode_reward += reward
        
        # Build info dict
        info = {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'distance_to_goal': distance_to_goal,
            'collision': state['collision_flag'],
            'position': state['position'].tolist(),
            'velocity': state['linear_velocity'].tolist(),
            'metrics': self.episode_metrics.copy(),
        }
        
        return observation, float(reward), terminated, truncated, info
    
    def close(self):
        """Clean up ROS2 resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown()
        if hasattr(self, 'node'):
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        logger.info("RosEnvV2 closed")
    
    def render(self):
        """Render is handled by Gazebo visualization."""
        pass


def make_ros_env(
    env_config: Optional[Dict[str, Any]] = None,
    reward_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RosEnvV2:
    """
    Factory function to create ROS environment with configuration.
    
    Args:
        env_config: Environment configuration
        reward_config: Reward function configuration
        **kwargs: Additional keyword arguments
    
    Returns:
        Configured RosEnvV2 instance
    """
    env_config = env_config or {}
    reward_config = reward_config or {}
    
    # Merge configurations
    config = {**env_config, **reward_config, **kwargs}
    
    return RosEnvV2(**config)
