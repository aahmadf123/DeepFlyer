"""
ROS Environment Base Class for DeepFlyer
Provides the foundation for all ROS-based RL environments with thread-safe state management
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import threading
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from abc import ABC, abstractmethod
import logging

# ROS2 imports (with fallback)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from std_msgs.msg import Header
    from geometry_msgs.msg import Point, Vector3, Quaternion, Twist
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object  # Fallback for type hints

from ..config import DeepFlyerConfig
from ..logger import Logger

logger = logging.getLogger(__name__)


class RosEnvState:
    """Thread-safe container for ROS environment state"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._data = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'orientation': np.zeros(4),  # quaternion [w, x, y, z]
            'angular_velocity': np.zeros(3),
            'battery_level': 1.0,
            'armed': False,
            'connected': False,
            'flight_mode': 'UNKNOWN',
            'last_update_time': 0.0
        }
    
    def update(self, key: str, value: Any) -> None:
        """Thread-safe update of state value"""
        with self._lock:
            self._data[key] = value
            self._data['last_update_time'] = time.time()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Thread-safe retrieval of state value"""
        with self._lock:
            return self._data.get(key, default)
    
    def get_all(self) -> Dict[str, Any]:
        """Thread-safe retrieval of all state data"""
        with self._lock:
            return self._data.copy()
    
    def is_fresh(self, max_age: float = 1.0) -> bool:
        """Check if state data is fresh"""
        with self._lock:
            return (time.time() - self._data['last_update_time']) < max_age


class RosEnv(gym.Env, ABC):
    """
    Base ROS environment class for DeepFlyer RL environments
    
    Provides:
    - Thread-safe ROS integration
    - Standard Gymnasium interface
    - State management and observation processing
    - Action space handling
    - Episode management
    - Mock fallback when ROS unavailable
    """
    
    def __init__(self,
                 node_name: str = "deepflyer_env",
                 namespace: str = "/deepflyer",
                 step_timeout: float = 1.0,
                 observation_timeout: float = 2.0,
                 use_ros: bool = True,
                 mock_when_unavailable: bool = True):
        """
        Initialize ROS environment
        
        Args:
            node_name: Name for ROS node
            namespace: ROS namespace for topics
            step_timeout: Timeout for step operations (seconds)
            observation_timeout: Timeout for fresh observations (seconds)
            use_ros: Whether to use actual ROS (False for testing)
            mock_when_unavailable: Use mock implementation when ROS unavailable
        """
        super().__init__()
        
        self.config = DeepFlyerConfig()
        self.namespace = namespace
        self.step_timeout = step_timeout
        self.observation_timeout = observation_timeout
        
        # State management
        self.state = RosEnvState()
        self.episode_active = False
        self.episode_step_count = 0
        self.episode_start_time = 0.0
        
        # Determine operating mode
        self.use_ros = use_ros and ROS_AVAILABLE
        self.mock_mode = not self.use_ros and mock_when_unavailable
        
        if not self.use_ros and not mock_when_unavailable:
            raise RuntimeError("ROS not available and mock mode disabled")
        
        # Initialize ROS or mock
        if self.use_ros:
            self._init_ros_interface(node_name)
        else:
            self._init_mock_interface()
        
        # Define observation and action spaces (subclasses should override)
        self._setup_spaces()
        
        logger.info(f"RosEnv initialized in {'ROS' if self.use_ros else 'mock'} mode")
    
    def _init_ros_interface(self, node_name: str) -> None:
        """Initialize ROS interface"""
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.node = Node(node_name)
            self.node.get_logger().info(f"ROS node '{node_name}' initialized")
            
            # Setup QoS profiles
            self.reliable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            
            self.best_effort_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            
            # Initialize ROS publishers/subscribers (implemented by subclasses)
            self._setup_ros_interface()
            
            # Start ROS spinning thread
            self.ros_thread = threading.Thread(target=self._ros_spin_thread, daemon=True)
            self.ros_thread.start()
            
            self.state.update('connected', True)
            
        except Exception as e:
            logger.error(f"Failed to initialize ROS interface: {e}")
            raise
    
    def _init_mock_interface(self) -> None:
        """Initialize mock interface for testing without ROS"""
        logger.info("Initializing mock ROS interface")
        
        # Mock node object
        class MockNode:
            def get_logger(self):
                return logger
        
        self.node = MockNode()
        
        # Initialize mock state
        self._setup_mock_state()
        
        # Start mock update thread
        self.mock_thread = threading.Thread(target=self._mock_update_thread, daemon=True)
        self.mock_thread.start()
        
        self.state.update('connected', True)
    
    def _setup_mock_state(self) -> None:
        """Setup initial mock state"""
        self.state.update('position', np.array([0.0, 0.0, 0.8]))  # Start at 0.8m altitude
        self.state.update('velocity', np.zeros(3))
        self.state.update('orientation', np.array([1.0, 0.0, 0.0, 0.0]))  # Identity quaternion
        self.state.update('angular_velocity', np.zeros(3))
        self.state.update('battery_level', 1.0)
        self.state.update('armed', True)
        self.state.update('flight_mode', 'OFFBOARD')
        
        # Mock physics parameters
        self.mock_physics = {
            'position': np.array([0.0, 0.0, 0.8]),
            'velocity': np.zeros(3),
            'last_action': np.zeros(self.action_space.shape[0]),
            'dt': 0.05  # 20Hz update rate
        }
    
    def _mock_update_thread(self) -> None:
        """Mock physics update thread"""
        while True:
            try:
                if self.episode_active:
                    self._update_mock_physics()
                time.sleep(self.mock_physics['dt'])
            except Exception as e:
                logger.error(f"Mock update thread error: {e}")
                break
    
    def _update_mock_physics(self) -> None:
        """Update mock drone physics based on last action"""
        dt = self.mock_physics['dt']
        
        # Simple integration of velocity
        self.mock_physics['position'] += self.mock_physics['velocity'] * dt
        
        # Add some damping
        self.mock_physics['velocity'] *= 0.95
        
        # Update state
        self.state.update('position', self.mock_physics['position'].copy())
        self.state.update('velocity', self.mock_physics['velocity'].copy())
        
        # Simulate battery drain
        current_battery = self.state.get('battery_level', 1.0)
        self.state.update('battery_level', max(0.0, current_battery - 0.0001))  # Slow drain
    
    @abstractmethod
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces (implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _setup_ros_interface(self) -> None:
        """Setup ROS publishers and subscribers (implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation (implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _process_action(self, action: np.ndarray) -> None:
        """Process and execute action (implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _calculate_reward(self, observation: np.ndarray, action: np.ndarray, info: Dict[str, Any]) -> float:
        """Calculate reward for current step (implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _check_episode_done(self, observation: np.ndarray, info: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if episode is terminated or truncated (implemented by subclasses)"""
        pass
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode tracking
        self.episode_active = True
        self.episode_step_count = 0
        self.episode_start_time = time.time()
        
        # Reset mock physics if in mock mode
        if self.mock_mode:
            self._setup_mock_state()
        
        # Wait for fresh state data
        start_wait = time.time()
        while not self.state.is_fresh(self.observation_timeout):
            if time.time() - start_wait > self.observation_timeout:
                logger.warning("Timeout waiting for fresh state data during reset")
                break
            time.sleep(0.01)
        
        # Get initial observation
        observation = self._get_observation()
        
        # Prepare info dict
        info = {
            'episode_step': self.episode_step_count,
            'episode_time': 0.0,
            'ros_connected': self.state.get('connected', False),
            'mock_mode': self.mock_mode
        }
        
        logger.info(f"Environment reset - Episode started")
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        if not self.episode_active:
            raise RuntimeError("Must call reset() before step()")
        
        step_start_time = time.time()
        self.episode_step_count += 1
        
        # Validate action
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Store action for mock physics
        if self.mock_mode:
            self.mock_physics['last_action'] = action.copy()
            # Convert action to velocity for mock physics
            # Assuming action is [lateral, vertical, speed] format
            if len(action) >= 3:
                self.mock_physics['velocity'][0] = action[2] * 0.6  # Forward speed
                self.mock_physics['velocity'][1] = action[0] * 0.8  # Lateral speed  
                self.mock_physics['velocity'][2] = action[1] * 0.4  # Vertical speed
        
        # Process action
        self._process_action(action)
        
        # Wait for state update
        time.sleep(0.05)  # Allow time for action to take effect
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        episode_time = time.time() - self.episode_start_time
        info = {
            'episode_step': self.episode_step_count,
            'episode_time': episode_time,
            'ros_connected': self.state.get('connected', False),
            'mock_mode': self.mock_mode,
            'step_time': time.time() - step_start_time
        }
        
        reward = self._calculate_reward(observation, action, info)
        
        # Check if episode is done
        terminated, truncated = self._check_episode_done(observation, info)
        
        if terminated or truncated:
            self.episode_active = False
            logger.info(f"Episode ended - Steps: {self.episode_step_count}, Time: {episode_time:.2f}s")
        
        return observation, reward, terminated, truncated, info
    
    def close(self) -> None:
        """Clean up environment resources"""
        self.episode_active = False
        
        if self.use_ros:
            try:
                if hasattr(self, 'node'):
                    self.node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                logger.warning(f"Error during ROS cleanup: {e}")
        
        logger.info("Environment closed")
    
    def _ros_spin_thread(self) -> None:
        """ROS spinning thread"""
        try:
            rclpy.spin(self.node)
        except Exception as e:
            logger.error(f"ROS spin thread error: {e}")
            self.state.update('connected', False)
    
    def get_ros_state(self) -> Dict[str, Any]:
        """Get current ROS state for debugging"""
        return self.state.get_all()
    
    def is_connected(self) -> bool:
        """Check if ROS connection is active"""
        return self.state.get('connected', False) and self.state.is_fresh(self.observation_timeout)


# Export main classes
__all__ = ['RosEnv', 'RosEnvState'] 