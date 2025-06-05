import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import logging
from dataclasses import dataclass
from threading import Lock
import time

# Try to import ROS packages
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image, Imu, CompressedImage
    from geometry_msgs.msg import Twist, PoseStamped, TwistStamped, Pose, Vector3
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Float32, Bool, Header
    from cv_bridge import CvBridge
    
    # Try to import MAVROS
    try:
        from mavros_msgs.msg import State
        from mavros_msgs.srv import CommandBool, SetMode
        MAVROS_AVAILABLE = True
    except ImportError:
        MAVROS_AVAILABLE = False
        
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object  # Dummy for type hints
    MAVROS_AVAILABLE = False
    
    # Import mock classes
    from .mock_ros import (
        MockNode as Node,
        MockImage as Image,
        MockPose as Pose,
        MockTwist as Twist,
        MockImu as Imu,
        MockMAVROSState as State,
        MockMAVROSService,
        MockZEDNode,
    )

logger = logging.getLogger(__name__)


@dataclass
class DroneState:
    """Thread-safe container for drone state data."""
    position: np.ndarray = None  # [x, y, z]
    orientation: np.ndarray = None  # quaternion [x, y, z, w]
    linear_velocity: np.ndarray = None  # [vx, vy, vz]
    angular_velocity: np.ndarray = None  # [wx, wy, wz]
    linear_acceleration: np.ndarray = None  # [ax, ay, az]
    front_camera_image: np.ndarray = None  # HxWxC
    down_camera_image: np.ndarray = None  # HxWxC
    collision_flag: bool = False
    distance_to_obstacle: float = float('inf')
    battery_level: float = 1.0
    timestamp: float = 0.0
    
    # MAVROS specific state
    connected: bool = False
    armed: bool = False
    guided: bool = False
    flight_mode: str = ""
    
    def __post_init__(self):
        self._lock = Lock()
        if self.position is None:
            self.position = np.zeros(3)
        if self.orientation is None:
            self.orientation = np.array([0, 0, 0, 1])  # identity quaternion
        if self.linear_velocity is None:
            self.linear_velocity = np.zeros(3)
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3)
        if self.linear_acceleration is None:
            self.linear_acceleration = np.zeros(3)
    
    def update(self, **kwargs):
        """Thread-safe state update."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get thread-safe snapshot of current state."""
        with self._lock:
            return {
                'position': self.position.copy() if self.position is not None else None,
                'orientation': self.orientation.copy() if self.orientation is not None else None,
                'linear_velocity': self.linear_velocity.copy() if self.linear_velocity is not None else None,
                'angular_velocity': self.angular_velocity.copy() if self.angular_velocity is not None else None,
                'linear_acceleration': self.linear_acceleration.copy() if self.linear_acceleration is not None else None,
                'front_camera_image': self.front_camera_image.copy() if self.front_camera_image is not None else None,
                'down_camera_image': self.down_camera_image.copy() if self.down_camera_image is not None else None,
                'collision_flag': self.collision_flag,
                'distance_to_obstacle': self.distance_to_obstacle,
                'battery_level': self.battery_level,
                'timestamp': self.timestamp,
                'connected': self.connected,
                'armed': self.armed,
                'guided': self.guided,
                'flight_mode': self.flight_mode,
            }


class RosEnvNode(Node):
    """ROS2 node for drone environment interaction."""
    
    def __init__(self, namespace: str = 'deepflyer', use_zed: bool = True):
        super().__init__(f'{namespace}_env_node')
        
        self.namespace = namespace
        self.state = DroneState()
        self.use_zed = use_zed
        
        # Create bridge for converting ROS images to OpenCV format
        if ROS_AVAILABLE:
            self.bridge = CvBridge()
        
        # Configure QoS profiles for reliable communication
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        ) if ROS_AVAILABLE else None
        
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        ) if ROS_AVAILABLE else None
        
        # Setup MAVROS subscribers if available
        if MAVROS_AVAILABLE:
            # Note that MAVROS topics follow a different convention
            self.mavros_state_sub = self.create_subscription(
                State, f'/mavros/state',
                self._mavros_state_callback, reliable_qos
            )
            
            # MAVROS local position is used for pose info
            self.pose_sub = self.create_subscription(
                PoseStamped, f'/mavros/local_position/pose', 
                self._pose_callback, reliable_qos
            )
            
            # MAVROS IMU data
            self.imu_sub = self.create_subscription(
                Imu, f'/mavros/imu/data',
                self._imu_callback, sensor_qos
            )
            
            # Initialize MAVROS services for later use
            if ROS_AVAILABLE:
                self.arm_service = self.create_client(CommandBool, '/mavros/cmd/arming')
                self.mode_service = self.create_client(SetMode, '/mavros/set_mode')
        else:
            # Standard pose/odom subscribers (non-MAVROS)
            self.pose_sub = self.create_subscription(
                PoseStamped, f'/{namespace}/pose', 
                self._pose_callback, reliable_qos
            )
            
            self.odom_sub = self.create_subscription(
                Odometry, f'/{namespace}/odom',
                self._odom_callback, sensor_qos
            )
            
            self.imu_sub = self.create_subscription(
                Imu, f'/{namespace}/imu',
                self._imu_callback, sensor_qos
            )
            
        # Camera subscribers - use ZED topics if specified
        if use_zed:
            # ZED Mini stereo camera topics
            self.front_camera_sub = self.create_subscription(
                Image, f'/zed_mini/zed_node/rgb/image_rect_color',
                self._front_camera_callback, sensor_qos
            )
            
            self.depth_sub = self.create_subscription(
                Image, f'/zed_mini/zed_node/depth/depth_registered',
                self._depth_callback, sensor_qos
            )
        else:
            # Standard camera topics
            self.front_camera_sub = self.create_subscription(
                Image, f'/{namespace}/camera/front/image_raw',
                self._front_camera_callback, sensor_qos
            )
            
            self.down_camera_sub = self.create_subscription(
                Image, f'/{namespace}/camera/down/image_raw',
                self._down_camera_callback, sensor_qos
            )
        
        # Standard collision and status subscribers
        self.collision_sub = self.create_subscription(
            Bool, f'/{namespace}/collision',
            self._collision_callback, reliable_qos
        )
        
        self.obstacle_distance_sub = self.create_subscription(
            Float32, f'/{namespace}/obstacle_distance',
            self._obstacle_distance_callback, sensor_qos
        )
        
        self.battery_sub = self.create_subscription(
            Float32, f'/{namespace}/battery_level',
            self._battery_callback, sensor_qos
        )
        
        # Publishers - use MAVROS topics if available
        if MAVROS_AVAILABLE:
            self.cmd_vel_pub = self.create_publisher(
                Twist, f'/mavros/setpoint_velocity/cmd_vel_unstamped', reliable_qos
            )
        else:
            self.cmd_vel_pub = self.create_publisher(
                Twist, f'/{namespace}/cmd_vel', reliable_qos
            )
        
        self.reset_pub = self.create_publisher(
            Bool, f'/{namespace}/reset', reliable_qos
        )
        
        self.get_logger().info(f"ROS2 environment node initialized: namespace={namespace}, use_zed={use_zed}")
    
    def _mavros_state_callback(self, msg: State):
        """Handle MAVROS state updates."""
        self.state.update(
            connected=msg.connected,
            armed=msg.armed,
            guided=msg.guided,
            flight_mode=msg.mode
        )
    
    def _pose_callback(self, msg: PoseStamped):
        """Handle pose updates."""
        position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        orientation = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        
        # Get timestamp from the message if available
        timestamp = 0.0
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            timestamp = time.time()
            
        self.state.update(
            position=position,
            orientation=orientation,
            timestamp=timestamp
        )
    
    def _odom_callback(self, msg: Odometry):
        """Handle odometry updates for velocity information."""
        linear_velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        angular_velocity = np.array([
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ])
        self.state.update(
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity
        )
    
    def _imu_callback(self, msg: Imu):
        """Handle IMU updates for acceleration data."""
        linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
        # If odom hasn't provided angular velocity, get it from IMU
        angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        
        self.state.update(
            linear_acceleration=linear_acceleration,
            angular_velocity=angular_velocity
        )
    
    def _front_camera_callback(self, msg: Image):
        """Handle front camera images."""
        try:
            if ROS_AVAILABLE and hasattr(self, 'bridge'):
                cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            else:
                cv_image = msg.data  # Mock image already has numpy array
            self.state.update(front_camera_image=cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert front camera image: {e}")
    
    def _down_camera_callback(self, msg: Image):
        """Handle downward camera images."""
        try:
            if ROS_AVAILABLE and hasattr(self, 'bridge'):
                cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            else:
                cv_image = msg.data  # Mock image already has numpy array
            self.state.update(down_camera_image=cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert down camera image: {e}")
    
    def _depth_callback(self, msg: Image):
        """Handle depth image from ZED camera."""
        try:
            if ROS_AVAILABLE and hasattr(self, 'bridge'):
                # ZED depth is 32-bit float in meters
                depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                
                # Process depth to find minimum distance (obstacle detection)
                # Ignore zeros and very small values (invalid measurements)
                min_depth = float('inf')
                valid_depths = depth_image[depth_image > 0.1]
                if valid_depths.size > 0:
                    min_depth = float(np.min(valid_depths))
                
                self.state.update(distance_to_obstacle=min_depth)
            else:
                # Mock depth processing
                min_depth = float(np.random.uniform(1.0, 10.0))
                self.state.update(distance_to_obstacle=min_depth)
                
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")
    
    def _collision_callback(self, msg: Bool):
        """Handle collision detection."""
        if hasattr(msg, 'data'):
            collision = bool(msg.data)
        else:
            # For mock objects
            collision = bool(msg)
        self.state.update(collision_flag=collision)
    
    def _obstacle_distance_callback(self, msg: Float32):
        """Handle obstacle distance updates."""
        if hasattr(msg, 'data'):
            distance = float(msg.data)
        else:
            # For mock objects
            distance = float(msg)
        self.state.update(distance_to_obstacle=distance)
    
    def _battery_callback(self, msg: Float32):
        """Handle battery level updates."""
        if hasattr(msg, 'data'):
            battery = float(msg.data)
        else:
            # For mock objects
            battery = float(msg)
        self.state.update(battery_level=battery)
    
    def send_velocity_command(self, linear: np.ndarray, angular: np.ndarray):
        """Send velocity command to drone."""
        cmd = Twist()
        
        if hasattr(cmd, 'linear') and hasattr(cmd.linear, 'x'):
            # ROS Twist message
            cmd.linear.x = float(linear[0])
            cmd.linear.y = float(linear[1])
            cmd.linear.z = float(linear[2])
            cmd.angular.x = float(angular[0])
            cmd.angular.y = float(angular[1])
            cmd.angular.z = float(angular[2])
        else:
            # Mock Twist object
            cmd.linear = linear
            cmd.angular = angular
            
        self.cmd_vel_pub.publish(cmd)
    
    def send_reset(self):
        """Send reset command to simulation."""
        if ROS_AVAILABLE:
            reset_msg = Bool()
            reset_msg.data = True
            self.reset_pub.publish(reset_msg)
        else:
            # For mock objects
            self.reset_pub.publish(True)
    
    def arm(self, arm: bool = True) -> bool:
        """Arm or disarm the drone using MAVROS."""
        if not MAVROS_AVAILABLE:
            if not ROS_AVAILABLE:
                # Use mock
                return MockMAVROSService.arm(arm)
            else:
                self.get_logger().warn("MAVROS not available, cannot arm/disarm")
                return False
            
        # Send arm command through service
        if ROS_AVAILABLE:
            request = CommandBool.Request()
            request.value = arm
            
            future = self.arm_service.call_async(request)
            # Wait for response
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
            return future.result().success if future.done() else False
        else:
            return False
    
    def set_mode(self, mode: str) -> bool:
        """Set flight mode using MAVROS."""
        if not MAVROS_AVAILABLE:
            if not ROS_AVAILABLE:
                # Use mock
                return MockMAVROSService.set_mode(mode)
            else:
                self.get_logger().warn(f"MAVROS not available, cannot set mode to {mode}")
                return False
            
        # Send mode command through service
        if ROS_AVAILABLE:
            request = SetMode.Request()
            request.custom_mode = mode
            
            future = self.mode_service.call_async(request)
            # Wait for response
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
            return future.result().mode_sent if future.done() else False
        else:
            return False


class RosEnv(gym.Env):
    """
    Production-ready ROS2-based Gym environment for drone RL.
    
    This environment interfaces with a Gazebo simulation through ROS2,
    providing observations and accepting velocity commands.
    """
    
    def __init__(
        self, 
        namespace: str = "deepflyer",
        observation_config: Optional[Dict[str, bool]] = None,
        action_mode: str = "continuous",  # "continuous" or "discrete"
        max_episode_steps: int = 500,
        step_duration: float = 0.05,  # 20 Hz control
        timeout: float = 5.0,  # seconds to wait for sensor data
        goal_position: Optional[List[float]] = None,
        target_altitude: Optional[float] = None,
        camera_resolution: Tuple[int, int] = (84, 84),  # Downsampled for RL
        use_zed: bool = True,  # Use ZED Mini camera
        use_mavros: bool = True,  # Use MAVROS for flight control
        auto_arm: bool = False,  # Automatically arm in reset
        auto_offboard: bool = False,  # Automatically set to OFFBOARD mode
    ):
        """
        Initialize ROS2 drone environment.
        
        Args:
            namespace: ROS2 namespace for topics
            observation_config: Dict specifying which observations to include
            action_mode: "continuous" or "discrete" action space
            max_episode_steps: Maximum steps per episode
            step_duration: Duration of each environment step
            timeout: Timeout for waiting for sensor data
            goal_position: Goal position [x, y, z] for navigation tasks
            target_altitude: Target altitude for altitude hold tasks
            camera_resolution: Resolution to downsample camera images to
            use_zed: Whether to use ZED Mini stereo camera topics
            use_mavros: Whether to use MAVROS for flight control
            auto_arm: Automatically arm the drone during reset
            auto_offboard: Automatically set OFFBOARD mode during reset
        """
        super().__init__()
        
        # Configuration
        self.namespace = namespace
        self.action_mode = action_mode
        self.max_episode_steps = max_episode_steps
        self.step_duration = step_duration
        self.timeout = timeout
        self.camera_resolution = camera_resolution
        self.use_zed = use_zed
        self.use_mavros = use_mavros and MAVROS_AVAILABLE
        self.auto_arm = auto_arm
        self.auto_offboard = auto_offboard
        
        # Task configuration
        self.goal_position = np.array(goal_position) if goal_position else np.array([5.0, 5.0, 1.5])
        self.target_altitude = target_altitude if target_altitude is not None else 1.5
        
        # Observation configuration
        self.observation_config = observation_config or {
            'position': True,
            'orientation': True,
            'linear_velocity': True,
            'angular_velocity': True,
            'front_camera': True,
            'down_camera': True,
            'collision': True,
            'obstacle_distance': True,
            'goal_relative': True,
        }
        
        # Initialize ROS2 if available
        if ROS_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()
            
            self.node = RosEnvNode(namespace, use_zed=use_zed)
            self.executor = rclpy.executors.SingleThreadedExecutor()
            self.executor.add_node(self.node)
            
            # Spin in background thread
            from threading import Thread
            self.ros_thread = Thread(target=self._spin_ros, daemon=True)
            self.ros_thread.start()
        else:
            # Use mock for testing without ROS
            self.node = RosEnvNode(namespace, use_zed=use_zed)
            logger.warning("ROS2 not available, using mock objects for testing")
        
        # Wait for initial sensor data
        self._wait_for_sensors()
        
        # Define spaces
        self._define_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        
        logger.info(f"RosEnv initialized with namespace: {namespace}")
    
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
        
        if self.observation_config.get('front_camera', True):
            obs_dict['front_camera'] = gym.spaces.Box(
                low=0, high=255, 
                shape=(*self.camera_resolution, 3), 
                dtype=np.uint8
            )
        
        if self.observation_config.get('down_camera', True) and not self.use_zed:
            # Only used if not using ZED Mini (which only has one camera)
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
        
        logger.info(f"Observation space: {self.observation_space}")
        logger.info(f"Action space: {self.action_space}")
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation from ROS2 state."""
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
        
        if self.observation_config.get('front_camera', True):
            img = state['front_camera_image']
            if img is not None:
                # Downsample image
                import cv2
                img_resized = cv2.resize(img, self.camera_resolution)
                obs['front_camera'] = img_resized.astype(np.uint8)
            else:
                obs['front_camera'] = np.zeros((*self.camera_resolution, 3), dtype=np.uint8)
        
        if self.observation_config.get('down_camera', True) and not self.use_zed:
            # Only used if not using ZED Mini (which only has one camera)
            img = state['down_camera_image']
            if img is not None:
                import cv2
                img_resized = cv2.resize(img, self.camera_resolution)
                obs['down_camera'] = img_resized.astype(np.uint8)
            else:
                obs['down_camera'] = np.zeros((*self.camera_resolution, 3), dtype=np.uint8)
        
        if self.observation_config.get('collision', True):
            obs['collision'] = np.array([float(state['collision_flag'])], dtype=np.float32)
        
        if self.observation_config.get('obstacle_distance', True):
            obs['obstacle_distance'] = np.array([state['distance_to_obstacle']], dtype=np.float32)
        
        if self.observation_config.get('goal_relative', True):
            obs['goal_relative'] = (self.goal_position - state['position']).astype(np.float32)
        
        return obs
    
    def _process_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process action into linear and angular velocity commands.
        
        Returns:
            (linear_velocity, angular_velocity) in m/s and rad/s
        """
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
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Send reset command to simulation
        self.node.send_reset()
        
        # Wait for reset to complete
        time.sleep(0.5)
        self._wait_for_sensors()
        
        # If using MAVROS and auto-arm/offboard are enabled, set up drone
        if self.use_mavros:
            if self.auto_offboard:
                self.node.set_mode("OFFBOARD")
                logger.info("Set mode to OFFBOARD")
                
            if self.auto_arm:
                success = self.node.arm(True)
                logger.info(f"Arming {'successful' if success else 'failed'}")
                
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Update goal position if provided in options
        if options and 'goal_position' in options:
            self.goal_position = np.array(options['goal_position'])
        
        observation = self._get_observation()
        info = {
            'episode_step': self.current_step,
            'goal_position': self.goal_position.tolist(),
            'target_altitude': self.target_altitude,
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Process and send action
        linear_vel, angular_vel = self._process_action(action)
        self.node.send_velocity_command(linear_vel, angular_vel)
        
        # Wait for step duration
        time.sleep(self.step_duration)
        
        # Get new observation
        observation = self._get_observation()
        state = self.node.state.get_snapshot()
        
        # Calculate reward (placeholder - use actual reward function)
        reward = self._calculate_reward(state, action)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if state['collision_flag']:
            terminated = True
            reward -= 10.0  # collision penalty
        
        # Check if goal reached
        distance_to_goal = np.linalg.norm(state['position'] - self.goal_position)
        if distance_to_goal < 0.2:  # within 20cm of goal
            terminated = True
            reward += 5.0  # goal bonus
        
        # Check truncation
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # Update episode reward
        self.episode_reward += reward
        
        # Build info dict
        info = {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'distance_to_goal': distance_to_goal,
            'collision': state['collision_flag'],
            'position': state['position'].tolist(),
            'velocity': state['linear_velocity'].tolist(),
        }
        
        # Add MAVROS specific info
        if self.use_mavros:
            info.update({
                'armed': state['armed'],
                'flight_mode': state['flight_mode'],
            })
        
        return observation, float(reward), terminated, truncated, info
    
    def _calculate_reward(self, state: Dict[str, Any], action: np.ndarray) -> float:
        """
        Calculate reward. This is a placeholder - actual implementation
        should use the reward functions from rl_agent.rewards.
        """
        # Distance to goal reward
        distance = np.linalg.norm(state['position'] - self.goal_position)
        max_distance = np.sqrt(10**2 + 10**2 + 3**2)  # room diagonal
        distance_reward = max(0.0, 1.0 - (distance / max_distance))
        
        # Collision penalty
        collision_penalty = -1.0 if state['collision_flag'] else 0.0
        
        # Time penalty
        time_penalty = -0.01
        
        return distance_reward + collision_penalty + time_penalty
    
    def close(self):
        """Clean up ROS2 resources."""
        if ROS_AVAILABLE:
            if hasattr(self, 'executor'):
                self.executor.shutdown()
            if hasattr(self, 'node'):
                self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        else:
            # Clean up mock resources
            if hasattr(self, 'node'):
                self.node.destroy_node()
                
        logger.info("RosEnv closed")
    
    def render(self):
        """Render is handled by Gazebo visualization."""
        pass
