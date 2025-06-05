"""Mock ROS components for testing without ROS installation."""

import numpy as np
import time
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MockImage:
    """Mock sensor_msgs/Image."""
    data: np.ndarray
    height: int
    width: int
    encoding: str = "rgb8"
    
    def __post_init__(self):
        if self.data is None:
            # Generate random image data for testing
            self.data = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)


@dataclass
class MockPose:
    """Mock geometry_msgs/Pose."""
    position: np.ndarray = None
    orientation: np.ndarray = None  # quaternion [x, y, z, w]
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 1.5])  # Default hover position
        if self.orientation is None:
            self.orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion


@dataclass
class MockTwist:
    """Mock geometry_msgs/Twist."""
    linear: np.ndarray = None
    angular: np.ndarray = None
    
    def __post_init__(self):
        if self.linear is None:
            self.linear = np.zeros(3)
        if self.angular is None:
            self.angular = np.zeros(3)


@dataclass
class MockImu:
    """Mock sensor_msgs/Imu."""
    linear_acceleration: np.ndarray = None
    angular_velocity: np.ndarray = None
    orientation: np.ndarray = None
    
    def __post_init__(self):
        if self.linear_acceleration is None:
            # Gravity + small noise
            self.linear_acceleration = np.array([0.0, 0.0, 9.81]) + np.random.randn(3) * 0.1
        if self.angular_velocity is None:
            self.angular_velocity = np.random.randn(3) * 0.01
        if self.orientation is None:
            self.orientation = np.array([0.0, 0.0, 0.0, 1.0])


class MockPublisher:
    """Mock ROS publisher."""
    def __init__(self, topic: str, msg_type: Any):
        self.topic = topic
        self.msg_type = msg_type
        self.published_msgs = []
        logger.info(f"Mock publisher created for topic: {topic}")
    
    def publish(self, msg: Any):
        """Store published messages for testing."""
        self.published_msgs.append(msg)
        logger.debug(f"Published to {self.topic}: {type(msg).__name__}")


class MockSubscriber:
    """Mock ROS subscriber with simulated data."""
    def __init__(self, topic: str, msg_type: Any, callback: Callable):
        self.topic = topic
        self.msg_type = msg_type
        self.callback = callback
        self.is_active = True
        logger.info(f"Mock subscriber created for topic: {topic}")
        
        # Start simulated data based on topic
        import threading
        self.thread = threading.Thread(target=self._simulate_data, daemon=True)
        self.thread.start()
    
    def _simulate_data(self):
        """Simulate sensor data at appropriate rates."""
        while self.is_active:
            if "/pose" in self.topic:
                # Simulate drone hovering with small movements
                pose = MockPose()
                pose.position += np.random.randn(3) * 0.01  # Small position noise
                self.callback(pose)
                time.sleep(0.02)  # 50 Hz
                
            elif "/imu" in self.topic:
                imu = MockImu()
                self.callback(imu)
                time.sleep(0.01)  # 100 Hz
                
            elif "/image" in self.topic:
                # Simulate camera images
                if "zed" in self.topic:
                    img = MockImage(None, 376, 672)  # ZED Mini WVGA resolution
                else:
                    img = MockImage(None, 480, 640)  # Default camera
                self.callback(img)
                time.sleep(0.067)  # ~15 Hz
                
            elif "/odom" in self.topic:
                # Simulate odometry
                twist = MockTwist()
                twist.linear = np.random.randn(3) * 0.1
                twist.angular = np.random.randn(3) * 0.05
                self.callback(twist)
                time.sleep(0.01)  # 100 Hz
                
            elif "/collision" in self.topic:
                # No collision by default
                self.callback(False)
                time.sleep(1.0)  # 1 Hz check
                
            elif "/obstacle_distance" in self.topic:
                # Random distance between 1 and 10 meters
                distance = np.random.uniform(1.0, 10.0)
                self.callback(distance)
                time.sleep(0.05)  # 20 Hz
                
            elif "/battery" in self.topic:
                # Slowly decreasing battery
                battery = 1.0 - (time.time() % 300) / 300  # 5 minute battery life
                self.callback(battery)
                time.sleep(1.0)  # 1 Hz
                
            else:
                time.sleep(0.1)  # Default rate
    
    def destroy(self):
        """Stop the simulation thread."""
        self.is_active = False


class MockNode:
    """Mock ROS2 node for testing."""
    def __init__(self, name: str):
        self.name = name
        self.publishers = {}
        self.subscribers = {}
        logger.info(f"Mock ROS node '{name}' created")
    
    def create_publisher(self, msg_type: Any, topic: str, qos: Any = None) -> MockPublisher:
        """Create a mock publisher."""
        pub = MockPublisher(topic, msg_type)
        self.publishers[topic] = pub
        return pub
    
    def create_subscription(self, msg_type: Any, topic: str, callback: Callable, qos: Any = None) -> MockSubscriber:
        """Create a mock subscriber."""
        sub = MockSubscriber(topic, msg_type, callback)
        self.subscribers[topic] = sub
        return sub
    
    def get_logger(self):
        """Return the logger."""
        return logger
    
    def destroy_node(self):
        """Clean up subscribers."""
        for sub in self.subscribers.values():
            sub.destroy()


# MAVROS specific mocks
class MockMAVROSState:
    """Mock MAVROS state message."""
    def __init__(self):
        self.connected = True
        self.armed = False
        self.guided = False
        self.mode = "STABILIZE"
        self.system_status = 3  # STANDBY


class MockMAVROSService:
    """Mock MAVROS service calls."""
    @staticmethod
    def arm(arm: bool = True) -> bool:
        """Mock arming service."""
        logger.info(f"Mock arming: {arm}")
        return True
    
    @staticmethod
    def set_mode(mode: str) -> bool:
        """Mock mode change service."""
        logger.info(f"Mock set mode: {mode}")
        return True


# ZED specific mocks
class MockZEDNode:
    """Mock ZED camera node with realistic parameters."""
    def __init__(self):
        self.resolution = "WVGA"  # 672x376
        self.fps = 30
        self.depth_mode = "PERFORMANCE"
        self.min_depth = 0.3
        self.max_depth = 20.0
        
    def get_camera_info(self) -> Dict[str, Any]:
        """Return camera parameters."""
        return {
            "resolution": self.resolution,
            "fps": self.fps,
            "depth_mode": self.depth_mode,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "baseline": 0.12,  # 120mm for ZED Mini
        }


# Utility functions
def generate_test_trajectory() -> np.ndarray:
    """Generate a simple test trajectory for the drone."""
    t = np.linspace(0, 2*np.pi, 100)
    trajectory = np.column_stack([
        5 + 2 * np.cos(t),  # x: circle with radius 2m centered at (5, 5)
        5 + 2 * np.sin(t),  # y
        1.5 + 0.5 * np.sin(2*t),  # z: oscillate between 1m and 2m
    ])
    return trajectory


def simulate_drone_dynamics(current_pos: np.ndarray, target_vel: np.ndarray, dt: float = 0.05) -> np.ndarray:
    """Simple drone dynamics simulation."""
    # First order dynamics with some lag
    tau = 0.3  # time constant
    alpha = dt / (tau + dt)
    new_pos = current_pos + alpha * target_vel * dt
    return new_pos


# Export mock classes
__all__ = [
    'MockImage',
    'MockPose',
    'MockTwist',
    'MockImu',
    'MockPublisher',
    'MockSubscriber',
    'MockNode',
    'MockMAVROSState',
    'MockMAVROSService',
    'MockZEDNode',
    'generate_test_trajectory',
    'simulate_drone_dynamics',
] 