import numpy as np
from typing import Optional, Dict, Tuple, List, Union, Callable
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
import os

# Try to import ROS packages
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    Node = object  # Dummy for type hints
    from .mock_ros import MockNode as Node, MockImage as Image

# Try to import ZED SDK if available
try:
    import pyzed.sl as sl
    ZED_SDK_AVAILABLE = True
except ImportError:
    ZED_SDK_AVAILABLE = False

# Try to import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

logger = logging.getLogger(__name__)


class ZEDResolution(Enum):
    """Available ZED camera resolutions."""
    HD2K = 0      # 2208*1242
    HD1080 = 1    # 1920*1080
    HD720 = 2     # 1280*720
    VGA = 3       # 672*376
    WVGA = 4      # 640*480


@dataclass
class ZEDData:
    """Thread-safe container for ZED camera data."""
    rgb_image: np.ndarray = None
    depth_image: np.ndarray = None
    point_cloud: np.ndarray = None
    confidence: np.ndarray = None
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4x4 transformation matrix
    timestamp: float = 0.0
    
    def __post_init__(self):
        self._lock = threading.Lock()
    
    def update(self, **kwargs):
        """Thread-safe update of camera data."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get_snapshot(self) -> Dict:
        """Thread-safe snapshot of current camera data."""
        with self._lock:
            return {
                'rgb_image': self.rgb_image.copy() if self.rgb_image is not None else None,
                'depth_image': self.depth_image.copy() if self.depth_image is not None else None,
                'point_cloud': self.point_cloud.copy() if self.point_cloud is not None else None,
                'confidence': self.confidence.copy() if self.confidence is not None else None,
                'pose': self.pose.copy(),
                'timestamp': self.timestamp,
            }


class ROSZEDInterface:
    """
    Interface for ZED Mini camera using ROS topics.
    
    This class helps integrate the ZED Mini with the DeepFlyer platform by
    subscribing to standard ZED ROS topics and processing the data.
    """
    
    def __init__(
        self, 
        node: Node,
        camera_name: str = "zed_mini",
        node_name: str = "zed_node",
        resolution: Tuple[int, int] = (640, 360),
    ):
        """
        Initialize ZED ROS interface.
        
        Args:
            node: ROS node to use for subscribing
            camera_name: Name of the camera (zed or zed_mini)
            node_name: Name of the ROS node running ZED
            resolution: Target resolution for downsampling
        """
        self.node = node
        self.camera_name = camera_name
        self.node_name = node_name
        self.resolution = resolution
        self.data = ZEDData()
        
        # Initialize CV bridge if ROS is available
        self.bridge = CvBridge() if ROS_AVAILABLE else None
        
        # Topic paths
        base_topic = f"/{camera_name}/{node_name}"
        self.rgb_topic = f"{base_topic}/rgb/image_rect_color"
        self.depth_topic = f"{base_topic}/depth/depth_registered"
        self.pointcloud_topic = f"{base_topic}/point_cloud/cloud_registered"
        self.confidence_topic = f"{base_topic}/confidence/confidence_map"
        self.pose_topic = f"{base_topic}/pose"
        
        # Set up subscriptions
        if hasattr(self.node, "create_subscription"):
            # RGB image
            self.rgb_sub = self.node.create_subscription(
                Image, self.rgb_topic, self._rgb_callback, 10
            )
            
            # Depth map
            self.depth_sub = self.node.create_subscription(
                Image, self.depth_topic, self._depth_callback, 10
            )
            
            # Confidence map (optional)
            self.conf_sub = self.node.create_subscription(
                Image, self.confidence_topic, self._confidence_callback, 10
            )
            
            logger.info(f"ZED ROS interface initialized: {camera_name}/{node_name}")
        else:
            logger.warning("Node does not support subscriptions, using mock interface")
    
    def _rgb_callback(self, msg: Image):
        """Process RGB image from ZED camera."""
        try:
            if ROS_AVAILABLE and self.bridge:
                # Convert ROS Image to OpenCV format
                cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
                
                # Resize if needed
                if cv_image.shape[0] != self.resolution[1] or cv_image.shape[1] != self.resolution[0]:
                    if OPENCV_AVAILABLE:
                        cv_image = cv2.resize(cv_image, self.resolution)
            else:
                # Use mock data
                cv_image = msg.data  # Already numpy array in mock
                
            self.data.update(
                rgb_image=cv_image,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Error processing RGB image: {e}")
    
    def _depth_callback(self, msg: Image):
        """Process depth image from ZED camera."""
        try:
            if ROS_AVAILABLE and self.bridge:
                # ZED depth is 32-bit float in meters
                depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                
                # Resize if needed
                if depth_image.shape[0] != self.resolution[1] or depth_image.shape[1] != self.resolution[0]:
                    if OPENCV_AVAILABLE:
                        depth_image = cv2.resize(depth_image, self.resolution)
            else:
                # Use mock data
                depth_image = msg.data  # Already numpy array in mock
                
            self.data.update(
                depth_image=depth_image,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Error processing depth image: {e}")
    
    def _confidence_callback(self, msg: Image):
        """Process confidence map from ZED camera."""
        try:
            if ROS_AVAILABLE and self.bridge:
                # ZED confidence is 8-bit grayscale
                confidence = self.bridge.imgmsg_to_cv2(msg, "mono8")
                
                # Resize if needed
                if confidence.shape[0] != self.resolution[1] or confidence.shape[1] != self.resolution[0]:
                    if OPENCV_AVAILABLE:
                        confidence = cv2.resize(confidence, self.resolution)
            else:
                # Use mock data
                confidence = msg.data  # Already numpy array in mock
                
            self.data.update(
                confidence=confidence,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Error processing confidence map: {e}")
    
    def get_data(self) -> Dict:
        """Get the latest camera data."""
        return self.data.get_snapshot()


class DirectZEDInterface:
    """
    Direct interface to ZED camera using the ZED SDK.
    
    This class provides direct access to the ZED camera without requiring ROS,
    making it useful for development and testing on systems without ROS.
    """
    
    def __init__(
        self,
        resolution: ZEDResolution = ZEDResolution.HD720,
        fps: int = 30,
        depth_mode: str = "ULTRA",  # PERFORMANCE, QUALITY, ULTRA
        min_depth: float = 0.3,     # Minimum depth in meters
        max_depth: float = 10.0,    # Maximum depth in meters
        target_resolution: Tuple[int, int] = (640, 360),  # For downsampling
    ):
        """
        Initialize direct ZED camera interface.
        
        Args:
            resolution: ZED camera resolution
            fps: Frames per second
            depth_mode: Depth mode (PERFORMANCE, QUALITY, ULTRA)
            min_depth: Minimum depth detection distance (meters)
            max_depth: Maximum depth detection distance (meters)
            target_resolution: Resolution to downsample images to
        """
        if not ZED_SDK_AVAILABLE:
            raise ImportError("ZED SDK not available. Please install pyzed.")
        
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.target_resolution = target_resolution
        self.data = ZEDData()
        
        # Initialize camera
        self.zed = sl.Camera()
        self.is_opened = False
        self._init_params()
        
        # Data containers
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        self.confidence = sl.Mat()
        
        # Tracking status
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
        
        # Positional tracking
        self.tracking_params = sl.PositionalTrackingParameters()
        self.pose = sl.Pose()
        self.tracking_enabled = False
        
        logger.info("DirectZEDInterface initialized")
    
    def _init_params(self):
        """Initialize camera parameters."""
        init_params = sl.InitParameters()
        
        # Set configuration parameters
        init_params.camera_resolution = {
            ZEDResolution.HD2K: sl.RESOLUTION.HD2K,
            ZEDResolution.HD1080: sl.RESOLUTION.HD1080,
            ZEDResolution.HD720: sl.RESOLUTION.HD720,
            ZEDResolution.VGA: sl.RESOLUTION.VGA,
            ZEDResolution.WVGA: sl.RESOLUTION.WVGA
        }[self.resolution]
        
        init_params.camera_fps = self.fps
        
        init_params.depth_mode = {
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
        }[self.depth_mode]
        
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = self.min_depth
        init_params.depth_maximum_distance = self.max_depth
        
        self.init_params = init_params
    
    def open(self) -> bool:
        """Open the ZED camera."""
        if self.is_opened:
            logger.warning("ZED camera already opened")
            return True
        
        # Open the camera
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"Failed to open ZED camera: {status}")
            return False
        
        # Camera opened successfully
        self.is_opened = True
        logger.info(f"ZED camera opened: {self.zed.get_camera_information().serial_number}")
        return True
    
    def enable_tracking(self) -> bool:
        """Enable positional tracking."""
        if not self.is_opened:
            logger.error("Camera not opened, cannot enable tracking")
            return False
        
        # Set tracking parameters (defaults are fine for most cases)
        status = self.zed.enable_positional_tracking(self.tracking_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"Failed to enable positional tracking: {status}")
            return False
        
        self.tracking_enabled = True
        logger.info("Positional tracking enabled")
        return True
    
    def grab(self) -> bool:
        """Grab a new frame from the ZED camera."""
        if not self.is_opened:
            logger.error("Camera not opened, cannot grab frame")
            return False
        
        # Grab frame
        status = self.zed.grab(self.runtime_params)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.error(f"Failed to grab frame: {status}")
            return False
        
        # Retrieve data
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        self.zed.retrieve_measure(self.confidence, sl.MEASURE.CONFIDENCE)
        
        # Update data
        rgb = self.image.get_data()
        depth = self.depth.get_data()
        point_cloud = self.point_cloud.get_data()
        confidence = self.confidence.get_data()
        
        # Resize if needed
        if rgb.shape[:2] != self.target_resolution[::-1]:  # Note: shape is (h,w) but resolution is (w,h)
            if OPENCV_AVAILABLE:
                rgb = cv2.resize(rgb, self.target_resolution)
                depth = cv2.resize(depth, self.target_resolution)
                confidence = cv2.resize(confidence, self.target_resolution)
                # Note: point cloud is not resized as it's a different format
        
        # Get pose if tracking is enabled
        if self.tracking_enabled:
            self.zed.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
            pose_data = self.pose.pose_data()
            # Convert to 4x4 transformation matrix
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = pose_data.rotation_matrix
            pose_matrix[:3, 3] = pose_data.translation
        else:
            pose_matrix = np.eye(4)
        
        self.data.update(
            rgb_image=rgb,
            depth_image=depth,
            point_cloud=point_cloud,
            confidence=confidence,
            pose=pose_matrix,
            timestamp=time.time()
        )
        
        return True
    
    def get_data(self) -> Dict:
        """Get the latest camera data."""
        return self.data.get_snapshot()
    
    def close(self):
        """Close the ZED camera."""
        if self.is_opened:
            if self.tracking_enabled:
                self.zed.disable_positional_tracking()
                self.tracking_enabled = False
            
            self.zed.close()
            self.is_opened = False
            logger.info("ZED camera closed")


class ZEDDepthProcessor:
    """
    Process ZED depth data for RL applications.
    
    This class helps extract useful information from ZED depth data,
    such as obstacle distances, collision detection, and environment maps.
    """
    
    def __init__(
        self,
        min_depth: float = 0.3,
        max_depth: float = 10.0,
        collision_threshold: float = 0.5,  # Distance for collision detection
    ):
        """
        Initialize ZED depth processor.
        
        Args:
            min_depth: Minimum valid depth in meters
            max_depth: Maximum valid depth in meters
            collision_threshold: Distance threshold for collision detection
        """
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.collision_threshold = collision_threshold
    
    def get_min_distance(self, depth_image: np.ndarray, region: str = "center") -> float:
        """
        Get minimum distance in specified region of depth image.
        
        Args:
            depth_image: Depth image from ZED camera (meters)
            region: Region to consider ("center", "all", "bottom", "top")
        
        Returns:
            Minimum distance in meters
        """
        if depth_image is None or depth_image.size == 0:
            return float('inf')
        
        # Define region of interest
        h, w = depth_image.shape
        
        if region == "center":
            # Central region (middle 1/3)
            roi = depth_image[h//3:2*h//3, w//3:2*w//3]
        elif region == "bottom":
            # Bottom half
            roi = depth_image[h//2:, :]
        elif region == "top":
            # Top half
            roi = depth_image[:h//2, :]
        else:
            # Full image
            roi = depth_image
        
        # Filter out invalid values (0 or negative)
        valid_depths = roi[(roi >= self.min_depth) & (roi <= self.max_depth)]
        
        if valid_depths.size == 0:
            return float('inf')
        
        return float(np.min(valid_depths))
    
    def detect_collision(self, depth_image: np.ndarray) -> bool:
        """
        Detect if there's an obstacle too close to the drone.
        
        Args:
            depth_image: Depth image from ZED camera (meters)
        
        Returns:
            True if collision detected, False otherwise
        """
        min_distance = self.get_min_distance(depth_image)
        return min_distance < self.collision_threshold
    
    def get_obstacle_directions(self, depth_image: np.ndarray) -> Dict[str, float]:
        """
        Get minimum distances in different directions.
        
        Args:
            depth_image: Depth image from ZED camera (meters)
        
        Returns:
            Dictionary with minimum distances in different directions
        """
        if depth_image is None or depth_image.size == 0:
            return {
                'front': float('inf'),
                'left': float('inf'),
                'right': float('inf'),
                'top': float('inf'),
                'bottom': float('inf')
            }
        
        h, w = depth_image.shape
        
        # Calculate minimum depths in different regions
        center_min = self.get_min_distance(depth_image, "center")
        
        # Divide image into regions
        left_region = depth_image[:, :w//3]
        right_region = depth_image[:, 2*w//3:]
        top_region = depth_image[:h//3, :]
        bottom_region = depth_image[2*h//3:, :]
        
        # Get minimum valid distances
        left_min = self.get_min_distance(left_region, "all")
        right_min = self.get_min_distance(right_region, "all")
        top_min = self.get_min_distance(top_region, "all")
        bottom_min = self.get_min_distance(bottom_region, "all")
        
        return {
            'front': center_min,
            'left': left_min,
            'right': right_min,
            'top': top_min,
            'bottom': bottom_min
        }
    
    def create_depth_representation(self, depth_image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Create a normalized or colorized representation of depth.
        
        Args:
            depth_image: Depth image from ZED camera (meters)
            normalize: Whether to normalize to [0, 1] or create RGB visualization
        
        Returns:
            Processed depth image
        """
        if depth_image is None or depth_image.size == 0:
            return None
        
        # Filter invalid values
        filtered = depth_image.copy()
        filtered[filtered < self.min_depth] = self.max_depth
        filtered[filtered > self.max_depth] = self.max_depth
        
        if normalize:
            # Normalize to [0, 1]
            normalized = (filtered - self.min_depth) / (self.max_depth - self.min_depth)
            return normalized
        else:
            # Create color visualization if OpenCV is available
            if OPENCV_AVAILABLE:
                normalized = (filtered - self.min_depth) / (self.max_depth - self.min_depth)
                normalized = (normalized * 255).astype(np.uint8)
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                return colored
            else:
                # Fall back to normalized grayscale
                return (filtered - self.min_depth) / (self.max_depth - self.min_depth)


def create_zed_interface(use_ros: bool = True, **kwargs) -> Union[ROSZEDInterface, DirectZEDInterface]:
    """
    Factory function to create appropriate ZED interface.
    
    Args:
        use_ros: Whether to use ROS interface or direct SDK
        **kwargs: Additional arguments for the chosen interface
    
    Returns:
        ZED interface instance
    """
    if use_ros and ROS_AVAILABLE:
        # Check if we have a ROS node
        if 'node' not in kwargs:
            # Try to create a standalone node
            if not rclpy.ok():
                rclpy.init()
            node = rclpy.create_node('zed_interface_node')
            kwargs['node'] = node
        
        return ROSZEDInterface(**kwargs)
    elif ZED_SDK_AVAILABLE:
        return DirectZEDInterface(**kwargs)
    else:
        raise ImportError("Neither ROS nor ZED SDK is available")


if __name__ == "__main__":
    # Simple test to verify module loads
    logger.info("ZED integration module loaded successfully")
    
    # List available interfaces
    interfaces = []
    if ROS_AVAILABLE:
        interfaces.append("ROS")
    if ZED_SDK_AVAILABLE:
        interfaces.append("ZED SDK")
    
    logger.info(f"Available interfaces: {', '.join(interfaces) or 'None'}") 