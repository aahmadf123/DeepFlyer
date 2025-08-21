"""
ZED Mini Stereo Camera Integration for DeepFlyer
Provides depth perception and visual SLAM capabilities
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# Try to import ZED SDK
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    logger.warning("ZED SDK not available, using mock interface")


class ZEDInterface(ABC):
    """Abstract base class for ZED camera interface"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize camera"""
        pass
    
    @abstractmethod
    def grab_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Grab RGB and depth frames"""
        pass
    
    @abstractmethod
    def get_depth_at_point(self, x: int, y: int) -> float:
        """Get depth value at specific pixel"""
        pass
    
    @abstractmethod
    def close(self):
        """Close camera connection"""
        pass


class ZEDMiniCamera(ZEDInterface):
    """ZED Mini stereo camera implementation"""
    
    def __init__(self, resolution: str = "HD720", fps: int = 30, depth_mode: str = "NEURAL"):
        """
        Initialize ZED Mini camera
        
        Args:
            resolution: Resolution mode (HD720, HD1080, VGA)
            fps: Frames per second (15, 30, 60)
            depth_mode: Depth computation mode (NEURAL, QUALITY, PERFORMANCE)
        """
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.camera = None
        self.runtime_params = None
        self.image = None
        self.depth = None
        self.point_cloud = None
        
        if not ZED_AVAILABLE:
            raise RuntimeError("ZED SDK not installed")
    
    def initialize(self) -> bool:
        """Initialize ZED Mini camera"""
        try:
            self.camera = sl.Camera()
            
            # Set initialization parameters
            init_params = sl.InitParameters()
            
            # Set resolution
            if self.resolution == "HD720":
                init_params.camera_resolution = sl.RESOLUTION.HD720
            elif self.resolution == "HD1080":
                init_params.camera_resolution = sl.RESOLUTION.HD1080
            else:
                init_params.camera_resolution = sl.RESOLUTION.VGA
            
            # Set FPS
            init_params.camera_fps = self.fps
            
            # Set depth mode
            if self.depth_mode == "NEURAL":
                init_params.depth_mode = sl.DEPTH_MODE.NEURAL
            elif self.depth_mode == "QUALITY":
                init_params.depth_mode = sl.DEPTH_MODE.QUALITY
            else:
                init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            
            # Set coordinate system
            init_params.coordinate_units = sl.UNIT.METER
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            
            # Enable depth computation
            init_params.depth_minimum_distance = 0.3  # 30cm minimum
            init_params.depth_maximum_distance = 10.0  # 10m maximum
            
            # Open camera
            err = self.camera.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                logger.error(f"Failed to open ZED camera: {err}")
                return False
            
            # Create runtime parameters
            self.runtime_params = sl.RuntimeParameters()
            self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
            
            # Initialize data containers
            self.image = sl.Mat()
            self.depth = sl.Mat()
            self.point_cloud = sl.Mat()
            
            logger.info(f"ZED Mini initialized: {self.resolution} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"ZED initialization failed: {e}")
            return False
    
    def grab_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Grab RGB and depth frames from ZED Mini
        
        Returns:
            rgb_image: RGB image as numpy array
            depth_map: Depth map as numpy array (meters)
        """
        if self.camera is None:
            return None, None
        
        # Grab new frame
        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve RGB image
            self.camera.retrieve_image(self.image, sl.VIEW.LEFT)
            rgb_image = self.image.get_data()[:, :, :3]  # Remove alpha channel
            
            # Retrieve depth map
            self.camera.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            depth_map = self.depth.get_data()
            
            return rgb_image, depth_map
        
        return None, None
    
    def get_depth_at_point(self, x: int, y: int) -> float:
        """
        Get depth value at specific pixel coordinate
        
        Args:
            x: X coordinate in image
            y: Y coordinate in image
        
        Returns:
            Depth in meters
        """
        if self.depth is None:
            return -1.0
        
        depth_value = self.depth.get_value(x, y)[1]  # Get depth value
        if np.isnan(depth_value) or np.isinf(depth_value):
            return -1.0
        
        return depth_value
    
    def get_3d_position(self, x: int, y: int) -> Optional[np.ndarray]:
        """
        Get 3D position of a pixel in camera coordinates
        
        Args:
            x: X coordinate in image
            y: Y coordinate in image
        
        Returns:
            3D position [x, y, z] in meters
        """
        if self.camera is None:
            return None
        
        # Retrieve point cloud
        self.camera.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        
        # Get 3D point
        point_3d = self.point_cloud.get_value(x, y)[1][:3]
        
        if np.any(np.isnan(point_3d)) or np.any(np.isinf(point_3d)):
            return None
        
        return point_3d
    
    def get_camera_pose(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get camera pose from visual-inertial tracking
        
        Returns:
            Dictionary with 'position' and 'orientation'
        """
        if self.camera is None:
            return None
        
        # Enable positional tracking if not enabled
        if not self.camera.is_position_tracking_enabled():
            tracking_params = sl.PositionalTrackingParameters()
            self.camera.enable_positional_tracking(tracking_params)
        
        # Get pose
        zed_pose = sl.Pose()
        if self.camera.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD) == sl.TRACKING_STATE.OK:
            position = zed_pose.get_translation().get()
            orientation = zed_pose.get_orientation().get()
            
            return {
                'position': np.array([position[0], position[1], position[2]]),
                'orientation': np.array([orientation[0], orientation[1], 
                                        orientation[2], orientation[3]])
            }
        
        return None
    
    def close(self):
        """Close ZED camera connection"""
        if self.camera is not None:
            self.camera.close()
            self.camera = None
            logger.info("ZED Mini camera closed")


class MockZEDCamera(ZEDInterface):
    """Mock ZED camera for testing without hardware"""
    
    def __init__(self, resolution: str = "HD720", fps: int = 30, depth_mode: str = "NEURAL"):
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.initialized = False
        
        # Image dimensions
        if resolution == "HD720":
            self.width, self.height = 1280, 720
        elif resolution == "HD1080":
            self.width, self.height = 1920, 1080
        else:
            self.width, self.height = 640, 480
    
    def initialize(self) -> bool:
        """Initialize mock camera"""
        self.initialized = True
        logger.info(f"Mock ZED camera initialized: {self.resolution} @ {self.fps}fps")
        return True
    
    def grab_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate mock RGB and depth frames"""
        if not self.initialized:
            return None, None
        
        # Generate mock RGB image (random noise for now)
        rgb_image = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        # Generate mock depth map (gradient from center)
        y, x = np.ogrid[:self.height, :self.width]
        center_x, center_y = self.width / 2, self.height / 2
        depth_map = np.sqrt((x - center_x)**2 + (y - center_y)**2) / 500.0 + 1.0
        depth_map = depth_map.astype(np.float32)
        
        return rgb_image, depth_map
    
    def get_depth_at_point(self, x: int, y: int) -> float:
        """Get mock depth value"""
        if not self.initialized:
            return -1.0
        
        # Simple distance from center
        center_x, center_y = self.width / 2, self.height / 2
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2) / 500.0 + 1.0
        return float(distance)
    
    def close(self):
        """Close mock camera"""
        self.initialized = False
        logger.info("Mock ZED camera closed")


class ZEDROSInterface(ZEDInterface):
    """ZED camera interface via ROS2 topics"""
    
    def __init__(self, node, namespace: str = "zed_mini"):
        """
        Initialize ZED ROS interface
        
        Args:
            node: ROS2 node instance
            namespace: ROS2 namespace for ZED topics
        """
        self.node = node
        self.namespace = namespace
        self.rgb_image = None
        self.depth_map = None
        
        # Import ROS dependencies
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        
        self.bridge = CvBridge()
        
        # Subscribe to topics
        self.rgb_sub = node.create_subscription(
            Image,
            f'/{namespace}/zed_node/rgb/image_rect_color',
            self._rgb_callback,
            10
        )
        
        self.depth_sub = node.create_subscription(
            Image,
            f'/{namespace}/zed_node/depth/depth_registered',
            self._depth_callback,
            10
        )
    
    def _rgb_callback(self, msg):
        """Process RGB image from ROS"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logger.error(f"Failed to process RGB image: {e}")
    
    def _depth_callback(self, msg):
        """Process depth map from ROS"""
        try:
            self.depth_map = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            logger.error(f"Failed to process depth map: {e}")
    
    def initialize(self) -> bool:
        """Initialize ROS interface"""
        logger.info(f"ZED ROS interface initialized on namespace: {self.namespace}")
        return True
    
    def grab_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get latest frames from ROS topics"""
        return self.rgb_image, self.depth_map
    
    def get_depth_at_point(self, x: int, y: int) -> float:
        """Get depth at point from last depth map"""
        if self.depth_map is None:
            return -1.0
        
        if 0 <= y < self.depth_map.shape[0] and 0 <= x < self.depth_map.shape[1]:
            return float(self.depth_map[y, x])
        
        return -1.0
    
    def close(self):
        """Close ROS interface"""
        # ROS subscriptions are cleaned up automatically
        pass


def create_zed_interface(mode: str = "direct", **kwargs) -> ZEDInterface:
    """
    Factory function to create appropriate ZED interface
    
    Args:
        mode: Interface mode ("direct", "ros", "mock")
        **kwargs: Additional arguments for the interface
    
    Returns:
        ZED interface instance
    """
    if mode == "direct":
        if ZED_AVAILABLE:
            return ZEDMiniCamera(**kwargs)
        else:
            logger.warning("ZED SDK not available, using mock interface")
            return MockZEDCamera(**kwargs)
    elif mode == "ros":
        return ZEDROSInterface(**kwargs)
    elif mode == "mock":
        return MockZEDCamera(**kwargs)
    else:
        raise ValueError(f"Unknown ZED interface mode: {mode}")