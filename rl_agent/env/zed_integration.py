"""
ZED Mini Camera Integration for DeepFlyer
Provides both ROS-based and direct SDK interfaces for stereo camera data
"""

import numpy as np
import cv2
import threading
import time
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# ROS2 imports (optional)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logging.warning("ROS2 not available. ZED will use direct SDK interface only.")

# ZED SDK imports (optional)
try:
    import pyzed.sl as sl
    ZED_SDK_AVAILABLE = True
except ImportError:
    ZED_SDK_AVAILABLE = False
    logging.warning("ZED SDK not available. ZED will use ROS interface only.")

logger = logging.getLogger(__name__)


@dataclass
class ZEDFrame:
    """Container for ZED camera frame data"""
    rgb_image: Optional[np.ndarray] = None
    depth_image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_id: int = 0
    
    # Intrinsic camera parameters
    fx: float = 0.0  # Focal length x
    fy: float = 0.0  # Focal length y
    cx: float = 0.0  # Principal point x
    cy: float = 0.0  # Principal point y
    
    # Camera pose (if available)
    position: Optional[np.ndarray] = None
    orientation: Optional[np.ndarray] = None
    
    # Quality metrics
    confidence: float = 0.0
    processing_time_ms: float = 0.0


class ZEDInterface(ABC):
    """Abstract base class for ZED camera interfaces"""
    
    @abstractmethod
    def start(self) -> bool:
        """Start the camera interface"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the camera interface"""
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[ZEDFrame]:
        """Get the latest camera frame"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if camera is connected and working"""
        pass


class ROSZEDInterface(ZEDInterface):
    """ROS2-based interface to ZED Mini camera using zed-ros2-wrapper"""
    
    def __init__(self, 
                 node: Optional[Node] = None,
                 namespace: str = "/zed_mini",
                 rgb_topic: str = "/zed_node/rgb/image_rect_color",
                 depth_topic: str = "/zed_node/depth/depth_registered",
                 camera_info_topic: str = "/zed_node/rgb/camera_info",
                 pose_topic: str = "/zed_node/pose",
                 frame_timeout: float = 1.0):
        """
        Initialize ROS-based ZED interface
        
        Args:
            node: ROS2 node for subscriptions (if None, creates internal node)
            namespace: ZED camera namespace
            rgb_topic: RGB image topic name
            depth_topic: Depth image topic name  
            camera_info_topic: Camera info topic name
            pose_topic: Camera pose topic name
            frame_timeout: Timeout for frame freshness (seconds)
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS2 not available. Cannot use ROSZEDInterface.")
        
        self.namespace = namespace
        self.frame_timeout = frame_timeout
        
        # Create or use provided node
        if node is None:
            rclpy.init()
            self.node = Node('zed_interface_node')
            self.owns_node = True
        else:
            self.node = node
            self.owns_node = False
        
        # Topic names
        self.rgb_topic = f"{namespace}{rgb_topic}"
        self.depth_topic = f"{namespace}{depth_topic}"
        self.camera_info_topic = f"{namespace}{camera_info_topic}"
        self.pose_topic = f"{namespace}{pose_topic}"
        
        # CV Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Data storage
        self.latest_frame = ZEDFrame()
        self.camera_info = None
        self.frame_lock = threading.Lock()
        
        # Subscribers
        self.rgb_sub = None
        self.depth_sub = None
        self.camera_info_sub = None
        self.pose_sub = None
        
        # Connection status
        self._connected = False
        self._last_rgb_time = 0.0
        self._last_depth_time = 0.0
        
        logger.info(f"ROSZEDInterface initialized with namespace: {namespace}")
    
    def start(self) -> bool:
        """Start ROS subscriptions to ZED topics"""
        try:
            # Create subscribers
            self.rgb_sub = self.node.create_subscription(
                Image, self.rgb_topic, self._rgb_callback, 10)
            
            self.depth_sub = self.node.create_subscription(
                Image, self.depth_topic, self._depth_callback, 10)
            
            self.camera_info_sub = self.node.create_subscription(
                CameraInfo, self.camera_info_topic, self._camera_info_callback, 10)
            
            self.pose_sub = self.node.create_subscription(
                PoseStamped, self.pose_topic, self._pose_callback, 10)
            
            # Start ROS spinning in separate thread
            self.ros_thread = threading.Thread(target=self._ros_spin_thread, daemon=True)
            self.ros_thread.start()
            
            # Wait for first frame
            start_time = time.time()
            while time.time() - start_time < 5.0:  # 5 second timeout
                if self._connected:
                    break
                time.sleep(0.1)
            
            if self._connected:
                logger.info("ZED ROS interface started successfully")
                return True
            else:
                logger.error("Failed to receive ZED data within timeout")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start ZED ROS interface: {e}")
            return False
    
    def stop(self) -> None:
        """Stop ROS subscriptions"""
        self._connected = False
        
        # Destroy subscribers
        if self.rgb_sub:
            self.node.destroy_subscription(self.rgb_sub)
        if self.depth_sub:
            self.node.destroy_subscription(self.depth_sub)
        if self.camera_info_sub:
            self.node.destroy_subscription(self.camera_info_sub)
        if self.pose_sub:
            self.node.destroy_subscription(self.pose_sub)
        
        # Cleanup node if we own it
        if self.owns_node:
            self.node.destroy_node()
            rclpy.shutdown()
        
        logger.info("ZED ROS interface stopped")
    
    def get_frame(self) -> Optional[ZEDFrame]:
        """Get the latest synchronized frame"""
        with self.frame_lock:
            current_time = time.time()
            
            # Check frame freshness
            if (current_time - self._last_rgb_time > self.frame_timeout or
                current_time - self._last_depth_time > self.frame_timeout):
                return None
            
            # Return copy of latest frame
            if self.latest_frame.rgb_image is not None and self.latest_frame.depth_image is not None:
                return ZEDFrame(
                    rgb_image=self.latest_frame.rgb_image.copy(),
                    depth_image=self.latest_frame.depth_image.copy(),
                    timestamp=self.latest_frame.timestamp,
                    frame_id=self.latest_frame.frame_id,
                    fx=self.latest_frame.fx,
                    fy=self.latest_frame.fy,
                    cx=self.latest_frame.cx,
                    cy=self.latest_frame.cy,
                    position=self.latest_frame.position,
                    orientation=self.latest_frame.orientation,
                    confidence=self.latest_frame.confidence,
                    processing_time_ms=self.latest_frame.processing_time_ms
                )
            
            return None
    
    def is_connected(self) -> bool:
        """Check if receiving fresh data from ZED"""
        current_time = time.time()
        return (self._connected and 
                current_time - self._last_rgb_time < self.frame_timeout and
                current_time - self._last_depth_time < self.frame_timeout)
    
    def _rgb_callback(self, msg: Image) -> None:
        """Process RGB image messages"""
        try:
            start_time = time.time()
            
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            with self.frame_lock:
                self.latest_frame.rgb_image = cv_image
                self.latest_frame.timestamp = time.time()
                self.latest_frame.frame_id = msg.header.seq if hasattr(msg.header, 'seq') else 0
                self.latest_frame.processing_time_ms = (time.time() - start_time) * 1000
                
                self._last_rgb_time = time.time()
                self._connected = True
                
        except Exception as e:
            logger.warning(f"Failed to process RGB image: {e}")
    
    def _depth_callback(self, msg: Image) -> None:
        """Process depth image messages"""
        try:
            # Convert ROS depth image to OpenCV (usually 32FC1)
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            with self.frame_lock:
                self.latest_frame.depth_image = cv_depth
                self._last_depth_time = time.time()
                
        except Exception as e:
            logger.warning(f"Failed to process depth image: {e}")
    
    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Process camera calibration info"""
        self.camera_info = msg
        
        with self.frame_lock:
            # Extract intrinsic parameters
            self.latest_frame.fx = msg.k[0]  # K[0,0]
            self.latest_frame.fy = msg.k[4]  # K[1,1] 
            self.latest_frame.cx = msg.k[2]  # K[0,2]
            self.latest_frame.cy = msg.k[5]  # K[1,2]
    
    def _pose_callback(self, msg: PoseStamped) -> None:
        """Process camera pose messages"""
        with self.frame_lock:
            # Extract position
            self.latest_frame.position = np.array([
                msg.pose.position.x,
                msg.pose.position.y, 
                msg.pose.position.z
            ])
            
            # Extract orientation (quaternion)
            self.latest_frame.orientation = np.array([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z
            ])
    
    def _ros_spin_thread(self) -> None:
        """ROS spinning thread"""
        try:
            rclpy.spin(self.node)
        except Exception as e:
            logger.error(f"ROS spin thread error: {e}")


class DirectZEDInterface(ZEDInterface):
    """Direct ZED SDK interface for maximum performance"""
    
    def __init__(self,
                 camera_id: int = 0,
                 resolution: str = "HD720",  # HD720, HD1080, VGA
                 fps: int = 60,
                 depth_mode: str = "PERFORMANCE",  # PERFORMANCE, MEDIUM, QUALITY
                 coordinate_system: str = "RIGHT_HANDED_Z_UP",
                 enable_tracking: bool = True,
                 enable_depth: bool = True):
        """
        Initialize direct ZED SDK interface
        
        Args:
            camera_id: Camera ID (0 for first camera)
            resolution: Camera resolution setting
            fps: Target frame rate
            depth_mode: Depth computation quality
            coordinate_system: Coordinate system for pose
            enable_tracking: Enable positional tracking
            enable_depth: Enable depth computation
        """
        if not ZED_SDK_AVAILABLE:
            raise RuntimeError("ZED SDK not available. Cannot use DirectZEDInterface.")
        
        self.camera_id = camera_id
        self.enable_tracking = enable_tracking
        self.enable_depth = enable_depth
        
        # Create ZED camera object
        self.zed = sl.Camera()
        
        # Create initialization parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.coordinate_system = getattr(sl.COORDINATE_SYSTEM, coordinate_system)
        self.init_params.enable_image_enhancement = True
        
        # Runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
        
        # Tracking parameters
        if enable_tracking:
            self.tracking_params = sl.PositionalTrackingParameters()
            self.tracking_params.enable_area_memory = True
            self.tracking_params.enable_pose_smoothing = True
        
        # Image containers
        self.rgb_sl = sl.Mat()
        self.depth_sl = sl.Mat()
        self.pose_sl = sl.Pose()
        
        # Status tracking
        self._connected = False
        self._frame_count = 0
        
        logger.info(f"DirectZEDInterface initialized with resolution: {resolution}, FPS: {fps}")
    
    def start(self) -> bool:
        """Start ZED camera"""
        try:
            # Open camera
            status = self.zed.open(self.init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                logger.error(f"Failed to open ZED camera: {status}")
                return False
            
            # Enable positional tracking if requested
            if self.enable_tracking:
                status = self.zed.enable_positional_tracking(self.tracking_params)
                if status != sl.ERROR_CODE.SUCCESS:
                    logger.warning(f"Failed to enable tracking: {status}")
                    # Continue without tracking
            
            # Get camera information
            self.camera_info = self.zed.get_camera_information()
            self._connected = True
            
            logger.info("ZED camera started successfully")
            logger.info(f"Camera model: {self.camera_info.camera_model}")
            logger.info(f"Serial number: {self.camera_info.serial_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start ZED camera: {e}")
            return False
    
    def stop(self) -> None:
        """Stop ZED camera"""
        if self._connected:
            if self.enable_tracking:
                self.zed.disable_positional_tracking()
            self.zed.close()
            self._connected = False
            logger.info("ZED camera stopped")
    
    def get_frame(self) -> Optional[ZEDFrame]:
        """Capture and return latest frame"""
        if not self._connected:
            return None
        
        try:
            start_time = time.time()
            
            # Grab frame
            status = self.zed.grab(self.runtime_params)
            if status != sl.ERROR_CODE.SUCCESS:
                return None
            
            # Retrieve images
            self.zed.retrieve_image(self.rgb_sl, sl.VIEW.LEFT)
            
            if self.enable_depth:
                self.zed.retrieve_measure(self.depth_sl, sl.MEASURE.DEPTH)
            
            # Get pose if tracking enabled
            pose_data = None
            orientation_data = None
            if self.enable_tracking:
                tracking_state = self.zed.get_position(self.pose_sl)
                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    translation = self.pose_sl.get_translation()
                    rotation = self.pose_sl.get_rotation_matrix()
                    pose_data = np.array([translation.get()[0], translation.get()[1], translation.get()[2]])
                    
                    # Convert rotation matrix to quaternion
                    orientation_data = self._rotation_matrix_to_quaternion(rotation.r)
            
            # Convert to numpy arrays
            rgb_np = self.rgb_sl.get_data()[:, :, :3]  # Remove alpha channel
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGBA2BGR)
            
            depth_np = None
            if self.enable_depth:
                depth_np = self.depth_sl.get_data()
            
            # Get camera intrinsics
            calibration = self.camera_info.camera_configuration.calibration_parameters.left_cam
            
            # Create frame object
            frame = ZEDFrame(
                rgb_image=rgb_np,
                depth_image=depth_np,
                timestamp=time.time(),
                frame_id=self._frame_count,
                fx=calibration.fx,
                fy=calibration.fy,
                cx=calibration.cx,
                cy=calibration.cy,
                position=pose_data,
                orientation=orientation_data,
                confidence=1.0,  # ZED provides high confidence data
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            self._frame_count += 1
            return frame
            
        except Exception as e:
            logger.warning(f"Failed to capture ZED frame: {e}")
            return None
    
    def is_connected(self) -> bool:
        """Check if camera is connected"""
        return self._connected
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to quaternion [w, x, y, z]"""
        trace = np.trace(R)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s  
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z])


class MockZEDInterface(ZEDInterface):
    """Mock ZED interface for testing without hardware"""
    
    def __init__(self, 
                 width: int = 1280,
                 height: int = 720,
                 fps: int = 30,
                 generate_synthetic_hoops: bool = True):
        """
        Initialize mock ZED interface
        
        Args:
            width: Image width
            height: Image height  
            fps: Simulated frame rate
            generate_synthetic_hoops: Whether to generate synthetic hoop images
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.generate_synthetic_hoops = generate_synthetic_hoops
        
        self._connected = False
        self._frame_count = 0
        self._start_time = 0.0
        
        # Synthetic camera parameters
        self.fx = width * 0.8
        self.fy = height * 0.8
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        logger.info(f"MockZEDInterface initialized: {width}x{height}@{fps}fps")
    
    def start(self) -> bool:
        """Start mock camera"""
        self._connected = True
        self._start_time = time.time()
        logger.info("Mock ZED camera started")
        return True
    
    def stop(self) -> None:
        """Stop mock camera"""
        self._connected = False
        logger.info("Mock ZED camera stopped")
    
    def get_frame(self) -> Optional[ZEDFrame]:
        """Generate synthetic frame"""
        if not self._connected:
            return None
        
        start_time = time.time()
        
        # Generate synthetic RGB image
        rgb_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb_image[:] = (100, 150, 200)  # Sky blue background
        
        # Generate synthetic depth (distance increases with height)
        depth_image = np.ones((self.height, self.width), dtype=np.float32)
        for y in range(self.height):
            depth_image[y, :] = 2.0 + (y / self.height) * 3.0  # 2-5 meter range
        
        # Add synthetic hoop if requested
        if self.generate_synthetic_hoops:
            self._add_synthetic_hoop(rgb_image, depth_image)
        
        # Generate synthetic pose (simple circular motion)
        elapsed_time = time.time() - self._start_time
        radius = 1.0
        pose = np.array([
            radius * np.cos(elapsed_time * 0.1),
            radius * np.sin(elapsed_time * 0.1),
            0.8  # Fixed altitude
        ])
        
        orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        frame = ZEDFrame(
            rgb_image=rgb_image,
            depth_image=depth_image,
            timestamp=time.time(),
            frame_id=self._frame_count,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            position=pose,
            orientation=orientation,
            confidence=0.9,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        self._frame_count += 1
        
        # Simulate frame rate
        time.sleep(1.0 / self.fps)
        
        return frame
    
    def is_connected(self) -> bool:
        """Check mock connection status"""
        return self._connected
    
    def _add_synthetic_hoop(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> None:
        """Add synthetic hoop to images"""
        # Simple circle in center of image
        center_x = self.width // 2 + int(50 * np.sin(time.time()))  # Slight movement
        center_y = self.height // 2
        radius = 80
        
        # Draw hoop on RGB image
        cv2.circle(rgb_image, (center_x, center_y), radius, (255, 100, 0), 8)
        cv2.circle(rgb_image, (center_x, center_y), radius - 15, (255, 150, 50), 3)
        
        # Create depth hole for hoop
        cv2.circle(depth_image, (center_x, center_y), radius - 10, 3.0, -1)


def create_zed_interface(interface_type: str = "auto", **kwargs) -> ZEDInterface:
    """
    Factory function to create appropriate ZED interface
    
    Args:
        interface_type: "ros", "direct", "mock", or "auto"
        **kwargs: Interface-specific arguments
        
    Returns:
        ZEDInterface instance
    """
    if interface_type == "ros":
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS2 not available for ZED interface")
        return ROSZEDInterface(**kwargs)
    
    elif interface_type == "direct":
        if not ZED_SDK_AVAILABLE:
            raise RuntimeError("ZED SDK not available for direct interface")
        return DirectZEDInterface(**kwargs)
    
    elif interface_type == "mock":
        return MockZEDInterface(**kwargs)
    
    elif interface_type == "auto":
        # Auto-select best available interface
        if ZED_SDK_AVAILABLE:
            logger.info("Auto-selecting DirectZEDInterface")
            return DirectZEDInterface(**kwargs)
        elif ROS_AVAILABLE:
            logger.info("Auto-selecting ROSZEDInterface")
            return ROSZEDInterface(**kwargs)
        else:
            logger.info("Auto-selecting MockZEDInterface")
            return MockZEDInterface(**kwargs)
    
    else:
        raise ValueError(f"Unknown interface type: {interface_type}")


# Export main classes and functions
__all__ = [
    'ZEDFrame', 'ZEDInterface', 'ROSZEDInterface', 
    'DirectZEDInterface', 'MockZEDInterface', 'create_zed_interface'
] 