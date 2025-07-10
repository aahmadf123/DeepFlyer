"""
Depth Processing for MVP Hoop Navigation

This module handles:
1. ZED camera depth data processing
2. YOLO11 hoop detection integration
3. Hoop center calculation and normalization
4. Distance estimation and normalization
5. Integration with MVP observation space

Provides the vision components for the 8D observation space:
- hoop_x_center_norm: [-1, 1] horizontal position in camera frame
- hoop_y_center_norm: [-1, 1] vertical position in camera frame  
- hoop_visible: [0, 1] binary detection flag
- hoop_distance_norm: [0, 1] normalized depth to hoop
"""

import numpy as np
import cv2
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time

try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available, using mock detection")

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    logging.warning("ZED SDK not available, using mock camera")

logger = logging.getLogger(__name__)


@dataclass
class HoopDetection:
    """Represents a detected hoop with all relevant information"""
    
    # Bounding box (pixel coordinates)
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    
    # Center coordinates (pixel)
    x_center: int
    y_center: int
    
    # Normalized center coordinates [-1, 1]
    x_center_norm: float
    y_center_norm: float
    
    # Distance information
    depth_mm: float           # Depth in millimeters
    distance_norm: float      # Normalized distance [0, 1]
    
    # Detection confidence
    confidence: float
    
    # Timestamp
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'bbox': [self.x_min, self.y_min, self.x_max, self.y_max],
            'center_pixel': [self.x_center, self.y_center],
            'center_norm': [self.x_center_norm, self.y_center_norm],
            'depth_mm': self.depth_mm,
            'distance_norm': self.distance_norm,
            'confidence': self.confidence,
            'timestamp': self.timestamp
        }


class ZEDDepthProcessor:
    """Processes ZED camera depth data for MVP hoop navigation"""
    
    def __init__(self, resolution: str = "720p", fps: int = 30):
        """
        Initialize ZED depth processor
        
        Args:
            resolution: Camera resolution ("720p", "1080p", "2K", "4K")
            fps: Frames per second
        """
        self.resolution = resolution
        self.fps = fps
        self.camera = None
        self.runtime_parameters = None
        self.is_connected = False
        
        # Image buffers
        self.rgb_image = sl.Mat()
        self.depth_image = sl.Mat()
        self.point_cloud = sl.Mat()
        
        # Camera parameters
        self.image_width = 0
        self.image_height = 0
        self.fx = 0.0  # Focal length X
        self.fy = 0.0  # Focal length Y
        self.cx = 0.0  # Principal point X
        self.cy = 0.0  # Principal point Y
        
        # Depth processing parameters
        self.max_depth_mm = 5000.0    # 5 meters max detection range
        self.min_depth_mm = 300.0     # 30cm minimum detection range
        
        logger.info(f"ZED Depth Processor initialized: {resolution}@{fps}fps")
    
    def connect(self) -> bool:
        """Connect to ZED camera"""
        if not ZED_AVAILABLE:
            logger.warning("ZED SDK not available, cannot connect to camera")
            return False
        
        try:
            # Create ZED camera object
            self.camera = sl.Camera()
            
            # Set configuration parameters
            init_params = sl.InitParameters()
            
            # Set resolution
            if self.resolution == "720p":
                init_params.camera_resolution = sl.RESOLUTION.HD720
                self.image_width, self.image_height = 1280, 720
            elif self.resolution == "1080p":
                init_params.camera_resolution = sl.RESOLUTION.HD1080
                self.image_width, self.image_height = 1920, 1080
            else:
                init_params.camera_resolution = sl.RESOLUTION.HD720
                self.image_width, self.image_height = 1280, 720
            
            init_params.camera_fps = self.fps
            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
            init_params.coordinate_units = sl.UNIT.MILLIMETER
            init_params.depth_minimum_distance = self.min_depth_mm
            init_params.depth_maximum_distance = self.max_depth_mm
            
            # Open camera
            status = self.camera.open(init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                logger.error(f"Failed to open ZED camera: {status}")
                return False
            
            # Get camera parameters
            camera_info = self.camera.get_camera_information()
            calibration = camera_info.camera_configuration.calibration_parameters.left_cam
            
            self.fx = calibration.fx
            self.fy = calibration.fy
            self.cx = calibration.cx
            self.cy = calibration.cy
            
            # Set runtime parameters
            self.runtime_parameters = sl.RuntimeParameters()
            self.runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
            
            self.is_connected = True
            logger.info(f"ZED camera connected successfully: {self.image_width}x{self.image_height}")
            logger.info(f"Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ZED camera: {e}")
            return False
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Capture frame from ZED camera
        
        Returns:
            Tuple of (rgb_image, depth_image) as numpy arrays, or None if failed
        """
        if not self.is_connected or not self.camera:
            return None
        
        try:
            # Grab frame
            if self.camera.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve RGB image
                self.camera.retrieve_image(self.rgb_image, sl.VIEW.LEFT)
                rgb_array = self.rgb_image.get_data()[:, :, :3]  # Remove alpha channel
                
                # Retrieve depth image
                self.camera.retrieve_measure(self.depth_image, sl.MEASURE.DEPTH)
                depth_array = self.depth_image.get_data()
                
                return rgb_array, depth_array
            else:
                logger.warning("Failed to grab frame from ZED camera")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing ZED frame: {e}")
            return None
    
    def get_depth_at_point(self, x: int, y: int, depth_image: np.ndarray) -> float:
        """
        Get depth value at specific pixel coordinates
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            depth_image: Depth image array
            
        Returns:
            Depth value in millimeters, or 0 if invalid
        """
        try:
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                depth_value = depth_image[y, x]
                
                # Check for valid depth
                if not np.isnan(depth_value) and not np.isinf(depth_value):
                    return float(depth_value)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error getting depth at ({x}, {y}): {e}")
            return 0.0
    
    def normalize_depth(self, depth_mm: float) -> float:
        """
        Normalize depth value to [0, 1] range
        
        Args:
            depth_mm: Depth in millimeters
            
        Returns:
            Normalized depth [0, 1] where 0 = close, 1 = far
        """
        if depth_mm <= 0:
            return 1.0  # Invalid depth = maximum distance
        
        # Clamp to valid range
        depth_clamped = np.clip(depth_mm, self.min_depth_mm, self.max_depth_mm)
        
        # Normalize to [0, 1]
        normalized = (depth_clamped - self.min_depth_mm) / (self.max_depth_mm - self.min_depth_mm)
        
        return float(normalized)
    
    def normalize_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """
        Normalize pixel coordinates to [-1, 1] range
        
        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            
        Returns:
            Tuple of (x_norm, y_norm) in [-1, 1] range
        """
        # Normalize to [-1, 1] with center at (0, 0)
        x_norm = (x - self.image_width / 2) / (self.image_width / 2)
        y_norm = (y - self.image_height / 2) / (self.image_height / 2)
        
        return float(x_norm), float(y_norm)
    
    def disconnect(self):
        """Disconnect from ZED camera"""
        if self.camera:
            self.camera.close()
            self.is_connected = False
            logger.info("ZED camera disconnected")


class YOLO11HoopDetector:
    """YOLO11 model for hoop detection"""
    
    def __init__(self, model_path: str = "weights/best.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLO11 hoop detector
        
        Args:
            model_path: Path to trained YOLO11 model
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False
        
        logger.info(f"YOLO11 Hoop Detector initialized: {model_path}")
    
    def load_model(self) -> bool:
        """Load YOLO11 model"""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available, using mock detection")
            return False
        
        try:
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            logger.info(f"YOLO11 model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load YOLO11 model: {e}")
            return False
    
    def detect_hoops(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect hoops in RGB image
        
        Args:
            rgb_image: RGB image array (H, W, 3)
            
        Returns:
            List of detection dictionaries with bbox, confidence, etc.
        """
        if not self.is_loaded and YOLO_AVAILABLE:
            logger.warning("YOLO model not loaded")
            return []
        
        if not YOLO_AVAILABLE:
            # Mock detection for testing
            return self._mock_detection(rgb_image)
        
        try:
            # Run inference
            results = self.model(rgb_image, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class': 'hoop'
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []
    
    def _mock_detection(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Mock detection for testing when YOLO is not available"""
        height, width = rgb_image.shape[:2]
        
        # Simulate a hoop detection in center of image
        center_x = width // 2
        center_y = height // 2
        box_size = min(width, height) // 6
        
        return [{
            'bbox': [
                center_x - box_size,
                center_y - box_size,
                center_x + box_size,
                center_y + box_size
            ],
            'confidence': 0.85,
            'class': 'hoop'
        }]


class MVPDepthProcessor:
    """Complete depth processing pipeline for MVP hoop navigation"""
    
    def __init__(self, 
                 zed_resolution: str = "720p",
                 yolo_model_path: str = "weights/best.pt",
                 confidence_threshold: float = 0.5):
        """
        Initialize MVP depth processor
        
        Args:
            zed_resolution: ZED camera resolution
            yolo_model_path: Path to YOLO11 model
            confidence_threshold: YOLO confidence threshold
        """
        # Initialize components
        self.zed_processor = ZEDDepthProcessor(resolution=zed_resolution)
        self.yolo_detector = YOLO11HoopDetector(yolo_model_path, confidence_threshold)
        
        # State tracking
        self.last_detection: Optional[HoopDetection] = None
        self.detection_history: List[HoopDetection] = []
        self.max_history = 10
        
        # Performance metrics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        logger.info("MVP Depth Processor initialized")
    
    def connect(self) -> bool:
        """Connect to camera and load models"""
        success = True
        
        # Connect to ZED camera
        if not self.zed_processor.connect():
            logger.warning("Failed to connect to ZED camera")
            success = False
        
        # Load YOLO model
        if not self.yolo_detector.load_model():
            logger.warning("Failed to load YOLO model")
            success = False
        
        return success
    
    def process_frame(self) -> Tuple[float, float, int, float]:
        """
        Process single frame to get MVP observation components
        
        Returns:
            Tuple of (hoop_x_center_norm, hoop_y_center_norm, hoop_visible, hoop_distance_norm)
        """
        self.frame_count += 1
        
        # Get frame from ZED camera
        frame_data = self.zed_processor.get_frame()
        if frame_data is None:
            # No frame available
            return 0.0, 0.0, 0, 1.0
        
        rgb_image, depth_image = frame_data
        
        # Detect hoops with YOLO
        detections = self.yolo_detector.detect_hoops(rgb_image)
        
        if not detections:
            # No hoops detected
            self.last_detection = None
            return 0.0, 0.0, 0, 1.0
        
        # Process best detection (highest confidence)
        best_detection = max(detections, key=lambda d: d['confidence'])
        hoop_detection = self._process_detection(best_detection, depth_image)
        
        # Update tracking
        self.last_detection = hoop_detection
        self.detection_history.append(hoop_detection)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
        
        self.detection_count += 1
        
        # Return MVP observation components
        return (
            hoop_detection.x_center_norm,
            hoop_detection.y_center_norm,
            1,  # hoop_visible
            hoop_detection.distance_norm
        )
    
    def _process_detection(self, detection: Dict[str, Any], depth_image: np.ndarray) -> HoopDetection:
        """
        Process YOLO detection to create HoopDetection object
        
        Args:
            detection: YOLO detection dictionary
            depth_image: Depth image array
            
        Returns:
            HoopDetection object with all computed information
        """
        # Extract bounding box
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Calculate center
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        
        # Normalize center coordinates
        x_center_norm, y_center_norm = self.zed_processor.normalize_coordinates(x_center, y_center)
        
        # Get depth at center
        depth_mm = self.zed_processor.get_depth_at_point(x_center, y_center, depth_image)
        
        # If center depth is invalid, try sampling around center
        if depth_mm <= 0:
            depth_mm = self._sample_depth_around_center(x_center, y_center, depth_image)
        
        # Normalize depth
        distance_norm = self.zed_processor.normalize_depth(depth_mm)
        
        return HoopDetection(
            x_min=x1,
            y_min=y1,
            x_max=x2,
            y_max=y2,
            x_center=x_center,
            y_center=y_center,
            x_center_norm=x_center_norm,
            y_center_norm=y_center_norm,
            depth_mm=depth_mm,
            distance_norm=distance_norm,
            confidence=confidence,
            timestamp=time.time()
        )
    
    def _sample_depth_around_center(self, x_center: int, y_center: int, 
                                   depth_image: np.ndarray, radius: int = 5) -> float:
        """
        Sample depth values around center point to find valid depth
        
        Args:
            x_center: Center x coordinate
            y_center: Center y coordinate
            depth_image: Depth image array
            radius: Sampling radius in pixels
            
        Returns:
            Best depth value found, or 0 if all invalid
        """
        valid_depths = []
        
        # Sample in a small circle around center
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    x = x_center + dx
                    y = y_center + dy
                    depth = self.zed_processor.get_depth_at_point(x, y, depth_image)
                    if depth > 0:
                        valid_depths.append(depth)
        
        if valid_depths:
            # Return median depth for robustness
            return float(np.median(valid_depths))
        else:
            return 0.0
    
    def get_last_detection(self) -> Optional[HoopDetection]:
        """Get the last valid hoop detection"""
        return self.last_detection
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        runtime = time.time() - self.start_time
        fps = self.frame_count / max(runtime, 0.001)
        detection_rate = self.detection_count / max(self.frame_count, 1)
        
        return {
            'frames_processed': self.frame_count,
            'detections_found': self.detection_count,
            'runtime_seconds': runtime,
            'fps': fps,
            'detection_rate': detection_rate,
            'last_detection': self.last_detection.to_dict() if self.last_detection else None
        }
    
    def disconnect(self):
        """Disconnect from all devices"""
        self.zed_processor.disconnect()
        logger.info("MVP Depth Processor disconnected")


# Convenience functions for integration
def create_mvp_depth_processor(**kwargs) -> MVPDepthProcessor:
    """Create MVP depth processor with default settings"""
    return MVPDepthProcessor(**kwargs)


def process_single_frame(processor) -> Dict[str, float]:
    """Process a single frame with the MVP depth processor"""
    try:
        rgb_image, depth_image = processor.capture_frame()
        if rgb_image is None or depth_image is None:
            return {
                'hoop_visible': 0.0,
                'hoop_x_center_norm': 0.0,
                'hoop_y_center_norm': 0.0,
                'hoop_distance_norm': 1.0
            }
        
        features = processor.process_frame(rgb_image, depth_image)
        return {
            'hoop_visible': features['hoop_detected'],
            'hoop_x_center_norm': features['hoop_x_center'],
            'hoop_y_center_norm': features['hoop_y_center'],
            'hoop_distance_norm': features['hoop_distance']
        }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return {
            'hoop_visible': 0.0,
            'hoop_x_center_norm': 0.0,
            'hoop_y_center_norm': 0.0,
            'hoop_distance_norm': 1.0
        } 