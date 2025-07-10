#!/usr/bin/env python3
"""
Vision Processor Node for MVP Hoop Navigation

This ROS2 node integrates:
- ZED camera RGB and depth data
- YOLO11 hoop detection
- 8D observation space generation
- Real-time vision processing for RL

Subscribes to:
- /zed_mini/zed_node/rgb/image_rect_color
- /zed_mini/zed_node/depth/depth_registered

Publishes to:
- /deepflyer/vision_features
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any
from cv_bridge import CvBridge

# ROS2 message imports
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Point

# Custom message imports
    from deepflyer_msgs.msg import VisionFeatures

# Import our depth processing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_agent.depth_processor import YOLO11HoopDetector, HoopDetection

logger = logging.getLogger(__name__)


class VisionProcessorNode(Node):
    """
    ROS2 node for vision processing with YOLO11 + ZED integration
    Provides real-time hoop detection for MVP trajectory navigation
    """
    
    def __init__(self):
        super().__init__('vision_processor_node')
        
        # Initialize parameters
        self.declare_parameter('yolo_model_path', 'weights/best.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('processing_fps', 20.0)
        self.declare_parameter('debug_visualization', False)
        
        # Get parameters
        self.yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.target_fps = self.get_parameter('processing_fps').get_parameter_value().double_value
        self.debug_viz = self.get_parameter('debug_visualization').get_parameter_value().bool_value
        
        # Initialize CV bridge
        self.cv_bridge = CvBridge()
        
        # Initialize YOLO detector
        self.yolo_detector = YOLO11HoopDetector(
            model_path=self.yolo_model_path,
            confidence_threshold=self.confidence_threshold
        )
        
        # Load YOLO model
        if not self.yolo_detector.load_model():
            self.get_logger().warn("Failed to load YOLO model, using mock detection")
        
        # Vision processing state
        self.latest_rgb_image: Optional[np.ndarray] = None
        self.latest_depth_image: Optional[np.ndarray] = None
        self.image_width = 0
        self.image_height = 0
        self.last_detection: Optional[HoopDetection] = None
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.processing_times = []
        self.start_time = time.time()
        
        # QoS profile for reliable image transport
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscribers
        self.rgb_subscription = self.create_subscription(
            Image,
            '/zed_mini/zed_node/rgb/image_rect_color',
            self.rgb_callback,
            image_qos
        )
        
        self.depth_subscription = self.create_subscription(
            Image,
            '/zed_mini/zed_node/depth/depth_registered',
            self.depth_callback,
            image_qos
        )
        
        # Create publisher
            self.vision_features_pub = self.create_publisher(
            VisionFeatures,
            '/deepflyer/vision_features',
            10
        )
        
        # Create processing timer
        self.processing_timer = self.create_timer(
            1.0 / self.target_fps,
            self.process_vision_frame
        )
        
        # Create stats timer (1 Hz)
        self.stats_timer = self.create_timer(1.0, self.publish_stats)
        
        self.get_logger().info(f"Vision Processor Node initialized")
        self.get_logger().info(f"YOLO model: {self.yolo_model_path}")
        self.get_logger().info(f"Target FPS: {self.target_fps}")
        self.get_logger().info(f"Confidence threshold: {self.confidence_threshold}")
    
    def rgb_callback(self, msg: Image):
        """Callback for RGB image from ZED camera"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb_image = cv_image
            
            # Update image dimensions
            self.image_height, self.image_width = cv_image.shape[:2]
            
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")
    
    def depth_callback(self, msg: Image):
        """Callback for depth image from ZED camera"""
        try:
            # Convert ROS depth image to OpenCV format
            # ZED depth is in millimeters, 32-bit float
            cv_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth_image = cv_depth
            
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
    
    def process_vision_frame(self):
        """Main vision processing loop"""
        if self.latest_rgb_image is None or self.latest_depth_image is None:
            # No data available yet
            return
        
        start_time = time.time()
        
        try:
            # Process the current frame
            vision_features = self.process_frame(
                self.latest_rgb_image.copy(),
                self.latest_depth_image.copy()
            )
            
            # Publish vision features
            if vision_features:
                self.vision_features_pub.publish(vision_features)
            
            # Track performance
            processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                self.frame_count += 1
        
        except Exception as e:
            self.get_logger().error(f"Error in vision processing: {e}")
    
    def process_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> Optional[VisionFeatures]:
        """
        Process RGB and depth images to extract hoop features
        
        Args:
            rgb_image: RGB image from ZED camera
            depth_image: Depth image from ZED camera (millimeters)
            
        Returns:
            VisionFeatures message or None if processing failed
        """
        # Detect hoops with YOLO11
        detections = self.yolo_detector.detect_hoops(rgb_image)
        
        # Create VisionFeatures message
        msg = VisionFeatures()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "zed_left_camera_frame"
        
        # Set image dimensions
        msg.image_width = self.image_width
        msg.image_height = self.image_height
        
        if not detections:
            # No hoops detected
            msg.hoop_detected = False
            msg.hoop_center_u = 0
            msg.hoop_center_v = 0
            msg.hoop_center_x_norm = 0.0
            msg.hoop_center_y_norm = 0.0
            msg.hoop_distance_meters = 0.0
            msg.hoop_distance_norm = 1.0  # Maximum distance when not detected
            msg.hoop_diameter_pixels = 0.0
            msg.hoop_area_ratio = 0.0
            msg.detection_confidence = 0.0
            msg.depth_confidence = 0.0
            msg.hoop_id = -1
            msg.next_hoop_visible = False
            
            self.last_detection = None
            return msg
        
        # Process best detection (highest confidence)
        best_detection = max(detections, key=lambda d: d['confidence'])
        hoop_detection = self._process_detection(best_detection, depth_image)
        
        # Fill VisionFeatures message
        msg.hoop_detected = True
        msg.hoop_center_u = hoop_detection.x_center
        msg.hoop_center_v = hoop_detection.y_center
        msg.hoop_center_x_norm = hoop_detection.x_center_norm
        msg.hoop_center_y_norm = hoop_detection.y_center_norm
        msg.hoop_distance_meters = hoop_detection.depth_mm / 1000.0  # Convert to meters
        msg.hoop_distance_norm = hoop_detection.distance_norm
        
        # Calculate hoop geometry
        bbox_width = hoop_detection.x_max - hoop_detection.x_min
        bbox_height = hoop_detection.y_max - hoop_detection.y_min
        msg.hoop_diameter_pixels = float(max(bbox_width, bbox_height))
        
        # Calculate area ratio
        bbox_area = bbox_width * bbox_height
        image_area = self.image_width * self.image_height
        msg.hoop_area_ratio = float(bbox_area / image_area) if image_area > 0 else 0.0
        
        # Set confidence values
        msg.detection_confidence = hoop_detection.confidence
        msg.depth_confidence = 1.0 if hoop_detection.depth_mm > 0 else 0.0
        
        # Additional information
        msg.hoop_id = 0  # Single hoop in MVP
        msg.next_hoop_visible = False  # Only one hoop in MVP
        
        # Update tracking
        self.last_detection = hoop_detection
        self.detection_count += 1
        
        # Debug visualization
        if self.debug_viz:
            self._draw_debug_visualization(rgb_image, hoop_detection)
        
        return msg
    
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
        
        # Normalize center coordinates to [-1, 1]
        x_center_norm = (x_center - self.image_width / 2) / (self.image_width / 2)
        y_center_norm = (y_center - self.image_height / 2) / (self.image_height / 2)
        
        # Get depth at center
        depth_mm = self._get_depth_at_point(x_center, y_center, depth_image)
        
        # If center depth is invalid, try sampling around center
        if depth_mm <= 0:
            depth_mm = self._sample_depth_around_center(x_center, y_center, depth_image)
        
        # Normalize depth to [0, 1] range
        distance_norm = self._normalize_depth(depth_mm)
        
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
    
    def _get_depth_at_point(self, x: int, y: int, depth_image: np.ndarray) -> float:
        """Get depth value at specific pixel coordinates"""
        try:
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                depth_value = depth_image[y, x]
                
                # Check for valid depth
                if not np.isnan(depth_value) and not np.isinf(depth_value) and depth_value > 0:
                    return float(depth_value)
            
            return 0.0
            
        except Exception as e:
            self.get_logger().warning(f"Error getting depth at ({x}, {y}): {e}")
            return 0.0
    
    def _sample_depth_around_center(self, x_center: int, y_center: int, 
                                   depth_image: np.ndarray, radius: int = 5) -> float:
        """Sample depth values around center point to find valid depth"""
        valid_depths = []
        
        # Sample in a small circle around center
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    x = x_center + dx
                    y = y_center + dy
                    depth = self._get_depth_at_point(x, y, depth_image)
                    if depth > 0:
                        valid_depths.append(depth)
        
        if valid_depths:
            # Return median depth for robustness
            return float(np.median(valid_depths))
        else:
            return 0.0
    
    def _normalize_depth(self, depth_mm: float, max_depth_mm: float = 5000.0, 
                        min_depth_mm: float = 300.0) -> float:
        """Normalize depth value to [0, 1] range"""
        if depth_mm <= 0:
            return 1.0  # Invalid depth = maximum distance
        
        # Clamp to valid range
        depth_clamped = np.clip(depth_mm, min_depth_mm, max_depth_mm)
        
        # Normalize to [0, 1]
        normalized = (depth_clamped - min_depth_mm) / (max_depth_mm - min_depth_mm)
        
        return float(normalized)
    
    def _draw_debug_visualization(self, rgb_image: np.ndarray, detection: HoopDetection):
        """Draw debug visualization on image"""
        try:
            # Draw bounding box
            cv2.rectangle(
                rgb_image,
                (detection.x_min, detection.y_min),
                (detection.x_max, detection.y_max),
                (0, 255, 0), 2
            )
            
            # Draw center point
            cv2.circle(
                rgb_image,
                (detection.x_center, detection.y_center),
                5, (0, 0, 255), -1
            )
            
            # Draw text information
            text = f"Conf: {detection.confidence:.2f}"
            cv2.putText(
                rgb_image, text,
                (detection.x_min, detection.y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            
            depth_text = f"Depth: {detection.depth_mm:.0f}mm"
            cv2.putText(
                rgb_image, depth_text,
                (detection.x_min, detection.y_min - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            
            # Show image
            cv2.imshow('Hoop Detection Debug', rgb_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().warning(f"Error in debug visualization: {e}")
    
    def publish_stats(self):
        """Publish processing statistics"""
        runtime = time.time() - self.start_time
        fps = self.frame_count / max(runtime, 0.001)
        detection_rate = self.detection_count / max(self.frame_count, 1)
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        self.get_logger().info(
            f"Vision Stats: {fps:.1f} FPS, {detection_rate:.1%} detection rate, "
            f"{avg_processing_time*1000:.1f}ms avg processing time"
        )
    
    def get_observation_components(self) -> Tuple[float, float, int, float]:
        """
        Get 8D observation space components for RL
        
        Returns:
            Tuple of (hoop_x_center_norm, hoop_y_center_norm, hoop_visible, hoop_distance_norm)
        """
        if self.last_detection is None:
            return 0.0, 0.0, 0, 1.0
        
        return (
            self.last_detection.x_center_norm,
            self.last_detection.y_center_norm,
            1,  # hoop_visible
            self.last_detection.distance_norm
        )


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        vision_processor = VisionProcessorNode()
        
        vision_processor.get_logger().info("Vision Processor Node started")
        
        # Spin the node
        rclpy.spin(vision_processor)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in vision processor node: {e}")
    finally:
        if 'vision_processor' in locals():
            vision_processor.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main() 