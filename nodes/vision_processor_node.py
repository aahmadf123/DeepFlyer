#!/usr/bin/env python3
"""
Vision Processor ROS2 Node
Standalone node for YOLO11 vision processing and hoop detection
"""

import numpy as np
import cv2
import time
import logging
from typing import Optional

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Custom messages
try:
    from deepflyer_msgs.msg import VisionFeatures
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False
    logging.warning("Custom messages not available")

# DeepFlyer imports
from rl_agent.env.vision_processor import YOLO11VisionProcessor, create_yolo11_processor
from rl_agent.env.zed_integration import create_zed_interface, ZEDFrame

logger = logging.getLogger(__name__)


class VisionProcessorNode(Node):
    """ROS2 node for vision processing with YOLO11"""
    
    def __init__(self):
        super().__init__('vision_processor_node')
        
        # Parameters
        self.declare_parameter('use_zed', True)
        self.declare_parameter('yolo_model_size', 'n')
        self.declare_parameter('confidence_threshold', 0.3)
        self.declare_parameter('processing_frequency', 30.0)
        self.declare_parameter('publish_debug_images', True)
        
        # Get parameters
        self.use_zed = self.get_parameter('use_zed').value
        self.yolo_model_size = self.get_parameter('yolo_model_size').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.processing_frequency = self.get_parameter('processing_frequency').value
        self.publish_debug_images = self.get_parameter('publish_debug_images').value
        
        # Initialize vision processor
        try:
            self.vision_processor = create_yolo11_processor(
                model_size=self.yolo_model_size,
                confidence=self.confidence_threshold
            )
            self.get_logger().info(f"YOLO11 processor initialized with model: yolo11{self.yolo_model_size}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize YOLO11: {e}")
            self.vision_processor = None
        
        # Initialize ZED interface if requested
        self.zed_interface = None
        if self.use_zed:
            try:
                self.zed_interface = create_zed_interface("auto")
                if not self.zed_interface.start():
                    raise RuntimeError("Failed to start ZED interface")
                self.get_logger().info("ZED interface started successfully")
            except Exception as e:
                self.get_logger().warning(f"ZED interface failed, using ROS topics: {e}")
                self.zed_interface = None
        
        # Setup ROS interface
        self._setup_ros_interface()
        
        # Processing state
        self.cv_bridge = CvBridge()
        self.last_rgb_image = None
        self.last_depth_image = None
        self.frame_count = 0
        self.processing_times = []
        
        # Start processing timer
        processing_period = 1.0 / self.processing_frequency
        self.processing_timer = self.create_timer(processing_period, self.process_frame)
        
        self.get_logger().info("Vision processor node started")
    
    def _setup_ros_interface(self):
        """Setup ROS publishers and subscribers"""
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        if CUSTOM_MSGS_AVAILABLE:
            self.vision_features_pub = self.create_publisher(
                VisionFeatures, '/deepflyer/vision_features', reliable_qos)
        
        if self.publish_debug_images:
            self.debug_image_pub = self.create_publisher(
                Image, '/deepflyer/debug_image', best_effort_qos)
        
        # Subscribers (if not using ZED interface directly)
        if not self.zed_interface:
            self.rgb_sub = self.create_subscription(
                Image, '/zed_mini/zed_node/rgb/image_rect_color',
                self.rgb_callback, best_effort_qos)
            
            self.depth_sub = self.create_subscription(
                Image, '/zed_mini/zed_node/depth/depth_registered', 
                self.depth_callback, best_effort_qos)
    
    def rgb_callback(self, msg: Image):
        """Callback for RGB image messages"""
        try:
            self.last_rgb_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warning(f"Failed to convert RGB image: {e}")
    
    def depth_callback(self, msg: Image):
        """Callback for depth image messages"""
        try:
            self.last_depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warning(f"Failed to convert depth image: {e}")
    
    def process_frame(self):
        """Main processing loop"""
        if not self.vision_processor:
            return
        
        start_time = time.time()
        
        # Get images
        rgb_image, depth_image = self._get_images()
        
        if rgb_image is None or depth_image is None:
            return
        
        try:
            # Process with YOLO11
            vision_features = self.vision_processor.process_frame(rgb_image, depth_image)
            
            if vision_features:
                # Publish vision features
                if CUSTOM_MSGS_AVAILABLE:
                    self._publish_vision_features(vision_features)
                
                # Publish debug image if enabled
                if self.publish_debug_images:
                    debug_image = self._create_debug_image(rgb_image, vision_features)
                    self._publish_debug_image(debug_image)
                
                # Update statistics
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                self.frame_count += 1
                
                # Log periodically
                if self.frame_count % 100 == 0:
                    avg_time = np.mean(self.processing_times)
                    self.get_logger().info(
                        f"Processed {self.frame_count} frames, "
                        f"avg processing time: {avg_time:.1f}ms"
                    )
        
        except Exception as e:
            self.get_logger().error(f"Vision processing error: {e}")
    
    def _get_images(self) -> tuple:
        """Get RGB and depth images from available source"""
        if self.zed_interface and self.zed_interface.is_connected():
            # Use ZED interface directly
            frame = self.zed_interface.get_frame()
            if frame:
                return frame.rgb_image, frame.depth_image
        else:
            # Use ROS topic images
            return self.last_rgb_image, self.last_depth_image
        
        return None, None
    
    def _publish_vision_features(self, vision_features):
        """Publish vision features message"""
        msg = VisionFeatures()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        
        # Primary hoop detection
        if vision_features.primary_hoop:
            msg.hoop_detected = True
            msg.hoop_center_x = vision_features.primary_hoop.center_x
            msg.hoop_center_y = vision_features.primary_hoop.center_y
            msg.hoop_diameter_pixels = vision_features.primary_hoop.diameter_pixels
            msg.hoop_distance = vision_features.primary_hoop.distance
            msg.confidence = vision_features.primary_hoop.confidence
        else:
            msg.hoop_detected = False
        
        # Alignment metrics
        msg.alignment_error = vision_features.hoop_alignment
        msg.centered_in_frame = abs(vision_features.hoop_alignment) < 0.2
        
        # Processing performance
        msg.processing_time_ms = int(vision_features.processing_time_ms)
        msg.fps = 1000.0 / max(vision_features.processing_time_ms, 1.0)
        msg.tracking_stable = vision_features.detection_stability > 0.8
        
        # Error handling
        msg.error_code = 0
        msg.error_message = ""
        
        self.vision_features_pub.publish(msg)
    
    def _create_debug_image(self, rgb_image: np.ndarray, vision_features) -> np.ndarray:
        """Create debug visualization image"""
        debug_image = rgb_image.copy()
        
        if vision_features.primary_hoop:
            hoop = vision_features.primary_hoop
            
            # Draw bounding box
            x1, y1 = int(hoop.bbox[0]), int(hoop.bbox[1])
            x2, y2 = int(hoop.bbox[2]), int(hoop.bbox[3])
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            center_x = int(hoop.center_x * rgb_image.shape[1])
            center_y = int(hoop.center_y * rgb_image.shape[0])
            cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw distance and confidence text
            text = f"Dist: {hoop.distance:.1f}m, Conf: {hoop.confidence:.2f}"
            cv2.putText(debug_image, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw alignment error
        alignment_text = f"Alignment: {vision_features.hoop_alignment:.2f}"
        cv2.putText(debug_image, alignment_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw frame count and processing time
        stats_text = f"Frame: {self.frame_count}, Time: {vision_features.processing_time_ms:.1f}ms"
        cv2.putText(debug_image, stats_text, (10, debug_image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_image
    
    def _publish_debug_image(self, debug_image: np.ndarray):
        """Publish debug visualization image"""
        try:
            debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            debug_msg.header.frame_id = "camera_link"
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warning(f"Failed to publish debug image: {e}")
    
    def destroy_node(self):
        """Clean up resources"""
        if self.zed_interface:
            self.zed_interface.stop()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = VisionProcessorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Vision processor node error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main() 