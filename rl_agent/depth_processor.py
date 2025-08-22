#!/usr/bin/env python3
"""
Depth processing and YOLO11 integration for hoop detection
Handles ZED Mini depth data processing and YOLO11 object detection
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class HoopDetection:
    """Single hoop detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    distance: float  # meters
    center_3d: Tuple[float, float, float]  # 3D position in camera frame
    alignment: float  # -1.0 (left) to 1.0 (right), 0.0 = centered
    size_ratio: float  # bbox area / image area

@dataclass 
class EnhancedHoopDetection(HoopDetection):
    """Enhanced hoop detection with comprehensive depth information"""
    distance_confidence: float  # Confidence in distance measurement (0.0-1.0)
    spatial_consistency: float  # How uniform the depth is across the hoop (0.0-1.0)  
    passable: bool             # Whether the hoop appears passable (no obstacles)
    obstacle_map: Optional[np.ndarray]  # Local obstacle map for navigation
    depth_std: float           # Standard deviation of depth measurements
    valid_pixel_ratio: float   # Ratio of valid depth pixels in detection


class EnhancedDepthProcessor:
    """Enhanced depth processing for YOLO11 detections with comprehensive spatial analysis"""
    
    def __init__(self):
        # Depth processing parameters
        self.depth_filter_kernel = 5  # For noise reduction
        self.min_valid_depth = 0.3    # meters
        self.max_valid_depth = 10.0   # meters
        
        # Spatial analysis parameters
        self.hoop_diameter = 1.2      # meters (approximate)
        self.safety_margin = 0.5      # meters
        
    def process_depth_for_detection(self, depth_map: np.ndarray, 
                                  bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """
        Process depth information within a bounding box
        
        Args:
            depth_map: Depth map in meters
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Dictionary with depth statistics and spatial information
        """
        x1, y1, x2, y2 = bbox
        
        # Extract depth region
        depth_roi = depth_map[y1:y2, x1:x2]
        
        # Filter invalid depths
        valid_mask = (depth_roi > self.min_valid_depth) & (depth_roi < self.max_valid_depth)
        valid_depths = depth_roi[valid_mask]
        
        if len(valid_depths) == 0:
            return {
                'distance': -1.0,
                'distance_confidence': 0.0,
                'spatial_consistency': 0.0,
                'passable': False,
                'obstacle_map': None
            }
        
        # Depth statistics
        mean_distance = np.mean(valid_depths)
        std_distance = np.std(valid_depths)
        median_distance = np.median(valid_depths)
        
        # Spatial consistency (how uniform is the depth?)
        spatial_consistency = 1.0 / (1.0 + std_distance)
        
        # Confidence based on valid pixel ratio
        distance_confidence = len(valid_depths) / depth_roi.size
        
        # Check if hoop is passable (no obstacles in the way)
        passable = self._check_passable_path(depth_roi, mean_distance)
        
        # Create obstacle map for navigation
        obstacle_map = self._create_obstacle_map(depth_roi)
        
        return {
            'distance': mean_distance,
            'distance_std': std_distance,
            'distance_median': median_distance,
            'distance_confidence': distance_confidence,
            'spatial_consistency': spatial_consistency,
            'valid_pixel_ratio': len(valid_depths) / depth_roi.size,
            'passable': passable,
            'obstacle_map': obstacle_map
        }
    
    def _check_passable_path(self, depth_roi: np.ndarray, target_distance: float) -> bool:
        """Check if there's a clear path through the hoop"""
        # Create mask for depths significantly closer than the hoop
        obstacle_threshold = target_distance - self.safety_margin
        obstacle_mask = (depth_roi < obstacle_threshold) & (depth_roi > self.min_valid_depth)
        
        # If more than 20% of the region has obstacles, consider it blocked
        obstacle_ratio = np.sum(obstacle_mask) / depth_roi.size
        return obstacle_ratio < 0.2
    
    def _create_obstacle_map(self, depth_roi: np.ndarray) -> np.ndarray:
        """Create obstacle map for local navigation"""
        # Simple obstacle detection based on depth gradients
        h, w = depth_roi.shape
        obstacle_map = np.zeros((h, w), dtype=np.uint8)
        
        # Mark invalid depths as obstacles
        invalid_mask = (depth_roi <= self.min_valid_depth) | (depth_roi >= self.max_valid_depth)
        obstacle_map[invalid_mask] = 255
        
        # Mark steep depth gradients as potential obstacles
        if h > 2 and w > 2:
            # Compute depth gradients
            grad_x = np.abs(np.diff(depth_roi, axis=1))
            grad_y = np.abs(np.diff(depth_roi, axis=0))
            
            # Threshold for steep gradients (indicating obstacles)
            grad_threshold = 0.5  # meters per pixel
            
            steep_x = grad_x > grad_threshold
            steep_y = grad_y > grad_threshold
            
            obstacle_map[:-1, 1:][steep_y] = 128
            obstacle_map[1:, :-1][steep_x] = 128
        
        return obstacle_map


class YOLO11HoopDetector:
    """YOLO11-based hoop detector with enhanced depth integration"""
    
    def __init__(self, model_path: str = "trained_models/yolo/best.pt", 
                 confidence_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Detection parameters
        self.min_hoop_area = 100  # minimum bbox area in pixels
        self.max_detections = 5   # maximum hoops to detect per frame
        
        # Enhanced depth processor
        self.depth_processor = EnhancedDepthProcessor()
        
    def load_model(self) -> bool:
        """Load YOLO11 model (production-only: requires explicit trained weights)."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"YOLO model weights not found at {self.model_path}. "
                    f"Place a trained weights file at this path."
                )
            self.model = YOLO(str(self.model_path))
            logger.info(f"Loaded YOLO model: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_hoops(self, rgb_image: np.ndarray, depth_image: Optional[np.ndarray] = None,
                    camera_info: Optional[Dict] = None) -> List[HoopDetection]:
        """
        Detect hoops in RGB image and calculate 3D positions using depth
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image in millimeters (H, W) 
            camera_info: Camera calibration parameters
            
        Returns:
            List of hoop detections
        """
        if self.model is None:
            raise RuntimeError("YOLO model not loaded. Call load_model() successfully before detection.")
        
        try:
            # Run YOLO detection
            results = self.model(rgb_image, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    # Extract bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    
                    # Filter small detections
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area < self.min_hoop_area:
                        continue
                    
                    # Calculate center and alignment
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    image_width = rgb_image.shape[1]
                    image_height = rgb_image.shape[0]
                    
                    # Alignment: -1.0 (left) to 1.0 (right)
                    alignment = (center_x - image_width / 2) / (image_width / 2)
                    
                    # Size ratio
                    size_ratio = bbox_area / (image_width * image_height)
                    
                    # Calculate distance and 3D position using depth
                    distance = 1.0  # Default distance
                    center_3d = (0.0, 0.0, distance)
                    
                    if depth_image is not None:
                        distance, center_3d = self._calculate_3d_position(
                            center_x, center_y, x1, y1, x2, y2,
                            depth_image, camera_info
                        )
                    
                    detection = HoopDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        distance=distance,
                        center_3d=center_3d,
                        alignment=alignment,
                        size_ratio=size_ratio
                    )
                    
                    detections.append(detection)
            
            # Sort by confidence and limit number
            detections.sort(key=lambda d: d.confidence, reverse=True)
            return detections[:self.max_detections]
            
        except Exception as e:
            logger.error(f"Error in hoop detection: {e}")
            raise
    
    def _calculate_3d_position(self, center_x: int, center_y: int, 
                              x1: int, y1: int, x2: int, y2: int,
                              depth_image: np.ndarray, 
                              camera_info: Optional[Dict]) -> Tuple[float, Tuple[float, float, float]]:
        """Calculate 3D position from depth image"""
        try:
            # Extract depth values in bounding box region
            depth_roi = depth_image[y1:y2, x1:x2]
            
            # Filter out invalid depth values (0 or very large)
            valid_depths = depth_roi[(depth_roi > 100) & (depth_roi < 10000)]  # 10cm to 10m
            
            if len(valid_depths) == 0:
                return 1.0, (0.0, 0.0, 1.0)
            
            # Use median depth for robustness
            depth_mm = np.median(valid_depths)
            distance = depth_mm / 1000.0  # Convert to meters
            
            # Calculate 3D position (simplified - would use proper camera calibration)
            if camera_info and 'fx' in camera_info and 'fy' in camera_info:
                fx = camera_info['fx']
                fy = camera_info['fy']
                cx = camera_info.get('cx', depth_image.shape[1] // 2)
                cy = camera_info.get('cy', depth_image.shape[0] // 2)
                
                # Project to 3D
                x_3d = (center_x - cx) * distance / fx
                y_3d = (center_y - cy) * distance / fy
                z_3d = distance
                
                return distance, (x_3d, y_3d, z_3d)
            else:
                # Simplified calculation without calibration
                return distance, (0.0, 0.0, distance)
                
        except Exception as e:
            logger.error(f"Error calculating 3D position: {e}")
            return 1.0, (0.0, 0.0, 1.0)
    
    
    def process_detections_to_features(self, detections: List[HoopDetection], 
                                     target_hoop_id: int = 0) -> np.ndarray:
        """
        Convert detections to 8D feature vector for RL agent
        
        Returns:
            8D numpy array: [hoop_x, hoop_y, hoop_visible, hoop_distance, 
                           drone_vx, drone_vy, drone_vz, yaw_rate]
        """
        features = np.zeros(8, dtype=np.float32)
        
        if not detections:
            # No hoop visible
            features[2] = 0.0  # hoop_visible = False
            features[3] = 1.0  # max normalized distance
            return features
        
        # Use the first (highest confidence) detection as target
        detection = detections[0]
        
        # Hoop position (normalized)
        features[0] = np.clip(detection.alignment, -1.0, 1.0)  # hoop_x
        features[1] = 0.0  # hoop_y (would need vertical alignment calculation)
        features[2] = 1.0  # hoop_visible = True
        
        # Distance (normalized to 0-1, assuming max 5m)
        features[3] = np.clip(detection.distance / 5.0, 0.0, 1.0)
        
        # Drone velocity components (would be filled by drone state)
        features[4:8] = 0.0  # Will be updated by drone state provider
        
        return features


class DepthProcessor:
    """Processes ZED Mini depth data for navigation"""
    
    def __init__(self):
        self.depth_scale = 1.0  # ZED outputs depth in mm
        self.max_reliable_depth = 10.0  # 10 meters max reliable range
        
    def process_depth_for_collision(self, depth_image: np.ndarray, 
                                  safety_radius: float = 1.0) -> Tuple[float, Optional[np.ndarray]]:
        """
        Process depth image for collision avoidance
        
        Args:
            depth_image: Depth image in millimeters
            safety_radius: Safety radius around drone (meters)
            
        Returns:
            Tuple of (min_distance, obstacle_direction)
        """
        # Convert to meters
        depth_m = depth_image.astype(np.float32) / 1000.0
        
        # Filter invalid values
        valid_mask = (depth_m > 0.1) & (depth_m < self.max_reliable_depth)
        
        if not np.any(valid_mask):
            return self.max_reliable_depth, None
        
        # Find minimum distance
        min_distance = np.min(depth_m[valid_mask])
        
        # Find direction to closest obstacle
        min_idx = np.unravel_index(np.argmin(depth_m, axis=None), depth_m.shape)
        h, w = depth_image.shape
        
        # Convert pixel coordinates to normalized direction
        direction_x = (min_idx[1] - w / 2) / (w / 2)  # -1 to 1
        direction_y = (min_idx[0] - h / 2) / (h / 2)  # -1 to 1
        
        obstacle_direction = np.array([direction_x, direction_y, 0.0])
        
        return min_distance, obstacle_direction