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
                    camera_info: Optional[Dict] = None) -> List[EnhancedHoopDetection]:
        """
        Detect hoops in RGB image and calculate enhanced 3D information using depth
        
        Args:
            rgb_image: RGB image (H, W, 3)
            depth_image: Depth image in millimeters (H, W) 
            camera_info: Camera calibration parameters
            
        Returns:
            List of enhanced hoop detections with comprehensive depth analysis
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
                    
                    # Enhanced depth processing if available
                    if depth_image is not None:
                        enhanced_depth_info = self.depth_processor.process_depth_for_detection(
                            depth_image, (x1, y1, x2, y2)
                        )
                        
                        distance = enhanced_depth_info['distance']
                        center_3d = self._calculate_3d_position_enhanced(
                            center_x, center_y, distance, camera_info
                        )
                        
                        # Create enhanced detection
                        detection = EnhancedHoopDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            distance=distance,
                            center_3d=center_3d,
                            alignment=alignment,
                            size_ratio=size_ratio,
                            distance_confidence=enhanced_depth_info['distance_confidence'],
                            spatial_consistency=enhanced_depth_info['spatial_consistency'],
                            passable=enhanced_depth_info['passable'],
                            obstacle_map=enhanced_depth_info['obstacle_map'],
                            depth_std=enhanced_depth_info.get('distance_std', 0.0),
                            valid_pixel_ratio=enhanced_depth_info['valid_pixel_ratio']
                        )
                    else:
                        # Fallback to basic detection without depth
                        distance = 1.0
                        center_3d = (0.0, 0.0, distance)
                        
                        detection = EnhancedHoopDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            distance=distance,
                            center_3d=center_3d,
                            alignment=alignment,
                            size_ratio=size_ratio,
                            distance_confidence=0.0,
                            spatial_consistency=0.0,
                            passable=True,  # Assume passable without depth
                            obstacle_map=None,
                            depth_std=0.0,
                            valid_pixel_ratio=0.0
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
    
    def _calculate_3d_position_enhanced(self, center_x: int, center_y: int, 
                                      distance: float, camera_info: Optional[Dict]) -> Tuple[float, float, float]:
        """Calculate enhanced 3D position with proper camera calibration"""
        try:
            if camera_info and 'fx' in camera_info and 'fy' in camera_info:
                fx = camera_info['fx']
                fy = camera_info['fy']
                cx = camera_info.get('cx', 320)  # Default for typical camera
                cy = camera_info.get('cy', 240)
                
                # Project to 3D using proper pinhole camera model
                x_3d = (center_x - cx) * distance / fx
                y_3d = (center_y - cy) * distance / fy
                z_3d = distance
                
                return (x_3d, y_3d, z_3d)
            else:
                # Fallback without calibration
                return (0.0, 0.0, distance)
                
        except Exception as e:
            logger.error(f"Error calculating enhanced 3D position: {e}")
            return (0.0, 0.0, distance)
    
    
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
    """Enhanced ZED Mini depth data processor with advanced navigation features"""
    
    def __init__(self):
        self.depth_scale = 1.0  # ZED outputs depth in mm
        self.max_reliable_depth = 10.0  # 10 meters max reliable range
        self.min_reliable_depth = 0.1   # 10cm minimum reliable range
        
        # Temporal filtering for depth stability
        self.depth_history = []
        self.history_size = 5
        
        # Advanced processing parameters
        self.median_filter_size = 5  # For noise reduction
        self.gradient_threshold = 0.5  # For edge detection (m/pixel)
        
    def process_depth_for_collision(self, depth_image: np.ndarray, 
                                  safety_radius: float = 1.0) -> Tuple[float, Optional[np.ndarray]]:
        """
        Enhanced depth processing for collision avoidance with temporal filtering
        
        Args:
            depth_image: Depth image in millimeters
            safety_radius: Safety radius around drone (meters)
            
        Returns:
            Tuple of (min_distance, obstacle_direction)
        """
        # Convert to meters and apply temporal filtering
        depth_m = self._apply_temporal_filtering(depth_image.astype(np.float32) / 1000.0)
        
        # Apply spatial filtering for noise reduction
        depth_filtered = self._apply_spatial_filtering(depth_m)
        
        # Filter invalid values
        valid_mask = (depth_filtered > self.min_reliable_depth) & (depth_filtered < self.max_reliable_depth)
        
        if not np.any(valid_mask):
            return self.max_reliable_depth, None
        
        # Find minimum distance with safety margin
        safe_depths = depth_filtered[valid_mask]
        min_distance = np.min(safe_depths)
        
        # Find direction to closest obstacle with enhanced accuracy
        obstacle_direction = self._calculate_obstacle_direction(depth_filtered, valid_mask)
        
        return min_distance, obstacle_direction
    
    def _apply_temporal_filtering(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply temporal filtering to reduce depth noise"""
        # Add current frame to history
        self.depth_history.append(depth_image.copy())
        
        # Maintain history size
        if len(self.depth_history) > self.history_size:
            self.depth_history.pop(0)
        
        # If we have multiple frames, use temporal median
        if len(self.depth_history) > 1:
            depth_stack = np.stack(self.depth_history, axis=0)
            filtered_depth = np.median(depth_stack, axis=0)
            return filtered_depth
        else:
            return depth_image
    
    def _apply_spatial_filtering(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply spatial filtering to reduce noise"""
        try:
            import cv2
            # Apply median filter to reduce noise
            filtered = cv2.medianBlur(depth_image.astype(np.float32), self.median_filter_size)
            return filtered
        except ImportError:
            # Fallback: simple averaging filter
            from scipy import ndimage
            return ndimage.uniform_filter(depth_image, size=3)
        except:
            # Last resort: return original
            return depth_image
    
    def _calculate_obstacle_direction(self, depth_image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Calculate direction to nearest obstacle with improved accuracy"""
        h, w = depth_image.shape
        
        # Create distance-weighted map
        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Weight by distance from center and depth value
        pixel_distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Combine depth and pixel distance for threat assessment
        threat_map = np.zeros_like(depth_image)
        threat_map[valid_mask] = 1.0 / (depth_image[valid_mask] + 0.1)  # Closer = higher threat
        
        # Weight by field of view position (center is more important)
        fov_weight = 1.0 / (1.0 + pixel_distances / max(w, h))
        threat_map *= fov_weight
        
        # Find direction of highest threat
        if np.any(threat_map > 0):
            max_threat_idx = np.unravel_index(np.argmax(threat_map), threat_map.shape)
            direction_x = (max_threat_idx[1] - center_x) / center_x  # -1 to 1
            direction_y = (max_threat_idx[0] - center_y) / center_y  # -1 to 1
            return np.array([direction_x, direction_y, 0.0])
        
        return np.array([0.0, 0.0, 0.0])
    
    def get_navigation_features(self, depth_image: np.ndarray) -> Dict[str, Any]:
        """
        Extract advanced navigation features from depth image
        
        Returns:
            Dictionary with navigation-relevant depth features
        """
        depth_m = depth_image.astype(np.float32) / 1000.0
        valid_mask = (depth_m > self.min_reliable_depth) & (depth_m < self.max_reliable_depth)
        
        if not np.any(valid_mask):
            return {
                'clear_path_ahead': False,
                'min_distance': self.max_reliable_depth,
                'average_distance': self.max_reliable_depth,
                'depth_variance': 0.0,
                'obstacle_density': 0.0,
                'navigation_confidence': 0.0
            }
        
        valid_depths = depth_m[valid_mask]
        
        # Basic statistics
        min_distance = np.min(valid_depths)
        avg_distance = np.mean(valid_depths)
        depth_variance = np.var(valid_depths)
        
        # Navigation analysis
        h, w = depth_image.shape
        center_region = depth_m[h//3:2*h//3, w//3:2*w//3]  # Central third
        center_valid_mask = (center_region > self.min_reliable_depth) & (center_region < self.max_reliable_depth)
        
        if np.any(center_valid_mask):
            center_min_distance = np.min(center_region[center_valid_mask])
            clear_path_ahead = center_min_distance > 2.0  # 2m clearance
        else:
            clear_path_ahead = False
            center_min_distance = 0.0
        
        # Obstacle density (percentage of pixels with obstacles within 3m)
        close_obstacles = np.sum((depth_m < 3.0) & valid_mask)
        total_valid = np.sum(valid_mask)
        obstacle_density = close_obstacles / max(total_valid, 1)
        
        # Navigation confidence based on depth quality and clearance
        navigation_confidence = min(1.0, center_min_distance / 3.0) * (1.0 - obstacle_density)
        
        return {
            'clear_path_ahead': clear_path_ahead,
            'min_distance': float(min_distance),
            'average_distance': float(avg_distance),
            'center_min_distance': float(center_min_distance),
            'depth_variance': float(depth_variance),
            'obstacle_density': float(obstacle_density),
            'navigation_confidence': float(navigation_confidence),
            'valid_pixel_ratio': float(total_valid / (h * w))
        }
    
    def detect_narrow_passages(self, depth_image: np.ndarray, 
                             passage_width_threshold: float = 1.5) -> List[Dict[str, Any]]:
        """
        Detect potential narrow passages for hoop navigation
        
        Args:
            depth_image: Depth image in millimeters
            passage_width_threshold: Minimum width for passage detection (meters)
            
        Returns:
            List of detected passages with their properties
        """
        depth_m = depth_image.astype(np.float32) / 1000.0
        h, w = depth_image.shape
        
        passages = []
        
        # Scan horizontal lines for gaps
        for y in range(h // 4, 3 * h // 4, h // 8):  # Sample several horizontal lines
            depth_line = depth_m[y, :]
            valid_depths = (depth_line > self.min_reliable_depth) & (depth_line < self.max_reliable_depth)
            
            if not np.any(valid_depths):
                continue
            
            # Find segments with sufficient depth
            far_pixels = depth_line > passage_width_threshold
            
            # Find continuous segments
            changes = np.diff(np.concatenate(([False], far_pixels, [False])).astype(int))
            segment_starts = np.where(changes == 1)[0]
            segment_ends = np.where(changes == -1)[0]
            
            for start, end in zip(segment_starts, segment_ends):
                segment_width = end - start
                if segment_width > w // 8:  # Minimum pixel width
                    center_x = (start + end) // 2
                    avg_depth = np.mean(depth_line[start:end])
                    
                    passage = {
                        'center_x': int(center_x),
                        'center_y': int(y),
                        'width_pixels': int(segment_width),
                        'width_normalized': float(segment_width / w),
                        'average_depth': float(avg_depth),
                        'confidence': min(1.0, segment_width / (w // 4))
                    }
                    passages.append(passage)
        
        # Sort by confidence and remove overlapping detections
        passages.sort(key=lambda p: p['confidence'], reverse=True)
        
        return passages[:3]  # Return top 3 passages