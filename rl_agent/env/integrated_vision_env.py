"""
Integrated MAVROS Environment with YOLO11 Vision Processing
Combines drone control with advanced computer vision for hoop detection
"""

import numpy as np
import cv2
import rospy
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import logging

from .mavros_env import MavrosEnv
from .vision_processor import YOLO11VisionProcessor, VisionFeatures, create_yolo11_processor
from .safety_layer import SafetyLayer
from ..logger import Logger

logger = logging.getLogger(__name__)


class IntegratedVisionMavrosEnv(MavrosEnv):
    """
    Enhanced MAVROS environment with YOLO11 vision processing
    Replaces simple computer vision with state-of-the-art object detection
    """
    
    def __init__(self, 
                 yolo_model_size: str = "n",  # 'n', 's', 'm', 'l', 'x'
                 confidence_threshold: float = 0.3,
                 use_custom_hoop_model: bool = False,
                 custom_model_path: Optional[str] = None,
                 vision_observation_space: bool = True,
                 **kwargs):
        """
        Initialize integrated environment with YOLO11 vision
        
        Args:
            yolo_model_size: YOLO11 model size ('n' fastest, 'x' most accurate)
            confidence_threshold: Minimum confidence for hoop detections
            use_custom_hoop_model: Whether to use custom-trained hoop model
            custom_model_path: Path to custom-trained YOLO11 model
            vision_observation_space: Include vision features in observation space
            **kwargs: Arguments passed to parent MavrosEnv
        """
        
        # Initialize parent MAVROS environment
        super().__init__(**kwargs)
        
        # Initialize YOLO11 vision processor
        self.vision_processor = create_yolo11_processor(
            model_size=yolo_model_size,
            confidence=confidence_threshold
        )
        
        # Setup custom hoop model if specified
        if use_custom_hoop_model and custom_model_path:
            try:
                self.vision_processor.setup_custom_hoop_detection(custom_model_path)
                logger.info(f"Using custom hoop model: {custom_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom model, using default: {e}")
        
        # Vision parameters
        self.vision_observation_space = vision_observation_space
        self.last_vision_features: Optional[VisionFeatures] = None
        
        # Update observation space to include vision features
        if self.vision_observation_space:
            self._update_observation_space_with_vision()
        
        # Performance monitoring
        self.vision_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'avg_processing_time': 0.0,
            'avg_confidence': 0.0
        }
        
        logger.info(f"Initialized YOLO11 vision environment with model size: {yolo_model_size}")
    
    def _update_observation_space_with_vision(self):
        """Update observation space to include vision features"""
        
        # Original observation space
        original_obs_space = self.observation_space
        
        # Vision features space
        vision_space = spaces.Dict({
            # Primary hoop features
            'hoop_detected': spaces.Discrete(2),  # 0 or 1
            'hoop_center_x': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # Normalized
            'hoop_center_y': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # Normalized
            'hoop_distance': spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),  # Meters
            'hoop_alignment': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),  # -1 left, +1 right
            'detection_confidence': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
            # Additional context
            'next_hoop_visible': spaces.Discrete(2),  # 0 or 1
            'detection_stability': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            
            # Multi-hoop information (up to 3 hoops)
            'num_hoops_detected': spaces.Discrete(4),  # 0-3 hoops
            'hoop_distances': spaces.Box(low=0.0, high=10.0, shape=(3,), dtype=np.float32),
            'hoop_confidences': spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
        })
        
        # Combined observation space
        if isinstance(original_obs_space, spaces.Dict):
            # Add vision to existing dict space
            combined_spaces = original_obs_space.spaces.copy()
            combined_spaces['vision'] = vision_space
            self.observation_space = spaces.Dict(combined_spaces)
        else:
            # Create new dict space with original as 'state'
            self.observation_space = spaces.Dict({
                'state': original_obs_space,
                'vision': vision_space
            })
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get observation including vision features"""
        
        # Get base observation from parent
        base_obs = super()._get_observation()
        
        if not self.vision_observation_space:
            return base_obs
        
        # Get vision features
        vision_obs = self._get_vision_observation()
        
        # Combine observations
        if isinstance(self.observation_space, spaces.Dict) and 'vision' in self.observation_space.spaces:
            if isinstance(base_obs, dict):
                # Add vision to existing dict
                base_obs['vision'] = vision_obs
                return base_obs
            else:
                # Wrap base obs as 'state'
                return {
                    'state': base_obs,
                    'vision': vision_obs
                }
        else:
            return base_obs
    
    def _get_vision_observation(self) -> Dict[str, np.ndarray]:
        """Extract vision features for RL observation"""
        
        # Default empty observation
        empty_obs = {
            'hoop_detected': np.array([0], dtype=np.int32),
            'hoop_center_x': np.array([0.0], dtype=np.float32),
            'hoop_center_y': np.array([0.0], dtype=np.float32),
            'hoop_distance': np.array([10.0], dtype=np.float32),  # Max distance
            'hoop_alignment': np.array([0.0], dtype=np.float32),
            'detection_confidence': np.array([0.0], dtype=np.float32),
            'next_hoop_visible': np.array([0], dtype=np.int32),
            'detection_stability': np.array([0.0], dtype=np.float32),
            'num_hoops_detected': np.array([0], dtype=np.int32),
            'hoop_distances': np.array([10.0, 10.0, 10.0], dtype=np.float32),
            'hoop_confidences': np.array([0.0, 0.0, 0.0], dtype=np.float32),
        }
        
        if self.last_vision_features is None:
            return empty_obs
        
        features = self.last_vision_features
        
        # Primary hoop information
        if features.primary_hoop is not None:
            hoop = features.primary_hoop
            
            # Normalize center coordinates to [-1, 1]
            if hasattr(self.vision_processor, 'image_center') and self.vision_processor.image_center:
                img_center = self.vision_processor.image_center
                norm_x = (hoop.center[0] - img_center[0]) / img_center[0]
                norm_y = (hoop.center[1] - img_center[1]) / img_center[1]
            else:
                norm_x, norm_y = 0.0, 0.0
            
            obs = {
                'hoop_detected': np.array([1], dtype=np.int32),
                'hoop_center_x': np.array([np.clip(norm_x, -1.0, 1.0)], dtype=np.float32),
                'hoop_center_y': np.array([np.clip(norm_y, -1.0, 1.0)], dtype=np.float32),
                'hoop_distance': np.array([min(hoop.distance, 10.0)], dtype=np.float32),
                'hoop_alignment': np.array([features.hoop_alignment], dtype=np.float32),
                'detection_confidence': np.array([features.detection_confidence], dtype=np.float32),
                'next_hoop_visible': np.array([1 if features.next_hoop_visible else 0], dtype=np.int32),
                'detection_stability': np.array([self.vision_processor.get_detection_stability()], dtype=np.float32),
            }
        else:
            obs = empty_obs.copy()
        
        # Multi-hoop information
        num_hoops = min(len(features.all_hoops), 3)
        obs['num_hoops_detected'] = np.array([num_hoops], dtype=np.int32)
        
        distances = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        confidences = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        for i, hoop in enumerate(features.all_hoops[:3]):
            distances[i] = min(hoop.distance, 10.0)
            confidences[i] = hoop.confidence
        
        obs['hoop_distances'] = distances
        obs['hoop_confidences'] = confidences
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step with vision processing"""
        
        # Process vision if camera data available
        self._process_vision_frame()
        
        # Execute normal step
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Add vision information to info
        if self.last_vision_features:
            info['vision'] = {
                'processing_time_ms': self.last_vision_features.processing_time_ms,
                'detection_confidence': self.last_vision_features.detection_confidence,
                'num_hoops': len(self.last_vision_features.all_hoops),
                'stability': self.vision_processor.get_detection_stability()
            }
        
        # Update performance stats
        self._update_vision_stats()
        
        return observation, reward, terminated, truncated, info
    
    def _process_vision_frame(self):
        """Process current camera frame with YOLO11"""
        try:
            # Get camera images (this would be implemented based on your camera interface)
            rgb_image, depth_image = self._get_camera_images()
            
            if rgb_image is not None and depth_image is not None:
                # Process with YOLO11
                self.last_vision_features = self.vision_processor.process_frame(rgb_image, depth_image)
                self.vision_stats['total_frames'] += 1
                
                if self.last_vision_features.primary_hoop is not None:
                    self.vision_stats['successful_detections'] += 1
            
        except Exception as e:
            logger.warning(f"Vision processing failed: {e}")
            self.last_vision_features = None
    
    def _get_camera_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get RGB and depth images from camera
        This should be implemented based on your camera interface (ZED, RealSense, etc.)
        """
        # TODO: Implement actual camera interface
        # For now, return None (would be replaced with actual camera code)
        return None, None
    
    def _update_vision_stats(self):
        """Update vision processing statistics"""
        if self.vision_stats['total_frames'] > 0:
            detection_rate = self.vision_stats['successful_detections'] / self.vision_stats['total_frames']
            
            if self.last_vision_features and self.last_vision_features.processing_time_ms > 0:
                # Running average of processing time
                current_time = self.last_vision_features.processing_time_ms
                alpha = 0.1  # Smoothing factor
                self.vision_stats['avg_processing_time'] = (
                    alpha * current_time + (1 - alpha) * self.vision_stats['avg_processing_time']
                )
            
            if self.last_vision_features and self.last_vision_features.detection_confidence > 0:
                # Running average of confidence
                current_conf = self.last_vision_features.detection_confidence
                alpha = 0.1
                self.vision_stats['avg_confidence'] = (
                    alpha * current_conf + (1 - alpha) * self.vision_stats['avg_confidence']
                )
    
    def get_vision_stats(self) -> Dict[str, float]:
        """Get vision processing performance statistics"""
        stats = self.vision_stats.copy()
        if stats['total_frames'] > 0:
            stats['detection_rate'] = stats['successful_detections'] / stats['total_frames']
        else:
            stats['detection_rate'] = 0.0
        
        return stats
    
    def render_vision(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Render vision detections on image for debugging/visualization
        
        Args:
            rgb_image: Raw RGB image from camera
            
        Returns:
            Annotated image with detection overlays
        """
        if self.last_vision_features is None:
            return rgb_image
        
        return self.vision_processor.draw_detections(rgb_image, self.last_vision_features)
    
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Reset environment and vision processing"""
        
        # Reset vision features
        self.last_vision_features = None
        
        # Reset parent environment
        observation, info = super().reset(**kwargs)
        
        # Add vision info
        info['vision_stats'] = self.get_vision_stats()
        
        return observation, info


# Convenience function for creating the integrated environment
def create_vision_mavros_env(yolo_model_size: str = "n", 
                           confidence_threshold: float = 0.3,
                           **kwargs) -> IntegratedVisionMavrosEnv:
    """
    Create integrated MAVROS environment with YOLO11 vision
    
    Args:
        yolo_model_size: YOLO11 model size ('n', 's', 'm', 'l', 'x')
        confidence_threshold: Detection confidence threshold
        **kwargs: Additional arguments for MavrosEnv
        
    Returns:
        Configured IntegratedVisionMavrosEnv
    """
    return IntegratedVisionMavrosEnv(
        yolo_model_size=yolo_model_size,
        confidence_threshold=confidence_threshold,
        vision_observation_space=True,
        **kwargs
    ) 