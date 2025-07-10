"""
YOLO11-based Vision Processing for DeepFlyer Hoop Detection
Enhanced computer vision system using state-of-the-art object detection
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class HoopDetection:
    """Container for hoop detection results"""
    center: Tuple[float, float]  # (x, y) in image coordinates
    confidence: float
    distance: float  # meters from ZED depth
    diameter_pixels: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    hoop_id: Optional[int] = None


@dataclass
class VisionFeatures:
    """Processed vision features for RL agent"""
    primary_hoop: Optional[HoopDetection]
    all_hoops: List[HoopDetection]
    hoop_alignment: float  # -1.0 (left) to 1.0 (right), 0.0 = centered
    next_hoop_visible: bool
    detection_confidence: float
    processing_time_ms: float


class YOLO11VisionProcessor:
    """
    Advanced vision processor using YOLO11 for hoop detection in DeepFlyer
    """
    
    def __init__(self, model_path: str = "weights/best.pt", 
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.5,
                 target_classes: List[str] = None,
                 device: str = "auto"):
        """
        Initialize YOLO11 vision processor
        
        Args:
            model_path: Path to YOLO11 model (defaults to DeepFlyer custom model, fallback to yolo11l.pt)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            target_classes: List of class names to detect (None for all classes)
            device: Device to run inference on ('auto', 'cpu', 'cuda', etc.)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes or ["sports ball", "frisbee", "donut"]  # Hoop-like objects
        
        # Initialize YOLO11 model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO11 model: {model_path}")
            
            # Set device
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            self.model.to(self.device)
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO11 model: {e}")
            raise
        
        # Vision processing parameters
        self.image_center = None
        self.depth_scale = 1000.0  # Convert mm to meters for ZED
        
        # Performance tracking
        self.detection_history = []
        self.max_history = 10
        
        # Custom hoop training flag
        self.custom_trained = False
        
    def setup_custom_hoop_detection(self, custom_model_path: str):
        """
        Setup custom-trained YOLO11 model specifically for orange hoops
        
        Args:
            custom_model_path: Path to custom-trained model for hoop detection
        """
        try:
            self.model = YOLO(custom_model_path)
            self.model.to(self.device)
            self.custom_trained = True
            self.target_classes = ["hoop"]  # Custom class
            logger.info(f"Loaded custom hoop detection model: {custom_model_path}")
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            raise
    
    def process_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> VisionFeatures:
        """
        Process ZED camera frame using YOLO11 for hoop detection
        
        Args:
            rgb_image: RGB image from ZED Mini
            depth_image: Depth map from ZED Mini
            
        Returns:
            VisionFeatures: Processed features for RL agent
        """
        start_time = time.time()
        
        # Store image dimensions
        if self.image_center is None:
            height, width = rgb_image.shape[:2]
            self.image_center = (width // 2, height // 2)
        
        # Run YOLO11 inference
        try:
            results = self.model(
                rgb_image,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                verbose=False
            )
            
            # Extract detections
            detections = self._extract_detections(results[0], depth_image)
            
            # Process detections for RL features
            vision_features = self._process_detections(detections)
            
            # Add processing time
            processing_time = (time.time() - start_time) * 1000
            vision_features.processing_time_ms = processing_time
            
            # Update detection history
            self._update_detection_history(vision_features)
            
            return vision_features
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return self._create_empty_features()
    
    def _extract_detections(self, result, depth_image: np.ndarray) -> List[HoopDetection]:
        """Extract hoop detections from YOLO11 results"""
        detections = []
        
        if result.boxes is None:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
            # Get class name
            class_name = self.model.names[int(cls)]
            
            # Filter for target classes (hoop-like objects)
            if not self.custom_trained and class_name not in self.target_classes:
                continue
            
            if self.custom_trained and class_name != "hoop":
                continue
            
            # Calculate center point
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center = (center_x, center_y)
            
            # Get depth at center point
            try:
                depth_value = depth_image[int(center_y), int(center_x)]
                distance = depth_value / self.depth_scale if depth_value > 0 else float('inf')
            except (IndexError, ValueError):
                distance = float('inf')
            
            # Calculate diameter in pixels
            diameter_pixels = min(x2 - x1, y2 - y1)
            
            detection = HoopDetection(
                center=center,
                confidence=float(conf),
                distance=distance,
                diameter_pixels=diameter_pixels,
                bbox=(x1, y1, x2, y2),
                hoop_id=i
            )
            
            detections.append(detection)
        
        return detections
    
    def _process_detections(self, detections: List[HoopDetection]) -> VisionFeatures:
        """Process detections into RL-ready features"""
        if not detections:
            return self._create_empty_features()
        
        # Sort detections by confidence and distance (prioritize close, confident detections)
        detections.sort(key=lambda d: (d.confidence * 0.7 + (1.0 / max(d.distance, 0.1)) * 0.3), reverse=True)
        
        # Primary hoop (closest/most confident)
        primary_hoop = detections[0]
        
        # Calculate alignment (-1.0 to 1.0, 0.0 = centered)
        hoop_alignment = self._calculate_alignment(primary_hoop.center)
        
        # Check if next hoop is visible
        next_hoop_visible = len(detections) > 1
        
        # Overall detection confidence
        detection_confidence = primary_hoop.confidence
        
        return VisionFeatures(
            primary_hoop=primary_hoop,
            all_hoops=detections,
            hoop_alignment=hoop_alignment,
            next_hoop_visible=next_hoop_visible,
            detection_confidence=detection_confidence,
            processing_time_ms=0.0  # Will be set by caller
        )
    
    def _calculate_alignment(self, hoop_center: Tuple[float, float]) -> float:
        """
        Calculate how well aligned the drone is with the hoop center
        
        Returns:
            alignment: -1.0 (far left) to 1.0 (far right), 0.0 = centered
        """
        if hoop_center is None or self.image_center is None:
            return 0.0
        
        horizontal_offset = hoop_center[0] - self.image_center[0]
        max_offset = self.image_center[0]  # Half of image width
        
        return np.clip(horizontal_offset / max_offset, -1.0, 1.0)
    
    def _create_empty_features(self) -> VisionFeatures:
        """Create empty vision features when no hoops detected"""
        return VisionFeatures(
            primary_hoop=None,
            all_hoops=[],
            hoop_alignment=0.0,
            next_hoop_visible=False,
            detection_confidence=0.0,
            processing_time_ms=0.0
        )
    
    def _update_detection_history(self, features: VisionFeatures):
        """Update detection history for stability analysis"""
        self.detection_history.append(features)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
    
    def get_detection_stability(self) -> float:
        """
        Calculate detection stability based on recent history
        
        Returns:
            stability: 0.0 (unstable) to 1.0 (very stable)
        """
        if len(self.detection_history) < 3:
            return 0.0
        
        # Count frames with successful detections
        successful_detections = sum(1 for f in self.detection_history[-5:] if f.primary_hoop is not None)
        return successful_detections / min(len(self.detection_history), 5)
    
    def draw_detections(self, image: np.ndarray, features: VisionFeatures) -> np.ndarray:
        """
        Draw detection results on image for visualization
        
        Args:
            image: Input image
            features: Vision features to visualize
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Draw all hoop detections
        for i, hoop in enumerate(features.all_hoops):
            # Color coding: primary hoop in green, others in blue
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(coord) for coord in hoop.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            center = (int(hoop.center[0]), int(hoop.center[1]))
            cv2.circle(annotated, center, 5, color, -1)
            
            # Add text info
            text = f"H{hoop.hoop_id}: {hoop.confidence:.2f} ({hoop.distance:.1f}m)"
            cv2.putText(annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
        
        # Draw alignment indicator
        if features.primary_hoop:
            # Draw line from image center to hoop center
            img_center = self.image_center
            hoop_center = (int(features.primary_hoop.center[0]), int(features.primary_hoop.center[1]))
            cv2.line(annotated, img_center, hoop_center, (0, 255, 255), 2)
            
            # Add alignment text
            align_text = f"Alignment: {features.hoop_alignment:.2f}"
            cv2.putText(annotated, align_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Add processing info
        proc_text = f"Processing: {features.processing_time_ms:.1f}ms"
        cv2.putText(annotated, proc_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return annotated


class HoopDetectionValidator:
    """
    Validator for hoop detection results to filter false positives
    """
    
    def __init__(self, min_area_ratio: float = 0.001, max_area_ratio: float = 0.3,
                 min_aspect_ratio: float = 0.7, max_aspect_ratio: float = 1.5):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
    
    def validate_detection(self, detection: HoopDetection, image_shape: Tuple[int, int]) -> bool:
        """
        Validate if detection is likely a real hoop
        
        Args:
            detection: HoopDetection to validate
            image_shape: (height, width) of the source image
            
        Returns:
            True if detection passes validation
        """
        # Calculate area ratio
        x1, y1, x2, y2 = detection.bbox
        detection_area = (x2 - x1) * (y2 - y1)
        image_area = image_shape[0] * image_shape[1]
        area_ratio = detection_area / image_area
        
        # Check area ratio
        if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
            return False
        
        # Check aspect ratio
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / max(height, 1)
        
        if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
            return False
        
        # Check distance reasonableness (hoops should be 0.5-10 meters away)
        if detection.distance < 0.5 or detection.distance > 10.0:
            return False
        
        return True


# Convenience function for easy integration
def create_yolo11_processor(model_path: str = "weights/best.pt", confidence: float = 0.3) -> YOLO11VisionProcessor:
    """
    Create YOLO11 vision processor for DeepFlyer hoop detection
    
    Args:
        model_path: Path to YOLO11 model (defaults to DeepFlyer custom-trained model)
        confidence: Confidence threshold for detections
        
    Returns:
        Configured YOLO11VisionProcessor
    """
    import os
    
    # Try to use the specified model, fallback to yolo11l.pt if not found
    final_model_path = model_path
    if not os.path.exists(model_path) and model_path == "weights/best.pt":
        logger.warning(f"Custom model {model_path} not found, falling back to yolo11l.pt")
        final_model_path = "yolo11l.pt"  # High accuracy fallback
    
    processor = YOLO11VisionProcessor(
        model_path=final_model_path,
        confidence_threshold=confidence,
        nms_threshold=0.5,
        device="auto"
    )
    
    # Set configuration based on model type
    if "best.pt" in final_model_path or "hoop" in final_model_path.lower():
        processor.custom_trained = True
        processor.target_classes = ["hoop"]  # Custom class for trained model
        logger.info("Using DeepFlyer custom-trained hoop detection model")
    else:
        processor.custom_trained = False
        processor.target_classes = ["sports ball", "frisbee", "donut"]  # General objects
        logger.info(f"Using general YOLO11 model: {final_model_path}")
    
    return processor 