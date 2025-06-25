#!/usr/bin/env python3
"""
Test Script for YOLO11 Vision Integration in DeepFlyer
Demonstrates the enhanced computer vision capabilities
"""

import os
import sys
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_agent.env.vision_processor import (
    YOLO11VisionProcessor, 
    create_yolo11_processor,
    HoopDetectionValidator
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLO11VisionDemo:
    """
    Demonstration class for YOLO11 vision processing
    """
    
    def __init__(self):
        """Initialize the demo"""
        self.vision_processor = None
        self.validator = HoopDetectionValidator()
        
        # Performance metrics
        self.metrics = {
            'total_frames': 0,
            'processing_times': [],
            'detection_counts': [],
            'confidence_scores': []
        }
    
    def setup_yolo11(self, model_size: str = "n", confidence: float = 0.3):
        """
        Setup YOLO11 vision processor
        
        Args:
            model_size: YOLO11 model size ('n', 's', 'm', 'l', 'x')
            confidence: Confidence threshold for detections
        """
        try:
            logger.info(f"Initializing YOLO11 model size: {model_size}")
            self.vision_processor = create_yolo11_processor(
                model_size=model_size,
                confidence=confidence
            )
            logger.info("✅ YOLO11 initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize YOLO11: {e}")
            logger.error("Make sure you have installed: pip install ultralytics torch torchvision")
            return False
    
    def test_synthetic_hoop_image(self):
        """Test with a synthetic hoop image"""
        logger.info("🎯 Testing with synthetic hoop image...")
        
        # Create a synthetic image with orange hoop-like objects
        image = self.create_synthetic_hoop_image()
        depth = np.full((image.shape[0], image.shape[1]), 2000, dtype=np.uint16)  # 2m depth
        
        # Process with YOLO11
        start_time = time.time()
        features = self.vision_processor.process_frame(image, depth)
        processing_time = (time.time() - start_time) * 1000
        
        # Display results
        self.display_results(image, features, processing_time, "Synthetic Test")
        
        return features
    
    def create_synthetic_hoop_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """
        Create a synthetic image with hoop-like objects for testing
        
        Args:
            width: Image width
            height: Image height
            
        Returns:
            Synthetic RGB image with orange circles (hoops)
        """
        # Create base image (sky-like background)
        image = np.full((height, width, 3), [135, 206, 235], dtype=np.uint8)  # Sky blue
        
        # Add some "ground" at bottom
        cv2.rectangle(image, (0, int(height * 0.7)), (width, height), (34, 139, 34), -1)  # Green
        
        # Add orange "hoops" (circles)
        hoops = [
            {'center': (320, 200), 'radius': 50, 'thickness': 15},  # Main hoop
            {'center': (150, 300), 'radius': 30, 'thickness': 10},  # Smaller hoop
            {'center': (500, 250), 'radius': 40, 'thickness': 12},  # Side hoop
        ]
        
        for hoop in hoops:
            # Draw orange hoop (outer circle)
            cv2.circle(image, hoop['center'], hoop['radius'], (0, 165, 255), hoop['thickness'])
            # Draw inner shadow for depth
            cv2.circle(image, hoop['center'], hoop['radius'] - hoop['thickness']//2, (0, 100, 200), 3)
        
        # Add some noise and texture
        noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def test_webcam_feed(self, duration: int = 30):
        """
        Test with live webcam feed
        
        Args:
            duration: Test duration in seconds
        """
        logger.info(f"🎥 Testing with webcam feed for {duration} seconds...")
        
        # Try to open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.warning("❌ Could not open webcam")
            return False
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        frame_count = 0
        
        logger.info("Press 'q' to quit early, 's' to save current frame")
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    break
                
                # Create mock depth image
                depth = np.full((frame.shape[0], frame.shape[1]), 2000, dtype=np.uint16)
                
                # Process with YOLO11
                proc_start = time.time()
                features = self.vision_processor.process_frame(frame, depth)
                processing_time = (time.time() - proc_start) * 1000
                
                # Draw detections
                annotated = self.vision_processor.draw_detections(frame, features)
                
                # Add performance info
                fps = frame_count / (time.time() - start_time) if time.time() - start_time > 0 else 0
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show frame
                cv2.imshow('YOLO11 Hoop Detection - DeepFlyer', annotated)
                
                # Update metrics
                self.update_metrics(features, processing_time)
                frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'yolo11_test_frame_{frame_count}.jpg', annotated)
                    logger.info(f"Saved frame {frame_count}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        logger.info(f"Processed {frame_count} frames in {time.time() - start_time:.1f} seconds")
        self.print_performance_summary()
        
        return True
    
    def benchmark_performance(self, num_frames: int = 100):
        """
        Benchmark YOLO11 performance with synthetic images
        
        Args:
            num_frames: Number of frames to process
        """
        logger.info(f"⚡ Benchmarking YOLO11 performance with {num_frames} frames...")
        
        # Generate test images
        test_images = []
        for i in range(num_frames):
            img = self.create_synthetic_hoop_image()
            depth = np.full((img.shape[0], img.shape[1]), 2000, dtype=np.uint16)
            test_images.append((img, depth))
        
        # Warm-up run
        logger.info("Warming up...")
        for i in range(5):
            img, depth = test_images[i % len(test_images)]
            self.vision_processor.process_frame(img, depth)
        
        # Benchmark run
        logger.info("Running benchmark...")
        start_time = time.time()
        
        for i, (img, depth) in enumerate(test_images):
            proc_start = time.time()
            features = self.vision_processor.process_frame(img, depth)
            processing_time = (time.time() - proc_start) * 1000
            
            self.update_metrics(features, processing_time)
            
            if (i + 1) % 20 == 0:
                logger.info(f"Processed {i + 1}/{num_frames} frames...")
        
        total_time = time.time() - start_time
        avg_fps = num_frames / total_time
        
        logger.info(f"✅ Benchmark complete!")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average FPS: {avg_fps:.1f}")
        
        self.print_performance_summary()
    
    def compare_with_traditional_cv(self):
        """Compare YOLO11 with traditional computer vision approach"""
        logger.info("📊 Comparing YOLO11 vs Traditional CV...")
        
        # Create test image
        test_image = self.create_synthetic_hoop_image()
        depth_image = np.full((test_image.shape[0], test_image.shape[1]), 2000, dtype=np.uint16)
        
        # Test YOLO11
        start_time = time.time()
        yolo_features = self.vision_processor.process_frame(test_image, depth_image)
        yolo_time = (time.time() - start_time) * 1000
        
        # Test Traditional CV (simplified version)
        start_time = time.time()
        traditional_detections = self.traditional_cv_detection(test_image)
        traditional_time = (time.time() - start_time) * 1000
        
        # Compare results
        print("\n" + "="*60)
        print("🔍 COMPUTER VISION COMPARISON")
        print("="*60)
        print(f"YOLO11:")
        print(f"  - Processing time: {yolo_time:.2f}ms")
        print(f"  - Detections: {len(yolo_features.all_hoops)}")
        print(f"  - Primary hoop confidence: {yolo_features.detection_confidence:.3f}")
        print(f"  - Detection stability: {self.vision_processor.get_detection_stability():.3f}")
        
        print(f"\nTraditional CV:")
        print(f"  - Processing time: {traditional_time:.2f}ms")
        print(f"  - Detections: {len(traditional_detections)}")
        
        # Visual comparison
        yolo_annotated = self.vision_processor.draw_detections(test_image.copy(), yolo_features)
        traditional_annotated = self.draw_traditional_detections(test_image.copy(), traditional_detections)
        
        # Show side by side
        comparison = np.hstack([traditional_annotated, yolo_annotated])
        
        # Add labels
        cv2.putText(comparison, "Traditional CV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(comparison, "YOLO11", (test_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Comparison: Traditional CV vs YOLO11', comparison)
        logger.info("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def traditional_cv_detection(self, image: np.ndarray) -> list:
        """
        Simple traditional computer vision for comparison
        (Simplified version of the original DeepFlyer approach)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Orange color range
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w//2, y + h//2)
                detections.append({
                    'center': center,
                    'bbox': (x, y, x+w, y+h),
                    'area': area
                })
        
        return detections
    
    def draw_traditional_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw traditional CV detections"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center = detection['center']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw center
            cv2.circle(image, center, 5, (0, 255, 255), -1)
            
            # Add area info
            cv2.putText(image, f"Area: {detection['area']:.0f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        return image
    
    def display_results(self, image: np.ndarray, features, processing_time: float, title: str):
        """Display detection results"""
        print(f"\n🎯 {title} Results:")
        print(f"   Processing time: {processing_time:.2f}ms")
        print(f"   Hoops detected: {len(features.all_hoops)}")
        
        if features.primary_hoop:
            hoop = features.primary_hoop
            print(f"   Primary hoop:")
            print(f"     - Center: ({hoop.center[0]:.1f}, {hoop.center[1]:.1f})")
            print(f"     - Distance: {hoop.distance:.2f}m")
            print(f"     - Confidence: {hoop.confidence:.3f}")
            print(f"     - Alignment: {features.hoop_alignment:.3f}")
        
        # Show annotated image
        annotated = self.vision_processor.draw_detections(image, features)
        cv2.imshow(f'YOLO11 Detection - {title}', annotated)
        
        logger.info("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def update_metrics(self, features, processing_time: float):
        """Update performance metrics"""
        self.metrics['total_frames'] += 1
        self.metrics['processing_times'].append(processing_time)
        self.metrics['detection_counts'].append(len(features.all_hoops))
        
        if features.primary_hoop:
            self.metrics['confidence_scores'].append(features.detection_confidence)
        else:
            self.metrics['confidence_scores'].append(0.0)
    
    def print_performance_summary(self):
        """Print performance summary"""
        if not self.metrics['processing_times']:
            return
        
        avg_time = np.mean(self.metrics['processing_times'])
        avg_fps = 1000.0 / avg_time if avg_time > 0 else 0
        avg_detections = np.mean(self.metrics['detection_counts'])
        avg_confidence = np.mean(self.metrics['confidence_scores'])
        detection_rate = sum(1 for c in self.metrics['detection_counts'] if c > 0) / len(self.metrics['detection_counts'])
        
        print("\n" + "="*50)
        print("📊 PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.metrics['total_frames']}")
        print(f"Average processing time: {avg_time:.2f}ms")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Average detections per frame: {avg_detections:.1f}")
        print(f"Detection rate: {detection_rate:.1%}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Min/Max processing time: {min(self.metrics['processing_times']):.2f}ms / {max(self.metrics['processing_times']):.2f}ms")


def main():
    """Main demo function"""
    print("🚁 YOLO11 Vision Integration Demo for DeepFlyer")
    print("=" * 60)
    
    demo = YOLO11VisionDemo()
    
    # Setup YOLO11
    if not demo.setup_yolo11(model_size="n", confidence=0.3):
        print("❌ Failed to setup YOLO11. Exiting...")
        return
    
    while True:
        print("\n🎮 Choose a demo option:")
        print("1. Test with synthetic hoop image")
        print("2. Test with webcam feed (30 seconds)")
        print("3. Performance benchmark")
        print("4. Compare YOLO11 vs Traditional CV")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        try:
            if choice == "1":
                demo.test_synthetic_hoop_image()
            
            elif choice == "2":
                demo.test_webcam_feed(duration=30)
            
            elif choice == "3":
                demo.benchmark_performance(num_frames=100)
            
            elif choice == "4":
                demo.compare_with_traditional_cv()
            
            elif choice == "5":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-5.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Error during demo: {e}")


if __name__ == "__main__":
    main() 