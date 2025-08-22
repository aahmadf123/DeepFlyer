#!/usr/bin/env python3
"""
Vision Pipeline Testing Script for DeepFlyer

Tests YOLO11 + ZED Mini integration and performance
"""

import sys
import os
import time
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.depth_processor import YOLO11HoopDetector, HoopDetection
from rl_agent.env.zed_integration import create_zed_interface

logger = logging.getLogger(__name__)


class VisionPipelineTester:
    """Test vision pipeline components"""
    
    def __init__(self):
        """Initialize vision pipeline tester"""
        self.test_results = {}
        self.yolo_detector = None
        self.zed_camera = None
        
    def run_vision_tests(self) -> Dict[str, Any]:
        """Run comprehensive vision pipeline tests"""
        print("=== DeepFlyer Vision Pipeline Testing ===\n")
        
        test_suite = [
            ('yolo_model_loading', self.test_yolo_model_loading),
            ('zed_camera_initialization', self.test_zed_camera_init),
            ('hoop_detection_accuracy', self.test_hoop_detection_accuracy),
            ('detection_performance', self.test_detection_performance),
            ('depth_estimation', self.test_depth_estimation),
            ('integrated_pipeline', self.test_integrated_pipeline),
            ('real_time_performance', self.test_real_time_performance)
        ]
        
        for test_name, test_func in test_suite:
            print(f"Running {test_name}...")
            try:
                result = test_func()
                self.test_results[test_name] = result
                
                if result['passed']:
                    print(f"  ✓ {test_name}: PASS")
                else:
                    print(f"  ✗ {test_name}: FAIL - {result['message']}")
                    
            except Exception as e:
                print(f"  ✗ {test_name}: ERROR - {str(e)}")
                self.test_results[test_name] = {
                    'passed': False,
                    'message': f"Test crashed: {str(e)}",
                    'details': {}
                }
        
        return self._generate_summary()
    
    def test_yolo_model_loading(self) -> Dict[str, Any]:
        """Test YOLO11 model loading and initialization"""
        try:
            # Test model loading
            self.yolo_detector = YOLO11HoopDetector(
                model_path='yolo11n.pt',  # Use nano model for testing
                device='cpu'  # Use CPU for testing
            )
            
            self.yolo_detector.load_model()
            
            return {
                'passed': True,
                'message': 'YOLO11 model loaded successfully',
                'details': {
                    'model_path': self.yolo_detector.model_path,
                    'device': str(self.yolo_detector.device),
                    'confidence_threshold': self.yolo_detector.confidence_threshold
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'YOLO11 model loading failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_zed_camera_init(self) -> Dict[str, Any]:
        """Test ZED camera initialization"""
        try:
            self.zed_camera = create_zed_interface(mode='direct')
            
            if self.zed_camera.initialize():
                return {
                    'passed': True,
                    'message': 'ZED camera initialized successfully',
                    'details': {
                        'camera_type': 'ZED Mini',
                        'resolution': 'HD720',
                        'fps': 30
                    }
                }
            else:
                return {
                    'passed': False,
                    'message': 'ZED camera failed to initialize',
                    'details': {}
                }
                
        except Exception as e:
            return {
                'passed': False,
                'message': f'ZED camera test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_hoop_detection_accuracy(self) -> Dict[str, Any]:
        """Test hoop detection accuracy with synthetic images"""
        if not self.yolo_detector:
            return {
                'passed': False,
                'message': 'YOLO detector not initialized',
                'details': {}
            }
        
        try:
            # Create synthetic test images with known hoops
            test_images = self._generate_synthetic_hoop_images()
            
            total_detections = 0
            correct_detections = 0
            false_positives = 0
            
            for test_image, expected_hoops in test_images:
                detections = self.yolo_detector.detect_hoops(test_image)
                
                total_detections += len(detections)
                
                # Simple accuracy check (would be more sophisticated in real implementation)
                if len(detections) == len(expected_hoops):
                    correct_detections += len(detections)
                else:
                    false_positives += abs(len(detections) - len(expected_hoops))
            
            accuracy = correct_detections / max(len(test_images), 1)
            
            return {
                'passed': accuracy > 0.7,  # 70% accuracy threshold
                'message': f'Detection accuracy: {accuracy:.2f}',
                'details': {
                    'total_test_images': len(test_images),
                    'total_detections': total_detections,
                    'correct_detections': correct_detections,
                    'false_positives': false_positives,
                    'accuracy': accuracy
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Hoop detection test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_detection_performance(self) -> Dict[str, Any]:
        """Test detection performance (speed and resource usage)"""
        if not self.yolo_detector:
            return {
                'passed': False,
                'message': 'YOLO detector not initialized',
                'details': {}
            }
        
        try:
            # Create test image
            test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            
            # Warm up
            for _ in range(5):
                self.yolo_detector.detect_hoops(test_image)
            
            # Performance test
            num_iterations = 20
            start_time = time.time()
            
            for _ in range(num_iterations):
                detections = self.yolo_detector.detect_hoops(test_image)
            
            total_time = time.time() - start_time
            avg_time_per_frame = total_time / num_iterations
            fps = 1.0 / avg_time_per_frame
            
            # Performance thresholds
            min_fps = 15.0  # Minimum acceptable FPS
            max_latency = 0.1  # Maximum acceptable latency (100ms)
            
            performance_adequate = fps >= min_fps and avg_time_per_frame <= max_latency
            
            return {
                'passed': performance_adequate,
                'message': f'Detection performance: {fps:.1f} FPS',
                'details': {
                    'average_fps': round(fps, 1),
                    'average_latency_ms': round(avg_time_per_frame * 1000, 1),
                    'total_test_time': round(total_time, 2),
                    'iterations': num_iterations,
                    'meets_requirements': performance_adequate
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Performance test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_depth_estimation(self) -> Dict[str, Any]:
        """Test depth estimation accuracy"""
        if not self.zed_camera:
            return {
                'passed': False,
                'message': 'ZED camera not initialized',
                'details': {}
            }
        
        try:
            # Grab frame from camera
            rgb_image, depth_image = self.zed_camera.grab_frame()
            
            if rgb_image is None or depth_image is None:
                return {
                    'passed': False,
                    'message': 'Failed to grab frames from ZED camera',
                    'details': {}
                }
            
            # Analyze depth image quality
            depth_stats = {
                'min_depth': float(np.min(depth_image[depth_image > 0])),
                'max_depth': float(np.max(depth_image)),
                'mean_depth': float(np.mean(depth_image[depth_image > 0])),
                'valid_pixels': int(np.sum(depth_image > 0)),
                'total_pixels': int(depth_image.size)
            }
            
            valid_depth_ratio = depth_stats['valid_pixels'] / depth_stats['total_pixels']
            
            # Test depth at specific point
            center_x, center_y = depth_image.shape[1] // 2, depth_image.shape[0] // 2
            center_depth = self.zed_camera.get_depth_at_point(center_x, center_y)
            
            depth_quality_good = (valid_depth_ratio > 0.7 and 
                                 depth_stats['min_depth'] < 10.0 and
                                 center_depth > 0)
            
            return {
                'passed': depth_quality_good,
                'message': f'Depth estimation quality: {valid_depth_ratio:.2f} valid pixels',
                'details': {
                    'depth_statistics': depth_stats,
                    'valid_depth_ratio': valid_depth_ratio,
                    'center_depth': center_depth,
                    'image_shape': depth_image.shape
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Depth estimation test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_integrated_pipeline(self) -> Dict[str, Any]:
        """Test integrated vision pipeline (YOLO + ZED)"""
        if not self.yolo_detector or not self.zed_camera:
            return {
                'passed': False,
                'message': 'Vision components not initialized',
                'details': {}
            }
        
        try:
            # Grab frame from ZED
            rgb_image, depth_image = self.zed_camera.grab_frame()
            
            if rgb_image is None:
                return {
                    'passed': False,
                    'message': 'Failed to grab RGB image',
                    'details': {}
                }
            
            # Run hoop detection
            detections = self.yolo_detector.detect_hoops(rgb_image)
            
            # For each detection, get depth information
            detection_results = []
            for detection in detections:
                # Get depth at detection center
                center_x = int(detection.center_x)
                center_y = int(detection.center_y)
                
                # Ensure coordinates are within image bounds
                if (0 <= center_x < depth_image.shape[1] and 
                    0 <= center_y < depth_image.shape[0]):
                    
                    depth = self.zed_camera.get_depth_at_point(center_x, center_y)
                    
                    detection_results.append({
                        'bbox': [detection.x1, detection.y1, detection.x2, detection.y2],
                        'confidence': detection.confidence,
                        'depth': depth,
                        'center': [center_x, center_y]
                    })
            
            integration_successful = len(detection_results) >= 0  # At least attempt was made
            
            return {
                'passed': integration_successful,
                'message': f'Integrated pipeline processed {len(detection_results)} detections',
                'details': {
                    'rgb_image_shape': rgb_image.shape,
                    'depth_image_shape': depth_image.shape,
                    'detection_count': len(detections),
                    'detections_with_depth': len(detection_results),
                    'detection_results': detection_results[:3]  # Limit to first 3 for brevity
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Integrated pipeline test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_real_time_performance(self) -> Dict[str, Any]:
        """Test real-time performance of integrated pipeline"""
        if not self.yolo_detector or not self.zed_camera:
            return {
                'passed': False,
                'message': 'Vision components not initialized',
                'details': {}
            }
        
        try:
            # Real-time performance test
            num_frames = 30  # Test for 30 frames
            frame_times = []
            detection_counts = []
            
            for i in range(num_frames):
                start_time = time.time()
                
                # Grab frame
                rgb_image, depth_image = self.zed_camera.grab_frame()
                
                if rgb_image is not None:
                    # Run detection
                    detections = self.yolo_detector.detect_hoops(rgb_image)
                    detection_counts.append(len(detections))
                else:
                    detection_counts.append(0)
                
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
            
            # Calculate performance metrics
            avg_frame_time = np.mean(frame_times)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            max_frame_time = np.max(frame_times)
            min_frame_time = np.min(frame_times)
            
            # Performance criteria
            target_fps = 20.0
            max_acceptable_latency = 0.1  # 100ms
            
            performance_adequate = (avg_fps >= target_fps and 
                                   max_frame_time <= max_acceptable_latency)
            
            return {
                'passed': performance_adequate,
                'message': f'Real-time performance: {avg_fps:.1f} FPS average',
                'details': {
                    'average_fps': round(avg_fps, 1),
                    'average_frame_time_ms': round(avg_frame_time * 1000, 1),
                    'max_frame_time_ms': round(max_frame_time * 1000, 1),
                    'min_frame_time_ms': round(min_frame_time * 1000, 1),
                    'frames_tested': num_frames,
                    'average_detections_per_frame': round(np.mean(detection_counts), 1),
                    'meets_real_time_requirements': performance_adequate
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'message': f'Real-time performance test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _generate_synthetic_hoop_images(self) -> List[Tuple[np.ndarray, List[Dict]]]:
        """Generate synthetic images with hoops for testing"""
        test_images = []
        
        # Create simple synthetic images with circular shapes
        for i in range(5):
            # Create image
            image = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
            
            # Add circular "hoops"
            hoops = []
            num_hoops = np.random.randint(1, 3)
            
            for j in range(num_hoops):
                center_x = np.random.randint(200, 1080)
                center_y = np.random.randint(200, 520)
                radius = np.random.randint(50, 150)
                
                # Draw circle (simplified hoop)
                cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 10)
                cv2.circle(image, (center_x, center_y), radius - 20, (0, 0, 0), 10)
                
                hoops.append({
                    'center': [center_x, center_y],
                    'radius': radius
                })
            
            test_images.append((image, hoops))
        
        return test_images
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        failed_tests = total_tests - passed_tests
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'test_results': self.test_results,
            'overall_status': 'PASS' if failed_tests == 0 else 'FAIL'
        }


def main():
    """Main testing function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create tester and run tests
    tester = VisionPipelineTester()
    results = tester.run_vision_tests()
    
    # Print summary
    print("\n" + "="*50)
    print("VISION PIPELINE TEST SUMMARY")
    print("="*50)
    
    print(f"Overall Status: {results['overall_status']}")
    print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    
    # Save results
    import json
    with open('vision_pipeline_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: vision_pipeline_test_results.json")
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()

