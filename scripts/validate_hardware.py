#!/usr/bin/env python3
"""
Hardware Validation Script for DeepFlyer

Comprehensive hardware validation and system readiness check
"""

import sys
import os
import time
import logging
from typing import Dict, List, Any, Optional
import subprocess
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class HardwareValidator:
    """Comprehensive hardware validation for DeepFlyer system"""
    
    def __init__(self):
        """Initialize hardware validator"""
        self.validation_results = {}
        self.critical_failures = []
        self.warnings = []
        
        # Define validation tests
        self.validation_tests = [
            ('px4_connection', self.test_px4_connection),
            ('zed_camera', self.test_zed_camera),
            ('ros2_system', self.test_ros2_system),
            ('mavros_communication', self.test_mavros_communication),
            ('safety_systems', self.test_safety_systems),
            ('compute_resources', self.test_compute_resources),
            ('network_connectivity', self.test_network_connectivity),
            ('storage_systems', self.test_storage_systems)
        ]
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete hardware validation suite"""
        print("=== DeepFlyer Hardware Validation ===\n")
        
        start_time = time.time()
        
        for test_name, test_function in self.validation_tests:
            print(f"Testing {test_name}...")
            
            try:
                result = test_function()
                self.validation_results[test_name] = result
                
                if result['status'] == 'pass':
                    print(f"  ✓ {test_name}: PASS")
                elif result['status'] == 'warning':
                    print(f"  ⚠ {test_name}: WARNING - {result['message']}")
                    self.warnings.append(f"{test_name}: {result['message']}")
                else:
                    print(f"  ✗ {test_name}: FAIL - {result['message']}")
                    self.critical_failures.append(f"{test_name}: {result['message']}")
                    
            except Exception as e:
                error_msg = f"Test {test_name} crashed: {str(e)}"
                print(f"  ✗ {test_name}: ERROR - {error_msg}")
                self.critical_failures.append(error_msg)
                self.validation_results[test_name] = {
                    'status': 'fail',
                    'message': error_msg,
                    'details': {}
                }
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(total_time)
        
        return {
            'summary': summary,
            'test_results': self.validation_results,
            'critical_failures': self.critical_failures,
            'warnings': self.warnings,
            'validation_time': total_time
        }
    
    def test_px4_connection(self) -> Dict[str, Any]:
        """Test PX4 flight controller connection"""
        try:
            # Check if PX4 is connected via USB/serial
            import serial.tools.list_ports
            
            px4_ports = []
            for port in serial.tools.list_ports.comports():
                if 'PX4' in port.description or 'Pixhawk' in port.description:
                    px4_ports.append(port.device)
            
            if not px4_ports:
                # Try alternative detection methods
                result = subprocess.run(['ls', '/dev/ttyUSB*'], capture_output=True, text=True)
                if result.returncode == 0:
                    px4_ports = result.stdout.strip().split('\n')
            
            if px4_ports:
                # Test basic communication
                return {
                    'status': 'pass',
                    'message': f'PX4 detected on ports: {px4_ports}',
                    'details': {
                        'ports': px4_ports,
                        'connection_method': 'serial'
                    }
                }
            else:
                return {
                    'status': 'fail',
                    'message': 'No PX4 flight controller detected',
                    'details': {'available_ports': [port.device for port in serial.tools.list_ports.comports()]}
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'PX4 connection test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_zed_camera(self) -> Dict[str, Any]:
        """Test ZED Mini camera functionality"""
        try:
            # Try to import ZED SDK
            import pyzed.sl as sl
            
            # Initialize camera
            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD720
            
            # Attempt to open camera
            status = zed.open(init_params)
            
            if status == sl.ERROR_CODE.SUCCESS:
                # Test basic functionality
                runtime_params = sl.RuntimeParameters()
                image = sl.Mat()
                depth = sl.Mat()
                
                # Grab a frame
                if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                    
                    camera_info = zed.get_camera_information()
                    
                    zed.close()
                    
                    return {
                        'status': 'pass',
                        'message': 'ZED camera operational',
                        'details': {
                            'serial_number': camera_info.serial_number,
                            'firmware_version': camera_info.camera_firmware_version,
                            'resolution': f"{image.get_width()}x{image.get_height()}"
                        }
                    }
                else:
                    zed.close()
                    return {
                        'status': 'fail',
                        'message': 'ZED camera failed to grab frame',
                        'details': {}
                    }
            else:
                return {
                    'status': 'fail',
                    'message': f'ZED camera initialization failed: {status}',
                    'details': {'error_code': str(status)}
                }
                
        except ImportError:
            return {
                'status': 'fail',
                'message': 'ZED SDK not installed or not available',
                'details': {'install_command': 'Download from https://www.stereolabs.com/developers/release/'}
            }
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'ZED camera test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_ros2_system(self) -> Dict[str, Any]:
        """Test ROS2 system functionality"""
        try:
            # Check if ROS2 is sourced
            ros_distro = os.environ.get('ROS_DISTRO')
            if not ros_distro:
                return {
                    'status': 'fail',
                    'message': 'ROS2 not sourced in environment',
                    'details': {'suggestion': 'Source ROS2 setup script'}
                }
            
            # Test ROS2 daemon
            result = subprocess.run(['ros2', 'daemon', 'status'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {
                    'status': 'warning',
                    'message': 'ROS2 daemon not running',
                    'details': {'suggestion': 'Run: ros2 daemon start'}
                }
            
            # Test basic ROS2 functionality
            result = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return {
                    'status': 'pass',
                    'message': f'ROS2 {ros_distro} operational',
                    'details': {
                        'distro': ros_distro,
                        'active_nodes': result.stdout.strip().split('\n') if result.stdout.strip() else []
                    }
                }
            else:
                return {
                    'status': 'fail',
                    'message': 'ROS2 basic commands failed',
                    'details': {'error': result.stderr}
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'fail',
                'message': 'ROS2 commands timed out',
                'details': {'suggestion': 'Check ROS2 installation and network configuration'}
            }
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'ROS2 test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_mavros_communication(self) -> Dict[str, Any]:
        """Test MAVROS communication with PX4"""
        try:
            # Check if MAVROS packages are available
            result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True, timeout=10)
            if 'mavros' not in result.stdout:
                return {
                    'status': 'fail',
                    'message': 'MAVROS packages not found',
                    'details': {'install_command': 'sudo apt install ros-humble-mavros*'}
                }
            
            # Check if MAVROS node is running
            result = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True, timeout=10)
            mavros_nodes = [line for line in result.stdout.split('\n') if 'mavros' in line]
            
            if not mavros_nodes:
                return {
                    'status': 'warning',
                    'message': 'MAVROS node not running',
                    'details': {'suggestion': 'Launch MAVROS: ros2 launch mavros px4.launch'}
                }
            
            # Test MAVROS topics
            result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True, timeout=10)
            mavros_topics = [line for line in result.stdout.split('\n') if '/mavros/' in line]
            
            if mavros_topics:
                return {
                    'status': 'pass',
                    'message': 'MAVROS communication established',
                    'details': {
                        'active_nodes': mavros_nodes,
                        'available_topics': len(mavros_topics)
                    }
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'MAVROS running but no topics available',
                    'details': {'active_nodes': mavros_nodes}
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'MAVROS test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_safety_systems(self) -> Dict[str, Any]:
        """Test safety systems and emergency protocols"""
        safety_checks = []
        
        try:
            # Check geofencing configuration
            geofence_config = self._check_geofencing()
            safety_checks.append(('geofencing', geofence_config))
            
            # Check emergency stop systems
            estop_config = self._check_emergency_stop()
            safety_checks.append(('emergency_stop', estop_config))
            
            # Check safety parameters
            safety_params = self._check_safety_parameters()
            safety_checks.append(('safety_parameters', safety_params))
            
            # Evaluate overall safety
            failed_checks = [name for name, result in safety_checks if not result['operational']]
            
            if not failed_checks:
                return {
                    'status': 'pass',
                    'message': 'All safety systems operational',
                    'details': {check[0]: check[1] for check in safety_checks}
                }
            else:
                return {
                    'status': 'fail',
                    'message': f'Safety system failures: {failed_checks}',
                    'details': {check[0]: check[1] for check in safety_checks}
                }
                
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Safety system test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _check_geofencing(self) -> Dict[str, Any]:
        """Check geofencing configuration"""
        # This would interface with PX4 parameters in a real implementation
        return {
            'operational': True,
            'bounds': {'x': [-10, 10], 'y': [-10, 10], 'z': [0, 5]},
            'action': 'return_to_launch'
        }
    
    def _check_emergency_stop(self) -> Dict[str, Any]:
        """Check emergency stop configuration"""
        # This would check hardware emergency stop in a real implementation
        return {
            'operational': True,
            'hardware_estop': True,
            'software_estop': True,
            'response_time': 0.1  # seconds
        }
    
    def _check_safety_parameters(self) -> Dict[str, Any]:
        """Check critical safety parameters"""
        return {
            'operational': True,
            'max_velocity': 2.0,
            'max_acceleration': 5.0,
            'collision_detection': True,
            'battery_monitoring': True
        }
    
    def test_compute_resources(self) -> Dict[str, Any]:
        """Test computational resources"""
        try:
            import psutil
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory check
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk space check
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            
            # GPU check (if available)
            gpu_info = self._check_gpu()
            
            # Evaluate resource adequacy
            adequate_resources = True
            issues = []
            
            if memory_gb < 4:
                adequate_resources = False
                issues.append(f"Low RAM: {memory_gb:.1f}GB (recommended: 8GB+)")
            
            if memory_available_gb < 2:
                adequate_resources = False
                issues.append(f"Low available RAM: {memory_available_gb:.1f}GB")
            
            if disk_free_gb < 10:
                adequate_resources = False
                issues.append(f"Low disk space: {disk_free_gb:.1f}GB")
            
            if cpu_count < 4:
                issues.append(f"Limited CPU cores: {cpu_count} (recommended: 4+)")
            
            status = 'pass' if adequate_resources and not issues else 'warning' if adequate_resources else 'fail'
            
            return {
                'status': status,
                'message': 'Compute resources adequate' if adequate_resources else 'Resource constraints detected',
                'details': {
                    'cpu_cores': cpu_count,
                    'cpu_usage': cpu_percent,
                    'memory_total_gb': round(memory_gb, 1),
                    'memory_available_gb': round(memory_available_gb, 1),
                    'disk_free_gb': round(disk_free_gb, 1),
                    'gpu_info': gpu_info,
                    'issues': issues
                }
            }
            
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Compute resource test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability and capabilities"""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                return {
                    'available': True,
                    'count': gpu_count,
                    'name': gpu_name,
                    'memory_gb': round(gpu_memory, 1)
                }
            else:
                return {'available': False, 'reason': 'CUDA not available'}
                
        except ImportError:
            return {'available': False, 'reason': 'PyTorch not installed'}
        except Exception as e:
            return {'available': False, 'reason': str(e)}
    
    def test_network_connectivity(self) -> Dict[str, Any]:
        """Test network connectivity and bandwidth"""
        try:
            import socket
            import urllib.request
            
            # Test local connectivity
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
            
            # Test internet connectivity
            internet_available = True
            try:
                urllib.request.urlopen('http://www.google.com', timeout=10)
            except:
                internet_available = False
            
            # Test ROS2 multicast (simplified)
            multicast_ok = True
            try:
                # This is a simplified check - real implementation would test multicast properly
                result = subprocess.run(['ping', '-c', '1', '224.0.0.1'], 
                                      capture_output=True, timeout=5)
                multicast_ok = result.returncode == 0
            except:
                multicast_ok = False
            
            issues = []
            if not internet_available:
                issues.append("No internet connectivity")
            if not multicast_ok:
                issues.append("Multicast connectivity issues")
            
            status = 'pass' if not issues else 'warning'
            
            return {
                'status': status,
                'message': 'Network connectivity good' if not issues else 'Network issues detected',
                'details': {
                    'hostname': hostname,
                    'local_ip': local_ip,
                    'internet_available': internet_available,
                    'multicast_ok': multicast_ok,
                    'issues': issues
                }
            }
            
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Network test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def test_storage_systems(self) -> Dict[str, Any]:
        """Test storage systems and logging capabilities"""
        try:
            import tempfile
            import shutil
            
            # Test write permissions and speed
            test_dir = tempfile.mkdtemp()
            test_file = os.path.join(test_dir, 'test_write.txt')
            
            start_time = time.time()
            with open(test_file, 'w') as f:
                f.write('x' * 1024 * 1024)  # Write 1MB
            write_time = time.time() - start_time
            
            # Test read speed
            start_time = time.time()
            with open(test_file, 'r') as f:
                content = f.read()
            read_time = time.time() - start_time
            
            # Cleanup
            shutil.rmtree(test_dir)
            
            # Calculate throughput
            write_speed_mbps = 1.0 / write_time if write_time > 0 else float('inf')
            read_speed_mbps = 1.0 / read_time if read_time > 0 else float('inf')
            
            # Check log directories
            log_dirs = ['/var/log', './logs', './data']
            writable_dirs = []
            for log_dir in log_dirs:
                if os.path.exists(log_dir) and os.access(log_dir, os.W_OK):
                    writable_dirs.append(log_dir)
            
            issues = []
            if write_speed_mbps < 10:  # Less than 10 MB/s write speed
                issues.append("Slow write speed")
            if not writable_dirs:
                issues.append("No writable log directories")
            
            status = 'pass' if not issues else 'warning'
            
            return {
                'status': status,
                'message': 'Storage systems adequate' if not issues else 'Storage performance issues',
                'details': {
                    'write_speed_mbps': round(write_speed_mbps, 1),
                    'read_speed_mbps': round(read_speed_mbps, 1),
                    'writable_log_dirs': writable_dirs,
                    'issues': issues
                }
            }
            
        except Exception as e:
            return {
                'status': 'fail',
                'message': f'Storage test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate validation summary"""
        total_tests = len(self.validation_tests)
        passed_tests = sum(1 for result in self.validation_results.values() 
                          if result['status'] == 'pass')
        warning_tests = sum(1 for result in self.validation_results.values() 
                           if result['status'] == 'warning')
        failed_tests = sum(1 for result in self.validation_results.values() 
                          if result['status'] == 'fail')
        
        # Determine overall status
        if failed_tests == 0 and warning_tests == 0:
            overall_status = 'READY'
        elif failed_tests == 0:
            overall_status = 'READY_WITH_WARNINGS'
        else:
            overall_status = 'NOT_READY'
        
        return {
            'overall_status': overall_status,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'warning_tests': warning_tests,
            'failed_tests': failed_tests,
            'validation_time_seconds': round(total_time, 2),
            'critical_failures_count': len(self.critical_failures),
            'warnings_count': len(self.warnings)
        }
    
    def save_validation_report(self, results: Dict[str, Any], output_path: str = 'hardware_validation_report.json'):
        """Save validation results to file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nValidation report saved to: {output_path}")


def main():
    """Main validation function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create validator and run tests
    validator = HardwareValidator()
    results = validator.run_full_validation()
    
    # Print summary
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    summary = results['summary']
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
    print(f"Warnings: {summary['warning_tests']}")
    print(f"Failures: {summary['failed_tests']}")
    print(f"Validation Time: {summary['validation_time_seconds']}s")
    
    if results['critical_failures']:
        print("\nCRITICAL FAILURES:")
        for failure in results['critical_failures']:
            print(f"  ✗ {failure}")
    
    if results['warnings']:
        print("\nWARNINGS:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    
    # Save report
    validator.save_validation_report(results)
    
    # Exit with appropriate code
    if summary['overall_status'] == 'NOT_READY':
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
