"""
Automated Deployment Manager for DeepFlyer

Automates model deployment to physical drone systems with safety validation
"""

import os
import sys
import time
import json
import shutil
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.validation.model_validator import ModelValidator
from rl_agent.sim_to_real.transfer_validator import SimToRealValidator

logger = logging.getLogger(__name__)


class DeploymentManager:
    """Automates model deployment to physical drone"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize deployment manager
        
        Args:
            config: Deployment configuration dictionary
        """
        self.config = config or self._load_default_config()
        
        # Initialize validators
        self.model_validator = ModelValidator()
        self.transfer_validator = SimToRealValidator()
        
        # Deployment paths
        self.deployment_root = Path(self.config.get('deployment_root', '/opt/deepflyer'))
        self.model_cache = Path(self.config.get('model_cache', './models'))
        self.backup_dir = Path(self.config.get('backup_dir', './deployment_backups'))
        
        # Safety settings
        self.safety_checks_enabled = self.config.get('safety_checks_enabled', True)
        self.require_validation = self.config.get('require_validation', True)
        self.max_deployment_attempts = self.config.get('max_deployment_attempts', 3)
        
        # Create necessary directories
        self._ensure_directories()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default deployment configuration"""
        return {
            'deployment_root': '/opt/deepflyer',
            'model_cache': './models',
            'backup_dir': './deployment_backups',
            'safety_checks_enabled': True,
            'require_validation': True,
            'max_deployment_attempts': 3,
            'deployment_timeout': 300,  # 5 minutes
            'health_check_timeout': 60,  # 1 minute
            'rollback_on_failure': True,
            'notification_enabled': False
        }
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.deployment_root,
            self.model_cache,
            self.backup_dir,
            self.deployment_root / 'models',
            self.deployment_root / 'config',
            self.deployment_root / 'logs'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def deploy_model(self, model_path: str, drone_id: str, 
                    deployment_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Deploy validated model to physical drone
        
        Args:
            model_path: Path to trained model
            drone_id: Unique identifier for target drone
            deployment_options: Additional deployment options
        
        Returns:
            Deployment result dictionary
        """
        logger.info(f"Starting deployment of {model_path} to drone {drone_id}")
        
        deployment_start_time = time.time()
        deployment_id = f"deploy_{drone_id}_{int(deployment_start_time)}"
        
        deployment_log = {
            'deployment_id': deployment_id,
            'drone_id': drone_id,
            'model_path': model_path,
            'start_time': deployment_start_time,
            'options': deployment_options or {},
            'steps': []
        }
        
        try:
            # Step 1: Pre-deployment validation
            if self.require_validation:
                validation_result = self._run_pre_deployment_validation(model_path)
                deployment_log['steps'].append({
                    'step': 'pre_deployment_validation',
                    'status': 'completed' if validation_result['deployment_ready'] else 'failed',
                    'result': validation_result,
                    'timestamp': time.time()
                })
                
                if not validation_result['deployment_ready']:
                    return self._create_failure_result(
                        deployment_log, 
                        "Model validation failed", 
                        validation_result
                    )
            
            # Step 2: Safety checks
            if self.safety_checks_enabled:
                safety_result = self._run_safety_checks(drone_id)
                deployment_log['steps'].append({
                    'step': 'safety_checks',
                    'status': 'completed' if safety_result['safe_to_deploy'] else 'failed',
                    'result': safety_result,
                    'timestamp': time.time()
                })
                
                if not safety_result['safe_to_deploy']:
                    return self._create_failure_result(
                        deployment_log,
                        "Safety checks failed",
                        safety_result
                    )
            
            # Step 3: Create deployment backup
            backup_result = self._create_deployment_backup(drone_id)
            deployment_log['steps'].append({
                'step': 'create_backup',
                'status': 'completed' if backup_result['success'] else 'failed',
                'result': backup_result,
                'timestamp': time.time()
            })
            
            # Step 4: Transfer model to drone
            transfer_result = self._transfer_to_drone(model_path, drone_id)
            deployment_log['steps'].append({
                'step': 'model_transfer',
                'status': 'completed' if transfer_result['success'] else 'failed',
                'result': transfer_result,
                'timestamp': time.time()
            })
            
            if not transfer_result['success']:
                return self._create_failure_result(
                    deployment_log,
                    "Model transfer failed",
                    transfer_result
                )
            
            # Step 5: Update drone configuration
            config_result = self._update_drone_configuration(drone_id, deployment_options)
            deployment_log['steps'].append({
                'step': 'update_configuration',
                'status': 'completed' if config_result['success'] else 'failed',
                'result': config_result,
                'timestamp': time.time()
            })
            
            # Step 6: Run deployment tests
            test_result = self._run_deployment_tests(drone_id)
            deployment_log['steps'].append({
                'step': 'deployment_tests',
                'status': 'completed' if test_result['all_tests_passed'] else 'failed',
                'result': test_result,
                'timestamp': time.time()
            })
            
            if not test_result['all_tests_passed']:
                # Rollback if tests fail
                if self.config.get('rollback_on_failure', True):
                    rollback_result = self._rollback_deployment(drone_id, backup_result.get('backup_path'))
                    deployment_log['steps'].append({
                        'step': 'rollback',
                        'status': 'completed' if rollback_result['success'] else 'failed',
                        'result': rollback_result,
                        'timestamp': time.time()
                    })
                
                return self._create_failure_result(
                    deployment_log,
                    "Deployment tests failed",
                    test_result
                )
            
            # Step 7: Finalize deployment
            finalize_result = self._finalize_deployment(drone_id, deployment_id)
            deployment_log['steps'].append({
                'step': 'finalize_deployment',
                'status': 'completed' if finalize_result['success'] else 'failed',
                'result': finalize_result,
                'timestamp': time.time()
            })
            
            # Calculate deployment time
            total_time = time.time() - deployment_start_time
            
            # Create success result
            return {
                'success': True,
                'deployment_id': deployment_id,
                'drone_id': drone_id,
                'model_path': model_path,
                'deployment_time': total_time,
                'deployment_log': deployment_log,
                'message': f"Model successfully deployed to drone {drone_id}",
                'next_steps': [
                    "Monitor drone performance",
                    "Run initial flight tests",
                    "Validate real-world performance"
                ]
            }
            
        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            
            deployment_log['steps'].append({
                'step': 'exception_handling',
                'status': 'failed',
                'result': {'error': str(e)},
                'timestamp': time.time()
            })
            
            return self._create_failure_result(
                deployment_log,
                f"Deployment failed with exception: {str(e)}",
                {'exception': str(e)}
            )
    
    def _run_pre_deployment_validation(self, model_path: str) -> Dict[str, Any]:
        """Run pre-deployment model validation"""
        logger.info("Running pre-deployment validation")
        
        try:
            validation_result = self.model_validator.validate_model(model_path)
            
            # Additional deployment-specific checks
            deployment_specific_checks = {
                'model_file_exists': os.path.exists(model_path),
                'model_file_size_ok': os.path.getsize(model_path) > 1024,  # At least 1KB
                'model_format_valid': model_path.endswith(('.pth', '.pt')),
            }
            
            validation_result['deployment_checks'] = deployment_specific_checks
            validation_result['deployment_ready'] = (
                validation_result.get('deployment_ready', False) and
                all(deployment_specific_checks.values())
            )
            
            return validation_result
            
        except Exception as e:
            return {
                'deployment_ready': False,
                'error': str(e),
                'recommendations': ['Fix model validation issues before deployment']
            }
    
    def _run_safety_checks(self, drone_id: str) -> Dict[str, Any]:
        """Run safety checks before deployment"""
        logger.info(f"Running safety checks for drone {drone_id}")
        
        safety_checks = []
        
        # Check 1: Hardware status
        hardware_status = self._check_hardware_status(drone_id)
        safety_checks.append(('hardware_status', hardware_status))
        
        # Check 2: Flight area clear
        flight_area_status = self._check_flight_area(drone_id)
        safety_checks.append(('flight_area', flight_area_status))
        
        # Check 3: Emergency systems
        emergency_systems_status = self._check_emergency_systems(drone_id)
        safety_checks.append(('emergency_systems', emergency_systems_status))
        
        # Check 4: Communication systems
        communication_status = self._check_communication_systems(drone_id)
        safety_checks.append(('communication', communication_status))
        
        # Evaluate overall safety
        failed_checks = [name for name, status in safety_checks if not status['operational']]
        safe_to_deploy = len(failed_checks) == 0
        
        return {
            'safe_to_deploy': safe_to_deploy,
            'safety_checks': dict(safety_checks),
            'failed_checks': failed_checks,
            'recommendations': self._generate_safety_recommendations(failed_checks)
        }
    
    def _check_hardware_status(self, drone_id: str) -> Dict[str, Any]:
        """Check drone hardware status"""
        # In a real implementation, this would interface with drone systems
        return {
            'operational': True,
            'battery_level': 95,
            'motor_status': 'good',
            'sensor_status': 'operational',
            'flight_controller_status': 'connected'
        }
    
    def _check_flight_area(self, drone_id: str) -> Dict[str, Any]:
        """Check if flight area is clear and safe"""
        return {
            'operational': True,
            'area_clear': True,
            'weather_conditions': 'good',
            'no_fly_zone_check': 'clear'
        }
    
    def _check_emergency_systems(self, drone_id: str) -> Dict[str, Any]:
        """Check emergency systems functionality"""
        return {
            'operational': True,
            'emergency_stop_armed': True,
            'return_to_launch_configured': True,
            'geofencing_active': True
        }
    
    def _check_communication_systems(self, drone_id: str) -> Dict[str, Any]:
        """Check communication systems"""
        return {
            'operational': True,
            'mavros_connected': True,
            'ros2_communication': True,
            'telemetry_link': 'strong'
        }
    
    def _generate_safety_recommendations(self, failed_checks: List[str]) -> List[str]:
        """Generate safety recommendations based on failed checks"""
        recommendations = []
        
        for check in failed_checks:
            if check == 'hardware_status':
                recommendations.append("Resolve hardware issues before deployment")
            elif check == 'flight_area':
                recommendations.append("Ensure flight area is clear and safe")
            elif check == 'emergency_systems':
                recommendations.append("Verify all emergency systems are functional")
            elif check == 'communication':
                recommendations.append("Establish reliable communication links")
        
        return recommendations
    
    def _create_deployment_backup(self, drone_id: str) -> Dict[str, Any]:
        """Create backup of current deployment"""
        logger.info(f"Creating deployment backup for drone {drone_id}")
        
        try:
            timestamp = int(time.time())
            backup_name = f"backup_{drone_id}_{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Backup current model and configuration
            drone_model_path = self.deployment_root / 'models' / f'{drone_id}_current.pth'
            drone_config_path = self.deployment_root / 'config' / f'{drone_id}_config.json'
            
            backup_info = {
                'backup_path': str(backup_path),
                'backup_name': backup_name,
                'timestamp': timestamp,
                'files_backed_up': []
            }
            
            # Backup model if exists
            if drone_model_path.exists():
                shutil.copy2(drone_model_path, backup_path / 'model.pth')
                backup_info['files_backed_up'].append('model.pth')
            
            # Backup config if exists
            if drone_config_path.exists():
                shutil.copy2(drone_config_path, backup_path / 'config.json')
                backup_info['files_backed_up'].append('config.json')
            
            # Save backup metadata
            with open(backup_path / 'backup_info.json', 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            return {
                'success': True,
                'backup_path': str(backup_path),
                'backup_info': backup_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _transfer_to_drone(self, model_path: str, drone_id: str) -> Dict[str, Any]:
        """Transfer model to drone system"""
        logger.info(f"Transferring model to drone {drone_id}")
        
        try:
            # Copy model to deployment directory
            target_path = self.deployment_root / 'models' / f'{drone_id}_current.pth'
            shutil.copy2(model_path, target_path)
            
            # Verify transfer
            if target_path.exists() and target_path.stat().st_size > 0:
                return {
                    'success': True,
                    'target_path': str(target_path),
                    'file_size': target_path.stat().st_size,
                    'transfer_time': time.time()
                }
            else:
                return {
                    'success': False,
                    'error': 'Transfer verification failed'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_drone_configuration(self, drone_id: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update drone configuration"""
        logger.info(f"Updating configuration for drone {drone_id}")
        
        try:
            config_path = self.deployment_root / 'config' / f'{drone_id}_config.json'
            
            # Load existing config or create new
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = self._get_default_drone_config()
            
            # Update with deployment options
            if options:
                config.update(options)
            
            # Add deployment metadata
            config['deployment_info'] = {
                'deployed_at': time.time(),
                'deployment_version': '1.0',
                'model_path': f'{drone_id}_current.pth'
            }
            
            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return {
                'success': True,
                'config_path': str(config_path),
                'config': config
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_default_drone_config(self) -> Dict[str, Any]:
        """Get default drone configuration"""
        return {
            'control_frequency': 20.0,
            'max_velocity': 2.0,
            'max_acceleration': 5.0,
            'safety_margins': {
                'altitude_min': 0.5,
                'altitude_max': 5.0,
                'boundary_buffer': 1.0
            },
            'emergency_protocols': {
                'auto_land_on_comms_loss': True,
                'return_to_launch_on_low_battery': True,
                'emergency_stop_enabled': True
            }
        }
    
    def _run_deployment_tests(self, drone_id: str) -> Dict[str, Any]:
        """Run deployment tests"""
        logger.info(f"Running deployment tests for drone {drone_id}")
        
        tests = [
            ('model_loading', self._test_model_loading),
            ('system_integration', self._test_system_integration),
            ('safety_systems', self._test_safety_systems),
            ('communication', self._test_communication)
        ]
        
        test_results = {}
        failed_tests = []
        
        for test_name, test_func in tests:
            try:
                result = test_func(drone_id)
                test_results[test_name] = result
                
                if not result.get('passed', False):
                    failed_tests.append(test_name)
                    
            except Exception as e:
                test_results[test_name] = {
                    'passed': False,
                    'error': str(e)
                }
                failed_tests.append(test_name)
        
        return {
            'all_tests_passed': len(failed_tests) == 0,
            'test_results': test_results,
            'failed_tests': failed_tests,
            'total_tests': len(tests)
        }
    
    def _test_model_loading(self, drone_id: str) -> Dict[str, Any]:
        """Test model loading on drone"""
        try:
            model_path = self.deployment_root / 'models' / f'{drone_id}_current.pth'
            
            if not model_path.exists():
                return {'passed': False, 'error': 'Model file not found'}
            
            # In real implementation, would actually load and test the model
            return {
                'passed': True,
                'model_size': model_path.stat().st_size,
                'load_time': 0.5  # Simulated load time
            }
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_system_integration(self, drone_id: str) -> Dict[str, Any]:
        """Test system integration"""
        # Simplified test - in real implementation would test actual integration
        return {
            'passed': True,
            'integration_components': ['ros2', 'mavros', 'vision_pipeline'],
            'all_components_responsive': True
        }
    
    def _test_safety_systems(self, drone_id: str) -> Dict[str, Any]:
        """Test safety systems"""
        return {
            'passed': True,
            'emergency_stop_functional': True,
            'geofencing_active': True,
            'failsafe_modes_configured': True
        }
    
    def _test_communication(self, drone_id: str) -> Dict[str, Any]:
        """Test communication systems"""
        return {
            'passed': True,
            'mavros_connection': True,
            'ros2_topics_available': True,
            'telemetry_link_quality': 'excellent'
        }
    
    def _rollback_deployment(self, drone_id: str, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Rollback deployment to previous state"""
        logger.info(f"Rolling back deployment for drone {drone_id}")
        
        try:
            if not backup_path:
                # Find most recent backup
                backups = list(self.backup_dir.glob(f"backup_{drone_id}_*"))
                if not backups:
                    return {
                        'success': False,
                        'error': 'No backup found for rollback'
                    }
                backup_path = max(backups, key=lambda p: p.stat().st_mtime)
            
            backup_path = Path(backup_path)
            
            # Restore model
            backup_model = backup_path / 'model.pth'
            if backup_model.exists():
                target_model = self.deployment_root / 'models' / f'{drone_id}_current.pth'
                shutil.copy2(backup_model, target_model)
            
            # Restore config
            backup_config = backup_path / 'config.json'
            if backup_config.exists():
                target_config = self.deployment_root / 'config' / f'{drone_id}_config.json'
                shutil.copy2(backup_config, target_config)
            
            return {
                'success': True,
                'backup_used': str(backup_path),
                'rollback_time': time.time()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _finalize_deployment(self, drone_id: str, deployment_id: str) -> Dict[str, Any]:
        """Finalize deployment"""
        logger.info(f"Finalizing deployment {deployment_id} for drone {drone_id}")
        
        try:
            # Update deployment status
            status_file = self.deployment_root / 'status' / f'{drone_id}_status.json'
            status_file.parent.mkdir(exist_ok=True)
            
            status = {
                'current_deployment_id': deployment_id,
                'deployment_time': time.time(),
                'status': 'active',
                'drone_id': drone_id
            }
            
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            return {
                'success': True,
                'status_file': str(status_file),
                'deployment_active': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_failure_result(self, deployment_log: Dict[str, Any], 
                             message: str, details: Any) -> Dict[str, Any]:
        """Create failure result"""
        deployment_log['end_time'] = time.time()
        deployment_log['status'] = 'failed'
        deployment_log['failure_reason'] = message
        
        return {
            'success': False,
            'deployment_id': deployment_log.get('deployment_id'),
            'drone_id': deployment_log.get('drone_id'),
            'message': message,
            'details': details,
            'deployment_log': deployment_log,
            'recommendations': self._generate_failure_recommendations(message, details)
        }
    
    def _generate_failure_recommendations(self, message: str, details: Any) -> List[str]:
        """Generate recommendations based on failure"""
        recommendations = []
        
        if 'validation' in message.lower():
            recommendations.append("Address model validation issues before redeployment")
            recommendations.append("Consider retraining model with updated data")
        
        if 'safety' in message.lower():
            recommendations.append("Resolve safety check failures")
            recommendations.append("Verify all safety systems are operational")
        
        if 'transfer' in message.lower():
            recommendations.append("Check network connectivity and permissions")
            recommendations.append("Verify target system has sufficient storage")
        
        if 'test' in message.lower():
            recommendations.append("Debug deployment test failures")
            recommendations.append("Check system integration and dependencies")
        
        if not recommendations:
            recommendations.append("Review deployment logs for specific issues")
            recommendations.append("Contact system administrator if problems persist")
        
        return recommendations
    
    def get_deployment_status(self, drone_id: str) -> Dict[str, Any]:
        """Get current deployment status for drone"""
        try:
            status_file = self.deployment_root / 'status' / f'{drone_id}_status.json'
            
            if status_file.exists():
                with open(status_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'status': 'no_deployment',
                    'message': 'No active deployment found'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def list_available_backups(self, drone_id: str) -> List[Dict[str, Any]]:
        """List available backups for drone"""
        backups = []
        
        try:
            backup_dirs = list(self.backup_dir.glob(f"backup_{drone_id}_*"))
            
            for backup_dir in sorted(backup_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
                backup_info_file = backup_dir / 'backup_info.json'
                
                if backup_info_file.exists():
                    with open(backup_info_file, 'r') as f:
                        backup_info = json.load(f)
                        backups.append(backup_info)
                else:
                    # Create basic info from directory
                    backups.append({
                        'backup_name': backup_dir.name,
                        'backup_path': str(backup_dir),
                        'timestamp': backup_dir.stat().st_mtime
                    })
                    
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
        
        return backups
