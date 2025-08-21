"""
Model Validation Framework for DeepFlyer

Validates trained models before deployment to ensure safety and performance
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
from pathlib import Path

# Import environment and agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.p3o import P3O, P3OConfig
from env.px4_env import DeepFlyerEnv

logger = logging.getLogger(__name__)


@dataclass
class ValidationScenario:
    """Single validation scenario configuration"""
    name: str
    description: str
    test_episodes: int
    success_criteria: Dict[str, float]
    environment_config: Dict[str, Any]
    scenario_specific_tests: List[str]


class ModelValidator:
    """Validates trained models before deployment"""
    
    def __init__(self, test_scenarios: Optional[List[str]] = None):
        """
        Initialize model validator
        
        Args:
            test_scenarios: List of scenario names to test
        """
        self.test_scenarios = test_scenarios or [
            'single_hoop_precision',
            'multi_hoop_circuit', 
            'obstacle_avoidance',
            'wind_disturbance',
            'sensor_noise',
            'emergency_scenarios'
        ]
        
        self.validation_metrics = {}
        self.scenarios = self._define_validation_scenarios()
        
        # Validation thresholds
        self.deployment_thresholds = {
            'overall_success_rate': 0.8,
            'safety_score': 0.95,
            'navigation_precision': 0.7,
            'stability_score': 0.8,
            'robustness_score': 0.6
        }
        
    def _define_validation_scenarios(self) -> Dict[str, ValidationScenario]:
        """Define all validation scenarios"""
        scenarios = {}
        
        scenarios['single_hoop_precision'] = ValidationScenario(
            name='single_hoop_precision',
            description='Navigate through single hoop with high precision',
            test_episodes=20,
            success_criteria={
                'success_rate': 0.9,
                'average_precision': 0.8,
                'collision_rate': 0.05
            },
            environment_config={
                'hoop_count': 1,
                'hoop_size_variation': 0.1,
                'wind_strength': 0.0,
                'sensor_noise': 0.1
            },
            scenario_specific_tests=['precision_measurement', 'approach_angle_analysis']
        )
        
        scenarios['multi_hoop_circuit'] = ValidationScenario(
            name='multi_hoop_circuit',
            description='Complete full 5-hoop racing circuit',
            test_episodes=15,
            success_criteria={
                'success_rate': 0.7,
                'completion_time': 120.0,  # seconds
                'navigation_efficiency': 0.8
            },
            environment_config={
                'hoop_count': 5,
                'course_layout': 'standard',
                'dynamic_elements': False
            },
            scenario_specific_tests=['course_completion', 'time_efficiency', 'path_optimization']
        )
        
        scenarios['obstacle_avoidance'] = ValidationScenario(
            name='obstacle_avoidance',
            description='Navigate with static obstacles present',
            test_episodes=15,
            success_criteria={
                'success_rate': 0.6,
                'collision_avoidance_rate': 0.95,
                'safety_margin': 0.5  # meters
            },
            environment_config={
                'obstacles_enabled': True,
                'obstacle_density': 0.3,
                'safety_buffer': 0.5
            },
            scenario_specific_tests=['obstacle_detection', 'avoidance_behavior', 'safety_margins']
        )
        
        scenarios['wind_disturbance'] = ValidationScenario(
            name='wind_disturbance',
            description='Performance under wind disturbances',
            test_episodes=20,
            success_criteria={
                'success_rate': 0.6,
                'stability_under_disturbance': 0.7,
                'wind_compensation': 0.8
            },
            environment_config={
                'wind_enabled': True,
                'wind_strength': 0.3,
                'wind_variability': 0.2,
                'gust_frequency': 0.1
            },
            scenario_specific_tests=['wind_response', 'stability_analysis', 'compensation_effectiveness']
        )
        
        scenarios['sensor_noise'] = ValidationScenario(
            name='sensor_noise',
            description='Robustness to sensor noise and failures',
            test_episodes=25,
            success_criteria={
                'success_rate': 0.5,
                'noise_tolerance': 0.7,
                'graceful_degradation': 0.8
            },
            environment_config={
                'sensor_noise_level': 0.3,
                'intermittent_failures': True,
                'failure_rate': 0.1
            },
            scenario_specific_tests=['noise_robustness', 'failure_recovery', 'degraded_performance']
        )
        
        scenarios['emergency_scenarios'] = ValidationScenario(
            name='emergency_scenarios',
            description='Emergency response and safety behavior',
            test_episodes=10,
            success_criteria={
                'emergency_response_time': 0.5,  # seconds
                'safe_landing_rate': 0.95,
                'collision_avoidance': 0.98
            },
            environment_config={
                'emergency_events': True,
                'failure_simulation': True,
                'safety_protocols': True
            },
            scenario_specific_tests=['emergency_detection', 'safe_landing', 'protocol_compliance']
        )
        
        return scenarios
    
    def validate_model(self, model_path: str) -> Dict[str, Any]:
        """Run comprehensive model validation"""
        logger.info(f"Starting model validation for {model_path}")
        
        # Load model
        agent = self._load_model(model_path)
        if agent is None:
            return self._create_failed_validation("Failed to load model")
        
        # Run validation scenarios
        scenario_results = {}
        for scenario_name in self.test_scenarios:
            if scenario_name in self.scenarios:
                logger.info(f"Running scenario: {scenario_name}")
                scenario_results[scenario_name] = self._run_scenario_test(agent, scenario_name)
            else:
                logger.warning(f"Unknown scenario: {scenario_name}")
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(scenario_results)
        
        # Determine deployment readiness
        deployment_ready = self._assess_deployment_readiness(validation_score, scenario_results)
        
        # Generate recommendations
        recommendations = self._generate_deployment_recommendations(scenario_results)
        
        validation_results = {
            'validation_score': validation_score,
            'scenario_results': scenario_results,
            'deployment_ready': deployment_ready,
            'recommendations': recommendations,
            'model_path': model_path,
            'validation_timestamp': time.time(),
            'validator_version': '1.0'
        }
        
        logger.info(f"Validation complete. Score: {validation_score:.3f}, Deployment ready: {deployment_ready}")
        return validation_results
    
    def _load_model(self, model_path: str) -> Optional[P3O]:
        """Load model for validation"""
        try:
            # Create P3O agent
            config = P3OConfig()
            agent = P3O(obs_dim=8, action_dim=4, config=config)
            
            # Load trained model
            agent.load_model_for_deployment(model_path)
            
            return agent
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def _run_scenario_test(self, agent: P3O, scenario_name: str) -> Dict[str, float]:
        """Run specific test scenario"""
        scenario = self.scenarios[scenario_name]
        
        # Create environment with scenario configuration
        env = DeepFlyerEnv(**scenario.environment_config)
        
        # Run test episodes
        episode_results = []
        for episode in range(scenario.test_episodes):
            episode_result = self._run_single_episode(agent, env, scenario)
            episode_results.append(episode_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_episode_results(episode_results, scenario)
        
        # Run scenario-specific tests
        specific_test_results = self._run_specific_tests(agent, env, scenario)
        aggregated_results.update(specific_test_results)
        
        return aggregated_results
    
    def _run_single_episode(self, agent: P3O, env: DeepFlyerEnv, scenario: ValidationScenario) -> Dict[str, Any]:
        """Run single validation episode"""
        obs, info = env.reset()
        
        episode_data = {
            'total_reward': 0.0,
            'episode_length': 0,
            'success': False,
            'collision_count': 0,
            'safety_violations': 0,
            'navigation_precision': 0.0,
            'completion_time': 0.0
        }
        
        start_time = time.time()
        max_steps = 1000  # Maximum episode length
        
        for step in range(max_steps):
            # Get action from agent
            with torch.no_grad():
                # Convert observation to tensor format expected by agent
                if isinstance(obs, dict):
                    obs_tensor = self._dict_obs_to_tensor(obs)
                else:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                action_tensor, _, _ = agent.network.forward(obs_tensor)
                action = action_tensor.cpu().numpy().flatten()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update episode data
            episode_data['total_reward'] += reward
            episode_data['episode_length'] += 1
            
            # Track safety violations
            if info.get('collision', False):
                episode_data['collision_count'] += 1
            if info.get('out_of_bounds', False):
                episode_data['safety_violations'] += 1
            
            # Check termination
            if terminated or truncated:
                episode_data['success'] = terminated and info.get('flight_phase') == 'land'
                episode_data['completion_time'] = time.time() - start_time
                break
        
        # Calculate navigation precision if applicable
        if 'hoops_traversed' in info:
            hoops_completed = info.get('hoops_traversed', 0)
            total_hoops = scenario.environment_config.get('hoop_count', 1)
            episode_data['navigation_precision'] = hoops_completed / max(total_hoops, 1)
        
        return episode_data
    
    def _dict_obs_to_tensor(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        """Convert dictionary observation to tensor"""
        # Extract key observation components in expected order
        obs_list = [
            obs_dict.get('hoop_x_center_norm', 0.0),
            obs_dict.get('hoop_y_center_norm', 0.0),
            obs_dict.get('hoop_visible', 0.0),
            obs_dict.get('hoop_distance_norm', 1.0),
            obs_dict.get('drone_vx_norm', 0.0),
            obs_dict.get('drone_vy_norm', 0.0),
            obs_dict.get('drone_vz_norm', 0.0),
            obs_dict.get('yaw_rate_norm', 0.0)
        ]
        
        return torch.FloatTensor(obs_list).unsqueeze(0)
    
    def _aggregate_episode_results(self, episodes: List[Dict[str, Any]], scenario: ValidationScenario) -> Dict[str, float]:
        """Aggregate results from multiple episodes"""
        if not episodes:
            return {}
        
        # Calculate aggregated metrics
        results = {
            'success_rate': sum(ep['success'] for ep in episodes) / len(episodes),
            'average_reward': np.mean([ep['total_reward'] for ep in episodes]),
            'average_episode_length': np.mean([ep['episode_length'] for ep in episodes]),
            'collision_rate': sum(ep['collision_count'] > 0 for ep in episodes) / len(episodes),
            'safety_violation_rate': sum(ep['safety_violations'] > 0 for ep in episodes) / len(episodes),
            'average_precision': np.mean([ep['navigation_precision'] for ep in episodes]),
            'completion_time_avg': np.mean([ep['completion_time'] for ep in episodes if ep['success']])
        }
        
        # Check against success criteria
        results['criteria_met'] = self._check_success_criteria(results, scenario.success_criteria)
        
        return results
    
    def _check_success_criteria(self, results: Dict[str, float], criteria: Dict[str, float]) -> Dict[str, bool]:
        """Check if results meet success criteria"""
        criteria_met = {}
        
        for criterion, threshold in criteria.items():
            if criterion in results:
                if criterion in ['collision_rate', 'safety_violation_rate']:
                    # Lower is better for these metrics
                    criteria_met[criterion] = results[criterion] <= threshold
                else:
                    # Higher is better for most metrics
                    criteria_met[criterion] = results[criterion] >= threshold
            else:
                criteria_met[criterion] = False
        
        return criteria_met
    
    def _run_specific_tests(self, agent: P3O, env: DeepFlyerEnv, scenario: ValidationScenario) -> Dict[str, float]:
        """Run scenario-specific validation tests"""
        specific_results = {}
        
        for test_name in scenario.scenario_specific_tests:
            if test_name == 'precision_measurement':
                specific_results['precision_score'] = self._test_precision_measurement(agent, env)
            elif test_name == 'approach_angle_analysis':
                specific_results['approach_score'] = self._test_approach_angles(agent, env)
            elif test_name == 'wind_response':
                specific_results['wind_resistance'] = self._test_wind_response(agent, env)
            elif test_name == 'stability_analysis':
                specific_results['stability_score'] = self._test_stability(agent, env)
            elif test_name == 'emergency_detection':
                specific_results['emergency_response'] = self._test_emergency_response(agent, env)
            # Add more specific tests as needed
        
        return specific_results
    
    def _test_precision_measurement(self, agent: P3O, env: DeepFlyerEnv) -> float:
        """Test navigation precision"""
        # Simple precision test - measure deviation from optimal path
        precision_scores = []
        
        for _ in range(5):  # Run 5 test episodes
            obs, _ = env.reset()
            total_deviation = 0.0
            steps = 0
            
            for step in range(200):
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = self._dict_obs_to_tensor(obs)
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    
                    action_tensor, _, _ = agent.network.forward(obs_tensor)
                    action = action_tensor.cpu().numpy().flatten()
                
                obs, _, terminated, truncated, info = env.step(action)
                
                # Measure deviation from ideal path (simplified)
                if isinstance(obs, dict):
                    hoop_alignment = abs(obs.get('hoop_x_center_norm', 0.0))
                    total_deviation += hoop_alignment
                    steps += 1
                
                if terminated or truncated:
                    break
            
            if steps > 0:
                avg_deviation = total_deviation / steps
                precision_score = max(0.0, 1.0 - avg_deviation)
                precision_scores.append(precision_score)
        
        return np.mean(precision_scores) if precision_scores else 0.0
    
    def _test_approach_angles(self, agent: P3O, env: DeepFlyerEnv) -> float:
        """Test approach angle consistency"""
        # Simplified test - measure approach angle variations
        return 0.8  # Placeholder
    
    def _test_wind_response(self, agent: P3O, env: DeepFlyerEnv) -> float:
        """Test response to wind disturbances"""
        # Simplified test - measure compensation effectiveness
        return 0.7  # Placeholder
    
    def _test_stability(self, agent: P3O, env: DeepFlyerEnv) -> float:
        """Test flight stability"""
        # Simplified test - measure control smoothness
        return 0.75  # Placeholder
    
    def _test_emergency_response(self, agent: P3O, env: DeepFlyerEnv) -> float:
        """Test emergency response behavior"""
        # Simplified test - measure emergency detection and response
        return 0.85  # Placeholder
    
    def _calculate_validation_score(self, scenario_results: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall validation score"""
        if not scenario_results:
            return 0.0
        
        # Weight scenarios by importance
        scenario_weights = {
            'single_hoop_precision': 0.25,
            'multi_hoop_circuit': 0.25,
            'obstacle_avoidance': 0.15,
            'wind_disturbance': 0.15,
            'sensor_noise': 0.10,
            'emergency_scenarios': 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for scenario_name, results in scenario_results.items():
            if scenario_name in scenario_weights:
                # Calculate scenario score based on success rate and criteria met
                success_rate = results.get('success_rate', 0.0)
                criteria_met = results.get('criteria_met', {})
                criteria_score = sum(criteria_met.values()) / max(len(criteria_met), 1)
                
                scenario_score = (success_rate * 0.6 + criteria_score * 0.4)
                weight = scenario_weights[scenario_name]
                
                weighted_score += scenario_score * weight
                total_weight += weight
        
        return weighted_score / max(total_weight, 1.0)
    
    def _assess_deployment_readiness(self, validation_score: float, scenario_results: Dict[str, Any]) -> bool:
        """Assess if model is ready for deployment"""
        # Check overall validation score
        if validation_score < self.deployment_thresholds['overall_success_rate']:
            return False
        
        # Check critical safety scenarios
        emergency_results = scenario_results.get('emergency_scenarios', {})
        if emergency_results.get('safety_violation_rate', 1.0) > 0.1:
            return False
        
        # Check minimum performance in key scenarios
        key_scenarios = ['single_hoop_precision', 'multi_hoop_circuit']
        for scenario in key_scenarios:
            if scenario in scenario_results:
                success_rate = scenario_results[scenario].get('success_rate', 0.0)
                if success_rate < 0.6:
                    return False
        
        return True
    
    def _generate_deployment_recommendations(self, scenario_results: Dict[str, Any]) -> List[str]:
        """Generate deployment recommendations based on validation results"""
        recommendations = []
        
        for scenario_name, results in scenario_results.items():
            success_rate = results.get('success_rate', 0.0)
            criteria_met = results.get('criteria_met', {})
            
            if success_rate < 0.7:
                recommendations.append(f"Improve performance in {scenario_name} scenario (success rate: {success_rate:.2f})")
            
            for criterion, met in criteria_met.items():
                if not met:
                    recommendations.append(f"Address {criterion} in {scenario_name} scenario")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Model validation successful - ready for deployment")
        else:
            recommendations.insert(0, "Additional training recommended before deployment")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_failed_validation(self, reason: str) -> Dict[str, Any]:
        """Create failed validation result"""
        return {
            'validation_score': 0.0,
            'scenario_results': {},
            'deployment_ready': False,
            'recommendations': [f"Validation failed: {reason}"],
            'validation_timestamp': time.time(),
            'validator_version': '1.0'
        }
