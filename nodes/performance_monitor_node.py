#!/usr/bin/env python3
"""
Performance Monitor Node for DeepFlyer

Comprehensive training performance monitoring and analysis
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from typing import Dict, List, Any, Optional
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# Custom messages
try:
    from deepflyer.msg import RewardFeedback, CourseState, VisionFeatures
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Container for episode performance metrics"""
    episode_number: int
    total_reward: float
    episode_length: int
    success: bool
    hoops_completed: int
    collision_count: int
    out_of_bounds_count: int
    average_speed: float
    navigation_efficiency: float
    timestamp: float


class PerformanceMonitor:
    """Comprehensive training performance monitoring"""
    
    def __init__(self):
        self.metrics_history: List[EpisodeMetrics] = []
        self.reward_components_history: List[Dict[str, float]] = []
        self.performance_thresholds = {
            'hoop_success_rate': 0.8,
            'average_episode_reward': 100.0,
            'navigation_accuracy': 0.9,
            'safety_violations': 0.05,
            'learning_stability': 0.1  # Max std dev in recent performance
        }
        
        # Learning trend analysis
        self.learning_window_size = 50
        self.stability_window_size = 20
        
        # Performance tracking
        self.start_time = time.time()
        self.last_analysis_time = time.time()
        
    def add_episode_metrics(self, metrics: EpisodeMetrics):
        """Add episode metrics to history"""
        self.metrics_history.append(metrics)
        
        # Keep only recent history to prevent memory issues
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-800:]
            
    def add_reward_breakdown(self, components: Dict[str, float]):
        """Add reward component breakdown"""
        self.reward_components_history.append(components)
        
        # Keep only recent history
        if len(self.reward_components_history) > 1000:
            self.reward_components_history = self.reward_components_history[-800:]
    
    def evaluate_training_progress(self, recent_episodes: int = 50) -> Dict[str, Any]:
        """Evaluate if training is progressing successfully"""
        if len(self.metrics_history) < 10:
            return self._create_insufficient_data_response()
            
        recent_metrics = self.metrics_history[-recent_episodes:]
        
        # Calculate key performance indicators
        metrics = {
            'success_rate': self._calculate_success_rate(recent_metrics),
            'reward_trend': self._calculate_reward_trend(recent_metrics),
            'learning_stability': self._calculate_stability(recent_metrics),
            'safety_performance': self._calculate_safety_metrics(recent_metrics),
            'efficiency_metrics': self._calculate_efficiency_metrics(recent_metrics)
        }
        
        # Estimate training completion
        completion_estimate = self._estimate_training_completion(recent_metrics)
        metrics['estimated_completion'] = completion_estimate
        
        # Generate recommendations
        recommendations = self._generate_training_recommendations(metrics)
        
        # Determine if training should continue
        should_continue = self._should_continue_training(metrics)
        
        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'should_continue_training': should_continue,
            'training_quality': self._assess_training_quality(metrics),
            'next_actions': self._suggest_next_actions(metrics)
        }
    
    def _calculate_success_rate(self, episodes: List[EpisodeMetrics]) -> Dict[str, float]:
        """Calculate various success rate metrics"""
        if not episodes:
            return {'overall': 0.0, 'recent_trend': 0.0}
            
        # Overall success rate
        successes = sum(1 for ep in episodes if ep.success)
        overall_rate = successes / len(episodes)
        
        # Recent trend (last 20 vs previous 20)
        if len(episodes) >= 40:
            recent_20 = episodes[-20:]
            previous_20 = episodes[-40:-20]
            recent_success = sum(1 for ep in recent_20 if ep.success) / 20
            previous_success = sum(1 for ep in previous_20 if ep.success) / 20
            trend = recent_success - previous_success
        else:
            trend = 0.0
            
        return {
            'overall': overall_rate,
            'recent_trend': trend,
            'threshold_met': overall_rate >= self.performance_thresholds['hoop_success_rate']
        }
    
    def _calculate_reward_trend(self, episodes: List[EpisodeMetrics]) -> Dict[str, float]:
        """Analyze reward trends and learning progress"""
        if len(episodes) < 10:
            return {'trend': 0.0, 'stability': 0.0}
            
        rewards = [ep.total_reward for ep in episodes]
        
        # Calculate trend using linear regression
        x = np.arange(len(rewards))
        trend_coeff = np.polyfit(x, rewards, 1)[0]
        
        # Calculate stability (inverse of standard deviation)
        recent_rewards = rewards[-self.stability_window_size:]
        stability = 1.0 / (1.0 + np.std(recent_rewards))
        
        # Average reward
        avg_reward = np.mean(rewards)
        
        return {
            'trend': float(trend_coeff),
            'stability': float(stability),
            'average_reward': float(avg_reward),
            'improving': trend_coeff > 0,
            'stable_learning': np.std(recent_rewards) < 20.0
        }
    
    def _calculate_stability(self, episodes: List[EpisodeMetrics]) -> Dict[str, float]:
        """Calculate learning stability metrics"""
        if len(episodes) < self.stability_window_size:
            return {'reward_stability': 0.0, 'performance_consistency': 0.0}
            
        recent_episodes = episodes[-self.stability_window_size:]
        
        # Reward stability
        rewards = [ep.total_reward for ep in recent_episodes]
        reward_std = np.std(rewards)
        reward_mean = np.mean(rewards)
        reward_cv = reward_std / max(abs(reward_mean), 1.0)  # Coefficient of variation
        
        # Performance consistency
        success_rate_windows = []
        window_size = 5
        for i in range(len(recent_episodes) - window_size + 1):
            window = recent_episodes[i:i+window_size]
            success_rate = sum(1 for ep in window if ep.success) / window_size
            success_rate_windows.append(success_rate)
            
        performance_consistency = 1.0 - np.std(success_rate_windows) if success_rate_windows else 0.0
        
        return {
            'reward_stability': max(0.0, 1.0 - reward_cv),
            'performance_consistency': float(performance_consistency),
            'is_stable': reward_cv < self.performance_thresholds['learning_stability']
        }
    
    def _calculate_safety_metrics(self, episodes: List[EpisodeMetrics]) -> Dict[str, float]:
        """Calculate safety-related performance metrics"""
        if not episodes:
            return {'violation_rate': 1.0, 'safety_score': 0.0}
            
        total_episodes = len(episodes)
        collision_episodes = sum(1 for ep in episodes if ep.collision_count > 0)
        oob_episodes = sum(1 for ep in episodes if ep.out_of_bounds_count > 0)
        
        violation_rate = (collision_episodes + oob_episodes) / total_episodes
        safety_score = 1.0 - violation_rate
        
        return {
            'violation_rate': violation_rate,
            'safety_score': safety_score,
            'collision_rate': collision_episodes / total_episodes,
            'out_of_bounds_rate': oob_episodes / total_episodes,
            'safety_threshold_met': violation_rate <= self.performance_thresholds['safety_violations']
        }
    
    def _calculate_efficiency_metrics(self, episodes: List[EpisodeMetrics]) -> Dict[str, float]:
        """Calculate navigation efficiency metrics"""
        if not episodes:
            return {'navigation_efficiency': 0.0, 'speed_consistency': 0.0}
            
        # Navigation efficiency
        efficiencies = [ep.navigation_efficiency for ep in episodes if ep.navigation_efficiency > 0]
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        
        # Speed consistency
        speeds = [ep.average_speed for ep in episodes if ep.average_speed > 0]
        speed_consistency = 1.0 - (np.std(speeds) / max(np.mean(speeds), 0.1)) if speeds else 0.0
        
        return {
            'navigation_efficiency': float(avg_efficiency),
            'speed_consistency': max(0.0, float(speed_consistency)),
            'efficiency_threshold_met': avg_efficiency >= self.performance_thresholds['navigation_accuracy']
        }
    
    def _estimate_training_completion(self, episodes: List[EpisodeMetrics]) -> Dict[str, Any]:
        """Estimate training completion and remaining time"""
        if len(episodes) < 20:
            return {'completion_percentage': 0.0, 'estimated_episodes_remaining': float('inf')}
            
        # Simple heuristic based on success rate trend
        recent_success_rate = sum(1 for ep in episodes[-20:] if ep.success) / 20
        
        # Estimate completion based on success rate approaching threshold
        target_success_rate = self.performance_thresholds['hoop_success_rate']
        
        if recent_success_rate >= target_success_rate:
            completion_percentage = 1.0
            estimated_remaining = 0
        else:
            # Estimate based on current improvement rate
            improvement_needed = target_success_rate - recent_success_rate
            
            # Calculate recent improvement rate
            if len(episodes) >= 40:
                prev_success_rate = sum(1 for ep in episodes[-40:-20] if ep.success) / 20
                improvement_rate = max(0.001, recent_success_rate - prev_success_rate)
                estimated_remaining = int(improvement_needed / improvement_rate * 20)
            else:
                estimated_remaining = 100
                
            completion_percentage = recent_success_rate / target_success_rate
        
        return {
            'completion_percentage': min(1.0, completion_percentage),
            'estimated_episodes_remaining': estimated_remaining,
            'current_success_rate': recent_success_rate,
            'target_success_rate': target_success_rate
        }
    
    def _generate_training_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable training recommendations"""
        recommendations = []
        
        # Success rate recommendations
        success_metrics = metrics['success_rate']
        if not success_metrics['threshold_met']:
            if success_metrics['recent_trend'] < 0:
                recommendations.append("Success rate is declining. Consider adjusting reward function or reducing difficulty.")
            else:
                recommendations.append("Continue training - success rate is improving but below threshold.")
        
        # Reward trend recommendations
        reward_metrics = metrics['reward_trend']
        if not reward_metrics['improving']:
            recommendations.append("Reward trend is flat or declining. Consider hyperparameter tuning.")
        if not reward_metrics['stable_learning']:
            recommendations.append("Learning is unstable. Try reducing learning rate or increasing batch size.")
        
        # Safety recommendations
        safety_metrics = metrics['safety_performance']
        if not safety_metrics['safety_threshold_met']:
            recommendations.append("High safety violation rate. Review safety constraints and penalty weights.")
        
        # Stability recommendations
        stability_metrics = metrics['learning_stability']
        if not stability_metrics['is_stable']:
            recommendations.append("Training is unstable. Consider curriculum learning or experience replay tuning.")
        
        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("Training progressing well. Continue current configuration.")
        
        return recommendations
    
    def _should_continue_training(self, metrics: Dict[str, Any]) -> bool:
        """Determine if training should continue"""
        # Continue if making progress
        if metrics['success_rate']['recent_trend'] > 0:
            return True
            
        # Continue if not yet meeting thresholds
        if not metrics['success_rate']['threshold_met']:
            return True
            
        # Stop if performance has plateaued
        completion = metrics['estimated_completion']
        if completion['completion_percentage'] >= 0.95:
            return False
            
        return True
    
    def _assess_training_quality(self, metrics: Dict[str, Any]) -> str:
        """Assess overall training quality"""
        scores = []
        
        # Success rate score
        if metrics['success_rate']['threshold_met']:
            scores.append(1.0)
        else:
            scores.append(metrics['success_rate']['overall'])
        
        # Learning stability score
        scores.append(metrics['learning_stability']['reward_stability'])
        
        # Safety score
        scores.append(metrics['safety_performance']['safety_score'])
        
        # Overall score
        overall_score = np.mean(scores)
        
        if overall_score >= 0.8:
            return "Excellent"
        elif overall_score >= 0.6:
            return "Good"
        elif overall_score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _suggest_next_actions(self, metrics: Dict[str, Any]) -> List[str]:
        """Suggest specific next actions"""
        actions = []
        
        completion = metrics['estimated_completion']
        
        if completion['completion_percentage'] >= 0.9:
            actions.append("Prepare for model validation and deployment")
            actions.append("Run final performance tests")
        elif completion['completion_percentage'] >= 0.7:
            actions.append("Consider transitioning to advanced difficulty")
            actions.append("Focus on optimization and fine-tuning")
        else:
            actions.append("Continue training with current curriculum")
            actions.append("Monitor performance trends closely")
        
        return actions
    
    def _create_insufficient_data_response(self) -> Dict[str, Any]:
        """Create response when insufficient data available"""
        return {
            'metrics': {
                'success_rate': {'overall': 0.0, 'threshold_met': False},
                'reward_trend': {'trend': 0.0, 'improving': False},
                'learning_stability': {'is_stable': False},
                'estimated_completion': {'completion_percentage': 0.0}
            },
            'recommendations': ["Continue training to gather sufficient data for analysis"],
            'should_continue_training': True,
            'training_quality': "Insufficient Data",
            'next_actions': ["Complete at least 20 episodes for meaningful analysis"]
        }


class PerformanceMonitorNode(Node):
    """ROS2 node for performance monitoring"""
    
    def __init__(self):
        super().__init__('performance_monitor_node')
        
        # Initialize performance monitor
        self.monitor = PerformanceMonitor()
        
        # Parameters
        self.declare_parameter('analysis_frequency', 30.0)  # seconds
        self.declare_parameter('save_metrics', True)
        self.declare_parameter('metrics_file', 'performance_metrics.json')
        
        self.analysis_frequency = self.get_parameter('analysis_frequency').value
        self.save_metrics = self.get_parameter('save_metrics').value
        self.metrics_file = self.get_parameter('metrics_file').value
        
        # Initialize ROS interfaces
        self._init_subscribers()
        self._init_timers()
        
        # State tracking
        self.current_episode_data = {}
        self.last_analysis = time.time()
        
        self.get_logger().info("Performance Monitor Node initialized")
    
    def _init_subscribers(self):
        """Initialize ROS subscribers"""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        if CUSTOM_MSGS_AVAILABLE:
            self.reward_sub = self.create_subscription(
                RewardFeedback,
                '/deepflyer/reward_feedback',
                self.reward_callback,
                qos_profile
            )
            
            self.course_sub = self.create_subscription(
                CourseState,
                '/deepflyer/course_state',
                self.course_state_callback,
                qos_profile
            )
    
    def _init_timers(self):
        """Initialize analysis timer"""
        self.analysis_timer = self.create_timer(
            self.analysis_frequency,
            self.run_performance_analysis
        )
    
    def reward_callback(self, msg):
        """Handle reward feedback messages"""
        # Extract reward components
        reward_components = {
            'total_reward': getattr(msg, 'total_reward', 0.0),
            'hoop_approach': getattr(msg, 'hoop_approach', 0.0),
            'hoop_passage': getattr(msg, 'hoop_passage', 0.0),
            'collision_penalty': getattr(msg, 'collision_penalty', 0.0)
        }
        
        self.monitor.add_reward_breakdown(reward_components)
    
    def course_state_callback(self, msg):
        """Handle course state messages"""
        # Update episode data
        self.current_episode_data.update({
            'episode_number': getattr(msg, 'episode_number', 0),
            'hoops_completed': getattr(msg, 'hoops_completed', 0),
            'total_reward': getattr(msg, 'total_reward', 0.0),
            'success': getattr(msg, 'episode_complete', False)
        })
    
    def run_performance_analysis(self):
        """Run periodic performance analysis"""
        current_time = time.time()
        
        if current_time - self.last_analysis >= self.analysis_frequency:
            analysis_results = self.monitor.evaluate_training_progress()
            
            # Log key findings
            quality = analysis_results['training_quality']
            should_continue = analysis_results['should_continue_training']
            
            self.get_logger().info(f"Training Quality: {quality}, Continue: {should_continue}")
            
            # Log recommendations
            for rec in analysis_results['recommendations'][:3]:  # Top 3 recommendations
                self.get_logger().info(f"Recommendation: {rec}")
            
            self.last_analysis = current_time
            
            # Save metrics if enabled
            if self.save_metrics:
                self._save_analysis_results(analysis_results)
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """Save analysis results to file"""
        try:
            timestamp = datetime.now().isoformat()
            output_data = {
                'timestamp': timestamp,
                'analysis_results': results
            }
            
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(output_data) + '\n')
                
        except Exception as e:
            self.get_logger().error(f"Failed to save metrics: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

