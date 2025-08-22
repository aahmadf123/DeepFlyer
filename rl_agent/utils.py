#!/usr/bin/env python3
"""
Utility classes for DeepFlyer RL training
Includes ClearML integration and hyperparameter optimization
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("Warning: ClearML not available. Install with: pip install clearml")


class EnhancedClearMLTracker:
    """Enhanced ClearML experiment tracking with comprehensive logging for DeepFlyer"""
    
    def __init__(self, project_name: str, task_name: str, tags: Optional[List[str]] = None):
        self.available = CLEARML_AVAILABLE
        self.task = None
        self.logger = None
        
        if not CLEARML_AVAILABLE:
            print("ClearML not available - running without experiment tracking")
            return
            
        try:
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                tags=tags or []
            )
            self.logger = self.task.get_logger()
            print(f"ClearML tracking initialized: {project_name}/{task_name}")
        except Exception as e:
            print(f"Failed to initialize ClearML: {e}")
            self.available = False
    
    def log_hyperparameters(self, params: Dict[str, Any], section: str = "General"):
        """Log hyperparameters to ClearML with section organization"""
        if self.task:
            try:
                # Flatten nested dictionaries for better organization
                flat_params = self._flatten_dict(params, parent_key=section)
                self.task.connect(flat_params)
            except Exception as e:
                print(f"Failed to log hyperparameters: {e}")
    
    def log_training_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log training metrics with proper categorization"""
        if not self.logger:
            return
        
        # Categorize metrics for better visualization
        for name, value in metrics.items():
            if 'loss' in name.lower():
                title = "Training Losses"
            elif 'reward' in name.lower():
                title = "Rewards"
            elif 'accuracy' in name.lower() or 'success' in name.lower():
                title = "Performance"
            elif 'safety' in name.lower() or 'intervention' in name.lower():
                title = "Safety"
            elif 'episode' in name.lower():
                title = "Episodes"
            else:
                title = "Training Metrics"
            
            try:
                self.logger.report_scalar(title=title, series=name, value=value, iteration=iteration)
            except Exception as e:
                print(f"Failed to log metric {name}: {e}")
    
    def log_episode_summary(self, episode: int, summary: Dict[str, Any]):
        """Log comprehensive episode summary"""
        if not self.logger:
            return
        
        try:
            # Episode performance
            if 'reward' in summary:
                self.logger.report_scalar("Episode Performance", "Total Reward", 
                                        summary['reward'], iteration=episode)
            
            if 'success' in summary:
                self.logger.report_scalar("Episode Performance", "Success Rate", 
                                        float(summary['success']), iteration=episode)
            
            if 'duration' in summary:
                self.logger.report_scalar("Episode Statistics", "Duration (s)", 
                                        summary['duration'], iteration=episode)
            
            if 'steps' in summary:
                self.logger.report_scalar("Episode Statistics", "Steps", 
                                        summary['steps'], iteration=episode)
            
            # Safety metrics
            if 'safety_interventions' in summary:
                self.logger.report_scalar("Safety", "Interventions", 
                                        summary['safety_interventions'], iteration=episode)
            
            # Flight performance
            if 'hoops_passed' in summary:
                self.logger.report_scalar("Flight Performance", "Hoops Passed", 
                                        summary['hoops_passed'], iteration=episode)
            
            if 'collision_occurred' in summary:
                self.logger.report_scalar("Flight Performance", "Collision Rate", 
                                        float(summary['collision_occurred']), iteration=episode)
                
        except Exception as e:
            print(f"Failed to log episode summary: {e}")
    
    def log_model_checkpoint(self, checkpoint_path: str, episode: int, metrics: Dict[str, float]):
        """Log model checkpoint information"""
        if not self.task:
            return
            
        try:
            # Upload model file
            self.task.upload_artifact(name=f"checkpoint_episode_{episode}", 
                                    artifact_object=checkpoint_path)
            
            # Log checkpoint metrics
            for name, value in metrics.items():
                self.logger.report_scalar("Checkpoints", f"Checkpoint_{name}", 
                                        value, iteration=episode)
                
        except Exception as e:
            print(f"Failed to log checkpoint: {e}")
    
    def log_hyperparameter_optimization(self, trial: int, config: Dict[str, Any], 
                                       performance: float, completed: bool = True):
        """Log hyperparameter optimization trial"""
        if not self.logger:
            return
            
        try:
            # Log trial performance
            self.logger.report_scalar("Hyperparameter Optimization", "Trial Performance", 
                                    performance, iteration=trial)
            
            # Log trial configuration as table
            config_table = []
            for key, value in config.items():
                config_table.append([key, str(value)])
            
            self.logger.report_table(
                title="Hyperparameter Trials",
                series=f"Trial_{trial}",
                iteration=trial,
                table_plot=[["Parameter", "Value"]] + config_table
            )
            
            # Mark trial status
            status = "Completed" if completed else "Failed"
            self.logger.report_text(f"Trial {trial}: {status} with performance {performance:.4f}")
            
        except Exception as e:
            print(f"Failed to log hyperparameter trial: {e}")
    
    def log_safety_analysis(self, episode: int, safety_data: Dict[str, Any]):
        """Log comprehensive safety analysis"""
        if not self.logger:
            return
            
        try:
            # Safety violations
            if 'violations' in safety_data:
                violation_count = len(safety_data['violations'])
                self.logger.report_scalar("Safety Analysis", "Violation Count", 
                                        violation_count, iteration=episode)
            
            # Emergency stops
            if 'emergency_stops' in safety_data:
                self.logger.report_scalar("Safety Analysis", "Emergency Stops", 
                                        safety_data['emergency_stops'], iteration=episode)
            
            # Geofence violations
            if 'geofence_violations' in safety_data:
                self.logger.report_scalar("Safety Analysis", "Geofence Violations", 
                                        safety_data['geofence_violations'], iteration=episode)
            
            # Collision risks
            if 'collision_risks' in safety_data:
                self.logger.report_scalar("Safety Analysis", "Collision Risks", 
                                        safety_data['collision_risks'], iteration=episode)
                
        except Exception as e:
            print(f"Failed to log safety analysis: {e}")
    
    def log_domain_randomization(self, episode: int, randomization_info: Dict[str, Any]):
        """Log domain randomization information"""
        if not self.logger:
            return
            
        try:
            # Randomization level
            if 'level' in randomization_info:
                level_map = {'minimal': 1, 'moderate': 2, 'aggressive': 3, 'student': 4}
                level_num = level_map.get(randomization_info['level'], 2)
                self.logger.report_scalar("Domain Randomization", "Level", 
                                        level_num, iteration=episode)
            
            # Visual randomization intensity
            if 'visual_intensity' in randomization_info:
                self.logger.report_scalar("Domain Randomization", "Visual Intensity", 
                                        randomization_info['visual_intensity'], iteration=episode)
            
            # Physics randomization intensity
            if 'physics_intensity' in randomization_info:
                self.logger.report_scalar("Domain Randomization", "Physics Intensity", 
                                        randomization_info['physics_intensity'], iteration=episode)
                
        except Exception as e:
            print(f"Failed to log domain randomization: {e}")
    
    def log_vision_analysis(self, episode: int, vision_data: Dict[str, Any]):
        """Log vision system analysis"""
        if not self.logger:
            return
            
        try:
            # Detection accuracy
            if 'detection_accuracy' in vision_data:
                self.logger.report_scalar("Vision Analysis", "Detection Accuracy", 
                                        vision_data['detection_accuracy'], iteration=episode)
            
            # Depth estimation quality
            if 'depth_quality' in vision_data:
                self.logger.report_scalar("Vision Analysis", "Depth Quality", 
                                        vision_data['depth_quality'], iteration=episode)
            
            # Hoop visibility duration
            if 'visibility_duration' in vision_data:
                self.logger.report_scalar("Vision Analysis", "Hoop Visibility (s)", 
                                        vision_data['visibility_duration'], iteration=episode)
                
        except Exception as e:
            print(f"Failed to log vision analysis: {e}")
    
    def log_comparison_plot(self, title: str, series_data: Dict[str, List[Tuple[int, float]]], 
                           iteration: int):
        """Log comparison plot for multiple series"""
        if not self.logger:
            return
            
        try:
            for series_name, data_points in series_data.items():
                for step, value in data_points:
                    self.logger.report_scalar(title, series_name, value, iteration=step)
        except Exception as e:
            print(f"Failed to log comparison plot: {e}")
    
    def log_text_summary(self, title: str, text: str, iteration: int):
        """Log text summary or report"""
        if not self.logger:
            return
            
        try:
            self.logger.report_text(text, title, iteration)
        except Exception as e:
            print(f"Failed to log text summary: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '/') -> Dict[str, Any]:
        """Flatten nested dictionary for ClearML"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def finalize_experiment(self, final_metrics: Dict[str, float]):
        """Finalize experiment with summary metrics"""
        if not self.task:
            return
            
        try:
            # Set final results
            for metric_name, value in final_metrics.items():
                self.task.set_parameter(f"Final/{metric_name}", value)
            
            # Mark task as completed
            print("Experiment tracking finalized")
        except Exception as e:
            print(f"Failed to finalize experiment: {e}")


# Legacy class for backward compatibility
class ClearMLTracker(EnhancedClearMLTracker):
    """Legacy ClearML tracker - use EnhancedClearMLTracker for new implementations"""
    
    def __init__(self, project_name: str, task_name: str, tags: Optional[List[str]] = None):
        super().__init__(project_name, task_name, tags)
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Legacy method - use log_training_metrics"""
        self.log_training_metrics(metrics, iteration)
    
    def log_performance(self, performance: float, iteration: int):
        """Legacy method"""
        if self.logger:
            self.logger.report_scalar("Performance", "Score", performance, iteration)


class HyperparameterOptimizer:
    """Random search hyperparameter optimizer for P3O"""
    
    def __init__(self, base_config, clearml_tracker: Optional[ClearMLTracker] = None):
        self.base_config = base_config
        self.clearml_tracker = clearml_tracker
        
        # Define hyperparameter search spaces
        self.search_spaces = {
            'learning_rate': (1e-5, 1e-3, 'log'),  # Log scale
            'clip_ratio': (0.1, 0.3, 'linear'),
            'procrastination_factor': (0.8, 0.99, 'linear'), 
            'batch_size': ([32, 64, 128, 256], 'choice'),
            'num_epochs': ([5, 10, 15, 20], 'choice'),
            'gamma': (0.95, 0.999, 'linear'),
            'gae_lambda': (0.9, 0.98, 'linear'),
            'value_loss_coef': (0.1, 1.0, 'linear'),
            'entropy_coef': (0.001, 0.1, 'log'),
            'action_noise': (0.01, 0.2, 'linear')
        }
        
        # Track best configuration
        self.best_config = None
        self.best_performance = -float('inf')
        self.current_trial = 0
        self.all_configs = []
        
    def suggest_config(self):
        """Generate random hyperparameter configuration"""
        from rl_agent.algorithms.p3o import P3OConfig
        
        config = P3OConfig()
        suggested_params = {}
        
        for param_name, search_space in self.search_spaces.items():
            if len(search_space) == 3 and search_space[2] == 'log':
                # Log scale sampling
                low, high = search_space[0], search_space[1]
                value = np.exp(np.random.uniform(np.log(low), np.log(high)))
            elif len(search_space) == 3 and search_space[2] == 'linear':
                # Linear scale sampling
                low, high = search_space[0], search_space[1]
                value = np.random.uniform(low, high)
            elif len(search_space) == 2 and search_space[1] == 'choice':
                # Choice from list
                choices = search_space[0]
                value = np.random.choice(choices)
            else:
                continue
                
            # Set parameter on config
            setattr(config, param_name, value)
            suggested_params[param_name] = value
        
        # Log to ClearML if available
        if self.clearml_tracker:
            self.clearml_tracker.log_hyperparameters(suggested_params)
        
        return config
    
    def report_performance(self, config, performance: float, additional_metrics: Optional[Dict] = None):
        """Report performance for a given configuration"""
        self.all_configs.append({
            'config': config,
            'performance': performance,
            'trial': self.current_trial,
            'metrics': additional_metrics or {}
        })
        
        # Update best if improved
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_config = config
            print(f"New best configuration found! Performance: {performance:.4f}")
        
        # Log to ClearML
        if self.clearml_tracker:
            self.clearml_tracker.log_performance(performance, self.current_trial)
    
    def get_best_config(self):
        """Get the best configuration found so far"""
        return self.best_config


class PerformanceTracker:
    """Track training performance metrics"""
    
    def __init__(self, save_dir: str = "experiments/performance"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.safety_interventions = []
        self.phase_transitions = []
        
    def log_episode(self, reward: float, length: int, loss: float = None, 
                   safety_interventions: int = 0, phases: List[str] = None):
        """Log episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if loss is not None:
            self.training_losses.append(loss)
        self.safety_interventions.append(safety_interventions)
        if phases:
            self.phase_transitions.append(phases)
    
    def get_recent_performance(self, window: int = 10) -> float:
        """Get average performance over recent episodes"""
        if len(self.episode_rewards) < window:
            return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        return np.mean(self.episode_rewards[-window:])
    
    def save_metrics(self, filename: str = None):
        """Save metrics to file"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"performance_{timestamp}.json"
        
        filepath = self.save_dir / filename
        
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'safety_interventions': self.safety_interventions,
            'phase_transitions': self.phase_transitions,
            'summary': {
                'total_episodes': len(self.episode_rewards),
                'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
                'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
                'total_safety_interventions': sum(self.safety_interventions)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Performance metrics saved to {filepath}")
        return filepath