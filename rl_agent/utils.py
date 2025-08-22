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


class ClearMLTracker:
    """ClearML experiment tracking integration"""
    
    def __init__(self, project_name: str, task_name: str, tags: Optional[List[str]] = None):
        if not CLEARML_AVAILABLE:
            self.task = None
            print("ClearML not available - running without experiment tracking")
            return
            
        self.task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags or []
        )
        self.logger = self.task.get_logger()
        
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to ClearML"""
        if self.task:
            self.task.connect(params)
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log training metrics"""
        if self.logger:
            for name, value in metrics.items():
                self.logger.report_scalar(title="Training", series=name, value=value, iteration=iteration)
    
    def log_performance(self, performance: float, iteration: int):
        """Log overall performance metric"""
        if self.logger:
            self.logger.report_scalar(title="Performance", series="Score", value=performance, iteration=iteration)


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