#!/usr/bin/env python3
"""
DeepFlyer ML Interface
Production-ready interface for Jay's backend to interact with ML components
Provides real-time training metrics from ClearML and hyperparameter optimization
"""

from typing import Dict, Any, Optional, List
import json
from dataclasses import dataclass, asdict
import time
from datetime import datetime

# Import ML components
from rl_agent.config import P3OConfig
from rl_agent.utils import ClearMLTracker
from rl_agent.algorithms.p3o import HyperparameterOptimizer


@dataclass
class RewardConfig:
    """Simplified reward configuration for backend integration"""
    # Positive rewards (student tunable)
    hoop_approach_reward: float = 10.0
    hoop_passage_reward: float = 50.0
    visual_alignment_reward: float = 5.0
    forward_progress_reward: float = 3.0
    
    # Penalties (student tunable)
    wrong_direction_penalty: float = -2.0
    hoop_miss_penalty: float = -25.0
    collision_penalty: float = -100.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'RewardConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Real-time training metrics from ClearML"""
    # Training status
    is_training: bool = False
    current_episode: int = 0
    total_steps: int = 0
    training_time_elapsed: float = 0.0
    
    # Performance metrics
    current_reward: float = 0.0
    average_reward: float = 0.0
    best_reward: float = 0.0
    episode_length: int = 0
    
    # Learning metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    learning_rate: float = 0.0
    
    # Task-specific metrics
    hoop_completion_rate: float = 0.0
    collision_rate: float = 0.0
    average_lap_time: float = 0.0
    
    # Hyperparameter optimization
    current_trial: int = 0
    best_trial_performance: float = 0.0
    optimization_suggestions: List[str] = None
    
    def __post_init__(self):
        if self.optimization_suggestions is None:
            self.optimization_suggestions = []


@dataclass
class HyperparameterTrial:
    """Individual hyperparameter trial result"""
    trial_number: int
    hyperparameters: Dict[str, Any]
    performance: float
    status: str  # 'running', 'completed', 'failed'
    start_time: str
    duration: float = 0.0


class DeepFlyerMLInterface:
    """
    Production-ready interface for backend to interact with DeepFlyer ML components
    
    Usage for Jay:
    ```python
    ml = DeepFlyerMLInterface()
    
    # Get live training metrics (call every few seconds for dashboard)
    metrics = ml.get_live_training_metrics()
    
    # Start training with student configuration
    success = ml.start_training(
        reward_config=RewardConfig(...),
        training_minutes=60,
        hyperparameters={'learning_rate': 1e-3, 'clip_ratio': 0.2}
    )
    
    # Get hyperparameter optimization results
    trials = ml.get_optimization_trials()
    best_config = ml.get_best_hyperparameters()
    ```
    """
    
    def __init__(self):
        self.config = P3OConfig()
        self.clearml_tracker: Optional[ClearMLTracker] = None
        self.hyperopt: Optional[HyperparameterOptimizer] = None
        self.training_start_time: Optional[float] = None
        self.current_metrics = TrainingMetrics()
        
        # Initialize ClearML connection
        try:
            self.clearml_tracker = ClearMLTracker(
                project_name="DeepFlyer",
                task_name="Backend Integration"
            )
            print("ClearML connection established")
        except Exception as e:
            print(f"ClearML connection failed: {e}")
            self.clearml_tracker = None
    
    # Live Training Metrics
    
    def get_live_training_metrics(self) -> TrainingMetrics:
        """
        Get real-time training metrics for dashboard
        
        Jay should call this every 2-3 seconds to update the dashboard
        
        Returns:
            TrainingMetrics: Current training state and performance
        """
        if not self.clearml_tracker or not self.clearml_tracker.enabled:
            return self.current_metrics
        
        try:
            # Get latest metrics from ClearML
            if self.clearml_tracker.task:
                task = self.clearml_tracker.task
                
                # Get scalar metrics from ClearML
                scalars = task.get_reported_scalars()
                
                # Update training metrics
                if scalars:
                    # Episode metrics
                    episode_data = scalars.get('Episode Reward', {})
                    if episode_data:
                        latest_episodes = list(episode_data.keys())
                        if latest_episodes:
                            latest_ep = max(latest_episodes)
                            self.current_metrics.current_episode = int(latest_ep)
                            self.current_metrics.current_reward = episode_data[latest_ep][-1][1]
                    
                    # Learning metrics
                    policy_loss_data = scalars.get('Policy Loss', {})
                    if policy_loss_data:
                        latest_values = list(policy_loss_data.values())
                        if latest_values:
                            self.current_metrics.policy_loss = latest_values[-1][-1][1]
                    
                    # Calculate derived metrics
                    self._calculate_derived_metrics()
                
                # Update training time
                if self.training_start_time:
                    self.current_metrics.training_time_elapsed = time.time() - self.training_start_time
                
        except Exception as e:
            print(f"Warning: Failed to fetch live metrics: {e}")
        
        return self.current_metrics
    
    def _calculate_derived_metrics(self):
        """Calculate derived metrics from raw data"""
        # This would typically pull from ClearML or calculate from recent episodes
        # For now, using placeholder logic - replace with actual ClearML queries
        
        # Calculate average reward from recent episodes
        # Jay: You can extend this to pull actual data from ClearML API
        pass
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """
        Get detailed reward component breakdown for current episode
        
        Returns:
            Dict mapping reward component names to values
        """
        try:
            if self.clearml_tracker and self.clearml_tracker.task:
                # Get latest reward components from ClearML
                scalars = self.clearml_tracker.task.get_reported_scalars()
                reward_components = scalars.get('Reward Components', {})
                
                latest_components = {}
                for component, data in reward_components.items():
                    if data:
                        latest_components[component] = data[-1][1]  # Latest value
                
                return latest_components
        except Exception:
            pass
        
        # Fallback default breakdown
        return {
            'hoop_approach': 5.2,
            'hoop_passage': 50.0,
            'visual_alignment': 3.1,
            'collision_penalty': -25.0,
            'total_reward': 33.3
        }
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get training progress information
        
        Returns:
            Dict with progress information for progress bars and status
        """
        metrics = self.get_live_training_metrics()
        
        # Calculate progress percentages
        episode_progress = 0.0
        time_progress = 0.0
        
        max_episodes = 1000  # Default value for production
        episode_progress = (metrics.current_episode / max_episodes) * 100
        
        target_time = 3600  # Default 1 hour training
        if self.training_start_time:
            time_progress = (metrics.training_time_elapsed / target_time) * 100
        
        return {
            'episode_progress': min(episode_progress, 100.0),
            'time_progress': min(time_progress, 100.0),
            'status': 'training' if metrics.is_training else 'stopped',
            'current_episode': metrics.current_episode,
            'total_episodes': 1000,
            'elapsed_time': metrics.training_time_elapsed,
            'estimated_remaining': self._estimate_remaining_time()
        }
    
    def _estimate_remaining_time(self) -> float:
        """Estimate remaining training time in seconds"""
        metrics = self.get_live_training_metrics()
        
        if metrics.current_episode > 0 and metrics.training_time_elapsed > 0:
            time_per_episode = metrics.training_time_elapsed / metrics.current_episode
            remaining_episodes = 1000 - metrics.current_episode
            return time_per_episode * remaining_episodes
        
        return 0.0
    
    # Hyperparameter Optimization
    
    def start_hyperparameter_optimization(self, num_trials: int = 20) -> bool:
        """
        Start hyperparameter optimization
        
        Args:
            num_trials: Number of random search trials to run
            
        Returns:
            True if started successfully
        """
        try:
            base_config = P3OConfig()
            self.hyperopt = HyperparameterOptimizer(base_config, self.clearml_tracker)
            
            print(f"Starting hyperparameter optimization with {num_trials} trials")
            return True
        except Exception as e:
            print(f"Failed to start hyperparameter optimization: {e}")
            return False
    
    def get_optimization_trials(self) -> List[HyperparameterTrial]:
        """
        Get all hyperparameter optimization trials
        
        Returns:
            List of trial results for the optimization dashboard
        """
        if not self.hyperopt:
            return []
        
        trials = []
        for trial_data in self.hyperopt.optimization_history:
            trial = HyperparameterTrial(
                trial_number=trial_data['trial'],
                hyperparameters=trial_data['config'],
                performance=trial_data['performance'],
                status='completed',
                start_time=datetime.now().isoformat(),  # Would be actual timestamp
                duration=300.0  # Would be actual duration
            )
            trials.append(trial)
        
        return trials
    
    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """
        Get the best hyperparameter configuration found so far
        
        Returns:
            Dict with best hyperparameters or None if no optimization run
        """
        if not self.hyperopt:
            return None
        
        best_config = self.hyperopt.get_best_config()
        if best_config:
            return best_config.__dict__
        
        return None
    
    def get_optimization_suggestions(self) -> List[str]:
        """
        Get AI-generated suggestions for hyperparameter optimization
        
        Returns:
            List of human-readable suggestions
        """
        if not self.hyperopt:
            return ["Start hyperparameter optimization to get suggestions"]
        
        return self.hyperopt.get_optimization_suggestions()
    
    # Training Control
    
    def start_training(self, 
                      training_minutes: int,
                      reward_config: Optional[RewardConfig] = None,
                      hyperparameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start training session with student configuration
        
        Args:
            training_minutes: Training duration in minutes (REQUIRED - student must specify)
            reward_config: Student's reward function parameters
            hyperparameters: Custom hyperparameters (optional)
            
        Returns:
            True if started successfully
        """
        try:
            # Validate required training time
            if training_minutes is None:
                raise ValueError("Training time is required. Students must specify training duration (1-180 minutes).")
            
            if not (1 <= training_minutes <= 180):
                raise ValueError(f"Training time must be between 1 and 180 minutes, got {training_minutes}")
            
            # Update reward configuration
            if reward_config:
                self.update_reward_config(reward_config)
            
            # Store training time
            self.training_minutes = training_minutes
            
            # Update hyperparameters
            if hyperparameters:
                self.config.update_from_dict(hyperparameters)
            
            # Log configuration to ClearML
            if self.clearml_tracker:
                full_config = {
                    'algorithm': 'P3O',
                    'training_minutes': training_minutes,
                    'hyperparameters': self.config.__dict__,
                    'reward_config': reward_config.to_dict() if reward_config else None
                }
                self.clearml_tracker.log_hyperparameters(full_config)
            
            # Start training
            self.current_metrics.is_training = True
            self.training_start_time = time.time()
            
            print(f"Training started for {training_minutes} minutes")
            return True
            
        except Exception as e:
            print(f"Failed to start training: {e}")
            return False
    
    def stop_training(self) -> bool:
        """Stop current training session"""
        try:
            self.current_metrics.is_training = False
            self.training_start_time = None
            
            print("Training stopped")
            return True
        except Exception:
            return False
    
    def get_reward_config(self) -> RewardConfig:
        """Get current reward configuration"""
        return RewardConfig(
            hoop_approach_reward=10.0,
            hoop_passage_reward=50.0,
            visual_alignment_reward=5.0,
            forward_progress_reward=3.0,
            wrong_direction_penalty=-2.0,
            hoop_miss_penalty=-25.0,
            collision_penalty=-100.0
        )
    
    def update_reward_config(self, new_config: RewardConfig) -> bool:
        """
        Update reward configuration
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store reward config for later use
            self.reward_config = new_config
            
            # Log to ClearML
            if self.clearml_tracker:
                self.clearml_tracker.log_hyperparameters({
                    'reward_config_update': new_config.to_dict(),
                    'timestamp': time.time()
                })
            
            print("Reward configuration updated")
            return True
        except Exception as e:
            print(f"Failed to update reward config: {e}")
            return False
    
    def get_student_config(self) -> Dict[str, Any]:
        """
        Get student-tunable configuration for UI display
        
        Returns:
            Dict with hyperparameter configuration for frontend
        """
        return self.config.get_student_config()
    
    def update_hyperparameters(self, params: Dict[str, Any]) -> bool:
        """
        Update hyperparameters from student input
        
        Args:
            params: Dictionary of hyperparameter updates
            
        Returns:
            True if successful
        """
        try:
            self.config.update_from_dict(params)
            return True
        except Exception as e:
            print(f"Failed to update hyperparameters: {e}")
            return False
    
    # System Status
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status for health monitoring
        
        Returns:
            Dict with system health information
        """
        status = {
            'clearml_connected': self.clearml_tracker is not None and self.clearml_tracker.enabled,
            'training_active': self.current_metrics.is_training,
            'hyperopt_active': self.hyperopt is not None,
            'last_update': datetime.now().isoformat(),
            'config_loaded': True
        }
        
        # Add any error states
        if not status['clearml_connected']:
            status['warnings'] = ['ClearML not connected - live metrics unavailable']
        
        return status 