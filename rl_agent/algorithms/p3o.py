"""
P3O (Procrastinated Proximal Policy Optimization) Algorithm
Enhanced version of PPO with procrastination factor for educational drone RL
MVP Configuration with student-tunable hyperparameters and random search optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
import random
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class P3OConfig:
    """Configuration for P3O algorithm with MVP hyperparameter ranges"""
    
    # MVP Hyperparameters (Student Tunable)
    learning_rate: float = 3e-4        # Range: 1e-4 to 3e-3
    clip_ratio: float = 0.2            # Range: 0.1 to 0.3
    entropy_coef: float = 0.01         # Range: 1e-3 to 0.1
    batch_size: int = 64               # Range: 64 to 256
    rollout_steps: int = 512           # Range: 512 to 2048
    num_epochs: int = 10               # Range: 3 to 10
    gamma: float = 0.99                # Range: 0.9 to 0.99
    gae_lambda: float = 0.95           # Range: 0.9 to 0.99
    
    # Fixed hyperparameters
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # P3O specific parameters
    alpha: float = 0.2                    # Blend factor between current and procrastinated policy
    procrastination_factor: float = 0.95  # How much to procrastinate (0.95 = 95% of original action)
    procrastination_decay: float = 0.999  # Decay rate for procrastination
    min_procrastination: float = 0.8      # Minimum procrastination factor
    
    # Network architecture
    hidden_dims: List[int] = None
    activation: str = "tanh"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary (for student UI)"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Validate and clamp to acceptable ranges
        self.validate_ranges()
    
    def validate_ranges(self) -> None:
        """Validate that all hyperparameters are within MVP acceptable ranges"""
        # Clamp learning rate
        self.learning_rate = np.clip(self.learning_rate, 1e-4, 3e-3)
        
        # Clamp clip ratio
        self.clip_ratio = np.clip(self.clip_ratio, 0.1, 0.3)
        
        # Clamp entropy coefficient
        self.entropy_coef = np.clip(self.entropy_coef, 1e-3, 0.1)
        
        # Clamp batch size to valid values
        valid_batch_sizes = [64, 128, 256]
        self.batch_size = min(valid_batch_sizes, key=lambda x: abs(x - self.batch_size))
        
        # Clamp rollout steps to valid values
        valid_rollout_steps = [512, 1024, 2048]
        self.rollout_steps = min(valid_rollout_steps, key=lambda x: abs(x - self.rollout_steps))
        
        # Clamp num epochs
        self.num_epochs = int(np.clip(self.num_epochs, 3, 10))
        
        # Clamp gamma
        self.gamma = np.clip(self.gamma, 0.9, 0.99)
        
        # Clamp GAE lambda
        self.gae_lambda = np.clip(self.gae_lambda, 0.9, 0.99)
    
    def get_student_config(self) -> Dict[str, Any]:
        """Get student-tunable configuration for UI display"""
        return {
            'learning_rate': {
                'value': self.learning_rate,
                'range': [1e-4, 3e-3],
                'description': 'Step size for optimizer',
                'type': 'float',
                'scale': 'log'
            },
            'clip_ratio': {
                'value': self.clip_ratio,
                'range': [0.1, 0.3],
                'description': 'Controls PPO-style policy update clipping',
                'type': 'float',
                'scale': 'linear'
            },
            'entropy_coef': {
                'value': self.entropy_coef,
                'range': [1e-3, 0.1],
                'description': 'Weight for entropy term to encourage exploration',
                'type': 'float',
                'scale': 'log'
            },
            'batch_size': {
                'value': self.batch_size,
                'range': [64, 256],
                'options': [64, 128, 256],
                'description': 'Minibatch size for updates',
                'type': 'select'
            },
            'rollout_steps': {
                'value': self.rollout_steps,
                'range': [512, 2048],
                'options': [512, 1024, 2048],
                'description': 'Environment steps per update',
                'type': 'select'
            },
            'num_epochs': {
                'value': self.num_epochs,
                'range': [3, 10],
                'description': 'Epochs per policy update',
                'type': 'int',
                'scale': 'linear'
            },
            'gamma': {
                'value': self.gamma,
                'range': [0.9, 0.99],
                'description': 'Discount factor for future rewards',
                'type': 'float',
                'scale': 'linear'
            },
            'gae_lambda': {
                'value': self.gae_lambda,
                'range': [0.9, 0.99],
                'description': 'GAE parameter for advantage estimation',
                'type': 'float',
                'scale': 'linear'
            }
        }
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default hyperparameter values"""
        return {
            'learning_rate': 3e-4,
            'clip_ratio': 0.2,
            'entropy_coef': 0.01,
            'batch_size': 64,
            'rollout_steps': 512,
            'num_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95
        }


class HyperparameterOptimizer:
    """Random search hyperparameter optimizer like AWS DeepRacer"""
    
    def __init__(self, base_config: P3OConfig, clearml_tracker=None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            base_config: Base P3O configuration 
            clearml_tracker: ClearML tracker for logging optimization results
        """
        self.base_config = base_config
        self.clearml = clearml_tracker
        
        # Optimization state
        self.optimization_history = []
        self.best_performance = -float('inf')
        self.best_config = None
        self.current_trial = 0
        
        # Define search ranges (as shown in user's screenshot)
        self.search_ranges = {
            'learning_rate': {'min': 1e-4, 'max': 3e-3, 'scale': 'log'},
            'clip_ratio': {'min': 0.1, 'max': 0.3, 'scale': 'linear'},
            'entropy_coef': {'min': 1e-3, 'max': 0.1, 'scale': 'log'},
            'batch_size': {'options': [64, 128, 256]},
            'rollout_steps': {'options': [512, 1024, 2048]},
            'num_epochs': {'min': 3, 'max': 10, 'scale': 'linear'},
            'gamma': {'min': 0.9, 'max': 0.99, 'scale': 'linear'},
            'gae_lambda': {'min': 0.9, 'max': 0.99, 'scale': 'linear'}
        }
        
        logger.info("Hyperparameter optimizer initialized")
    
    def sample_random_config(self) -> Dict[str, Any]:
        """Sample random hyperparameter configuration"""
        config = {}
        
        for param, range_info in self.search_ranges.items():
            if 'options' in range_info:
                # Discrete choice
                config[param] = random.choice(range_info['options'])
            else:
                # Continuous range
                min_val, max_val = range_info['min'], range_info['max']
                scale = range_info.get('scale', 'linear')
                
                if scale == 'log':
                    # Log scale sampling
                    log_min, log_max = np.log10(min_val), np.log10(max_val)
                    log_val = random.uniform(log_min, log_max)
                    config[param] = 10 ** log_val
                else:
                    # Linear scale sampling
                    config[param] = random.uniform(min_val, max_val)
        
        return config
    
    def suggest_config(self) -> P3OConfig:
        """Suggest next hyperparameter configuration to try"""
        self.current_trial += 1
        
        # Sample random configuration
        random_config = self.sample_random_config()
        
        # Create new P3OConfig
        suggested_config = P3OConfig(**self.base_config.__dict__)
        suggested_config.update_from_dict(random_config)
        
        # Log to ClearML
        if self.clearml:
            trial_info = {
                'trial_number': self.current_trial,
                'hyperparameters': random_config,
                'optimization_type': 'random_search'
            }
            self.clearml.log_hyperparameters(trial_info)
        
        logger.info(f"Trial {self.current_trial}: Suggested config: {random_config}")
        return suggested_config
    
    def report_performance(self, config: P3OConfig, performance_metric: float, 
                          additional_metrics: Dict[str, float] = None):
        """
        Report performance of a hyperparameter configuration
        
        Args:
            config: The configuration that was tested
            performance_metric: Primary performance metric (e.g., average reward)
            additional_metrics: Additional metrics to log
        """
        # Store in history
        trial_data = {
            'trial': self.current_trial,
            'config': config.__dict__.copy(),
            'performance': performance_metric,
            'additional_metrics': additional_metrics or {}
        }
        self.optimization_history.append(trial_data)
        
        # Update best configuration
        if performance_metric > self.best_performance:
            self.best_performance = performance_metric
            self.best_config = config.__dict__.copy()
            
            logger.info(f"New best configuration found! Performance: {performance_metric:.4f}")
            
            # Log best config to ClearML
            if self.clearml:
                self.clearml.log_hyperparameters({
                    'best_performance': performance_metric,
                    'best_config': self.best_config,
                    'trial_of_best': self.current_trial
                })
        
        # Log trial results to ClearML
        if self.clearml:
            metrics = {
                'performance_metric': performance_metric,
                'trial_number': self.current_trial
            }
            if additional_metrics:
                metrics.update(additional_metrics)
            
            self.clearml.log_metrics(metrics, self.current_trial)
        
        logger.info(f"Trial {self.current_trial} performance: {performance_metric:.4f}")
    
    def get_best_config(self) -> Optional[P3OConfig]:
        """Get the best configuration found so far"""
        if self.best_config is None:
            return None
        
        best_config = P3OConfig()
        best_config.update_from_dict(self.best_config)
        return best_config
    
    def get_optimization_suggestions(self, recent_trials: int = 5) -> List[str]:
        """
        Get optimization suggestions based on recent trials
        
        Args:
            recent_trials: Number of recent trials to analyze
            
        Returns:
            List of human-readable suggestions
        """
        if len(self.optimization_history) < recent_trials:
            return ["Need more trials to provide suggestions"]
        
        suggestions = []
        recent_data = self.optimization_history[-recent_trials:]
        
        # Analyze recent performance trends
        performances = [trial['performance'] for trial in recent_data]
        avg_performance = np.mean(performances)
        
        if avg_performance < self.best_performance * 0.8:
            suggestions.append("Recent trials performing poorly. Consider trying different learning rate range.")
        
        # Analyze learning rate trends
        learning_rates = [trial['config']['learning_rate'] for trial in recent_data]
        if all(lr < 1e-3 for lr in learning_rates) and avg_performance < self.best_performance * 0.9:
            suggestions.append("Try higher learning rates (> 1e-3) for faster learning")
        
        # Analyze clip ratio
        clip_ratios = [trial['config']['clip_ratio'] for trial in recent_data]
        if all(cr > 0.25 for cr in clip_ratios) and avg_performance < self.best_performance * 0.9:
            suggestions.append("High clip ratios may cause instability. Try lower values (< 0.2)")
        
        return suggestions if suggestions else ["Optimization progressing well. Continue with random search."]
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        results = {
            'best_performance': self.best_performance,
            'best_config': self.best_config,
            'optimization_history': self.optimization_history,
            'total_trials': self.current_trial
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {filepath}")


class MVPTrainingConfig:
    """Training configuration for MVP with student-configurable training time"""
    
    def __init__(self):
        # Training time parameters - MUST be set by student
        self.train_time_minutes = None      # Required: Student must set this
        self.steps_per_second = 20          # 20Hz control rate
        self.max_episodes = 1000            # Safety limit
        
        # Auto-calculated parameters (computed when training time is set)
        self.max_steps = None
        
        # Training monitoring
        self.save_interval_minutes = 10     # Save checkpoint every 10 minutes
        self.eval_interval_minutes = 5      # Evaluate every 5 minutes
        
    def set_training_time(self, minutes: int) -> None:
        """Set training time in minutes (student UI)"""
        self.train_time_minutes = np.clip(minutes, 10, 180)  # 10 minutes to 3 hours
        self.max_steps = self._calculate_max_steps()
        logger.info(f"Training time set to {self.train_time_minutes} minutes ({self.max_steps} steps)")
    
    def _calculate_max_steps(self) -> int:
        """Calculate maximum training steps from time"""
        if self.train_time_minutes is None:
            raise ValueError("Training time must be set before calculating max steps. Call set_training_time() first.")
        # max_steps = steps_per_second * 60 * train_time_minutes
        return self.steps_per_second * 60 * self.train_time_minutes
    
    def get_checkpoint_steps(self) -> int:
        """Get steps between checkpoints"""
        if self.train_time_minutes is None:
            raise ValueError("Training time must be set before getting checkpoint steps. Call set_training_time() first.")
        return self.steps_per_second * 60 * self.save_interval_minutes
    
    def get_eval_steps(self) -> int:
        """Get steps between evaluations"""
        if self.train_time_minutes is None:
            raise ValueError("Training time must be set before getting eval steps. Call set_training_time() first.")
        return self.steps_per_second * 60 * self.eval_interval_minutes
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for UI display"""
        return {
            'train_time_minutes': {
                'value': self.train_time_minutes,
                'range': [10, 180],
                'description': 'Training duration in minutes',
                'type': 'int',
                'scale': 'linear',
                'required': True,
                'placeholder': 'Enter training time (10-180 minutes)'
            },
            'max_steps': {
                'value': self.max_steps,
                'description': 'Total training steps (auto-calculated)',
                'type': 'readonly'
            },
            'steps_per_second': {
                'value': self.steps_per_second,
                'description': 'Environment control rate (Hz)',
                'type': 'readonly'
            },
            'save_interval_minutes': {
                'value': self.save_interval_minutes,
                'range': [1, 30],
                'description': 'Minutes between model saves',
                'type': 'int'
            },
            'eval_interval_minutes': {
                'value': self.eval_interval_minutes,
                'range': [1, 15],
                'description': 'Minutes between evaluations',
                'type': 'int'
            }
        }


# Legacy compatibility functions
def create_default_p3o_config() -> P3OConfig:
    """Create default P3O configuration for MVP"""
    return P3OConfig()


def create_mvp_training_config(train_time_minutes: int) -> MVPTrainingConfig:
    """Create MVP training configuration with required training time"""
    if train_time_minutes is None:
        raise ValueError("Training time is required. Students must specify training duration (10-180 minutes).")
    config = MVPTrainingConfig()
    config.set_training_time(train_time_minutes)
    return config


def validate_student_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean student configuration input"""
    config = P3OConfig()
    
    # Update with student values
    safe_config = {}
    for key, value in config_dict.items():
        if hasattr(config, key):
            try:
                # Type conversion and validation
                if key in ['batch_size', 'rollout_steps', 'num_epochs']:
                    safe_config[key] = int(value)
                else:
                    safe_config[key] = float(value)
            except (ValueError, TypeError):
                logger.warning(f"Invalid value for {key}: {value}, using default")
                continue
        else:
            logger.warning(f"Unknown hyperparameter: {key}")
    
    # Create config and validate
    config.update_from_dict(safe_config)
    return safe_config


class P3OPolicy(nn.Module):
    """P3O Policy Network with procrastination mechanism"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: P3OConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Activation function
        if config.activation == "tanh":
            self.activation = nn.Tanh()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
        
        # Build policy network
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                self.activation
            ])
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Policy head (mean and log_std)
        self.policy_mean = nn.Linear(input_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head
        self.value_head = nn.Linear(input_dim, 1)
        
        # Previous action storage for procrastination
        self.register_buffer('prev_action', torch.zeros(action_dim))
        self.procrastination_factor = config.procrastination_factor
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Small initialization for policy head
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning action distribution parameters and value
        
        Returns:
            action_mean: Mean of action distribution
            action_log_std: Log standard deviation of action distribution  
            value: State value estimate
        """
        features = self.feature_extractor(obs)
        
        action_mean = self.policy_mean(features)
        action_log_std = self.policy_log_std.expand_as(action_mean)
        value = self.value_head(features)
        
        return action_mean, action_log_std, value
    
    def sample_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action using procrastinated policy
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_mean, action_log_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
        else:
            # Create normal distribution
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            
            # Sample raw action
            raw_action = dist.sample()
            
            # Apply procrastination: blend with previous action
            action = (self.config.alpha * raw_action + 
                     (1 - self.config.alpha) * self.procrastination_factor * self.prev_action)
            
            # Update previous action
            self.prev_action = action.detach().clone()
        
        # Compute log probability
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations
        
        Args:
            obs: Observation tensor
            actions: Action tensor
            
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Action distribution entropy
        """
        action_mean, action_log_std, values = self.forward(obs)
        
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy
    
    def update_procrastination(self):
        """Update procrastination factor with decay"""
        self.procrastination_factor = max(
            self.config.min_procrastination,
            self.procrastination_factor * self.config.procrastination_decay
        )


class P3OValueNetwork(nn.Module):
    """Separate value network for P3O (optional)"""
    
    def __init__(self, obs_dim: int, config: P3OConfig):
        super().__init__()
        
        # Build value network
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh()
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        self.value_net = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate"""
        return self.value_net(obs).squeeze(-1)


class P3O:
    """P3O (Procrastinated Proximal Policy Optimization) Algorithm"""
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 config: Optional[P3OConfig] = None,
                 device: str = "cpu"):
        """
        Initialize P3O algorithm
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension  
            config: P3O configuration
            device: Device to run on ("cpu" or "cuda")
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or P3OConfig()
        self.device = torch.device(device)
        
        # Initialize policy network
        self.policy = P3OPolicy(obs_dim, action_dim, self.config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.config.learning_rate
        )
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
        logger.info(f"P3O initialized with obs_dim={obs_dim}, action_dim={action_dim}")
    
    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from student UI"""
        self.config.update_from_dict(config_dict)
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.learning_rate
        
        logger.info(f"Updated P3O config: {config_dict}")
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action for given observation
        
        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.sample_action(obs_tensor, deterministic)
        
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()
    
    def update(self, replay_buffer, batch_size: Optional[int] = None, n_epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Update policy using collected experience
        
        Args:
            replay_buffer: Experience replay buffer
            batch_size: Batch size for updates (uses config if None)
            n_epochs: Number of update epochs (uses config if None)
            
        Returns:
            training_metrics: Dictionary of training metrics
        """
        # Use config values if not specified
        batch_size = batch_size or self.config.batch_size
        n_epochs = n_epochs or self.config.num_epochs
        
        if len(replay_buffer) < batch_size:
            return {}
        
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size)
        
        obs = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(obs, rewards, next_obs, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy for multiple epochs
        total_metrics = {}
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(len(obs))
            
            epoch_metrics = []
            
            # Mini-batch updates
            for start_idx in range(0, len(obs), batch_size):
                end_idx = min(start_idx + batch_size, len(obs))
                mb_indices = indices[start_idx:end_idx]
                
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                
                # Compute losses
                metrics = self._compute_losses(
                    mb_obs, mb_actions, mb_returns, mb_advantages, mb_old_log_probs
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                metrics['total_loss'].backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                epoch_metrics.append({k: v.item() if torch.is_tensor(v) else v 
                                    for k, v in metrics.items()})
            
            # Average metrics for this epoch
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
            
            total_metrics.update(avg_metrics)
        
        # Update procrastination factor
        self.policy.update_procrastination()
        
        # Store training statistics
        for key, value in total_metrics.items():
            self.training_stats[key].append(value)
        
        return total_metrics
    
    def _compute_gae(self, obs: torch.Tensor, rewards: torch.Tensor, 
                     next_obs: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        with torch.no_grad():
            _, _, values = self.policy(obs)
            _, _, next_values = self.policy(next_obs)
            
            values = values.squeeze(-1)
            next_values = next_values.squeeze(-1)
            
            # Compute deltas
            deltas = rewards + self.config.gamma * next_values * (~dones) - values
            
            # Compute GAE
            advantages = torch.zeros_like(rewards)
            gae = 0
            
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.config.gamma * self.config.gae_lambda * gae * (~dones[t])
                advantages[t] = gae
            
            returns = advantages + values
        
        return returns, advantages
    
    def _compute_losses(self, obs: torch.Tensor, actions: torch.Tensor,
                       returns: torch.Tensor, advantages: torch.Tensor,
                       old_log_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute P3O losses"""
        # Get current policy outputs
        log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)
        
        # Compute ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        
        # Policy loss (PPO clipped objective)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy loss (for exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.config.value_loss_coef * value_loss + 
                     self.config.entropy_coef * entropy_loss)
        
        # Compute additional metrics
        with torch.no_grad():
            kl_divergence = (old_log_probs - log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_ratio).float().mean()
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
            'kl_divergence': kl_divergence,
            'clip_fraction': clip_fraction
        }
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        logger.info(f"Model loaded from {path}")
    
    def get_training_stats(self) -> Dict[str, List[float]]:
        """Get training statistics"""
        return self.training_stats.copy() 