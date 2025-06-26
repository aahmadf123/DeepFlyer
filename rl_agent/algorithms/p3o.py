"""
P3O (Procrastinated Proximal Policy Optimization) Algorithm
Enhanced version of PPO with procrastination factor for educational drone RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class P3OConfig:
    """Configuration for P3O algorithm"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
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
    
    def update(self, replay_buffer, batch_size: int = 64, n_epochs: int = 10) -> Dict[str, float]:
        """
        Update policy using collected experience
        
        Args:
            replay_buffer: Experience replay buffer
            batch_size: Batch size for updates
            n_epochs: Number of update epochs
            
        Returns:
            training_metrics: Dictionary of training metrics
        """
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
                     next_obs: torch.Tensor, dones: torch.Tensor,
                     gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
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
                gae = deltas[t] + self.config.gamma * gae_lambda * gae * (~dones[t])
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