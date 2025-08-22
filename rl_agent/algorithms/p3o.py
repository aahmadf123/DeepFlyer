"""
P3O (Procrastinated Proximal Policy Optimization) Algorithm for DeepFlyer
Direct control implementation for drone racing through hoops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class P3OConfig:
    """P3O hyperparameters configuration"""
    # Network architecture
    hidden_dims: List[int] = (256, 256)
    activation: str = "tanh"
    
    # P3O specific parameters
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    procrastination_factor: float = 0.95  # Alpha for action blending
    
    # Training parameters
    batch_size: int = 64
    num_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Action noise for exploration
    action_noise: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'learning_rate': self.learning_rate,
            'clip_ratio': self.clip_ratio,
            'procrastination_factor': self.procrastination_factor,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'action_noise': self.action_noise
        }


class P3ONetwork(nn.Module):
    """P3O Policy and Value Network"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: P3OConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Shared feature extractor
        layers = []
        input_dim = obs_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh() if config.activation == "tanh" else nn.ReLU())
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Policy head (mean and log_std)
        self.policy_mean = nn.Linear(input_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Previous action for procrastination
        self.register_buffer('prev_action', torch.zeros(action_dim))
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        # Small initial policy
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and value"""
        features = self.feature_extractor(obs)
        action_mean = self.policy_mean(features)
        value = self.value_head(features)
        return action_mean, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action with procrastination mechanism"""
        action_mean, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
        else:
            # Sample from Gaussian distribution
            std = self.policy_log_std.exp()
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
        
        # Apply procrastination (blend with previous action)
        if self.prev_action.shape == action.shape:
            alpha = self.config.procrastination_factor
            procrastinated_action = alpha * self.prev_action + (1 - alpha) * action
        else:
            procrastinated_action = action
        
        # Update previous action
        self.prev_action = procrastinated_action.detach()
        
        # Calculate log probability
        std = self.policy_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        log_prob = dist.log_prob(procrastinated_action).sum(dim=-1, keepdim=True)
        
        return procrastinated_action, log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for P3O loss calculation"""
        action_mean, value = self.forward(obs)
        
        std = self.policy_log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class P3O:
    """P3O Algorithm for drone control"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: Optional[P3OConfig] = None, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or P3OConfig()
        self.device = torch.device(device)
        
        # Initialize network
        self.network = P3ONetwork(obs_dim, action_dim, self.config).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': []
        }
        
        logger.info(f"P3O initialized: obs_dim={obs_dim}, action_dim={action_dim}")
    
    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict action for given observation"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, _, _ = self.network.get_action(obs_tensor, deterministic)
        
        return action.cpu().numpy().squeeze()
    
    def add_to_buffer(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, 
                     reward: float, done: bool):
        """Add experience to replay buffer (handled externally)"""
        # This is handled by the replay buffer in the node
        pass
    
    def update(self, replay_buffer) -> Dict[str, float]:
        """Update P3O policy using collected experiences"""
        if len(replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        batch = replay_buffer.sample(self.config.batch_size)
        
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Calculate advantages using GAE
        with torch.no_grad():
            _, next_values = self.network.forward(next_obs)
            _, values = self.network.forward(obs)
            
            # TD error
            td_target = rewards + self.config.gamma * next_values.squeeze() * (1 - dones)
            td_error = td_target - values.squeeze()
            
            # GAE
            advantages = self._calculate_gae(td_error, dones)
            returns = advantages + values.squeeze()
        
        # Store old log probs for KL calculation
        with torch.no_grad():
            old_log_probs, _, _ = self.network.evaluate_actions(obs, actions)
        
        # P3O update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.num_epochs):
            # Evaluate current policy
            log_probs, values, entropy = self.network.evaluate_actions(obs, actions)
            
            # Calculate ratio for P3O clipping
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Ensure advantages match ratio dimensions
            if advantages.dim() == 1 and ratio.dim() == 2:
                advantages_matched = advantages.unsqueeze(1)
            elif advantages.dim() == 2 and ratio.dim() == 1:
                advantages_matched = advantages.squeeze(1)
            else:
                advantages_matched = advantages
            
            # Clipped surrogate loss
            surr1 = ratio * advantages_matched
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages_matched
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.config.value_loss_coef * value_loss + 
                         self.config.entropy_coef * entropy_loss)
            
            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Calculate KL divergence for monitoring
        with torch.no_grad():
            new_log_probs, _, _ = self.network.evaluate_actions(obs, actions)
            kl_div = (old_log_probs - new_log_probs).mean().item()
        
        # Store statistics
        stats = {
            'policy_loss': total_policy_loss / self.config.num_epochs,
            'value_loss': total_value_loss / self.config.num_epochs,
            'entropy': total_entropy / self.config.num_epochs,
            'kl_divergence': kl_div
        }
        
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        return stats
    
    def _calculate_gae(self, td_errors: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate Generalized Advantage Estimation"""
        advantages = torch.zeros_like(td_errors)
        gae = 0
        
        for t in reversed(range(len(td_errors))):
            if dones[t]:
                gae = 0
            gae = td_errors[t] + self.config.gamma * self.config.gae_lambda * gae
            advantages[t] = gae
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', {})
        logger.info(f"Model loaded from {path}")
    
    def save_model(self, checkpoint_path: str, episode: int, metrics: Dict[str, float]):
        """Save complete model checkpoint for deployment"""
        # Ensure directory exists
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode': episode,
            'config': self.config.__dict__,
            'metrics': metrics,
            'timestamp': time.time(),
            'p3o_version': '1.0'
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
        # Save deployment-ready model (smaller, inference-optimized)
        deployment_model = {
            'model_state_dict': self.network.state_dict(),
            'config': self.config.__dict__,
            'normalization_params': self._get_normalization_params(),
            'deployment_ready': True,
            'timestamp': time.time()
        }
        deployment_path = checkpoint_path.replace('.pth', '_deployment.pth')
        torch.save(deployment_model, deployment_path)
        logger.info(f"Deployment model saved to {deployment_path}")
        
        return deployment_path
    
    def load_model_for_deployment(self, model_path: str):
        """Load model optimized for real-time inference"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model weights
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()  # Set to evaluation mode
        
        # Set configuration
        if 'config' in checkpoint:
            for key, value in checkpoint['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        logger.info(f"Deployment model loaded from {model_path}")
        return checkpoint.get('config', {})
    
    def _get_normalization_params(self) -> Dict[str, Any]:
        """Get normalization parameters for deployment"""
        return {
            'obs_mean': getattr(self, 'obs_mean', np.zeros(8)),
            'obs_std': getattr(self, 'obs_std', np.ones(8)),
            'action_scale': getattr(self, 'action_scale', 1.0),
            'action_bounds': {
                'low': [-2.0, -2.0, -1.0, -1.0],  # [vx, vy, vz, yaw_rate]
                'high': [2.0, 2.0, 1.0, 1.0]
            }
        }
    
    def export_for_deployment(self, model_path: str, output_path: str = None):
        """Export model for production deployment with optimization"""
        if output_path is None:
            output_path = model_path.replace('.pth', '_optimized.pth')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.network.eval()
        
        # Create optimized deployment package
        optimized_model = {
            'model_state_dict': self.network.state_dict(),
            'config': checkpoint.get('config', {}),
            'normalization_params': self._get_normalization_params(),
            'deployment_metadata': {
                'framework': 'pytorch',
                'model_type': 'p3o_policy',
                'input_shape': [8],  # 8D observation space
                'output_shape': [4],  # 4D action space
                'performance_metrics': checkpoint.get('metrics', {}),
                'optimization_level': 'production'
            }
        }
        
        torch.save(optimized_model, output_path)
        logger.info(f"Optimized deployment model exported to {output_path}")
        return output_path