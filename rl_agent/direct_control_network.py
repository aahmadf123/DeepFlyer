"""
Neural network for direct drone control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DirectControlNetwork(nn.Module):
    """Neural network for direct drone control."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,  # thrust, roll, pitch, yaw
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        Initialize direct control network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (default 4: thrust, roll, pitch, yaw)
            hidden_dim: Hidden layer dimension
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers for control actions
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            mean: Mean of the Gaussian policy [batch_size, action_dim]
            log_std: Log standard deviation [batch_size, action_dim]
        """
        x = self.encoder(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample control action from the policy.
        
        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: If True, return the mean action
            
        Returns:
            action: Sampled control action [batch_size, action_dim]
            log_prob: Log probability of the action [batch_size, 1]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            return torch.tanh(mean), None
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Constrain actions to [-1, 1]
        
        # Compute log probability, using the formula for change of variables
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the log probability of an action given a state.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            mean: Mean action [batch_size, action_dim]
            log_prob: Log probability of the action [batch_size, 1]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        
        # Invert tanh transformation: action = tanh(x_t) => x_t = atanh(action)
        # Use approximation for numerical stability
        eps = 1e-6
        x_t = 0.5 * torch.log((1 + action + eps) / (1 - action + eps))
        
        # Compute log probability including change of variables
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return mean, log_prob
    
    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy of the policy for a given state.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            entropy: Entropy of the policy [batch_size, 1]
        """
        _, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Entropy of a Gaussian is 0.5 + 0.5*log(2*pi) + log(sigma)
        entropy = 0.5 + 0.5 * torch.log(2 * torch.pi) + log_std
        return entropy.sum(1, keepdim=True) 