"""
Neural network for the RL supervisor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SupervisorNetwork(nn.Module):
    """Neural network for the RL supervisor."""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        Initialize supervisor network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output layers for PID gains
        self.mean = nn.Linear(hidden_dim, 1)  # Only P gain for now
        self.log_std = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor [batch_size, state_dim]
            
        Returns:
            mean: Mean of the Gaussian policy [batch_size, 1]
            log_std: Log standard deviation [batch_size, 1]
        """
        x = self.encoder(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample PID gain from the policy.
        
        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: If True, return the mean gain
            
        Returns:
            gain: Sampled PID gain [batch_size, 1]
            log_prob: Log probability of the gain [batch_size, 1]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            gain = mean
            return gain, None
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        gain = torch.sigmoid(x_t)  # Constrain gain to (0, 1)
        
        # Compute log probability, using the formula for change of variables
        log_prob = normal.log_prob(x_t) - torch.log(gain * (1 - gain) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return gain, log_prob 