"""
RL Supervisor Agent for PID control.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Union, Any, Optional
import gymnasium as gym
from collections import deque
import random

from rl_agent.models.base_model import BaseModel
from rl_agent.supervisor_network import SupervisorNetwork
from rl_agent.pid_controller import PIDController


class SupervisorAgent(BaseModel):
    """
    RL Supervisor Agent.
    
    This agent observes the current state and errors, and adjusts PID gains
    to improve path following performance.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "auto",
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        **kwargs
    ):
        """
        Initialize the supervisor agent.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
            hidden_dim: Hidden dimension size for networks
            lr: Learning rate
            gamma: Discount factor
            batch_size: Batch size for training
            buffer_size: Maximum replay buffer size
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, action_space, device, **kwargs)
        
        # Get dimensions
        if isinstance(observation_space, gym.spaces.Box):
            self.state_dim = int(np.prod(observation_space.shape))
        else:
            raise ValueError("Observation space must be continuous")
        
        # Store hyperparameters
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Create networks
        self.actor = SupervisorNetwork(
            self.state_dim,
            hidden_dim
        ).to(self.device)
        
        self.critic = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Create PID controller
        self.pid = PIDController()
        
        # Training metrics
        self.actor_losses = []
        self.critic_losses = []
    
    def predict(
        self, 
        observation: Union[np.ndarray, Dict[str, np.ndarray]], 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Predict PID gain based on observation.
        
        Args:
            observation: The current observation from the environment
            deterministic: Whether to return deterministic actions
            
        Returns:
            gain: The predicted PID gain
            states: None (no additional state information)
        """
        # Handle dict observations
        if isinstance(observation, dict):
            # Flatten dict observation into vector
            obs_vector = []
            for key in sorted(observation.keys()):
                obs_vector.append(observation[key].flatten())
            observation = np.concatenate(obs_vector)
        
        # Convert to tensor
        state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        
        # Get PID gain from actor
        with torch.no_grad():
            gain, _ = self.actor.sample(state, deterministic)
        
        # Update PID controller
        self.pid.update_gains(gain.item())
        
        return gain.cpu().numpy(), None
    
    def add_to_buffer(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray, 
        reward: float, 
        done: bool
    ) -> None:
        """
        Add a transition to the replay buffer.
        
        Args:
            state: Current state
            action: Action taken (PID gain)
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        self.replay_buffer.append((state, action, next_state, reward, done))
    
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            **kwargs: Additional training arguments
            
        Returns:
            training_info: Dictionary containing training metrics
        """
        self.training_started = True
        self.total_timesteps = total_timesteps
        
        return {
            "actor_loss": self.actor_losses[-1] if self.actor_losses else None,
            "critic_loss": self.critic_losses[-1] if self.critic_losses else None
        }
    
    def learn(self, batch_size: int = None) -> Dict[str, float]:
        """
        Perform one iteration of learning on a batch of data.
        
        Args:
            batch_size: Size of the batch to learn from
            
        Returns:
            metrics: Dictionary of learning metrics
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.replay_buffer) < batch_size:
            return {
                "actor_loss": float('nan'),
                "critic_loss": float('nan')
            }
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.vstack(rewards)).to(self.device)
        dones = torch.FloatTensor(np.vstack(dones)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        
        with torch.no_grad():
            next_values = self.critic(next_states)
            value_targets = rewards + self.gamma * next_values * (1 - dones)
        
        values = self.critic(states)
        critic_loss = F.mse_loss(values, value_targets)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        
        # Compute advantages
        with torch.no_grad():
            advantages = value_targets - values
        
        # Get action probabilities
        _, log_probs = self.actor.sample(states)
        
        # Policy loss
        actor_loss = -(log_probs * advantages).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item()
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """
        Get model state for saving.
        
        Returns:
            state_dict: Dictionary containing model state
        """
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "pid": {
                "kp": self.pid.kp,
                "ki": self.pid.ki,
                "kd": self.pid.kd
            }
        }
    
    def load_model_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load model state from state dictionary.
        
        Args:
            state_dict: Dictionary containing model state
        """
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        
        # Update PID gains
        pid_state = state_dict["pid"]
        self.pid.update_gains(
            pid_state["kp"],
            pid_state["ki"],
            pid_state["kd"]
        ) 