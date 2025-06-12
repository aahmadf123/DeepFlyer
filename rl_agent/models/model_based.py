import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Union, Any, Optional, List
import gymnasium as gym
from collections import deque
import random
import time

from rl_agent.models.base_model import BaseModel


class DynamicsModel(nn.Module):
    """Neural network model that predicts next state given current state and action."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict the next state given the current state and action.
        
        Args:
            state: Current state tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            next_state_delta: Predicted change in state [batch_size, state_dim]
        """
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class RewardModel(nn.Module):
    """Neural network model that predicts reward given state, action, next_state."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim * 2 + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict the reward given state, action, and next state.
        
        Args:
            state: Current state tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            next_state: Next state tensor [batch_size, state_dim]
            
        Returns:
            reward: Predicted reward [batch_size, 1]
        """
        x = torch.cat([state, action, next_state], dim=1)
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Neural network policy model."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
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
        x = self.trunk(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            state: State tensor [batch_size, state_dim]
            deterministic: If True, return the mean action
            
        Returns:
            action: Sampled action [batch_size, action_dim]
            log_prob: Log probability of the action [batch_size, 1]
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            return action, None
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Compute log probability, using the formula for change of variables
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class ModelBasedAgent(BaseModel):
    """
    Model-Based Reinforcement Learning Agent.
    
    This agent learns a dynamics model and reward model of the environment,
    then uses these models for planning and policy improvement.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "auto",
        hidden_dim: int = 256,
        dynamics_lr: float = 1e-3,
        reward_lr: float = 1e-3,
        policy_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        planning_horizon: int = 10,
        n_planning_iters: int = 5,
        batch_size: int = 256,
        buffer_size: int = 1_000_000,
        model_update_freq: int = 250,
        policy_update_freq: int = 1,
        **kwargs
    ):
        """
        Initialize the model-based agent.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
            hidden_dim: Hidden dimension size for all networks
            dynamics_lr: Learning rate for dynamics model
            reward_lr: Learning rate for reward model
            policy_lr: Learning rate for policy network
            gamma: Discount factor
            tau: Soft update coefficient
            planning_horizon: Number of steps to look ahead during planning
            n_planning_iters: Number of planning iterations
            batch_size: Batch size for training
            buffer_size: Maximum replay buffer size
            model_update_freq: Frequency of model updates
            policy_update_freq: Frequency of policy updates
            **kwargs: Additional arguments
        """
        super().__init__(observation_space, action_space, device, **kwargs)
        
        # Verify observation and action spaces
        assert isinstance(observation_space, gym.spaces.Box), "Observation space must be continuous"
        assert isinstance(action_space, gym.spaces.Box), "Action space must be continuous"
        
        # Get dimensions
        self.state_dim = int(np.prod(observation_space.shape))
        self.action_dim = int(np.prod(action_space.shape))
        
        # Store hyperparameters
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.planning_horizon = planning_horizon
        self.n_planning_iters = n_planning_iters
        self.batch_size = batch_size
        self.model_update_freq = model_update_freq
        self.policy_update_freq = policy_update_freq
        
        # Create models
        self.dynamics_model = DynamicsModel(
            self.state_dim, 
            self.action_dim, 
            hidden_dim
        ).to(self.device)
        
        self.reward_model = RewardModel(
            self.state_dim, 
            self.action_dim, 
            hidden_dim
        ).to(self.device)
        
        self.policy = PolicyNetwork(
            self.state_dim,
            self.action_dim,
            hidden_dim
        ).to(self.device)
        
        # Create optimizers
        self.dynamics_optimizer = optim.Adam(
            self.dynamics_model.parameters(), 
            lr=dynamics_lr
        )
        
        self.reward_optimizer = optim.Adam(
            self.reward_model.parameters(), 
            lr=reward_lr
        )
        
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=policy_lr
        )
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Training metrics
        self.dynamics_losses = []
        self.reward_losses = []
        self.policy_losses = []
        
        # Normalization stats
        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None
        
    def predict(
        self, 
        observation: Union[np.ndarray, Dict[str, np.ndarray]], 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Predict action based on observation.
        
        Args:
            observation: The current observation from the environment
            deterministic: Whether to return deterministic actions
            
        Returns:
            actions: The predicted actions
            states: None (no additional state information)
        """
        # Handle dict observations
        if isinstance(observation, dict):
            # Flatten dict observation into vector
            obs_vector = []
            for key in sorted(observation.keys()):
                obs_vector.append(observation[key].flatten())
            observation = np.concatenate(obs_vector)
        
        # Convert to tensor and normalize
        state = torch.FloatTensor(observation).to(self.device).unsqueeze(0)
        if self.state_mean is not None and self.state_std is not None:
            state_mean = torch.FloatTensor(self.state_mean).to(self.device)
            state_std = torch.FloatTensor(self.state_std).to(self.device)
            state = (state - state_mean) / (state_std + 1e-8)
        
        # Get action from policy
        with torch.no_grad():
            action, _ = self.policy.sample(state, deterministic)
        
        # Denormalize action if needed
        if self.action_mean is not None and self.action_std is not None:
            action_mean = torch.FloatTensor(self.action_mean).to(self.device)
            action_std = torch.FloatTensor(self.action_std).to(self.device)
            action = action * action_std + action_mean
        
        # Convert to numpy and clip to action space
        action = action.cpu().numpy()[0]
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )
        
        return action, None
    
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            **kwargs: Additional training arguments
            
        Returns:
            training_info: Dictionary containing training metrics
        """
        # This method would integrate with the environment for training
        # For this implementation, we'll assume the environment interaction
        # is handled externally and focus on the learn method
        self.training_started = True
        self.total_timesteps = total_timesteps
        
        return {
            "dynamics_loss": self.dynamics_losses[-1] if self.dynamics_losses else None,
            "reward_loss": self.reward_losses[-1] if self.reward_losses else None,
            "policy_loss": self.policy_losses[-1] if self.policy_losses else None
        }
    
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
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        self.replay_buffer.append((state, action, next_state, reward, done))
        
        # Update normalization statistics if buffer has enough data
        if len(self.replay_buffer) >= 1000 and (
            self.state_mean is None or 
            len(self.replay_buffer) % 10000 == 0
        ):
            self._update_normalization_stats()
    
    def _update_normalization_stats(self) -> None:
        """Update normalization statistics from replay buffer data."""
        # Sample states, actions, next_states
        states = []
        actions = []
        next_states = []
        
        # Sample random transitions
        indices = np.random.choice(
            len(self.replay_buffer), 
            min(10000, len(self.replay_buffer)), 
            replace=False
        )
        
        for i in indices:
            state, action, next_state, _, _ = self.replay_buffer[i]
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
        
        # Compute statistics
        states = np.vstack(states)
        actions = np.vstack(actions)
        next_states = np.vstack(next_states)
        
        self.state_mean = np.mean(states, axis=0)
        self.state_std = np.std(states, axis=0) + 1e-8
        self.action_mean = np.mean(actions, axis=0)
        self.action_std = np.std(actions, axis=0) + 1e-8
    
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
                "dynamics_loss": float('nan'),
                "reward_loss": float('nan'),
                "policy_loss": float('nan')
            }
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Convert to tensors and normalize
        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.vstack(rewards)).to(self.device)
        dones = torch.FloatTensor(np.vstack(dones)).to(self.device)
        
        # Normalize states and actions if statistics are available
        if self.state_mean is not None and self.state_std is not None:
            state_mean = torch.FloatTensor(self.state_mean).to(self.device)
            state_std = torch.FloatTensor(self.state_std).to(self.device)
            states_norm = (states - state_mean) / (state_std + 1e-8)
            next_states_norm = (next_states - state_mean) / (state_std + 1e-8)
        else:
            states_norm = states
            next_states_norm = next_states
            
        if self.action_mean is not None and self.action_std is not None:
            action_mean = torch.FloatTensor(self.action_mean).to(self.device)
            action_std = torch.FloatTensor(self.action_std).to(self.device)
            actions_norm = (actions - action_mean) / (action_std + 1e-8)
        else:
            actions_norm = actions
        
        # Update dynamics model
        self.dynamics_optimizer.zero_grad()
        pred_next_state_delta = self.dynamics_model(states_norm, actions_norm)
        dynamics_loss = nn.MSELoss()(
            pred_next_state_delta, 
            next_states_norm - states_norm
        )
        dynamics_loss.backward()
        self.dynamics_optimizer.step()
        
        # Update reward model
        self.reward_optimizer.zero_grad()
        pred_rewards = self.reward_model(states_norm, actions_norm, next_states_norm)
        reward_loss = nn.MSELoss()(pred_rewards, rewards)
        reward_loss.backward()
        self.reward_optimizer.step()
        
        # Update policy (less frequently)
        policy_loss = torch.tensor(0.0)
        if self.total_timesteps % self.policy_update_freq == 0:
            self.policy_optimizer.zero_grad()
            
            # Sample actions from current policy
            policy_actions, log_probs = self.policy.sample(states_norm)
            
            # Predict next states using dynamics model
            pred_next_states_delta = self.dynamics_model(states_norm, policy_actions)
            pred_next_states = states_norm + pred_next_states_delta
            
            # Predict rewards
            pred_rewards = self.reward_model(states_norm, policy_actions, pred_next_states)
            
            # Policy loss is negative expected reward
            policy_loss = -pred_rewards.mean()
            
            # Add entropy regularization
            if log_probs is not None:
                policy_loss += 0.01 * log_probs.mean()
                
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # Store losses
        self.dynamics_losses.append(dynamics_loss.item())
        self.reward_losses.append(reward_loss.item())
        self.policy_losses.append(policy_loss.item())
        
        return {
            "dynamics_loss": dynamics_loss.item(),
            "reward_loss": reward_loss.item(),
            "policy_loss": policy_loss.item()
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """
        Get model state for saving.
        
        Returns:
            state_dict: Dictionary containing model state
        """
        return {
            "dynamics_model": self.dynamics_model.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "policy": self.policy.state_dict(),
            "dynamics_optimizer": self.dynamics_optimizer.state_dict(),
            "reward_optimizer": self.reward_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std
        }
    
    def load_model_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load model state from state dictionary.
        
        Args:
            state_dict: Dictionary containing model state
        """
        self.dynamics_model.load_state_dict(state_dict["dynamics_model"])
        self.reward_model.load_state_dict(state_dict["reward_model"])
        self.policy.load_state_dict(state_dict["policy"])
        self.dynamics_optimizer.load_state_dict(state_dict["dynamics_optimizer"])
        self.reward_optimizer.load_state_dict(state_dict["reward_optimizer"])
        self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])
        self.state_mean = state_dict.get("state_mean")
        self.state_std = state_dict.get("state_std")
        self.action_mean = state_dict.get("action_mean")
        self.action_std = state_dict.get("action_std")