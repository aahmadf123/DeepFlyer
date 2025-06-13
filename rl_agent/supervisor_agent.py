"""
RL Supervisor Agent for PID control.

This module implements the P3O (Procrastinated Policy-based Observer) algorithm
for tuning PID controllers.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Union, Any, Optional, List
import gymnasium as gym
from collections import deque
import random

from rl_agent.models.base_model import BaseModel
from rl_agent.supervisor_network import SupervisorNetwork
from rl_agent.pid_controller import PIDController


class SupervisorAgent(BaseModel):
    """
    RL Supervisor Agent using P3O algorithm.
    
    This agent observes the current state and errors, and adjusts PID gains
    to improve path following performance using the P3O algorithm, which combines
    the advantages of on-policy and off-policy methods.
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
        clip_ratio: float = 0.2,  # PPO clip parameter
        entropy_coef: float = 0.01,  # Entropy coefficient
        n_updates: int = 10,  # Number of policy updates per learning iteration
        tau: float = 0.005,  # Soft update coefficient for target networks
        alpha: float = 0.1,  # Blend factor for P3O (mixing on-policy and off-policy)
        procrastination_factor: float = 0.95,  # Factor for procrastinated updates
        **kwargs
    ):
        """
        Initialize the P3O supervisor agent.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
            hidden_dim: Hidden dimension size for networks
            lr: Learning rate
            gamma: Discount factor
            batch_size: Batch size for training
            buffer_size: Maximum replay buffer size
            clip_ratio: PPO clip parameter
            entropy_coef: Entropy coefficient to encourage exploration
            n_updates: Number of policy updates per learning iteration
            tau: Soft update coefficient for target networks
            alpha: Blend factor for P3O (mixing on-policy and off-policy)
            procrastination_factor: Factor for procrastinated updates
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
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.n_updates = n_updates
        self.tau = tau
        self.alpha = alpha
        self.procrastination_factor = procrastination_factor
        
        # Create networks
        self.actor = SupervisorNetwork(
            self.state_dim,
            hidden_dim
        ).to(self.device)
        
        # On-policy critic (for PPO style updates)
        self.critic_on = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Off-policy critic (for SAC style updates)
        self.critic_off = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Target networks for off-policy learning
        self.critic_off_target = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(self.device)
        
        # Initialize target network weights to match main networks
        self.critic_off_target.load_state_dict(self.critic_off.state_dict())
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_on_optimizer = optim.Adam(self.critic_on.parameters(), lr=lr)
        self.critic_off_optimizer = optim.Adam(self.critic_off.parameters(), lr=lr)
        
        # Initialize replay buffers
        self.on_policy_buffer = []  # For on-policy updates (cleared after each update)
        self.off_policy_buffer = deque(maxlen=buffer_size)  # For off-policy updates
        
        # Create PID controller
        self.pid = PIDController()
        
        # Training metrics
        self.actor_losses = []
        self.critic_on_losses = []
        self.critic_off_losses = []
        self.entropy_values = []
        
        # Procrastination counter
        self.steps_since_on_policy_update = 0
    
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
        
        # Scale gain to action space range
        if isinstance(self.action_space, gym.spaces.Box):
            low = self.action_space.low
            high = self.action_space.high
            # Rescale from [0,1] to [low, high]
            gain = gain.cpu().numpy() * (high - low) + low
        
        # Update PID controller
        self.pid.update_gains(gain.item())
        
        return gain, None
    
    def add_to_buffer(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        next_state: np.ndarray, 
        reward: float, 
        done: bool
    ) -> None:
        """
        Add a transition to both replay buffers.
        
        Args:
            state: Current state
            action: Action taken (PID gain)
            next_state: Next state
            reward: Reward received
            done: Whether the episode is done
        """
        # Add to off-policy buffer for SAC-style updates
        self.off_policy_buffer.append((state, action, next_state, reward, done))
        
        # Add to on-policy buffer for PPO-style updates
        # Include log probabilities for PPO importance sampling
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            _, log_prob = self.actor.sample(state_tensor, deterministic=False)
            value_on = self.critic_on(state_tensor).item()
            value_off = self.critic_off(state_tensor).item()
            
        self.on_policy_buffer.append(
            (state, action, next_state, reward, done, log_prob.item(), value_on, value_off)
        )
        
        # Increment procrastination counter
        self.steps_since_on_policy_update += 1
    
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
            "critic_on_loss": self.critic_on_losses[-1] if self.critic_on_losses else None,
            "critic_off_loss": self.critic_off_losses[-1] if self.critic_off_losses else None,
            "entropy": self.entropy_values[-1] if self.entropy_values else None
        }
    
    def learn(self, batch_size: int = None) -> Dict[str, float]:
        """
        Perform one iteration of learning using P3O.
        
        Args:
            batch_size: Size of the batch to learn from
            
        Returns:
            metrics: Dictionary of learning metrics
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        metrics = {}
        
        # Check if we have enough data for off-policy updates
        if len(self.off_policy_buffer) >= batch_size:
            off_policy_metrics = self._learn_off_policy(batch_size)
            metrics.update(off_policy_metrics)
        
        # Determine if we should do an on-policy update based on procrastination factor
        do_on_policy_update = (
            len(self.on_policy_buffer) > 0 and 
            (random.random() < (1 - self.procrastination_factor ** self.steps_since_on_policy_update))
        )
        
        if do_on_policy_update:
            on_policy_metrics = self._learn_on_policy()
            metrics.update(on_policy_metrics)
            
            # Reset procrastination counter
            self.steps_since_on_policy_update = 0
            
            # Clear on-policy buffer after update
            self.on_policy_buffer = []
        
        return metrics
    
    def _learn_on_policy(self) -> Dict[str, float]:
        """
        Perform on-policy updates (PPO-style).
        
        Returns:
            metrics: Dictionary of on-policy learning metrics
        """
        if not self.on_policy_buffer:
            return {"on_policy_loss": float('nan')}
            
        # Extract data from on-policy buffer
        states, actions, next_states, rewards, dones, old_log_probs, values_on, values_off = zip(*self.on_policy_buffer)
        
        # Convert to tensors
        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.vstack(rewards)).to(self.device)
        dones = torch.FloatTensor(np.vstack(dones)).to(self.device)
        old_log_probs = torch.FloatTensor(np.vstack(old_log_probs)).to(self.device)
        
        # Compute returns and advantages
        with torch.no_grad():
            next_values = self.critic_on(next_states)
            returns = rewards + self.gamma * next_values * (1 - dones)
            values = self.critic_on(states)
            advantages = returns - values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Perform multiple epochs of updates
        for _ in range(self.n_updates):
            # Get current action probabilities
            _, new_log_probs = self.actor.evaluate(states, actions)
            
            # Compute ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO objectives
            obj1 = ratio * advantages
            obj2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Policy loss
            policy_loss = -torch.min(obj1, obj2).mean()
            
            # Entropy bonus for exploration
            entropy = self.actor.entropy(states).mean()
            policy_loss = policy_loss - self.entropy_coef * entropy
            
            # Update actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            # Update on-policy critic
            self.critic_on_optimizer.zero_grad()
            value_pred = self.critic_on(states)
            value_loss = F.mse_loss(value_pred, returns)
            value_loss.backward()
            self.critic_on_optimizer.step()
        
        # Record metrics
        self.actor_losses.append(policy_loss.item())
        self.critic_on_losses.append(value_loss.item())
        self.entropy_values.append(entropy.item())
        
        return {
            "on_policy_loss": policy_loss.item(),
            "on_policy_value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }
    
    def _learn_off_policy(self, batch_size: int) -> Dict[str, float]:
        """
        Perform off-policy updates (SAC-style).
        
        Args:
            batch_size: Size of the batch to learn from
            
        Returns:
            metrics: Dictionary of off-policy learning metrics
        """
        # Sample batch from replay buffer
        batch = random.sample(self.off_policy_buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.vstack(states)).to(self.device)
        next_states = torch.FloatTensor(np.vstack(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.vstack(rewards)).to(self.device)
        dones = torch.FloatTensor(np.vstack(dones)).to(self.device)
        actions = torch.FloatTensor(np.vstack(actions)).to(self.device)
        
        # Update off-policy critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q_target = self.critic_off_target(next_states)
            next_q_value = rewards + self.gamma * next_q_target * (1 - dones)
        
        # Current Q value
        current_q_value = self.critic_off(states)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q_value, next_q_value)
        
        # Update off-policy critic
        self.critic_off_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_off_optimizer.step()
        
        # Soft update target networks
        for target_param, param in zip(self.critic_off_target.parameters(), self.critic_off.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        
        # Record metrics
        self.critic_off_losses.append(critic_loss.item())
        
        return {
            "off_policy_value_loss": critic_loss.item()
        }
    
    def get_model_state(self) -> Dict[str, Any]:
        """
        Get model state for saving.
        
        Returns:
            state_dict: Dictionary containing model state
        """
        return {
            "actor": self.actor.state_dict(),
            "critic_on": self.critic_on.state_dict(),
            "critic_off": self.critic_off.state_dict(),
            "critic_off_target": self.critic_off_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_on_optimizer": self.critic_on_optimizer.state_dict(),
            "critic_off_optimizer": self.critic_off_optimizer.state_dict(),
            "pid": {
                "kp": self.pid.kp,
                "ki": self.pid.ki,
                "kd": self.pid.kd
            },
            "steps_since_on_policy_update": self.steps_since_on_policy_update
        }
    
    def load_model_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load model state from state dictionary.
        
        Args:
            state_dict: Dictionary containing model state
        """
        self.actor.load_state_dict(state_dict["actor"])
        self.critic_on.load_state_dict(state_dict["critic_on"])
        self.critic_off.load_state_dict(state_dict["critic_off"])
        self.critic_off_target.load_state_dict(state_dict["critic_off_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_on_optimizer.load_state_dict(state_dict["critic_on_optimizer"])
        self.critic_off_optimizer.load_state_dict(state_dict["critic_off_optimizer"])
        
        # Update PID gains
        pid_state = state_dict["pid"]
        self.pid.update_gains(
            pid_state["kp"],
            pid_state["ki"],
            pid_state["kd"]
        )
        
        # Load procrastination counter
        if "steps_since_on_policy_update" in state_dict:
            self.steps_since_on_policy_update = state_dict["steps_since_on_policy_update"] 