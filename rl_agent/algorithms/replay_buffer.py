"""
Replay Buffer for P3O Training
Efficient experience storage and sampling
"""

import numpy as np
from typing import Dict, Tuple
import random


class ReplayBuffer:
    """Experience replay buffer for P3O training"""
    
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int = 10000):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        
        # Preallocate arrays for efficiency
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of experiences"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'obs': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_obs': self.next_observations[indices],
            'dones': self.dones[indices]
        }
    
    def sample_trajectory(self, length: int) -> Dict[str, np.ndarray]:
        """Sample a continuous trajectory"""
        if self.size < length:
            return self.sample(self.size)
        
        start_idx = np.random.randint(0, self.size - length)
        indices = np.arange(start_idx, start_idx + length)
        
        return {
            'obs': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_obs': self.next_observations[indices],
            'dones': self.dones[indices]
        }
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if self.size == 0:
            return {}
        
        return {
            'buffer_size': self.size,
            'buffer_capacity': self.buffer_size,
            'avg_reward': np.mean(self.rewards[:self.size]),
            'std_reward': np.std(self.rewards[:self.size]),
            'max_reward': np.max(self.rewards[:self.size]),
            'min_reward': np.min(self.rewards[:self.size]),
            'done_ratio': np.mean(self.dones[:self.size])
        }