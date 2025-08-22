"""
Enhanced Replay Buffer for P3O Training
Efficient experience storage with GAE and trajectory support for DeepRacer-style drone RL
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class P3OReplayBuffer:
    """Enhanced experience replay buffer for P3O training with GAE support"""
    
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int = 10000, 
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Preallocate arrays for efficiency
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        
        # GAE-specific arrays
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        # Episode tracking
        self.episode_starts = np.zeros(buffer_size, dtype=np.float32)
        self.episode_lengths = []
        self.episode_rewards = []
        
        self.ptr = 0
        self.size = 0
        self.episode_start_ptr = 0
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool, value: float = 0.0, 
            log_prob: float = 0.0):
        """Add experience to buffer with P3O-specific data"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        
        # Track episode start
        if self.ptr == 0 or self.dones[(self.ptr - 1) % self.buffer_size]:
            self.episode_starts[self.ptr] = 1.0
            self.episode_start_ptr = self.ptr
        else:
            self.episode_starts[self.ptr] = 0.0
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
        
        # Complete episode statistics
        if done:
            episode_length = self.ptr - self.episode_start_ptr
            episode_reward = np.sum(self.rewards[self.episode_start_ptr:self.ptr])
            self.episode_lengths.append(episode_length)
            self.episode_rewards.append(episode_reward)
            
    def compute_gae(self, next_value: float = 0.0):
        """
        Compute Generalized Advantage Estimation (GAE) for all stored experiences
        
        Args:
            next_value: Value estimate for the state after the last stored state
        """
        # Find all complete episodes in buffer
        episode_boundaries = []
        for i in range(self.size):
            if self.episode_starts[i] == 1.0:
                episode_boundaries.append(i)
        
        # Add final boundary
        if episode_boundaries and episode_boundaries[-1] < self.size - 1:
            episode_boundaries.append(self.size)
        
        # Compute GAE for each episode
        for i in range(len(episode_boundaries) - 1):
            start_idx = episode_boundaries[i]
            end_idx = episode_boundaries[i + 1]
            
            # Compute advantages using GAE
            advantages = np.zeros(end_idx - start_idx)
            last_gae_lam = 0
            
            # Work backwards through the episode
            for t in reversed(range(end_idx - start_idx)):
                idx = start_idx + t
                
                if t == end_idx - start_idx - 1:  # Last step
                    next_non_terminal = 1.0 - self.dones[idx]
                    next_values = next_value if idx == self.size - 1 else self.values[idx + 1]
                else:
                    next_non_terminal = 1.0 - self.dones[idx]
                    next_values = self.values[idx + 1]
                
                # Temporal difference error
                td_error = (self.rewards[idx] + self.gamma * next_values * next_non_terminal - 
                           self.values[idx])
                
                # GAE calculation
                advantages[t] = last_gae_lam = (td_error + 
                                               self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam)
            
            # Store computed advantages
            self.advantages[start_idx:end_idx] = advantages
            
            # Compute returns (advantages + values)
            self.returns[start_idx:end_idx] = advantages + self.values[start_idx:end_idx]
        
        # Normalize advantages
        if self.size > 1:
            mean_adv = np.mean(self.advantages[:self.size])
            std_adv = np.std(self.advantages[:self.size])
            if std_adv > 1e-8:
                self.advantages[:self.size] = (self.advantages[:self.size] - mean_adv) / std_adv
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch of experiences with P3O data"""
        if batch_size > self.size:
            batch_size = self.size
            
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'obs': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_obs': self.next_observations[indices],
            'dones': self.dones[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices]
        }
    
    def get_all_data(self) -> Dict[str, np.ndarray]:
        """Get all stored data for P3O training"""
        return {
            'obs': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'rewards': self.rewards[:self.size],
            'next_obs': self.next_observations[:self.size],
            'dones': self.dones[:self.size],
            'values': self.values[:self.size],
            'log_probs': self.log_probs[:self.size],
            'advantages': self.advantages[:self.size],
            'returns': self.returns[:self.size]
        }
    
    def sample_trajectory(self, length: int) -> Dict[str, np.ndarray]:
        """Sample a continuous trajectory"""
        if self.size < length:
            return self.get_all_data()
        
        start_idx = np.random.randint(0, self.size - length)
        indices = np.arange(start_idx, start_idx + length)
        
        return {
            'obs': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_obs': self.next_observations[indices],
            'dones': self.dones[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices]
        }
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive buffer statistics"""
        if self.size == 0:
            return {}
        
        stats = {
            'buffer_size': self.size,
            'buffer_capacity': self.buffer_size,
            'avg_reward': np.mean(self.rewards[:self.size]),
            'std_reward': np.std(self.rewards[:self.size]),
            'max_reward': np.max(self.rewards[:self.size]),
            'min_reward': np.min(self.rewards[:self.size]),
            'done_ratio': np.mean(self.dones[:self.size]),
            'avg_value': np.mean(self.values[:self.size]),
            'avg_advantage': np.mean(self.advantages[:self.size]),
            'std_advantage': np.std(self.advantages[:self.size]),
            'num_episodes': len(self.episode_rewards)
        }
        
        # Episode statistics
        if self.episode_rewards:
            stats.update({
                'avg_episode_reward': np.mean(self.episode_rewards),
                'std_episode_reward': np.std(self.episode_rewards),
                'max_episode_reward': np.max(self.episode_rewards),
                'min_episode_reward': np.min(self.episode_rewards),
                'avg_episode_length': np.mean(self.episode_lengths),
                'std_episode_length': np.std(self.episode_lengths)
            })
        
        return stats
    

# Legacy ReplayBuffer class for backwards compatibility
class ReplayBuffer(P3OReplayBuffer):
    """Legacy replay buffer - use P3OReplayBuffer for new implementations"""
    
    def __init__(self, obs_dim: int, action_dim: int, buffer_size: int = 10000):
        super().__init__(obs_dim, action_dim, buffer_size)
        logger.warning("Using legacy ReplayBuffer. Consider upgrading to P3OReplayBuffer for better P3O support.")