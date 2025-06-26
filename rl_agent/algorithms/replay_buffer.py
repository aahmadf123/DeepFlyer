"""
Replay Buffer for DeepFlyer RL
Efficient storage and sampling of experience tuples
"""

import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience replay buffer for RL training"""
    
    def __init__(self, 
                 capacity: int = 100000,
                 observation_shape: Tuple[int, ...] = None,
                 action_shape: Tuple[int, ...] = None):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            observation_shape: Shape of observations
            action_shape: Shape of actions
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        
        # Storage for experiences
        self.observations = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_observations = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.log_probs = deque(maxlen=capacity)
        self.values = deque(maxlen=capacity)
        
        # Additional info storage
        self.info = deque(maxlen=capacity)
        
        self.size = 0
        
        logger.info(f"ReplayBuffer initialized with capacity: {capacity}")
    
    def add(self, 
            observation: np.ndarray,
            action: np.ndarray, 
            reward: float,
            next_observation: np.ndarray,
            done: bool,
            log_prob: float = 0.0,
            value: float = 0.0,
            info: Optional[Dict[str, Any]] = None) -> None:
        """
        Add experience to buffer
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Episode termination flag
            log_prob: Log probability of action (for on-policy methods)
            value: State value estimate
            info: Additional information
        """
        self.observations.append(observation.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.next_observations.append(next_observation.copy())
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.info.append(info or {})
        
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            batch: Dictionary containing batch data
        """
        if batch_size > self.size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {self.size}")
        
        # Sample random indices
        indices = random.sample(range(self.size), batch_size)
        
        # Extract batch data
        batch = {
            'observations': np.array([self.observations[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'next_observations': np.array([self.next_observations[i] for i in indices]),
            'dones': np.array([self.dones[i] for i in indices]),
            'log_probs': np.array([self.log_probs[i] for i in indices]),
            'values': np.array([self.values[i] for i in indices])
        }
        
        return batch
    
    def sample_recent(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample most recent experiences (for on-policy methods)
        
        Args:
            batch_size: Number of recent experiences to sample
            
        Returns:
            batch: Dictionary containing recent batch data
        """
        if batch_size > self.size:
            batch_size = self.size
        
        # Get most recent experiences
        start_idx = max(0, self.size - batch_size)
        indices = list(range(start_idx, self.size))
        
        batch = {
            'observations': np.array([self.observations[i] for i in indices]),
            'actions': np.array([self.actions[i] for i in indices]),
            'rewards': np.array([self.rewards[i] for i in indices]),
            'next_observations': np.array([self.next_observations[i] for i in indices]),
            'dones': np.array([self.dones[i] for i in indices]),
            'log_probs': np.array([self.log_probs[i] for i in indices]),
            'values': np.array([self.values[i] for i in indices])
        }
        
        return batch
    
    def clear(self) -> None:
        """Clear all experiences from buffer"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.info.clear()
        self.size = 0
    
    def __len__(self) -> int:
        """Get current buffer size"""
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training"""
        return self.size >= min_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if self.size == 0:
            return {'size': 0, 'capacity': self.capacity}
        
        rewards = [self.rewards[i] for i in range(self.size)]
        
        stats = {
            'size': self.size,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'done_ratio': np.mean([self.dones[i] for i in range(self.size)])
        }
        
        return stats


class EpisodeBuffer:
    """Buffer for storing complete episodes"""
    
    def __init__(self, max_episodes: int = 1000):
        """
        Initialize episode buffer
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)
        
        logger.info(f"EpisodeBuffer initialized with max_episodes: {max_episodes}")
    
    def add_episode(self, 
                   observations: List[np.ndarray],
                   actions: List[np.ndarray],
                   rewards: List[float],
                   infos: List[Dict[str, Any]]) -> None:
        """
        Add complete episode to buffer
        
        Args:
            observations: List of observations in episode
            actions: List of actions in episode
            rewards: List of rewards in episode
            infos: List of info dicts in episode
        """
        episode = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'infos': infos,
            'length': len(observations),
            'total_reward': sum(rewards),
            'timestamp': np.datetime64('now')
        }
        
        self.episodes.append(episode)
    
    def get_recent_episodes(self, n_episodes: int) -> List[Dict[str, Any]]:
        """Get n most recent episodes"""
        n_episodes = min(n_episodes, len(self.episodes))
        return list(self.episodes)[-n_episodes:]
    
    def get_best_episodes(self, n_episodes: int) -> List[Dict[str, Any]]:
        """Get n episodes with highest total reward"""
        if len(self.episodes) == 0:
            return []
        
        sorted_episodes = sorted(self.episodes, key=lambda ep: ep['total_reward'], reverse=True)
        n_episodes = min(n_episodes, len(sorted_episodes))
        return sorted_episodes[:n_episodes]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get episode statistics"""
        if len(self.episodes) == 0:
            return {'num_episodes': 0}
        
        rewards = [ep['total_reward'] for ep in self.episodes]
        lengths = [ep['length'] for ep in self.episodes]
        
        stats = {
            'num_episodes': len(self.episodes),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths)
        }
        
        return stats
    
    def clear(self) -> None:
        """Clear all episodes"""
        self.episodes.clear()
    
    def __len__(self) -> int:
        """Get number of stored episodes"""
        return len(self.episodes)


__all__ = ['ReplayBuffer', 'EpisodeBuffer'] 