"""
Direct Control Agent for DeepFlyer
Maps P3O actions directly to PX4 velocity commands
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from .algorithms.p3o import P3O, P3OConfig
from .algorithms.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class DirectControlConfig:
    """Configuration for direct control"""
    # Observation space (8D: hoop_x, hoop_y, hoop_visible, hoop_distance, vx, vy, vz, yaw_rate)
    obs_dim: int = 8
    
    # Action space (4D: vx, vy, vz, yaw_rate)
    action_dim: int = 4
    
    # Control limits (m/s and rad/s)
    max_velocity: float = 2.0  # m/s
    max_yaw_rate: float = 1.0  # rad/s
    
    # Safety constraints
    min_altitude: float = 0.3  # meters
    max_altitude: float = 3.0  # meters
    geofence_radius: float = 5.0  # meters
    
    # ZED Mini parameters
    camera_fov: float = 90.0  # degrees
    max_detection_range: float = 10.0  # meters
    
    # Control smoothing
    action_smoothing: float = 0.8  # Low-pass filter coefficient


class DirectControlAgent:
    """Direct RL control agent for drone racing"""
    
    def __init__(self, config: Optional[DirectControlConfig] = None, 
                 p3o_config: Optional[P3OConfig] = None):
        self.config = config or DirectControlConfig()
        self.p3o_config = p3o_config or P3OConfig()
        
        # Initialize P3O agent
        self.p3o = P3O(
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            config=self.p3o_config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            buffer_size=10000
        )
        
        # Previous action for smoothing
        self.prev_action = np.zeros(self.config.action_dim)
        
        # Episode tracking
        self.episode_obs = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_step = 0
        
        logger.info("DirectControlAgent initialized for drone racing")
    
    def process_observation(self, vision_data: Dict, drone_state: Dict) -> np.ndarray:
        """
        Process raw sensor data into 8D observation vector
        
        Args:
            vision_data: Dictionary with ZED Mini + YOLO detection
                - hoop_detected: bool
                - hoop_center_x: float [-1, 1] normalized
                - hoop_center_y: float [-1, 1] normalized
                - hoop_distance: float [0, 1] normalized
            drone_state: Dictionary with drone telemetry
                - velocity: [vx, vy, vz] in m/s
                - yaw_rate: rad/s
        
        Returns:
            8D observation vector
        """
        obs = np.zeros(self.config.obs_dim, dtype=np.float32)
        
        # Hoop detection features (4D)
        if vision_data.get('hoop_detected', False):
            obs[0] = vision_data['hoop_center_x']  # Horizontal offset
            obs[1] = vision_data['hoop_center_y']  # Vertical offset
            obs[2] = 1.0  # Hoop visible flag
            obs[3] = vision_data['hoop_distance']  # Normalized distance
        else:
            obs[0] = 0.0
            obs[1] = 0.0
            obs[2] = 0.0  # Hoop not visible
            obs[3] = 1.0  # Max distance
        
        # Drone velocity features (4D)
        velocity = drone_state.get('velocity', [0, 0, 0])
        obs[4] = np.clip(velocity[0] / self.config.max_velocity, -1, 1)  # vx normalized
        obs[5] = np.clip(velocity[1] / self.config.max_velocity, -1, 1)  # vy normalized
        obs[6] = np.clip(velocity[2] / self.config.max_velocity, -1, 1)  # vz normalized
        obs[7] = np.clip(drone_state.get('yaw_rate', 0) / self.config.max_yaw_rate, -1, 1)
        
        return obs
    
    def get_action(self, observation: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Get action from P3O agent with direct velocity mapping
        
        Args:
            observation: 8D observation vector
            training: Whether in training mode (adds exploration noise)
        
        Returns:
            4D action vector [vx, vy, vz, yaw_rate] in SI units
        """
        # Get normalized action from P3O [-1, 1]
        raw_action = self.p3o.predict(observation, deterministic=not training)
        
        # Add exploration noise if training
        if training:
            noise = np.random.normal(0, self.config.action_smoothing * 0.1, size=self.config.action_dim)
            raw_action = np.clip(raw_action + noise, -1, 1)
        
        # Apply action smoothing
        alpha = self.config.action_smoothing
        smoothed_action = alpha * self.prev_action + (1 - alpha) * raw_action
        self.prev_action = smoothed_action.copy()
        
        # Map to velocity commands
        action = np.zeros(self.config.action_dim)
        action[0] = smoothed_action[0] * self.config.max_velocity  # vx (forward/back)
        action[1] = smoothed_action[1] * self.config.max_velocity  # vy (left/right)
        action[2] = smoothed_action[2] * self.config.max_velocity  # vz (up/down)
        action[3] = smoothed_action[3] * self.config.max_yaw_rate  # yaw_rate
        
        return action
    
    def apply_safety_constraints(self, action: np.ndarray, drone_state: Dict) -> np.ndarray:
        """
        Apply safety constraints to action
        
        Args:
            action: 4D action vector [vx, vy, vz, yaw_rate]
            drone_state: Current drone state with position
        
        Returns:
            Safe action vector
        """
        safe_action = action.copy()
        position = drone_state.get('position', [0, 0, 0])
        
        # Altitude constraints
        if position[2] < self.config.min_altitude and safe_action[2] < 0:
            safe_action[2] = max(0, safe_action[2])  # Prevent going lower
        elif position[2] > self.config.max_altitude and safe_action[2] > 0:
            safe_action[2] = min(0, safe_action[2])  # Prevent going higher
        
        # Geofence constraints (circular)
        horizontal_dist = np.sqrt(position[0]**2 + position[1]**2)
        if horizontal_dist > self.config.geofence_radius * 0.9:
            # Apply inward force
            direction_to_center = -np.array([position[0], position[1]]) / (horizontal_dist + 1e-6)
            safe_action[0] = safe_action[0] * 0.5 + direction_to_center[0] * self.config.max_velocity * 0.5
            safe_action[1] = safe_action[1] * 0.5 + direction_to_center[1] * self.config.max_velocity * 0.5
        
        return safe_action
    
    def store_experience(self, obs: np.ndarray, action: np.ndarray, 
                        next_obs: np.ndarray, reward: float, done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.add(obs, action, reward, next_obs, done)
        
        # Track episode data
        self.episode_obs.append(obs)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_step += 1
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform one training step if enough data available"""
        if len(self.replay_buffer) < self.p3o_config.batch_size:
            return None
        
        # Update P3O policy
        stats = self.p3o.update(self.replay_buffer)
        return stats
    
    def reset_episode(self):
        """Reset episode tracking"""
        self.episode_obs = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_step = 0
        self.prev_action = np.zeros(self.config.action_dim)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics"""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'total_reward': sum(self.episode_rewards),
            'episode_length': self.episode_step,
            'avg_reward': np.mean(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards)
        }
    
    def save(self, path: str):
        """Save agent model"""
        self.p3o.save(path)
    
    def load(self, path: str):
        """Load agent model"""
        self.p3o.load(path)