"""
Training environment for the RL supervisor.

This environment simulates a drone following a path using a PID controller,
with the RL agent adjusting the PID gains.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional, List

from rl_agent.pid_controller import PIDController
from rl_agent.error_calculator import ErrorCalculator
from rl_agent.reward_function import RewardFunction


class SupervisorEnv(gym.Env):
    """
    Environment for training the RL supervisor.
    
    This environment simulates a drone following a path using a PID controller,
    with the RL agent adjusting the PID gains.
    """
    
    def __init__(
        self,
        max_steps: int = 1000,
        dt: float = 0.1,
        path_length: float = 10.0,
        with_disturbance: bool = False,
        disturbance_std: float = 0.1
    ):
        """
        Initialize the environment.
        
        Args:
            max_steps: Maximum number of steps per episode
            dt: Time step
            path_length: Length of the path
            with_disturbance: Whether to add wind disturbance
            disturbance_std: Standard deviation of the wind disturbance
        """
        super().__init__()
        
        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt
        self.path_length = path_length
        self.with_disturbance = with_disturbance
        self.disturbance_std = disturbance_std
        
        # State dimensions
        self.state_dim = 11  # pos(3) + vel(3) + orient(3) + errors(2)
        
        # Define action space (PID gains)
        self.action_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(1,),  # Just P gain for now
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Create controllers and calculators
        self.pid = PIDController()
        self.error_calculator = ErrorCalculator()
        self.reward_function = RewardFunction()
        
        # Initialize state
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        self.cross_track_error = 0.0
        self.heading_error = 0.0
        
        # Path definition
        self.origin = np.array([0.0, 0.0, 1.0])
        self.target = np.array([self.path_length, 0.0, 1.0])
        self.error_calculator.set_path(self.origin.tolist(), self.target.tolist())
        
        # Episode tracking
        self.steps = 0
        self.done = False
        
        # Drone dynamics parameters
        self.mass = 1.0  # kg
        self.max_vel = 2.0  # m/s
        self.max_accel = 1.0  # m/s^2
        self.drag_coeff = 0.1
        
        # Wind disturbance
        self.wind_direction = np.random.uniform(0, 2 * np.pi)
        self.wind_speed = 0.0
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset state
        self.position = self.origin.copy()
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)
        
        # Reset errors
        self.cross_track_error, self.heading_error = self.error_calculator.compute_errors(
            self.position,
            self.orientation[2]
        )
        
        # Reset episode tracking
        self.steps = 0
        self.done = False
        
        # Reset controllers
        self.pid = PIDController()
        self.reward_function.reset()
        
        # Reset wind disturbance
        if self.with_disturbance:
            self.wind_direction = np.random.uniform(0, 2 * np.pi)
            self.wind_speed = np.random.uniform(0.0, 1.0)
        else:
            self.wind_speed = 0.0
        
        # Get observation
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: PID gain
            
        Returns:
            observation: New observation
            reward: Reward
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        # Update PID controller with new gain
        self.pid.update_gains(float(action[0]))
        
        # Compute control using PID
        linear_vel, angular_vel = self.pid.compute_control(
            self.cross_track_error,
            self.heading_error,
            self.dt
        )
        
        # Apply control to update state
        self._update_state(linear_vel, angular_vel)
        
        # Compute errors
        self.cross_track_error, self.heading_error = self.error_calculator.compute_errors(
            self.position,
            self.orientation[2]
        )
        
        # Compute reward
        reward, reward_info = self.reward_function.compute_reward(
            self.cross_track_error,
            self.heading_error,
            linear_vel,
            angular_vel
        )
        
        # Update episode tracking
        self.steps += 1
        
        # Check termination conditions
        progress = self.error_calculator.compute_progress(self.position)
        
        # Terminate if reached the end of the path
        if progress >= 0.99:
            self.done = True
            reward += 10.0  # Bonus for completing the path
        
        # Terminate if too far from the path
        if abs(self.cross_track_error) > 5.0:
            self.done = True
            reward -= 5.0  # Penalty for going too far off the path
        
        # Terminate if max steps reached
        truncated = self.steps >= self.max_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Prepare info
        info = {
            "progress": progress,
            "cross_track_error": self.cross_track_error,
            "heading_error": self.heading_error,
            "pid_gain": self.pid.kp,
            "reward_components": reward_info
        }
        
        return observation, reward, self.done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            observation: Current observation
        """
        return np.concatenate([
            self.position,
            self.velocity,
            self.orientation,
            np.array([self.cross_track_error, self.heading_error])
        ])
    
    def _update_state(self, linear_vel: float, angular_vel: float):
        """
        Update the state based on control inputs.
        
        Args:
            linear_vel: Linear velocity command
            angular_vel: Angular velocity command
        """
        # Update orientation
        self.orientation[2] += angular_vel * self.dt
        # Normalize yaw to [-pi, pi]
        self.orientation[2] = np.arctan2(
            np.sin(self.orientation[2]),
            np.cos(self.orientation[2])
        )
        
        # Compute acceleration from control
        heading_vector = np.array([
            np.cos(self.orientation[2]),
            np.sin(self.orientation[2]),
            0.0
        ])
        
        # Apply linear velocity command
        target_velocity = heading_vector * linear_vel
        
        # Add wind disturbance
        if self.with_disturbance:
            wind_vector = np.array([
                np.cos(self.wind_direction),
                np.sin(self.wind_direction),
                0.0
            ]) * self.wind_speed
            
            # Add random gusts
            if np.random.random() < 0.05:  # 5% chance of a gust
                gust = np.random.normal(0, self.disturbance_std, 3)
                wind_vector += gust
            
            # Apply wind to velocity
            target_velocity += wind_vector
        
        # Simple dynamics: velocity approaches target with some lag
        accel = (target_velocity - self.velocity) * 2.0  # Gain for responsiveness
        
        # Limit acceleration
        accel_norm = np.linalg.norm(accel)
        if accel_norm > self.max_accel:
            accel = accel * (self.max_accel / accel_norm)
        
        # Update velocity
        self.velocity += accel * self.dt
        
        # Apply drag
        self.velocity -= self.velocity * self.drag_coeff * self.dt
        
        # Limit velocity
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > self.max_vel:
            self.velocity = self.velocity * (self.max_vel / vel_norm)
        
        # Update position
        self.position += self.velocity * self.dt 