"""
DeepFlyer Hoop Navigation Environment
Main RL environment for drone hoop navigation training
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
import logging

# Optional ROS imports - environment works without them
try:
    from .safety_layer import BeginnerSafetyLayer
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False
    
try:
    from ..rewards import HoopRewardFunction, HoopRewardConfig
    REWARDS_AVAILABLE = True
except ImportError:
    REWARDS_AVAILABLE = False

try:
    from ..config import DeepFlyerConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeepFlyerEnv(gym.Env):
    """
    DeepFlyer Hoop Navigation Environment - Gymnasium Compliant
    
    A custom Gymnasium environment for training RL agents to navigate drones
    through hoops using ZED camera and YOLO detection.
    
    Observation Space: Dict with vision features and drone state
    Action Space: Continuous 4D velocity commands [vx, vy, vz, yaw_rate]
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 size: int = 5,  # Grid size for simulation
                 max_episode_steps: int = 500,
                 enable_safety: bool = True,
                 config_dict: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize DeepFlyer hoop navigation environment
        
        Args:
            render_mode: Rendering mode ("human", "rgb_array", or None)
            size: Environment size (for simulation bounds)
            max_episode_steps: Maximum steps per episode
            enable_safety: Enable safety constraints
            config_dict: Configuration overrides
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Store parameters
        self.render_mode = render_mode
        self.size = size
        self.max_episode_steps = max_episode_steps
        self.enable_safety = enable_safety
        
        # Load configuration (optional)
        if CONFIG_AVAILABLE:
            self.config = DeepFlyerConfig()
            if config_dict:
                self.config.update(config_dict)
        else:
            self.config = None
        
        # Initialize reward function (optional)
        if REWARDS_AVAILABLE:
            self.reward_function = HoopRewardFunction()
        else:
            self.reward_function = None
        
        # Initialize safety layer (optional)
        if SAFETY_AVAILABLE and enable_safety:
            self.safety_layer = BeginnerSafetyLayer()
        else:
            self.safety_layer = None
        
        # Define observation space (8D as per DeepFlyer spec)
        self.observation_space = gym.spaces.Dict({
            "hoop_x_center_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "hoop_y_center_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "hoop_visible": gym.spaces.Discrete(2),  # 0 or 1
            "hoop_distance_norm": gym.spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "drone_vx_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "drone_vy_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "drone_vz_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "yaw_rate_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
        })
        
        # Define action space (4D velocity commands)
        self.action_space = gym.spaces.Box(
            low=np.array([-2.0, -2.0, -1.0, -1.0], dtype=np.float32),  # [vx, vy, vz, yaw_rate]
            high=np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.current_hoop = 0
        self.hoops_passed = 0
        self.episode_start_time = 0.0
        
        # Hoop navigation state
        self._agent_location = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # [x, y, z]
        self._target_location = np.array([2.0, 2.0, 1.0], dtype=np.float32)  # [x, y, z]
        self._agent_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._yaw_rate = 0.0
        
        # Initialize ROS components if available
        self._init_ros_components()
        
        logger.info("DeepFlyer hoop navigation environment initialized (Gymnasium compliant)")
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset environment for new episode.
        
        Args:
            seed: Seed for random number generator
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional info dict
        """
        # Call parent reset (required by Gymnasium)
        super().reset(seed=seed)
        
        # Reset reward function state (if available)
        if self.reward_function:
            self.reward_function.reset_episode()
        
        # Reset episode state
        self.current_step = 0
        self.current_hoop = 0
        self.hoops_passed = 0
        self.episode_start_time = 0.0
        
        # Reset agent state randomly within bounds
        if seed is not None:
            np.random.seed(seed)
        
        # Random spawn position
        self._agent_location = np.array([
            np.random.uniform(0.5, self.size - 0.5),
            np.random.uniform(0.5, self.size - 0.5),
            1.0  # Fixed altitude
        ], dtype=np.float32)
        
        # Random target position (different from agent)
        while True:
            self._target_location = np.array([
                np.random.uniform(0.5, self.size - 0.5),
                np.random.uniform(0.5, self.size - 0.5),
                1.0
            ], dtype=np.float32)
            if np.linalg.norm(self._target_location - self._agent_location) > 1.0:
                break
        
        # Reset velocity
        self._agent_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._yaw_rate = 0.0
        
        # Get initial observation
        observation = self._get_obs()
        
        # Build info dict
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """
        Execute one environment step.
        
        Args:
            action: 4D velocity command [vx, vy, vz, yaw_rate]
            
        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: Whether episode ended (success/failure)
            truncated: Whether episode ended (time limit)
            info: Additional information
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Apply safety constraints if enabled
        if self.safety_layer:
            # Convert to dict format for safety layer
            state_dict = {
                'position': self._agent_location,
                'velocity': self._agent_velocity,
                'bounds': [0, self.size, 0, self.size, 0.5, 2.0]  # [x_min, x_max, y_min, y_max, z_min, z_max]
            }
            action = self.safety_layer.constrain_action(action, state_dict)
        
        # Update agent state (simple physics simulation)
        dt = 0.05  # 20 Hz
        self._agent_velocity = action[:3].astype(np.float32)
        self._yaw_rate = float(action[3])
        
        # Update position
        self._agent_location += self._agent_velocity * dt
        
        # Apply bounds (keep agent in environment)
        self._agent_location = np.clip(self._agent_location, 
                                     [0.0, 0.0, 0.5], 
                                     [self.size, self.size, 2.0])
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Check if reached target (within 0.3m)
        distance_to_target = np.linalg.norm(self._agent_location - self._target_location)
        if distance_to_target < 0.3:
            terminated = True
            self.hoops_passed += 1
        
        # Check collision with boundaries
        if (self._agent_location[0] <= 0.1 or self._agent_location[0] >= self.size - 0.1 or
            self._agent_location[1] <= 0.1 or self._agent_location[1] >= self.size - 0.1 or
            self._agent_location[2] <= 0.2):
            terminated = True
        
        # Check episode length
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # Get observation
        observation = self._get_obs()
        
        # Calculate reward (use simple reward if reward function not available)
        if self.reward_function:
            state = self._extract_state_for_reward(observation)
            reward = self.reward_function.compute_reward(state, action)
        else:
            # Simple distance-based reward
            reward = -distance_to_target
            if terminated and distance_to_target < 0.3:
                reward += 10.0  # Success bonus
        
        # Build info dict
        info = self._get_info()
        if self.reward_function:
            info['reward_components'] = self.reward_function.component_values
        info['distance_to_target'] = distance_to_target
        info['episode_step'] = self.current_step
        
        return observation, float(reward), terminated, truncated, info
    
    def _get_obs(self):
        """
        Get current observation in the format defined by observation_space.
        
        Returns:
            dict: Observation dictionary with all required keys
        """
        # Calculate hoop visibility and position
        distance_to_target = np.linalg.norm(self._agent_location - self._target_location)
        direction_to_target = self._target_location - self._agent_location
        
        # Normalize direction (simulate camera view)
        if distance_to_target > 0:
            direction_norm = direction_to_target / distance_to_target
        else:
            direction_norm = np.array([0.0, 0.0, 0.0])
        
        # Simulate hoop detection (visible if within range and in front)
        hoop_visible = 1 if distance_to_target < 3.0 else 0
        
        return {
            "hoop_x_center_norm": np.float32(np.clip(direction_norm[0], -1.0, 1.0)),
            "hoop_y_center_norm": np.float32(np.clip(direction_norm[1], -1.0, 1.0)),
            "hoop_visible": hoop_visible,
            "hoop_distance_norm": np.float32(np.clip(distance_to_target / 5.0, 0.0, 1.0)),
            "drone_vx_norm": np.float32(np.clip(self._agent_velocity[0] / 2.0, -1.0, 1.0)),
            "drone_vy_norm": np.float32(np.clip(self._agent_velocity[1] / 2.0, -1.0, 1.0)),
            "drone_vz_norm": np.float32(np.clip(self._agent_velocity[2] / 1.0, -1.0, 1.0)),
            "yaw_rate_norm": np.float32(np.clip(self._yaw_rate, -1.0, 1.0)),
        }
    
    def _get_info(self):
        """
        Get auxiliary information for debugging (not used for learning).
        
        Returns:
            dict: Info dictionary with debugging information
        """
        distance_to_target = np.linalg.norm(self._agent_location - self._target_location)
        
        return {
            "distance_to_target": float(distance_to_target),
            "agent_position": self._agent_location.tolist(),
            "target_position": self._target_location.tolist(),
            "agent_velocity": self._agent_velocity.tolist(),
            "yaw_rate": float(self._yaw_rate),
            "hoops_passed": self.hoops_passed,
            "episode_step": self.current_step,
        }
    
    def _extract_state_for_reward(self, obs):
        """Extract state dict for reward function from observation"""
        return {
            'hoop_visible': obs['hoop_visible'],
            'hoop_x_center_norm': obs['hoop_x_center_norm'],
            'hoop_y_center_norm': obs['hoop_y_center_norm'], 
            'hoop_distance_norm': obs['hoop_distance_norm'],
            'collision': False,  # Would come from physics simulation
            'speed': np.linalg.norm(self._agent_velocity),
            'all_systems_normal': True,
            'hoop_passages_completed': self.hoops_passed,
            'flight_phase': 'NAVIGATE_TO_HOOP',
            'drone_vx_norm': obs['drone_vx_norm'],
            'drone_vy_norm': obs['drone_vy_norm'],
            'drone_vz_norm': obs['drone_vz_norm'],
            'yaw_rate_norm': obs['yaw_rate_norm']
        }
    
    def _init_ros_components(self):
        """Initialize ROS2 components if available (optional for simulation)"""
        try:
            # Try to initialize ROS components for real hardware
            # This is optional - environment works without ROS for simulation
            pass
        except Exception as e:
            logger.debug(f"ROS components not available (simulation mode): {e}")
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            rgb_array if render_mode is "rgb_array", otherwise None
        """
        if self.render_mode == "human":
            # Simple ASCII rendering
            print(f"\n--- DeepFlyer Environment (Step {self.current_step}) ---")
            for y in range(self.size - 1, -1, -1):
                row = ""
                for x in range(self.size):
                    agent_here = (abs(self._agent_location[0] - x) < 0.5 and 
                                abs(self._agent_location[1] - y) < 0.5)
                    target_here = (abs(self._target_location[0] - x) < 0.5 and 
                                 abs(self._target_location[1] - y) < 0.5)
                    
                    if agent_here and target_here:
                        row += "@ "  # Agent at target
                    elif agent_here:
                        row += "A "  # Agent
                    elif target_here:
                        row += "T "  # Target
                    else:
                        row += ". "  # Empty
                print(row)
            
            print(f"Agent: ({self._agent_location[0]:.1f}, {self._agent_location[1]:.1f}, {self._agent_location[2]:.1f})")
            print(f"Target: ({self._target_location[0]:.1f}, {self._target_location[1]:.1f}, {self._target_location[2]:.1f})")
            print(f"Distance: {np.linalg.norm(self._agent_location - self._target_location):.2f}")
            print(f"Velocity: ({self._agent_velocity[0]:.1f}, {self._agent_velocity[1]:.1f}, {self._agent_velocity[2]:.1f})")
            print()
            
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            # For now, return a simple placeholder
            rgb_array = np.zeros((400, 400, 3), dtype=np.uint8)
            return rgb_array
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'ros_node'):
            # Clean up ROS resources if they exist
            pass


def make_deepflyer_env(**kwargs):
    """
    Convenience function to create DeepFlyer environment
    
    Usage:
        env = make_deepflyer_env(enable_safety=True)
    """
    return DeepFlyerEnv(**kwargs)


# Register environment with Gymnasium
gym.register(
    id="DeepFlyer/HoopNavigation-v0",
    entry_point=DeepFlyerEnv,
    max_episode_steps=500,
    kwargs={
        'size': 5,
        'enable_safety': True,
        'render_mode': None
    }
)

# Also register a version with rendering enabled
gym.register(
    id="DeepFlyer/HoopNavigation-v1",
    entry_point=DeepFlyerEnv,
    max_episode_steps=500,
    kwargs={
        'size': 5,
        'enable_safety': True,
        'render_mode': 'human'
    }
)


__all__ = ['DeepFlyerEnv', 'make_deepflyer_env'] 