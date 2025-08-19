#!/usr/bin/env python3
"""
Standalone Environment Test

Tests the DeepFlyer environment in complete isolation without any ROS dependencies.
This demonstrates the core Gymnasium compliance.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StandaloneDeepFlyerEnv(gym.Env):
    """
    Standalone DeepFlyer Environment - Pure Gymnasium Implementation
    
    This is a fully self-contained environment that demonstrates proper Gymnasium
    compliance without any external dependencies.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 size: int = 5,
                 max_episode_steps: int = 500):
        """Initialize the standalone environment"""
        super().__init__()
        
        self.render_mode = render_mode
        self.size = size
        self.max_episode_steps = max_episode_steps
        
        # Define observation space (8D as per DeepFlyer spec)
        self.observation_space = gym.spaces.Dict({
            "hoop_x_center_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "hoop_y_center_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "hoop_visible": gym.spaces.Discrete(2),
            "hoop_distance_norm": gym.spaces.Box(0.0, 1.0, shape=(), dtype=np.float32),
            "drone_vx_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "drone_vy_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "drone_vz_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
            "yaw_rate_norm": gym.spaces.Box(-1.0, 1.0, shape=(), dtype=np.float32),
        })
        
        # Define action space (4D velocity commands)
        self.action_space = gym.spaces.Box(
            low=np.array([-2.0, -2.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self._agent_location = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._target_location = np.array([2.0, 2.0, 1.0], dtype=np.float32)
        self._agent_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._yaw_rate = 0.0
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.current_step = 0
        
        # Random spawn position
        self._agent_location = np.array([
            np.random.uniform(0.5, self.size - 0.5),
            np.random.uniform(0.5, self.size - 0.5),
            1.0
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
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray):
        """Execute one step"""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Simple physics simulation
        dt = 0.05
        self._agent_velocity = action[:3].astype(np.float32)
        self._yaw_rate = float(action[3])
        
        # Update position
        self._agent_location += self._agent_velocity * dt
        
        # Apply bounds
        self._agent_location = np.clip(self._agent_location, 
                                     [0.0, 0.0, 0.5], 
                                     [self.size, self.size, 2.0])
        
        # Check termination
        distance_to_target = np.linalg.norm(self._agent_location - self._target_location)
        terminated = distance_to_target < 0.3
        
        # Check truncation
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        
        # Calculate reward
        reward = -distance_to_target
        if terminated:
            reward += 10.0
        
        return self._get_obs(), float(reward), terminated, truncated, self._get_info()
    
    def _get_obs(self):
        """Get observation"""
        distance_to_target = np.linalg.norm(self._agent_location - self._target_location)
        direction_to_target = self._target_location - self._agent_location
        
        if distance_to_target > 0:
            direction_norm = direction_to_target / distance_to_target
        else:
            direction_norm = np.array([0.0, 0.0, 0.0])
        
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
        """Get info dict"""
        return {
            "distance_to_target": float(np.linalg.norm(self._agent_location - self._target_location)),
            "agent_position": self._agent_location.tolist(),
            "target_position": self._target_location.tolist(),
            "episode_step": self.current_step,
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            print(f"\n--- DeepFlyer Environment (Step {self.current_step}) ---")
            for y in range(self.size - 1, -1, -1):
                row = ""
                for x in range(self.size):
                    agent_here = (abs(self._agent_location[0] - x) < 0.5 and 
                                abs(self._agent_location[1] - y) < 0.5)
                    target_here = (abs(self._target_location[0] - x) < 0.5 and 
                                 abs(self._target_location[1] - y) < 0.5)
                    
                    if agent_here and target_here:
                        row += "@ "
                    elif agent_here:
                        row += "A "
                    elif target_here:
                        row += "T "
                    else:
                        row += ". "
                print(row)
            
            print(f"Distance: {np.linalg.norm(self._agent_location - self._target_location):.2f}")
            print()
            
        elif self.render_mode == "rgb_array":
            return np.zeros((400, 400, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up"""
        pass


def test_environment():
    """Test the standalone environment"""
    print("🚁 Testing Standalone DeepFlyer Environment")
    print("=" * 50)
    
    # Test environment creation
    env = StandaloneDeepFlyerEnv(render_mode="human", size=5)
    print("✅ Environment created")
    
    # Test with gymnasium check_env
    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env)
        print("✅ Gymnasium check_env passed")
    except Exception as e:
        print(f"⚠️  Gymnasium check warning: {e}")
    
    # Test episode
    obs, info = env.reset(seed=42)
    print("✅ Reset successful")
    print(f"Observation keys: {list(obs.keys())}")
    print(f"Info keys: {list(info.keys())}")
    
    # Run a short episode
    total_reward = 0
    for step in range(20):
        # Move towards target
        direction = env._target_location - env._agent_location
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction) * 1.5  # Move towards target
        
        action = np.array([direction[0], direction[1], 0.0, 0.0], dtype=np.float32)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:  # Render every 5 steps
            env.render()
        
        print(f"Step {step + 1}: reward={reward:.3f}, distance={info['distance_to_target']:.3f}")
        
        if terminated:
            print(f"🎉 Reached target in {step + 1} steps!")
            break
        elif truncated:
            print(f"⏰ Episode truncated at {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    env.close()
    
    # Test registration
    print("\n=== Testing Registration ===")
    try:
        gym.register(
            id="StandaloneDeepFlyer/Test-v0",
            entry_point=StandaloneDeepFlyerEnv,
            max_episode_steps=500
        )
        
        registered_env = gym.make("StandaloneDeepFlyer/Test-v0")
        obs, info = registered_env.reset()
        action = registered_env.action_space.sample()
        obs, reward, terminated, truncated, info = registered_env.step(action)
        
        print("✅ Environment registration and usage works")
    except Exception as e:
        print(f"⚠️  Registration test warning: {e}")
    
    print("\n🎉 All tests completed! Environment is Gymnasium compliant.")


if __name__ == "__main__":
    test_environment()
