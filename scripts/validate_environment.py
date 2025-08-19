#!/usr/bin/env python3
"""
Environment Validation Script for DeepFlyer

Tests the environment against Gymnasium standards and runs basic functionality checks.
This follows the debugging guidelines from the official Gymnasium documentation.
"""

import gymnasium as gym
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_agent.env import DeepFlyerEnv


def test_environment_validity():
    """Test environment with gymnasium's check_env"""
    print("=== Testing Environment Validity ===")
    
    try:
        from gymnasium.utils.env_checker import check_env
        
        # Test basic environment
        env = DeepFlyerEnv()
        check_env(env)
        print("✅ Environment passes all Gymnasium checks!")
        
        # Test registered environment
        env_registered = gym.make("DeepFlyer/HoopNavigation-v0")
        check_env(env_registered)
        print("✅ Registered environment passes all checks!")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False


def test_manual_episode():
    """Test a manual episode with known actions"""
    print("\n=== Testing Manual Episode ===")
    
    try:
        env = gym.make("DeepFlyer/HoopNavigation-v1", render_mode="human")  # With rendering
        
        # Test reset with seed for reproducibility
        obs, info = env.reset(seed=42)
        print(f"✅ Reset successful")
        print(f"Observation keys: {list(obs.keys())}")
        print(f"Info keys: {list(info.keys())}")
        
        # Validate observation space
        assert env.observation_space.contains(obs), "Observation not in observation space!"
        print("✅ Observation space validation passed")
        
        # Test each action dimension
        actions_to_test = [
            np.array([1.0, 0.0, 0.0, 0.0]),    # Move forward
            np.array([0.0, 1.0, 0.0, 0.0]),    # Move right  
            np.array([0.0, 0.0, 1.0, 0.0]),    # Move up
            np.array([0.0, 0.0, 0.0, 1.0]),    # Yaw
            np.array([-1.0, -1.0, -0.5, -0.5]) # Combined negative
        ]
        
        print(f"\n🎮 Testing {len(actions_to_test)} different actions:")
        
        for i, action in enumerate(actions_to_test):
            # Validate action space
            assert env.action_space.contains(action), f"Action {action} not in action space!"
            
            old_pos = obs['agent_position'] if 'agent_position' in info else "unknown"
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = info.get('agent_position', "unknown")
            
            print(f"  Action {i+1}: {action} -> reward={reward:.3f}, terminated={terminated}")
            print(f"    Position: {old_pos} -> {new_pos}")
            print(f"    Distance to target: {info.get('distance_to_target', 'unknown'):.3f}")
            
            # Validate return types
            assert isinstance(reward, (float, int)), f"Reward must be numeric, got {type(reward)}"
            assert isinstance(terminated, bool), f"Terminated must be bool, got {type(terminated)}"
            assert isinstance(truncated, bool), f"Truncated must be bool, got {type(truncated)}"
            assert isinstance(info, dict), f"Info must be dict, got {type(info)}"
            
            if terminated or truncated:
                print("  📍 Episode ended, resetting...")
                obs, info = env.reset()
                break
        
        env.close()
        print("✅ Manual episode test passed")
        return True
        
    except Exception as e:
        print(f"❌ Manual episode test failed: {e}")
        return False


def test_space_definitions():
    """Test observation and action space definitions"""
    print("\n=== Testing Space Definitions ===")
    
    try:
        env = DeepFlyerEnv()
        
        # Check observation space
        obs_space = env.observation_space
        print(f"Observation space: {obs_space}")
        
        # Check if it's a Dict space
        assert isinstance(obs_space, gym.spaces.Dict), "Observation space must be Dict"
        
        # Check required keys
        required_keys = {
            "hoop_x_center_norm", "hoop_y_center_norm", "hoop_visible", 
            "hoop_distance_norm", "drone_vx_norm", "drone_vy_norm", 
            "drone_vz_norm", "yaw_rate_norm"
        }
        obs_keys = set(obs_space.spaces.keys())
        missing_keys = required_keys - obs_keys
        
        assert not missing_keys, f"Missing observation keys: {missing_keys}"
        print("✅ All required observation keys present")
        
        # Check action space
        action_space = env.action_space
        print(f"Action space: {action_space}")
        
        assert isinstance(action_space, gym.spaces.Box), "Action space must be Box"
        assert action_space.shape == (4,), f"Action space should be 4D, got {action_space.shape}"
        print("✅ Action space correctly defined")
        
        # Test sampling
        for _ in range(10):
            obs = obs_space.sample()
            action = action_space.sample()
            
            assert obs_space.contains(obs), "Sampled observation not in space"
            assert action_space.contains(action), "Sampled action not in space"
        
        print("✅ Space sampling works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Space definition test failed: {e}")
        return False


def test_environment_registration():
    """Test environment registration with gym.make()"""
    print("\n=== Testing Environment Registration ===")
    
    try:
        # Test both registered versions
        env_v0 = gym.make("DeepFlyer/HoopNavigation-v0")
        env_v1 = gym.make("DeepFlyer/HoopNavigation-v1")
        
        print("✅ Both environment versions can be created with gym.make()")
        
        # Test with custom parameters
        env_custom = gym.make("DeepFlyer/HoopNavigation-v0", size=10, max_episode_steps=1000)
        assert env_custom.size == 10, "Custom parameters not applied"
        assert env_custom.max_episode_steps == 1000, "Custom episode steps not applied"
        
        print("✅ Custom parameters work correctly")
        
        # Test vectorized environments
        vec_env = gym.make_vec("DeepFlyer/HoopNavigation-v0", num_envs=3)
        print(f"✅ Vectorized environment created: {vec_env}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment registration test failed: {e}")
        return False


def test_reward_consistency():
    """Test reward function consistency"""
    print("\n=== Testing Reward Consistency ===")
    
    try:
        env = DeepFlyerEnv()
        
        # Test multiple episodes for consistency
        total_rewards = []
        
        for episode in range(3):
            obs, info = env.reset(seed=42 + episode)  # Different seeds
            episode_reward = 0
            
            for step in range(50):  # Short episodes
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Check reward bounds (should be reasonable)
                assert -100 <= reward <= 100, f"Reward {reward} seems unreasonable"
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            print(f"  Episode {episode + 1}: {step + 1} steps, total reward: {episode_reward:.2f}")
        
        print(f"✅ Reward function working consistently")
        print(f"Average episode reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reward consistency test failed: {e}")
        return False


def main():
    """Run all environment validation tests"""
    print("🚁 DeepFlyer Environment Validation")
    print("=" * 50)
    
    tests = [
        test_space_definitions,
        test_environment_validity,
        test_environment_registration,
        test_manual_episode,
        test_reward_consistency,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Environment is Gymnasium compliant.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())
