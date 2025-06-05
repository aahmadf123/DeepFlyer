#!/usr/bin/env python3
"""Test script for ROS2/Gazebo environment integration."""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rl_agent.env import make_env, make_drone_env


def test_ros_connection():
    """Test basic ROS2 connectivity."""
    print("=" * 60)
    print("Testing ROS2 Environment Connection")
    print("=" * 60)
    
    try:
        # Try to create ROS environment
        env = make_env("ros:deepflyer")
        print("✓ ROS2 environment created successfully")
        
        # Get initial observation
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Info: {info}")
        
        # Test a few steps with hover action
        print("\nTesting hover action for 5 steps...")
        for i in range(5):
            if env.action_space.shape:  # Continuous
                action = np.zeros(env.action_space.shape)
            else:  # Discrete
                action = 0  # Hover
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.3f}, terminated={terminated}, "
                  f"position={info.get('position', 'N/A')}")
            
            if terminated:
                break
        
        env.close()
        print("✓ Environment closed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nMake sure:")
        print("  1. ROS2 is installed and sourced")
        print("  2. Gazebo is running with DeepFlyer simulation")
        print("  3. Required topics are being published")
        return False


def test_reward_functions():
    """Test different reward functions with ROS environment."""
    print("\n" + "=" * 60)
    print("Testing Reward Functions")
    print("=" * 60)
    
    reward_functions = [
        "reach_target",
        "collision_avoidance",
        "save_energy",
        "fly_steady",
        "fly_smoothly",
        "be_fast"
    ]
    
    for reward_fn in reward_functions:
        try:
            env = make_drone_env(
                namespace="deepflyer",
                reward_function=reward_fn,
                goal_position=[5.0, 5.0, 1.5]
            )
            obs, info = env.reset()
            
            # Take one step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"✓ {reward_fn}: reward={reward:.3f}")
            env.close()
            
        except Exception as e:
            print(f"✗ {reward_fn}: {e}")


def test_multi_objective():
    """Test multi-objective reward function."""
    print("\n" + "=" * 60)
    print("Testing Multi-Objective Reward")
    print("=" * 60)
    
    try:
        env = make_env(
            "ros:deepflyer",
            reward_function="multi_objective",
            reward_weights={
                'reach': 1.0,
                'collision': 2.0,
                'energy': 0.5,
                'speed': 0.3
            }
        )
        
        obs, info = env.reset()
        print("✓ Multi-objective environment created")
        
        # Test a few random actions
        total_reward = 0
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                break
        
        print(f"✓ Average reward over 10 steps: {total_reward/10:.3f}")
        env.close()
        
    except Exception as e:
        print(f"✗ Error: {e}")


def test_safety_limits():
    """Test safety monitoring features."""
    print("\n" + "=" * 60)
    print("Testing Safety Features")
    print("=" * 60)
    
    try:
        env = make_drone_env(
            namespace="deepflyer",
            enable_safety_monitor=True,
            safety_config={
                'max_velocity': 1.0,
                'min_altitude': 0.5,
                'max_altitude': 2.0
            }
        )
        
        obs, info = env.reset()
        print("✓ Safety monitor enabled")
        
        # Try maximum velocity command
        if env.action_space.shape:  # Continuous
            max_action = np.ones(env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(max_action)
            print(f"✓ Velocity after max command: {info.get('velocity', 'N/A')}")
        
        env.close()
        
    except Exception as e:
        print(f"✗ Error: {e}")


def test_fallback_mode():
    """Test fallback when ROS is not available."""
    print("\n" + "=" * 60)
    print("Testing Fallback Mode")
    print("=" * 60)
    
    # This should work even without ROS
    env = make_env("CartPole-v1", reward_function="reach_target")
    print("✓ CartPole with drone reward created")
    
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✓ Step successful: reward={reward:.3f}")
    env.close()


def main():
    """Run all tests."""
    print("DeepFlyer ROS2 Environment Test Suite")
    print("=====================================\n")
    
    # Check if ROS is available
    try:
        import rclpy
        print("✓ ROS2 Python packages found")
    except ImportError:
        print("✗ ROS2 Python packages not found")
        print("  Install with: pip install rclpy")
        print("\nRunning fallback test only...\n")
        test_fallback_mode()
        return
    
    # Run tests
    tests = [
        test_ros_connection,
        test_reward_functions,
        test_multi_objective,
        test_safety_limits,
        test_fallback_mode
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test {test.__name__} failed with: {e}")
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print("=" * 60)


if __name__ == "__main__":
    main() 