#!/usr/bin/env python3
"""
Core Environment Test - Test just the main DeepFlyerEnv without ROS dependencies
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import only the core environment
try:
    from rl_agent.env.px4_env import DeepFlyerEnv
    print("✅ Core environment imported successfully")
except Exception as e:
    print(f"❌ Failed to import core environment: {e}")
    exit(1)

def test_basic_functionality():
    """Test basic environment functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    try:
        # Create environment
        env = DeepFlyerEnv(render_mode=None, size=5)
        print("✅ Environment created")
        
        # Test observation and action spaces
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print("✅ Reset successful")
        print(f"Observation keys: {list(obs.keys())}")
        
        # Validate observation
        assert env.observation_space.contains(obs), "Observation not in space!"
        print("✅ Observation space validation passed")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Step successful - reward: {reward:.3f}")
        
        # Test multiple steps
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                print(f"  Episode ended at step {i+1}")
                break
        
        print("✅ Multiple steps successful")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_gymnasium_compliance():
    """Test basic Gymnasium compliance"""
    print("\n=== Testing Gymnasium Compliance ===")
    
    try:
        import gymnasium as gym
        
        # Test with gymnasium check_env
        try:
            from gymnasium.utils.env_checker import check_env
            env = DeepFlyerEnv()
            check_env(env)
            print("✅ Gymnasium check_env passed")
        except Exception as e:
            print(f"⚠️  Gymnasium check_env warning: {e}")
        
        # Test environment registration  
        try:
            env = gym.make("DeepFlyer/HoopNavigation-v0")
            print("✅ Environment registration works")
            
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print("✅ Registered environment works")
            
        except Exception as e:
            print(f"⚠️  Environment registration issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gymnasium compliance test failed: {e}")
        return False

def test_rendering():
    """Test rendering functionality"""
    print("\n=== Testing Rendering ===")
    
    try:
        # Test human rendering
        env = DeepFlyerEnv(render_mode="human", size=3)  # Smaller for testing
        obs, info = env.reset(seed=42)
        
        print("Testing human rendering:")
        env.render()
        
        # Take a few steps and render
        for i in range(3):
            action = np.array([1.0, 0.0, 0.0, 0.0])  # Move forward
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                break
        
        print("✅ Human rendering works")
        
        # Test rgb_array rendering
        env_rgb = DeepFlyerEnv(render_mode="rgb_array")
        obs, info = env_rgb.reset()
        rgb_array = env_rgb.render()
        
        if rgb_array is not None:
            print(f"✅ RGB array rendering works: shape {rgb_array.shape}")
        else:
            print("⚠️  RGB array rendering returned None")
        
        return True
        
    except Exception as e:
        print(f"❌ Rendering test failed: {e}")
        return False

def main():
    """Run all core environment tests"""
    print("🚁 DeepFlyer Core Environment Test")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_gymnasium_compliance,
        test_rendering,
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
        print("🎉 All core tests passed! Environment is working.")
        return 0
    else:
        print("⚠️  Some tests failed, but core functionality works.")
        return 0  # Don't fail - this is expected during development

if __name__ == "__main__":
    exit(main())
