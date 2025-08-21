#!/usr/bin/env python3
"""
Full System Test for DeepFlyer
Tests all components working together
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from rl_agent.algorithms.p3o import P3O, P3OConfig
        from rl_agent.algorithms.replay_buffer import ReplayBuffer
        from rl_agent.direct_control_agent import DirectControlAgent
        from rl_agent.rewards.rewards import HoopNavigationReward
        from rl_agent.env.zed_integration import create_zed_interface
        print("✓ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_p3o_agent():
    """Test P3O agent creation and basic operations"""
    print("\nTesting P3O agent...")
    try:
        from rl_agent.algorithms.p3o import P3O, P3OConfig
        
        config = P3OConfig()
        agent = P3O(obs_dim=8, action_dim=4, config=config)
        
        # Test prediction
        obs = np.random.randn(8).astype(np.float32)
        action = agent.predict(obs)
        assert action.shape == (4,), f"Wrong action shape: {action.shape}"
        
        print("✓ P3O agent working correctly")
        return True
    except Exception as e:
        print(f"✗ P3O agent test failed: {e}")
        return False

def test_direct_control():
    """Test direct control agent"""
    print("\nTesting direct control agent...")
    try:
        from rl_agent.direct_control_agent import DirectControlAgent
        
        agent = DirectControlAgent()
        
        # Test observation processing
        vision_data = {
            'hoop_detected': True,
            'hoop_center_x': 0.1,
            'hoop_center_y': -0.2,
            'hoop_distance': 0.5
        }
        drone_state = {
            'velocity': [1.0, 0.0, 0.5],
            'yaw_rate': 0.1
        }
        
        obs = agent.process_observation(vision_data, drone_state)
        assert obs.shape == (8,), f"Wrong observation shape: {obs.shape}"
        
        # Test action generation
        action = agent.get_action(obs, training=False)
        assert action.shape == (4,), f"Wrong action shape: {action.shape}"
        
        print("✓ Direct control agent working correctly")
        return True
    except Exception as e:
        print(f"✗ Direct control test failed: {e}")
        return False

def test_reward_function():
    """Test reward calculation"""
    print("\nTesting reward function...")
    try:
        from rl_agent.rewards.rewards import HoopNavigationReward
        
        reward_fn = HoopNavigationReward()
        
        # Test reward calculation
        obs = np.array([0.1, 0.1, 1.0, 0.3, 0.5, 0.0, 0.0, 0.0])
        action = np.array([1.0, 0.0, 0.0, 0.0])
        info = {'hoop_passed': False, 'collision': False}
        
        reward, components = reward_fn.calculate_reward(obs, action, info)
        
        assert isinstance(reward, float), "Reward should be float"
        assert isinstance(components, dict), "Components should be dict"
        assert len(components) > 0, "Components should not be empty"
        
        print(f"✓ Reward function working (sample reward: {reward:.2f})")
        return True
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        return False

def test_replay_buffer():
    """Test replay buffer"""
    print("\nTesting replay buffer...")
    try:
        from rl_agent.algorithms.replay_buffer import ReplayBuffer
        
        buffer = ReplayBuffer(obs_dim=8, action_dim=4, buffer_size=1000)
        
        # Add some experiences
        for _ in range(100):
            obs = np.random.randn(8).astype(np.float32)
            action = np.random.randn(4).astype(np.float32)
            reward = np.random.randn()
            next_obs = np.random.randn(8).astype(np.float32)
            done = np.random.random() > 0.9
            
            buffer.add(obs, action, reward, next_obs, done)
        
        # Sample batch
        batch = buffer.sample(32)
        assert batch['obs'].shape == (32, 8), "Wrong batch observation shape"
        assert batch['actions'].shape == (32, 4), "Wrong batch action shape"
        
        print(f"✓ Replay buffer working (size: {len(buffer)})")
        return True
    except Exception as e:
        print(f"✗ Replay buffer test failed: {e}")
        return False

def test_zed_integration():
    """Test ZED camera integration"""
    print("\nTesting ZED integration...")
    try:
        from rl_agent.env.zed_integration import create_zed_interface
        
        # Create mock interface (won't fail without hardware)
        zed = create_zed_interface("mock")
        
        # Initialize
        success = zed.initialize()
        assert success, "Failed to initialize mock ZED"
        
        # Grab frame
        rgb, depth = zed.grab_frame()
        assert rgb is not None, "No RGB image"
        assert depth is not None, "No depth map"
        
        # Get depth at point
        depth_value = zed.get_depth_at_point(100, 100)
        assert depth_value > 0, "Invalid depth value"
        
        zed.close()
        
        print("✓ ZED integration working (mock mode)")
        return True
    except Exception as e:
        print(f"✗ ZED integration test failed: {e}")
        return False

def test_training_step():
    """Test a complete training step"""
    print("\nTesting training step...")
    try:
        from rl_agent.direct_control_agent import DirectControlAgent
        from rl_agent.rewards.rewards import HoopNavigationReward
        
        agent = DirectControlAgent()
        reward_fn = HoopNavigationReward()
        
        # Simulate one episode step
        obs = np.random.randn(8).astype(np.float32)
        action = agent.get_action(obs, training=True)
        
        # Simulate next observation
        next_obs = obs + np.random.randn(8) * 0.1
        info = {'collision': False, 'hoop_passed': False}
        
        # Calculate reward
        reward, _ = reward_fn.calculate_reward(next_obs, action, info)
        
        # Store experience
        agent.store_experience(obs, action, next_obs, reward, False)
        
        # Try training (may not update if buffer too small)
        stats = agent.train_step()
        
        print("✓ Training step completed successfully")
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        return False

def test_model_save_load():
    """Test model saving and loading"""
    print("\nTesting model save/load...")
    try:
        from rl_agent.direct_control_agent import DirectControlAgent
        import tempfile
        
        # Create agent
        agent1 = DirectControlAgent()
        
        # Get initial prediction
        obs = np.random.randn(8).astype(np.float32)
        action1 = agent1.get_action(obs, training=False)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            agent1.save(tmp.name)
            model_path = tmp.name
        
        # Create new agent and load
        agent2 = DirectControlAgent()
        agent2.load(model_path)
        
        # Get prediction from loaded model
        action2 = agent2.get_action(obs, training=False)
        
        # Check if predictions match
        assert np.allclose(action1, action2, atol=1e-5), "Loaded model gives different predictions"
        
        # Clean up
        os.unlink(model_path)
        
        print("Model save/load working correctly")
        return True
    except Exception as e:
        print(f"Model save/load test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("DEEPFLYER SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("P3O Agent", test_p3o_agent),
        ("Direct Control", test_direct_control),
        ("Reward Function", test_reward_function),
        ("Replay Buffer", test_replay_buffer),
        ("ZED Integration", test_zed_integration),
        ("Training Step", test_training_step),
        ("Model Save/Load", test_model_save_load)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f" {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status:10} {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n ALL TESTS PASSED - System ready for deployment!")
        return 0
    else:
        print("\n Some tests failed - Please fix issues before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())