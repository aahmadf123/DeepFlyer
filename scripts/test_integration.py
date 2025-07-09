#!/usr/bin/env python3
"""
DeepFlyer Integration Test Script
Tests ML components integration without requiring full ROS setup
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.config import DeepFlyerConfig
from rl_agent.algorithms.p3o import P3O
from rl_agent.env.vision_processor import create_yolo11_processor
from api.ml_interface import DeepFlyerMLInterface, RewardConfig


def test_config_loading():
    """Test configuration loading"""
    print("🔧 Testing configuration loading...")
    
    try:
        config = DeepFlyerConfig()
        assert config.RL_CONFIG is not None
        assert config.REWARD_CONFIG is not None
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_yolo_model_loading():
    """Test YOLO model loading"""
    print("🎯 Testing YOLO model loading...")
    
    try:
        # Test with custom trained model
        processor = create_yolo11_processor(
            model_path="weights/best.pt",
            confidence_threshold=0.3
        )
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        features = processor.process_frame(dummy_image)
        
        assert hasattr(features, 'hoop_detected')
        assert hasattr(features, 'hoop_distance')
        print("✅ YOLO model loaded and tested successfully")
        return True
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        return False


def test_p3o_agent():
    """Test P3O agent initialization"""
    print("🧠 Testing P3O agent...")
    
    try:
        config = DeepFlyerConfig()
        observation_space = 12
        action_space = 3
        
        agent = P3O(
            observation_dim=observation_space,
            action_dim=action_space,
            learning_rate=config.RL_CONFIG["learning_rate"],
            buffer_size=1000
        )
        
        # Test with dummy observation
        dummy_obs = np.random.randn(observation_space)
        action = agent.predict(dummy_obs)
        
        assert len(action) == action_space
        assert all(-1 <= a <= 1 for a in action)
        print("✅ P3O agent initialized and tested successfully")
        return True
    except Exception as e:
        print(f"❌ P3O agent test failed: {e}")
        return False


def test_ml_interface():
    """Test ML interface functionality"""
    print("🔌 Testing ML interface...")
    
    try:
        ml_interface = DeepFlyerMLInterface()
        
        # Test reward config
        config = ml_interface.get_reward_config()
        assert isinstance(config, RewardConfig)
        
        # Test config update
        new_config = RewardConfig(hoop_approach_reward=15.0)
        success = ml_interface.update_reward_config(new_config)
        assert success
        
        # Test metrics
        metrics = ml_interface.get_training_metrics()
        assert hasattr(metrics, 'episode')
        
        # Test live data
        live_data = ml_interface.get_live_data()
        assert 'metrics' in live_data
        assert 'reward_config' in live_data
        
        print("✅ ML interface tested successfully")
        return True
    except Exception as e:
        print(f"❌ ML interface test failed: {e}")
        return False


def test_reward_calculation():
    """Test reward calculation logic"""
    print("🎯 Testing reward calculation...")
    
    try:
        from rl_agent.rewards.rewards import compute_reward_components
        
        # Dummy drone state
        drone_state = {
            'position': [1.0, 0.5, 1.0],
            'velocity': [0.5, 0.0, 0.0],
            'collision': False
        }
        
        # Dummy hoop state
        hoop_state = {
            'position': [2.0, 0.5, 1.0],
            'detected': True,
            'distance': 1.5,
            'alignment': 0.8
        }
        
        rewards = compute_reward_components(drone_state, hoop_state)
        assert isinstance(rewards, dict)
        assert 'total' in rewards
        
        print("✅ Reward calculation tested successfully")
        return True
    except Exception as e:
        print(f"❌ Reward calculation test failed: {e}")
        return False


def test_clearml_integration():
    """Test ClearML integration"""
    print("📊 Testing ClearML integration...")
    
    try:
        from rl_agent.utils import ClearMLTracker
        
        tracker = ClearMLTracker(
            project_name="DeepFlyer-Test",
            task_name="Integration-Test"
        )
        
        # Test logging
        tracker.log_scalar("test_metric", 1.0, 0)
        tracker.log_hyperparameters({"test_param": 42})
        
        print("✅ ClearML integration tested successfully")
        return True
    except Exception as e:
        print(f"❌ ClearML integration test failed: {e}")
        return False


def run_performance_test():
    """Run performance benchmark"""
    print("⚡ Running performance test...")
    
    try:
        # YOLO processing speed test
        processor = create_yolo11_processor("weights/best.pt", 0.3)
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        for _ in range(10):
            features = processor.process_frame(dummy_image)
        processing_time = (time.time() - start_time) / 10
        
        fps = 1.0 / processing_time
        print(f"✅ YOLO processing: {fps:.1f} FPS (target: >15 FPS)")
        
        if fps < 15:
            print("⚠️  Warning: YOLO processing too slow for real-time")
        
        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("🚁 DeepFlyer Integration Tests")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_yolo_model_loading,
        test_p3o_agent,
        test_ml_interface,
        test_reward_calculation,
        test_clearml_integration,
        run_performance_test
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test crashed: {e}\n")
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    exit(main()) 