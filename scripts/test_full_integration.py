#!/usr/bin/env python3
"""
Full Integration Test for DeepFlyer

Tests the complete pipeline is ready for integration.
Run this before handing off to teammates.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports() -> bool:
    """Test all critical imports work"""
    print("Testing imports...")
    try:
        # Core ML
        import torch
        import gymnasium as gym
        from ultralytics import YOLO
        
        # RL Agent
        from rl_agent.algorithms.p3o import P3O, P3OConfig
        from rl_agent.rewards import reward_function, HoopRewardFunction
        from rl_agent.config import DeepFlyerConfig
        from rl_agent.utils import ClearMLTracker
        
        # Environment
        from rl_agent.env.px4_base_env import PX4BaseEnv
        from rl_agent.env.zed_integration import ZEDInterface
        
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_loading() -> bool:
    """Test configuration files exist and load"""
    print("\nTesting configuration...")
    try:
        # Check student tuning config
        config_file = Path("config/student_tuning.json")
        if not config_file.exists():
            print(f"✗ Missing config file: {config_file}")
            return False
        
        with open(config_file) as f:
            config = json.load(f)
        
        required_sections = ['p3o_hyperparameters', 'reward_parameters', 'training_settings']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        print(f"✓ Configuration loaded: {len(config)} sections")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_p3o_agent() -> bool:
    """Test P3O agent initialization and operation"""
    print("\nTesting P3O agent...")
    try:
        from rl_agent.algorithms.p3o import P3O, P3OConfig
        
        config = P3OConfig()
        agent = P3O(obs_dim=8, action_dim=4, config=config)
        
        # Test action selection
        obs = np.random.randn(8).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        
        if action.shape != (4,):
            print(f"✗ Wrong action shape: {action.shape}")
            return False
        
        print(f"✓ P3O agent working: action={action.shape}, value={value:.2f}")
        return True
    except Exception as e:
        print(f"✗ P3O test failed: {e}")
        return False


def test_reward_function() -> bool:
    """Test reward function with various inputs"""
    print("\nTesting reward function...")
    try:
        from rl_agent.rewards import reward_function
        
        test_cases = [
            {
                'name': 'Hoop visible and aligned',
                'params': {
                    'hoop_x_center_norm': 0.05,
                    'hoop_y_center_norm': 0.05,
                    'hoop_visible': 1,
                    'hoop_distance_norm': 0.3,
                    'drone_vx_norm': 0.5,
                    'drone_vy_norm': 0.0,
                    'drone_vz_norm': 0.0,
                    'yaw_rate_norm': 0.0,
                    'all_systems_normal': True,
                    'speed': 1.0,
                    'collision': False,
                    'hoop_passages_completed': 0,
                    'flight_phase': 'NAVIGATE_TO_HOOP'
                },
                'expected_range': (5.0, 20.0)
            },
            {
                'name': 'Collision penalty',
                'params': {
                    'hoop_x_center_norm': 0.0,
                    'hoop_y_center_norm': 0.0,
                    'hoop_visible': 1,
                    'hoop_distance_norm': 0.1,
                    'collision': True,
                    'all_systems_normal': True,
                    'speed': 0.0,
                    'hoop_passages_completed': 0,
                    'flight_phase': 'NAVIGATE_TO_HOOP'
                },
                'expected_range': (0.0, 0.01)
            }
        ]
        
        for test in test_cases:
            reward = reward_function(test['params'])
            min_val, max_val = test['expected_range']
            
            if not (min_val <= reward <= max_val):
                print(f"✗ {test['name']}: reward={reward:.2f} not in range {test['expected_range']}")
                return False
            
            print(f"✓ {test['name']}: reward={reward:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Reward function test failed: {e}")
        return False


def test_hyperopt_tools() -> bool:
    """Test hyperparameter optimization tools"""
    print("\nTesting hyperopt tools...")
    try:
        from rl_agent.algorithms.p3o import HyperparameterOptimizer, P3OConfig
        
        base_config = P3OConfig()
        optimizer = HyperparameterOptimizer(base_config)
        
        # Test random sampling
        for i in range(3):
            config = optimizer.suggest_config()
            if not isinstance(config, P3OConfig):
                print(f"✗ Invalid config type: {type(config)}")
                return False
        
        # Test performance reporting
        optimizer.report_performance(config, 100.0)
        
        print(f"✓ Hyperopt tools working: {optimizer.current_trial} trials")
        return True
    except Exception as e:
        print(f"✗ Hyperopt test failed: {e}")
        return False


def test_clearml_optional() -> bool:
    """Test ClearML integration (optional)"""
    print("\nTesting ClearML (optional)...")
    try:
        from rl_agent.utils import ClearMLTracker
        
        # Try to create tracker (may fail if not configured)
        tracker = ClearMLTracker(
            project_name="DeepFlyer-Test",
            task_name="Integration-Test"
        )
        
        if tracker.enabled:
            print("✓ ClearML available and configured")
        else:
            print("⚠ ClearML not available (optional)")
        
        return True
    except Exception as e:
        print(f"⚠ ClearML test skipped: {e}")
        return True  # Optional, so still pass


def test_file_structure() -> bool:
    """Test critical files exist"""
    print("\nTesting file structure...")
    
    critical_files = [
        "rl_agent/algorithms/p3o.py",
        "rl_agent/rewards.py",
        "rl_agent/config.py",
        "rl_agent/trajectory.py",
        "nodes/rl_agent_node.py",
        "nodes/vision_processor_node.py",
        "launch/deepflyer_ml.launch.py",
        "config/student_tuning.json",
        "scripts/hyperopt_runner.py",
        "requirements.txt",
        "package.xml",
        "CMakeLists.txt"
    ]
    
    missing = []
    for file in critical_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"✗ Missing {len(missing)} files:")
        for f in missing[:5]:  # Show first 5
            print(f"  - {f}")
        return False
    
    print(f"✓ All {len(critical_files)} critical files present")
    return True


def test_no_mvp_references() -> bool:
    """Check that MVP references have been removed"""
    print("\nChecking for MVP references...")
    
    # Quick check in a few key files
    files_to_check = [
        "rl_agent/rewards.py",
        "rl_agent/algorithms/p3o.py",
        "launch/deepflyer_ml.launch.py"
    ]
    
    found_mvp = False
    for file in files_to_check:
        if Path(file).exists():
            with open(file) as f:
                content = f.read()
                if 'MVP' in content or 'mvp' in content:
                    print(f"⚠ Found MVP reference in {file}")
                    found_mvp = True
    
    if found_mvp:
        print("⚠ Some MVP references remain (non-critical)")
    else:
        print("✓ No MVP references found")
    
    return True  # Non-critical


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("DEEPFLYER INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config_loading),
        ("P3O Agent", test_p3o_agent),
        ("Reward Function", test_reward_function),
        ("Hyperopt Tools", test_hyperopt_tools),
        ("ClearML", test_clearml_optional),
        ("File Structure", test_file_structure),
        ("MVP Cleanup", test_no_mvp_references)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Ready for integration!")
        return 0
    else:
        print("\n⚠ Some tests failed - Review issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
