import numpy as np
import pytest
from rl_agent.rewards import (
    follow_trajectory_reward,
    heading_error_reward,
    create_cross_track_and_heading_reward,
    RewardFunction,
)

def test_follow_trajectory_reward():
    # Create a simple trajectory
    trajectory = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
    ]
    
    # Test when on the path
    state = {"position": [0.5, 0.0, 0.0]}
    params = {"trajectory": trajectory, "max_error": 2.0}
    reward = follow_trajectory_reward(state, {}, parameters=params)
    assert pytest.approx(reward, rel=1e-6) == 1.0  # No error, full reward
    
    # Test when off the path
    state = {"position": [0.5, 1.0, 0.0]}  # 1m off the path
    reward = follow_trajectory_reward(state, {}, parameters=params)
    assert pytest.approx(reward, rel=1e-6) == 0.5  # Half of max_error, half reward

def test_heading_error_reward():
    # Test with no error
    state = {"heading_error": 0.0}
    reward = heading_error_reward(state, {})
    assert pytest.approx(reward, rel=1e-6) == 0.0  # No error, no penalty
    
    # Test with maximum error
    state = {"heading_error": np.pi}
    reward = heading_error_reward(state, {})
    assert pytest.approx(reward, rel=1e-6) == -1.0  # Max error, full penalty
    
    # Test with half error
    state = {"heading_error": np.pi/2}
    reward = heading_error_reward(state, {})
    assert pytest.approx(reward, rel=1e-6) == -0.5  # Half error, half penalty

def test_create_cross_track_and_heading_reward():
    # Create a reward function with the helper
    trajectory = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    
    reward_fn = create_cross_track_and_heading_reward(
        cross_track_weight=1.0,
        heading_weight=0.5,
        max_error=2.0,
        max_heading_error=np.pi,
        trajectory=trajectory
    )
    
    # Check that it's a RewardFunction instance
    assert isinstance(reward_fn, RewardFunction)
    
    # Check that it has exactly 2 components
    assert len(reward_fn.components) == 2
    
    # Check component types
    assert reward_fn.components[0].component_type.value == "follow_trajectory"
    assert reward_fn.components[1].component_type.value == "heading_error"
    
    # Check weights
    assert reward_fn.components[0].weight == 1.0
    assert reward_fn.components[1].weight == 0.5
    
    # Test computing a reward
    state = {
        "position": [0.5, 0.5, 0.0],  # 0.5m off path
        "heading_error": np.pi/2,      # 90 degrees off
    }
    
    reward = reward_fn.compute_reward(state, np.zeros(4))
    
    # Expected: 1.0 * (1 - 0.5/2.0) + 0.5 * (-0.5) = 0.75 - 0.25 = 0.5
    assert pytest.approx(reward, rel=1e-6) == 0.5 