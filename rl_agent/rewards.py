"""
DeepFlyer Reinforcement Learning Reward Functions.

This module provides the minimal reward functions for drone control:
- Path following (cross-track error)
- Heading alignment (heading error)
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, Callable, Optional, Union

# Registry for reward functions
REGISTRY = {}

class RewardComponentType(Enum):
    """Types of reward components for classifying and configuring rewards."""
    FOLLOW_TRAJECTORY = "follow_trajectory"
    HEADING_ERROR = "heading_error"

class RewardComponent:
    def __init__(
        self, 
        component_type: RewardComponentType,
        fn: Callable[[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]], float],
        weight: float = 1.0,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.component_type = component_type
        self.fn = fn
        self.weight = weight
        self.parameters = parameters or {}
        self.value = 0.0  # Last computed value
    
    def compute(
        self, 
        state: Dict[str, Any], 
        action: Dict[str, Any],
        next_state: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        self.value = self.fn(state, action, next_state, self.parameters, info)
        return self.weight * self.value

class RewardFunction:
    def __init__(self):
        self.components = []
        self.component_values = {}
    def add_component(
        self, 
        component_type: Union[str, RewardComponentType], 
        weight: float = 1.0,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        if isinstance(component_type, str):
            component_type = RewardComponentType(component_type)
        component_fn = get_component_fn(component_type)
        component = RewardComponent(
            component_type=component_type,
            fn=component_fn,
            weight=weight,
            parameters=parameters
        )
        self.components.append(component)
    def compute_reward(
        self, 
        state: Dict[str, Any], 
        action: Any,
        next_state: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        if not isinstance(action, dict):
            action_dict = {"raw": action}
        else:
            action_dict = action
        total_reward = 0.0
        self.component_values = {}
        for component in self.components:
            component_reward = component.compute(state, action_dict, next_state, info)
            total_reward += component_reward
            self.component_values[component.component_type.value] = component_reward
        return total_reward
    def reset_stats(self) -> None:
        self.component_values = {}

def follow_trajectory_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Reward for following a predefined trajectory (cross-track error only).
    """
    params = parameters or {}
    position = np.array(state.get("position", [0, 0, 0]))
    trajectory = params.get("trajectory")
    if trajectory is None:
        return 0.0
    if isinstance(trajectory, list):
        trajectory = [np.array(point) for point in trajectory]
    min_distance = float('inf')
    total_length = 0.0
    for i in range(len(trajectory) - 1):
        segment_length = np.linalg.norm(trajectory[i+1] - trajectory[i])
        total_length += segment_length
    for i in range(len(trajectory) - 1):
        segment_start = trajectory[i]
        segment_end = trajectory[i+1]
        segment_vec = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vec)
        to_position = position - segment_start
        projection = np.dot(to_position, segment_vec) / segment_length
        projection = max(0, min(segment_length, projection))
        projected_point = segment_start + (projection / segment_length) * segment_vec
        distance = np.linalg.norm(position - projected_point)
        if distance < min_distance:
            min_distance = distance
    max_error = params.get("max_error", 2.0)
    cross_track_reward = max(0.0, 1.0 - (min_distance / max_error))
    return cross_track_reward

def heading_error_reward(
    state: Dict[str, Any],
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Reward (penalty) for heading error.
    """
    params = parameters or {}
    heading_error = abs(state.get("heading_error", 0.0))
    max_heading_error = params.get("max_heading_error", np.pi)
    return -heading_error / max_heading_error

def get_component_fn(component_type: RewardComponentType) -> Callable:
    component_map = {
        RewardComponentType.FOLLOW_TRAJECTORY: follow_trajectory_reward,
        RewardComponentType.HEADING_ERROR: heading_error_reward,
    }
    if component_type not in component_map:
        raise ValueError(f"No function registered for component type: {component_type}")
    return component_map[component_type]

def create_cross_track_and_heading_reward(cross_track_weight=1.0, heading_weight=0.1, max_error=2.0, max_heading_error=np.pi, trajectory=None):
    reward_fn = RewardFunction()
    reward_fn.add_component(
        RewardComponentType.FOLLOW_TRAJECTORY,
        weight=cross_track_weight,
        parameters={
            "cross_track_weight": 1.0,
            "progress_weight": 0.0,
            "max_error": max_error,
            "trajectory": trajectory,
        }
    )
    reward_fn.add_component(
        RewardComponentType.HEADING_ERROR,
        weight=heading_weight,
        parameters={"max_heading_error": max_heading_error}
    )
    return reward_fn

def register_reward(name: str, reward_fn: Callable) -> None:
    REGISTRY[name] = {
        'fn': reward_fn,
        'description': reward_fn.__doc__ or "No description"
    }

register_reward("follow_trajectory", follow_trajectory_reward)
register_reward("heading_error", heading_error_reward) 