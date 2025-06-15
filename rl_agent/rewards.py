"""
DeepFlyer Reinforcement Learning Reward Functions.

This module provides reward functions for drone control, focusing on:
- Trajectory following
- Collision avoidance
- Smooth and efficient flight
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

# Registry for reward functions
REGISTRY = {}

class RewardComponentType(Enum):
    """Types of reward components for classifying and configuring rewards."""
    REACH_TARGET = "reach_target"
    AVOID_CRASHES = "avoid_crashes"
    SAVE_ENERGY = "save_energy"
    FLY_STEADY = "fly_steady"
    FLY_SMOOTHLY = "fly_smoothly"
    BE_FAST = "be_fast"
    FOLLOW_TRAJECTORY = "follow_trajectory"
    MINIMIZE_JERK = "minimize_jerk"
    TERMINAL = "terminal"


class RewardComponent:
    """
    A single component of a composite reward function.
    
    Each component focuses on a specific aspect of the drone's behavior,
    such as following a trajectory or avoiding crashes.
    """
    
    def __init__(
        self, 
        component_type: RewardComponentType,
        fn: Callable[[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]], float],
        weight: float = 1.0,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a reward component.
        
        Args:
            component_type: Type of the reward component
            fn: Function that computes the reward value
            weight: Weight of this component in the composite reward
            parameters: Additional parameters for the reward function
        """
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
        """
        Compute the reward value for this component.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state (if available)
            info: Additional information
            
        Returns:
            Weighted reward value
        """
        # Compute raw reward
        self.value = self.fn(state, action, next_state, self.parameters, info)
        
        # Apply weight
        return self.weight * self.value


class RewardFunction:
    """
    Composite reward function combining multiple components.
    
    This class allows building custom reward functions by combining
    and weighting different components.
    """
    
    def __init__(self):
        """Initialize an empty reward function."""
        self.components = []
        self.component_values = {}  # For logging/debugging
    
    def add_component(
        self, 
        component_type: Union[str, RewardComponentType], 
        weight: float = 1.0,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a reward component to the function.
        
        Args:
            component_type: Type of component to add (string or enum)
            weight: Weight of this component
            parameters: Additional parameters for the component
        """
        # Convert string to enum if needed
        if isinstance(component_type, str):
            try:
                component_type = RewardComponentType(component_type)
            except ValueError:
                raise ValueError(f"Unknown reward component type: {component_type}")
        
        # Get component function from registry
        component_fn = get_component_fn(component_type)
        
        # Create and add component
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
        """
        Compute the total reward by combining all components.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state (if available)
            info: Additional information
            
        Returns:
            Total reward
        """
        # Convert numpy array action to dict for reward functions
        if not isinstance(action, dict):
            action_dict = {"raw": action}
        else:
            action_dict = action
            
        # Compute rewards for each component
        total_reward = 0.0
        self.component_values = {}
        
        for component in self.components:
            component_reward = component.compute(state, action_dict, next_state, info)
            total_reward += component_reward
            
            # Store values for debugging/logging
            self.component_values[component.component_type.value] = component_reward
        
        return total_reward
    
    def reset_stats(self) -> None:
        """Reset component statistics."""
        self.component_values = {}


def reach_target_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Reward for getting closer to the goal position.
    
    Args:
        state: Current state including position and goal
        action: Not used
        next_state: Not used
        parameters: Optional parameters 
        info: Not used
        
    Returns:
        Normalized reward based on distance to goal
    """
    params = parameters or {}
    position = np.array(state.get("position", [0, 0, 0]))
    
    # Get goal position from state or parameters
    if "goal_position" in state:
        goal = np.array(state["goal_position"])
    elif "goal" in state:
        goal = np.array(state["goal"])
    elif "goal" in params:
        goal = np.array(params["goal"])
    else:
        return 0.0  # No goal defined
    
    # Calculate distance to goal
    distance = np.linalg.norm(position - goal)
    
    # Normalize by max diagonal (default 10m)
    max_diagonal = state.get("max_room_diagonal", 10.0)
    
    # Higher reward for being closer to goal
    reward = max(0.0, 1.0 - (distance / max_diagonal))
    
    # Bonus for reaching the goal
    if distance < 0.2:  # Within 20cm
        reward += 1.0
        
    return reward


def follow_trajectory_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Reward for following a predefined trajectory.
    
    Combines cross-track error (distance from path) and progress along path.
    
    Args:
        state: Current state
        action: Not used
        next_state: Not used
        parameters: Must include 'trajectory' as list of waypoints
        info: Not used
        
    Returns:
        Combined reward for trajectory following
    """
    params = parameters or {}
    position = np.array(state.get("position", [0, 0, 0]))
    
    # Get trajectory from parameters
    trajectory = params.get("trajectory")
    if trajectory is None:
        # Fall back to reach_target if no trajectory
        return reach_target_reward(state, action, next_state, parameters, info)
        
    # Convert list to numpy array if needed
    if isinstance(trajectory, list):
        trajectory = [np.array(point) for point in trajectory]
    
    # Find closest segment in trajectory
    min_distance = float('inf')
    closest_segment = None
    progress = 0.0
    total_length = 0.0
    
    # Calculate total path length
    for i in range(len(trajectory) - 1):
        segment_length = np.linalg.norm(trajectory[i+1] - trajectory[i])
        total_length += segment_length
    
    # Find closest segment and progress
    path_progress = 0.0
    for i in range(len(trajectory) - 1):
        segment_start = trajectory[i]
        segment_end = trajectory[i+1]
        segment_vec = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vec)
        
        # Vector from segment start to position
        to_position = position - segment_start
        
        # Project position onto segment
        projection = np.dot(to_position, segment_vec) / segment_length
        projection = max(0, min(segment_length, projection))
        
        # Projected point
        projected_point = segment_start + (projection / segment_length) * segment_vec
        
        # Distance to segment
        distance = np.linalg.norm(position - projected_point)
        
        if distance < min_distance:
            min_distance = distance
            closest_segment = i
            
            # Progress along this segment
            segment_progress = projection / segment_length
            
            # Overall progress (accumulated length + progress along this segment)
            accumulated_length = 0
            for j in range(i):
                accumulated_length += np.linalg.norm(trajectory[j+1] - trajectory[j])
            
            path_progress = (accumulated_length + projection) / max(total_length, 1e-6)
    
    # Cross-track error component (distance from path)
    max_error = params.get("max_error", 2.0)  # Max expected error in meters
    cross_track_reward = max(0.0, 1.0 - (min_distance / max_error))
    
    # Progress component
    progress_reward = path_progress
    
    # Terminal reward if at end of trajectory
    terminal_reward = 0.0
    if path_progress > 0.99:  # Within 1% of end
        terminal_reward = 2.0
    
    # Combine rewards (weighted sum)
    cross_track_weight = params.get("cross_track_weight", 0.6)
    progress_weight = params.get("progress_weight", 0.4)
    
    return (cross_track_weight * cross_track_reward + 
            progress_weight * progress_reward +
            terminal_reward)


def avoid_crashes_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Reward (or penalty) for avoiding crashes and obstacles.
    
    Args:
        state: Current state
        action: Not used
        next_state: Not used
        parameters: Optional parameters
        info: Not used
        
    Returns:
        Reward/penalty based on collision or proximity to obstacles
    """
    params = parameters or {}
    
    # Check for collision
    if state.get("collision_flag", False):
        return params.get("collision_penalty", -1.0)
    
    # Check distance to obstacle
    obstacle_distance = state.get("distance_to_obstacle", float('inf'))
    min_safe_distance = params.get("min_safe_distance", 0.5)  # 50cm
    
    if obstacle_distance < min_safe_distance:
        # Penalty proportional to proximity
        proximity_factor = 1.0 - (obstacle_distance / min_safe_distance)
        return -0.5 * proximity_factor
    
    return 0.0  # No penalty when safe


def fly_smoothly_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Reward for smooth flight (minimizing jerk).
    
    Args:
        state: Current state including velocities
        action: Not used
        next_state: Not used
        parameters: Optional parameters
        info: Additional info including previous action
        
    Returns:
        Reward based on smoothness of flight
    """
    params = parameters or {}
    info = info or {}
    
    # Check if we have previous velocity
    if "prev_velocity" not in state:
        return 0.0
    
    curr_velocity = np.array(state.get("linear_velocity", [0, 0, 0]))
    prev_velocity = np.array(state["prev_velocity"])
    dt = state.get("dt", 0.05)  # Default 50ms
    
    # Calculate jerk (change in acceleration)
    linear_jerk = np.linalg.norm(curr_velocity - prev_velocity) / dt
    
    # Check angular velocity change if available
    angular_jerk = 0.0
    if "angular_velocity" in state and "prev_angular_velocity" in state:
        curr_angular = state["angular_velocity"]
        prev_angular = state["prev_angular_velocity"]
        angular_jerk = abs(curr_angular - prev_angular) / dt
    
    # Normalize by maximum expected jerk
    max_lin_jerk = state.get("max_lin_jerk", params.get("max_lin_jerk", 0.5))
    max_ang_jerk = state.get("max_ang_jerk", params.get("max_ang_jerk", 0.5))
    
    lin_penalty = min(1.0, linear_jerk / max_lin_jerk)
    ang_penalty = min(1.0, angular_jerk / max_ang_jerk)
    
    # Also penalize large changes in control input
    action_penalty = 0.0
    if "prev_action" in info and "raw" in action:
        prev_action = info["prev_action"]
        if prev_action is not None:
            action_change = np.linalg.norm(action["raw"] - prev_action)
            action_penalty = min(0.5, action_change)
    
    # Combine penalties (higher is worse)
    total_penalty = 0.4 * lin_penalty + 0.3 * ang_penalty + 0.3 * action_penalty
    
    # Convert to reward (higher is better)
    return max(0.0, 1.0 - total_penalty)


def be_fast_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Reward for completing the task quickly and efficiently.
    
    Args:
        state: Current state
        action: Not used
        next_state: Not used
        parameters: Optional parameters
        info: Not used
        
    Returns:
        Speed reward
    """
    params = parameters or {}
    
    # If at goal, reward based on time taken
    if state.get("at_goal", False):
        time_elapsed = state.get("time_elapsed", 0.0)
        max_time = state.get("max_time_allowed", params.get("max_time_allowed", 10.0))
        time_factor = min(1.0, time_elapsed / max_time)
        return 1.0 + (1.0 - time_factor)  # Up to 2.0 if very fast
    
    # Otherwise, reward based on speed toward goal
    velocity = np.array(state.get("linear_velocity", [0, 0, 0]))
    speed = np.linalg.norm(velocity)
    max_speed = state.get("max_speed", params.get("max_speed", 2.0))
    
    # Normalize speed
    return min(1.0, speed / max_speed)


def get_component_fn(component_type: RewardComponentType) -> Callable:
    """
    Get the reward function for a component type.
    
    Args:
        component_type: Type of reward component
        
    Returns:
        Reward calculation function
    """
    component_map = {
        RewardComponentType.REACH_TARGET: reach_target_reward,
        RewardComponentType.AVOID_CRASHES: avoid_crashes_reward,
        RewardComponentType.FLY_SMOOTHLY: fly_smoothly_reward,
        RewardComponentType.BE_FAST: be_fast_reward,
        RewardComponentType.FOLLOW_TRAJECTORY: follow_trajectory_reward,
    }
    
    if component_type not in component_map:
        raise ValueError(f"No function registered for component type: {component_type}")
    
    return component_map[component_type]


def create_default_reward_function() -> RewardFunction:
    """
    Create a default composite reward function.
    
    Returns:
        Default reward function with balanced components
    """
    reward_fn = RewardFunction()
    
    # Add components with default weights
    reward_fn.add_component(RewardComponentType.REACH_TARGET, weight=1.0)
    reward_fn.add_component(RewardComponentType.AVOID_CRASHES, weight=1.0)
    reward_fn.add_component(RewardComponentType.FLY_SMOOTHLY, weight=0.5)
    reward_fn.add_component(RewardComponentType.BE_FAST, weight=0.3)
    
    return reward_fn


def create_trajectory_following_reward() -> RewardFunction:
    """
    Create a reward function optimized for trajectory following.
    
    Returns:
        Reward function focused on trajectory following
    """
    reward_fn = RewardFunction()
    
    # Add components with weights prioritizing trajectory following
    reward_fn.add_component(
        RewardComponentType.FOLLOW_TRAJECTORY, 
        weight=1.5,
        parameters={
            "cross_track_weight": 0.7,  # Emphasize staying on path
            "progress_weight": 0.3,     # Less emphasis on speed
            "max_error": 2.0            # Maximum expected deviation (meters)
        }
    )
    reward_fn.add_component(RewardComponentType.AVOID_CRASHES, weight=1.0)
    reward_fn.add_component(RewardComponentType.FLY_SMOOTHLY, weight=0.8)
    reward_fn.add_component(RewardComponentType.BE_FAST, weight=0.2)
    
    return reward_fn


def create_wind_resistant_reward() -> RewardFunction:
    """
    Create a reward function optimized for flying in wind conditions.
    
    Returns:
        Reward function that emphasizes smooth flight and trajectory following
    """
    reward_fn = RewardFunction()
    
    # Add components with weights prioritizing stability in wind
    reward_fn.add_component(
        RewardComponentType.FOLLOW_TRAJECTORY, 
        weight=1.2,
        parameters={
            "cross_track_weight": 0.8,  # Strong emphasis on path adherence
            "progress_weight": 0.2      # Less emphasis on speed
        }
    )
    reward_fn.add_component(RewardComponentType.AVOID_CRASHES, weight=1.0)
    reward_fn.add_component(RewardComponentType.FLY_SMOOTHLY, weight=1.0)  # Emphasize smoothness
    reward_fn.add_component(RewardComponentType.BE_FAST, weight=0.1)       # Speed is less important
    
    return reward_fn


# Register reward functions
def register_reward(name: str, reward_fn: Callable) -> None:
    """Register a reward function in the global registry."""
    REGISTRY[name] = {
        'fn': reward_fn,
        'description': reward_fn.__doc__ or "No description"
    }

# Register basic rewards
register_reward("reach_target", reach_target_reward)
register_reward("avoid_crashes", avoid_crashes_reward)
register_reward("fly_smoothly", fly_smoothly_reward)
register_reward("follow_trajectory", follow_trajectory_reward)
register_reward("be_fast", be_fast_reward) 