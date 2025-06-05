# rl_agent/rewards.py
# Implementations of preset reward functions

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass
from enum import Enum
import inspect

logger = logging.getLogger(__name__)


class RewardComponent(Enum):
    """Types of reward components."""
    REACH_TARGET = "reach_target"
    AVOID_CRASHES = "avoid_crashes"
    FLY_STEADY = "fly_steady"
    MINIMIZE_TIME = "minimize_time"
    ENERGY_EFFICIENCY = "energy_efficiency"
    SMOOTH_CONTROL = "smooth_control"
    FOLLOW_TRAJECTORY = "follow_trajectory"
    ALTITUDE_CONTROL = "altitude_control"
    CUSTOM = "custom"


@dataclass
class RewardConfig:
    """Configuration for a reward component."""
    component_type: RewardComponent
    weight: float = 1.0
    parameters: Dict[str, Any] = None
    active: bool = True
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class RewardRegistry:
    """
    Registry of reward functions.
    
    This class maintains a collection of reward functions that can be
    used by environments for reinforcement learning.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._reward_functions = {}
    
    def register(
        self,
        name: str,
        component_type: RewardComponent,
        function: Callable,
        description: str = "",
    ):
        """
        Register a reward function.
        
        Args:
            name: Name of the reward function
            component_type: Type of reward component
            function: The reward function to register
            description: Description of what the reward function does
        """
        self._reward_functions[name] = {
            'component_type': component_type,
            'function': function,
            'description': description,
            'signature': inspect.signature(function),
        }
        logger.debug(f"Registered reward function '{name}' of type {component_type.value}")
    
    def get(self, name: str) -> Optional[Callable]:
        """
        Get a reward function by name.
        
        Args:
            name: Name of the reward function
            
        Returns:
            The reward function or None if not found
        """
        if name in self._reward_functions:
            return self._reward_functions[name]['function']
        else:
            logger.warning(f"Reward function '{name}' not found")
            return None
    
    def get_all(self) -> Dict[str, Dict]:
        """
        Get all registered reward functions.
        
        Returns:
            Dictionary of reward function entries
        """
        return self._reward_functions
    
    def get_names(self) -> List[str]:
        """
        Get names of all registered reward functions.
        
        Returns:
            List of reward function names
        """
        return list(self._reward_functions.keys())
    
    def get_by_type(self, component_type: RewardComponent) -> Dict[str, Dict]:
        """
        Get reward functions of a specific type.
        
        Args:
            component_type: Type of reward component
            
        Returns:
            Dictionary of reward functions of the specified type
        """
        return {
            name: info for name, info in self._reward_functions.items()
            if info['component_type'] == component_type
        }
    
    def describe(self, name: str) -> Optional[str]:
        """
        Get description of a reward function.
        
        Args:
            name: Name of the reward function
            
        Returns:
            Description of the reward function or None if not found
        """
        if name in self._reward_functions:
            return self._reward_functions[name]['description']
        else:
            return None


class RewardFunction:
    """
    Configurable reward function for reinforcement learning.
    
    This class combines multiple reward components with weights to create
    a composite reward function.
    """
    
    def __init__(
        self,
        registry: RewardRegistry,
        components: Optional[List[RewardConfig]] = None,
        normalize: bool = True,
    ):
        """
        Initialize reward function with components.
        
        Args:
            registry: Registry containing reward components
            components: List of reward component configurations
            normalize: Whether to normalize final reward to [-1, 1]
        """
        self.registry = registry
        self.components = components or []
        self.normalize = normalize
        
        # For tracking reward statistics
        self.min_reward = 0.0
        self.max_reward = 0.0
        self.sum_reward = 0.0
        self.count = 0
        self.component_values = {}
    
    def add_component(
        self,
        name: str,
        weight: float = 1.0,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a reward component.
        
        Args:
            name: Name of the reward function in the registry
            weight: Weight of this component
            parameters: Parameters to pass to the reward function
        """
        if name not in self.registry.get_names():
            logger.warning(f"Reward function '{name}' not found in registry")
            return
            
        component_type = self.registry._reward_functions[name]['component_type']
        
        self.components.append(
            RewardConfig(
                component_type=component_type,
                weight=weight,
                parameters=parameters or {},
                active=True
            )
        )
    
    def remove_component(self, index: int) -> bool:
        """
        Remove a reward component by index.
        
        Args:
            index: Index of component to remove
            
        Returns:
            True if removed, False if index invalid
        """
        if 0 <= index < len(self.components):
            self.components.pop(index)
            return True
        return False
    
    def compute_reward(
        self,
        state: Dict[str, Any],
        action: Optional[np.ndarray] = None,
        next_state: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute combined reward from all components.
        
        Args:
            state: Current state dictionary
            action: Action taken
            next_state: Resulting state
            info: Additional info dictionary
            
        Returns:
            Combined reward value
        """
        if not self.components:
            logger.warning("No reward components defined")
            return 0.0
        
        total_reward = 0.0
        self.component_values = {}
        
        for i, config in enumerate(self.components):
            if not config.active:
                continue
                
            # Find the function name from the component type
            function_names = [
                name for name, info in self.registry.get_all().items()
                if info['component_type'] == config.component_type
            ]
            
            if not function_names:
                logger.warning(f"No reward functions found for component type {config.component_type.value}")
                continue
                
            # Use the first function that matches the component type
            function_name = function_names[0]
            function = self.registry.get(function_name)
            
            if function is None:
                continue
                
            # Build arguments for the function
            kwargs = {
                'state': state,
            }
            
            # Add optional arguments if the function accepts them
            sig = self.registry._reward_functions[function_name]['signature']
            
            if 'action' in sig.parameters and action is not None:
                kwargs['action'] = action
                
            if 'next_state' in sig.parameters and next_state is not None:
                kwargs['next_state'] = next_state
                
            if 'info' in sig.parameters and info is not None:
                kwargs['info'] = info
                
            # Add custom parameters from config
            for param_name, param_value in config.parameters.items():
                if param_name in sig.parameters:
                    kwargs[param_name] = param_value
                    
            try:
                component_reward = function(**kwargs)
                
                # Store individual component value for analysis
                self.component_values[function_name] = component_reward
                
                # Apply weight and add to total
                total_reward += component_reward * config.weight
                
            except Exception as e:
                logger.error(f"Error computing reward for component {function_name}: {e}")
        
        # Update statistics
        self.count += 1
        self.sum_reward += total_reward
        self.min_reward = min(self.min_reward, total_reward)
        self.max_reward = max(self.max_reward, total_reward)
        
        # Normalize if requested
        if self.normalize and self.count > 1:
            # Use exponential moving average for normalization to adapt over time
            normalize_min = self.min_reward
            normalize_max = max(self.max_reward, normalize_min + 0.1)  # Ensure positive range
            
            range_size = normalize_max - normalize_min
            if range_size > 0:
                total_reward = 2.0 * (total_reward - normalize_min) / range_size - 1.0
                total_reward = np.clip(total_reward, -1.0, 1.0)
        
        return float(total_reward)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get reward statistics.
        
        Returns:
            Dictionary with reward statistics
        """
        mean_reward = self.sum_reward / max(1, self.count)
        return {
            'min_reward': self.min_reward,
            'max_reward': self.max_reward,
            'mean_reward': mean_reward,
            'count': self.count,
        }
    
    def reset_stats(self):
        """Reset reward statistics."""
        self.min_reward = 0.0
        self.max_reward = 0.0
        self.sum_reward = 0.0
        self.count = 0
    
    def __str__(self) -> str:
        """Get string representation of the reward function."""
        components = [
            f"{i}: {c.component_type.value} (w={c.weight})"
            for i, c in enumerate(self.components)
        ]
        return f"RewardFunction with {len(components)} components: {', '.join(components)}"


# Create global registry
REGISTRY = RewardRegistry()


# ============================
# Common reward functions
# ============================

def reach_target(
    state: Dict[str, Any],
    goal_position: Optional[np.ndarray] = None,
    distance_threshold: float = 0.2,
    min_distance: float = 0.0,
    max_distance: float = 10.0,
    use_sparse: bool = False,
    success_reward: float = 1.0,
    power: float = 1.0,
) -> float:
    """
    Reward for reaching a target position.
    
    Args:
        state: Current state dictionary with 'position' key
        goal_position: Target position [x, y, z], or use state['goal_position']
        distance_threshold: Distance considered "reached" for sparse reward
        min_distance: Distance at which dense reward is maximum
        max_distance: Distance at which dense reward starts
        use_sparse: Whether to use sparse (0/1) or dense reward
        success_reward: Reward value when goal is reached
        power: Power to raise normalized distance to (1=linear, 2=quadratic)
        
    Returns:
        Reward value
    """
    # Get positions
    if 'position' not in state:
        return 0.0
        
    position = state['position']
    
    # Get goal position
    if goal_position is None:
        if 'goal_position' in state:
            goal_position = state['goal_position']
        else:
            return 0.0
    
    # Calculate distance to goal
    distance = np.linalg.norm(position - goal_position)
    
    # Sparse reward
    if use_sparse:
        return float(success_reward) if distance <= distance_threshold else 0.0
    
    # Dense reward - higher as drone gets closer to goal
    if distance <= min_distance:
        return float(success_reward)
    elif distance >= max_distance:
        return 0.0
    else:
        # Normalize distance to [0, 1] range and invert
        normalized_dist = (max_distance - distance) / (max_distance - min_distance)
        # Apply power function for non-linear scaling
        return float(success_reward * (normalized_dist ** power))


def avoid_crashes(
    state: Dict[str, Any],
    min_distance: float = 0.5,
    max_distance: float = 2.0,
    collision_penalty: float = -1.0,
) -> float:
    """
    Reward for avoiding crashes/collisions.
    
    Args:
        state: Current state dictionary with obstacle distance
        min_distance: Distance below which maximum penalty applies
        max_distance: Distance above which no penalty applies
        collision_penalty: Penalty value for collision
        
    Returns:
        Reward value (negative for being close to obstacles)
    """
    # Check for collision flag
    if 'collision_flag' in state and state['collision_flag']:
        return float(collision_penalty)
    
    # Get obstacle distance - check multiple possible keys
    obstacle_distance = None
    
    for key in ['obstacle_distance', 'distance_to_obstacle']:
        if key in state:
            value = state[key]
            if isinstance(value, np.ndarray) and value.size > 0:
                obstacle_distance = float(value.min())
            else:
                obstacle_distance = float(value)
            break
            
    if obstacle_distance is None:
        return 0.0  # No obstacle information available
    
    # Apply penalty based on distance
    if obstacle_distance <= min_distance:
        return float(collision_penalty)
    elif obstacle_distance >= max_distance:
        return 0.0  # No penalty when far from obstacles
    else:
        # Scale penalty linearly between min and max distance
        normalized_dist = (obstacle_distance - min_distance) / (max_distance - min_distance)
        return float(collision_penalty * (1 - normalized_dist))


def fly_steady(
    state: Dict[str, Any],
    action: Optional[np.ndarray] = None,
    prev_action: Optional[np.ndarray] = None,
    max_velocity: float = 1.0,
    max_angular_velocity: float = 0.5,
    velocity_weight: float = 0.6,
    angular_weight: float = 0.4,
    max_action_diff: float = 0.3,
    action_diff_weight: float = 0.2,
) -> float:
    """
    Reward for flying steadily with minimal jerky movements.
    
    Args:
        state: Current state dictionary
        action: Current action
        prev_action: Previous action (used for action smoothness)
        max_velocity: Maximum linear velocity for normalization
        max_angular_velocity: Maximum angular velocity for normalization
        velocity_weight: Weight for linear velocity stability
        angular_weight: Weight for angular velocity stability
        max_action_diff: Maximum action difference for normalization
        action_diff_weight: Weight for action smoothness
        
    Returns:
        Reward value for flying steadily
    """
    # Initialize components
    smoothness_reward = 0.0
    
    # Check for velocity in state
    if 'linear_velocity' in state:
        linear_vel = state['linear_velocity']
        # Penalize high velocities
        vel_magnitude = np.linalg.norm(linear_vel)
        vel_penalty = min(1.0, vel_magnitude / max_velocity)
        smoothness_reward += velocity_weight * (1.0 - vel_penalty)
    
    # Check for angular velocity
    if 'angular_velocity' in state:
        angular_vel = state['angular_velocity']
        # Penalize high angular velocities
        ang_magnitude = np.linalg.norm(angular_vel)
        ang_penalty = min(1.0, ang_magnitude / max_angular_velocity)
        smoothness_reward += angular_weight * (1.0 - ang_penalty)
    
    # Check action smoothness if both current and previous actions available
    if action is not None and prev_action is not None:
        action_diff = np.linalg.norm(action - prev_action)
        # Reward smooth action transitions
        action_smoothness = max(0.0, 1.0 - action_diff / max_action_diff)
        smoothness_reward += action_diff_weight * action_smoothness
    
    # Ensure we have at least some components
    if not any(key in state for key in ['linear_velocity', 'angular_velocity']):
        return 0.0
        
    # Normalize based on provided weights to ensure range [0, 1]
    total_weight = (
        (velocity_weight if 'linear_velocity' in state else 0.0) +
        (angular_weight if 'angular_velocity' in state else 0.0) +
        (action_diff_weight if action is not None and prev_action is not None else 0.0)
    )
    
    if total_weight > 0:
        smoothness_reward = smoothness_reward / total_weight
        
    return float(smoothness_reward)


def follow_trajectory(
    state: Dict[str, Any],
    trajectory: List[np.ndarray],
    max_deviation: float = 1.0,
    completion_reward: float = 1.0,
) -> float:
    """
    Reward for following a predefined trajectory.
    
    Args:
        state: Current state dictionary with position and trajectory progress
        trajectory: List of waypoints [x, y, z]
        max_deviation: Maximum allowed deviation from trajectory
        completion_reward: Reward for completing trajectory
        
    Returns:
        Reward value for trajectory following
    """
    if 'position' not in state or 'trajectory_idx' not in state:
        return 0.0
    
    position = state['position']
    current_idx = state['trajectory_idx']
    
    # Check if trajectory completed
    if current_idx >= len(trajectory) - 1:
        return float(completion_reward)
    
    # Get current and next waypoint
    current_waypoint = trajectory[current_idx]
    next_waypoint = trajectory[current_idx + 1]
    
    # Calculate the vector from current to next waypoint
    trajectory_vector = next_waypoint - current_waypoint
    trajectory_length = np.linalg.norm(trajectory_vector)
    
    if trajectory_length < 1e-6:
        # Waypoints too close, skip
        return 0.0
        
    # Normalize trajectory vector
    trajectory_direction = trajectory_vector / trajectory_length
    
    # Calculate vector from current waypoint to drone
    drone_vector = position - current_waypoint
    
    # Project drone position onto trajectory line
    projection = np.dot(drone_vector, trajectory_direction)
    
    # Calculate perpendicular distance from drone to trajectory line
    if projection < 0:
        # Drone is behind current waypoint
        deviation = np.linalg.norm(position - current_waypoint)
    elif projection > trajectory_length:
        # Drone is beyond next waypoint
        deviation = np.linalg.norm(position - next_waypoint)
    else:
        # Drone is somewhere along trajectory segment
        projected_point = current_waypoint + projection * trajectory_direction
        deviation = np.linalg.norm(position - projected_point)
    
    # Calculate reward based on deviation
    if deviation > max_deviation:
        return 0.0
    else:
        return float(completion_reward * (1.0 - deviation / max_deviation))


def maintain_altitude(
    state: Dict[str, Any],
    target_altitude: float,
    tolerance: float = 0.2,
    max_deviation: float = 2.0,
) -> float:
    """
    Reward for maintaining a specific altitude.
    
    Args:
        state: Current state dictionary with position
        target_altitude: Target altitude to maintain
        tolerance: Tolerance within which maximum reward is given
        max_deviation: Maximum deviation after which reward becomes zero
        
    Returns:
        Reward value for altitude control
    """
    if 'position' not in state:
        return 0.0
        
    # Extract current altitude (z-coordinate)
    current_altitude = float(state['position'][2])
    
    # Calculate deviation from target
    deviation = abs(current_altitude - target_altitude)
    
    # Maximum reward if within tolerance
    if deviation <= tolerance:
        return 1.0
        
    # Zero reward if beyond max deviation
    if deviation >= max_deviation:
        return 0.0
        
    # Linear interpolation between tolerance and max_deviation
    normalized = (max_deviation - deviation) / (max_deviation - tolerance)
    return float(normalized)


def minimize_energy(
    state: Dict[str, Any],
    action: np.ndarray,
    energy_coeff: float = 0.01,
) -> float:
    """
    Reward for minimizing energy usage (smaller control inputs).
    
    Args:
        state: Current state dictionary
        action: Control action taken
        energy_coeff: Coefficient for energy penalty
        
    Returns:
        Reward value (negative for high energy usage)
    """
    if action is None:
        return 0.0
        
    # Calculate squared magnitude of action as proxy for energy
    energy_used = np.sum(action**2)
    
    # Return negative reward (penalty) for energy usage
    return float(-energy_coeff * energy_used)


def time_efficiency(
    state: Dict[str, Any],
    time_penalty: float = -0.01,
    reached_goal: bool = False,
    goal_reward: float = 1.0,
) -> float:
    """
    Penalty for taking time to complete the task.
    
    Args:
        state: Current state dictionary
        time_penalty: Penalty per timestep
        reached_goal: Whether goal is reached in this step
        goal_reward: Reward for reaching goal
        
    Returns:
        Reward/penalty value
    """
    # Check if goal is reached from state instead of parameter if available
    if 'reached_goal' in state:
        reached_goal = bool(state['reached_goal'])
        
    # Give goal reward if reached
    if reached_goal:
        return float(goal_reward)
        
    # Otherwise, apply time penalty
    return float(time_penalty)


# Register the standard reward functions
REGISTRY.register(
    name="reach_target",
    component_type=RewardComponent.REACH_TARGET,
    function=reach_target,
    description="Reward for reaching a target position in 3D space"
)

REGISTRY.register(
    name="avoid_crashes",
    component_type=RewardComponent.AVOID_CRASHES,
    function=avoid_crashes,
    description="Penalty for getting close to obstacles or crashing"
)

REGISTRY.register(
    name="fly_steady",
    component_type=RewardComponent.FLY_STEADY,
    function=fly_steady,
    description="Reward for flying with minimal jerky movements"
)

REGISTRY.register(
    name="follow_trajectory",
    component_type=RewardComponent.FOLLOW_TRAJECTORY,
    function=follow_trajectory,
    description="Reward for following a predefined trajectory"
)

REGISTRY.register(
    name="maintain_altitude",
    component_type=RewardComponent.ALTITUDE_CONTROL,
    function=maintain_altitude,
    description="Reward for maintaining a specific altitude"
)

REGISTRY.register(
    name="minimize_energy",
    component_type=RewardComponent.ENERGY_EFFICIENCY,
    function=minimize_energy,
    description="Reward for minimizing energy usage (smaller control inputs)"
)

REGISTRY.register(
    name="time_efficiency",
    component_type=RewardComponent.MINIMIZE_TIME,
    function=time_efficiency,
    description="Penalty for taking time to complete the task"
)


def create_default_reward_function() -> RewardFunction:
    """
    Create a default reward function with standard components.
    
    Returns:
        RewardFunction with standard components
    """
    reward_fn = RewardFunction(REGISTRY)
    
    # Add standard components with default weights
    reward_fn.add_component("reach_target", weight=1.0)
    reward_fn.add_component("avoid_crashes", weight=1.0)
    reward_fn.add_component("fly_steady", weight=0.3)
    reward_fn.add_component("time_efficiency", weight=0.1)
    
    return reward_fn


if __name__ == "__main__":
    # Test reward functions with dummy state
    dummy_state = {
        'position': np.array([1.0, 1.0, 1.0]),
        'goal_position': np.array([5.0, 5.0, 2.0]),
        'linear_velocity': np.array([0.1, 0.2, 0.0]),
        'angular_velocity': np.array([0.0, 0.0, 0.1]),
        'obstacle_distance': 1.5,
    }
    
    # Test registry and reward functions
    for name in REGISTRY.get_names():
        fn = REGISTRY.get(name)
        value = fn(dummy_state)
        print(f"{name}: {value}")
        
    # Test composite reward function
    reward_fn = create_default_reward_function()
    composite_reward = reward_fn.compute_reward(dummy_state)
    print(f"Composite reward: {composite_reward}")
    print(f"Component values: {reward_fn.component_values}") 