"""
DeepFlyer Reinforcement Learning Reward Functions.

This module provides reward functions for the MVP hoop navigation task:
- Hoop detection and alignment
- Navigation and passage through hoops
- Return trajectory and landing
"""

import numpy as np
from enum import Enum
from typing import Dict, Any, Callable, Optional, Union

# MVP Hoop Navigation Reward Functions

class MVPRewardComponentType(Enum):
    """MVP reward component types for hoop navigation task."""
    HOOP_DETECTED = "hoop_detected"
    HORIZONTAL_ALIGN = "horizontal_align"
    VERTICAL_ALIGN = "vertical_align"
    DEPTH_CLOSER = "depth_closer"
    HOOP_PASSAGE = "hoop_passage"
    ROUNDTRIP_FINISH = "roundtrip_finish"
    COLLISION_PENALTY = "collision_penalty"
    MISSED_HOOP_PENALTY = "missed_hoop_penalty"
    DRIFT_LOST_PENALTY = "drift_lost_penalty"
    TIME_PENALTY = "time_penalty"

class MVPRewardConfig:
    """Configuration for MVP reward function with student-tunable ranges."""
    
    def __init__(self):
        # Positive Rewards (Student Tunable)
        self.hoop_detected_reward = 1.0        # Range: 1-5
        self.horizontal_align_reward = 5.0     # Range: 1-10
        self.vertical_align_reward = 5.0       # Range: 1-10
        self.depth_closer_reward = 10.0        # Range: 5-20
        self.hoop_passage_reward = 100.0       # Range: 50-200
        self.roundtrip_finish_reward = 200.0   # Range: 200-300
        
        # Penalties (Student Tunable)
        self.collision_penalty = -25.0         # Range: -10 to -100
        self.missed_hoop_penalty = -25.0       # Range: -10 to -50
        self.drift_lost_penalty = -10.0        # Range: -5 to -25
        self.time_penalty = -1.0               # Range: -0.5 to -2
        
        # Alignment thresholds
        self.alignment_threshold = 0.1         # |center| < 0.1 for alignment
        self.approach_threshold = 0.8          # Distance threshold for "closer"
        self.passage_threshold = 0.3           # Distance threshold for passage
        
    def update_from_dict(self, config_dict: Dict[str, float]):
        """Update configuration from dictionary (for student UI)."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate_ranges(self):
        """Validate that all values are within acceptable ranges."""
        # Clamp positive rewards to ranges
        self.hoop_detected_reward = np.clip(self.hoop_detected_reward, 1.0, 5.0)
        self.horizontal_align_reward = np.clip(self.horizontal_align_reward, 1.0, 10.0)
        self.vertical_align_reward = np.clip(self.vertical_align_reward, 1.0, 10.0)
        self.depth_closer_reward = np.clip(self.depth_closer_reward, 5.0, 20.0)
        self.hoop_passage_reward = np.clip(self.hoop_passage_reward, 50.0, 200.0)
        self.roundtrip_finish_reward = np.clip(self.roundtrip_finish_reward, 200.0, 300.0)
        
        # Clamp penalties to ranges (note: negative values)
        self.collision_penalty = np.clip(self.collision_penalty, -100.0, -10.0)
        self.missed_hoop_penalty = np.clip(self.missed_hoop_penalty, -50.0, -10.0)
        self.drift_lost_penalty = np.clip(self.drift_lost_penalty, -25.0, -5.0)
        self.time_penalty = np.clip(self.time_penalty, -2.0, -0.5)

def mvp_hoop_detected_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Reward for detecting a hoop in the camera frame."""
    hoop_visible = state.get("hoop_visible", 0)
    config = parameters.get("config", MVPRewardConfig())
    
    if hoop_visible > 0:
        return config.hoop_detected_reward
    return 0.0

def mvp_horizontal_align_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Reward for horizontally aligning with hoop center."""
    hoop_x_center = state.get("hoop_x_center_norm", 0.0)
    hoop_visible = state.get("hoop_visible", 0)
    config = parameters.get("config", MVPRewardConfig())
    
    if hoop_visible > 0 and abs(hoop_x_center) < config.alignment_threshold:
        return config.horizontal_align_reward
    return 0.0

def mvp_vertical_align_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Reward for vertically aligning with hoop center."""
    hoop_y_center = state.get("hoop_y_center_norm", 0.0)
    hoop_visible = state.get("hoop_visible", 0)
    config = parameters.get("config", MVPRewardConfig())
    
    if hoop_visible > 0 and abs(hoop_y_center) < config.alignment_threshold:
        return config.vertical_align_reward
    return 0.0

def mvp_depth_closer_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Reward for approaching the hoop (depth decreasing)."""
    hoop_distance = state.get("hoop_distance_norm", 1.0)
    hoop_visible = state.get("hoop_visible", 0)
    config = parameters.get("config", MVPRewardConfig())
    
    # Store previous distance for comparison
    prev_distance = parameters.get("prev_hoop_distance", 1.0)
    parameters["prev_hoop_distance"] = hoop_distance
    
    if hoop_visible > 0 and hoop_distance < prev_distance and hoop_distance < config.approach_threshold:
        # Reward proportional to distance improvement
        improvement = prev_distance - hoop_distance
        return config.depth_closer_reward * improvement * 5.0  # Scale factor
    return 0.0

def mvp_hoop_passage_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Reward for successfully passing through the hoop."""
    hoop_distance = state.get("hoop_distance_norm", 1.0)
    hoop_visible = state.get("hoop_visible", 0)
    passage_detected = info.get("hoop_passage_detected", False) if info else False
    config = parameters.get("config", MVPRewardConfig())
    
    # Check if drone just passed through hoop (distance very close + aligned)
    hoop_x_center = state.get("hoop_x_center_norm", 0.0)
    hoop_y_center = state.get("hoop_y_center_norm", 0.0)
    
    aligned = (abs(hoop_x_center) < config.alignment_threshold * 2 and 
               abs(hoop_y_center) < config.alignment_threshold * 2)
    
    if passage_detected or (hoop_distance < config.passage_threshold and aligned):
        # Mark that passage occurred
        parameters["hoop_passages"] = parameters.get("hoop_passages", 0) + 1
        return config.hoop_passage_reward
    return 0.0

def mvp_roundtrip_finish_reward(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Reward for completing the full roundtrip (both passages + landing)."""
    passages = parameters.get("hoop_passages", 0)
    landed_at_origin = info.get("landed_at_origin", False) if info else False
    config = parameters.get("config", MVPRewardConfig())
    
    if passages >= 2 and landed_at_origin:
        return config.roundtrip_finish_reward
    return 0.0

def mvp_collision_penalty(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Penalty for collision with objects or walls."""
    collision = state.get("collision", False) or (info.get("collision", False) if info else False)
    config = parameters.get("config", MVPRewardConfig())
    
    if collision:
        return config.collision_penalty
    return 0.0

def mvp_missed_hoop_penalty(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Penalty for passing beside the hoop instead of through it."""
    hoop_distance = state.get("hoop_distance_norm", 1.0)
    hoop_x_center = state.get("hoop_x_center_norm", 0.0)
    hoop_y_center = state.get("hoop_y_center_norm", 0.0)
    config = parameters.get("config", MVPRewardConfig())
    
    # Check if drone passed close to hoop but not aligned (missed)
    close_to_hoop = hoop_distance < config.passage_threshold * 1.5
    misaligned = (abs(hoop_x_center) > config.alignment_threshold * 3 or 
                  abs(hoop_y_center) > config.alignment_threshold * 3)
    
    prev_distance = parameters.get("prev_hoop_distance", 1.0)
    moving_away = hoop_distance > prev_distance
    
    if close_to_hoop and misaligned and moving_away:
        return config.missed_hoop_penalty
    return 0.0

def mvp_drift_lost_penalty(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Penalty for losing sight of the hoop."""
    hoop_visible = state.get("hoop_visible", 0)
    prev_visible = parameters.get("prev_hoop_visible", 0)
    parameters["prev_hoop_visible"] = hoop_visible
    config = parameters.get("config", MVPRewardConfig())
    
    # Penalty if hoop was visible but now lost
    if prev_visible > 0 and hoop_visible == 0:
        return config.drift_lost_penalty
    return 0.0

def mvp_time_penalty(
    state: Dict[str, Any], 
    action: Dict[str, Any],
    next_state: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None
) -> float:
    """Time penalty per timestep to encourage efficiency."""
    config = parameters.get("config", MVPRewardConfig())
    return config.time_penalty

class MVPRewardFunction:
    """Complete MVP reward function combining all components."""
    
    def __init__(self, config: Optional[MVPRewardConfig] = None):
        self.config = config or MVPRewardConfig()
        self.config.validate_ranges()
        
        # Component functions
        self.components = {
            MVPRewardComponentType.HOOP_DETECTED: mvp_hoop_detected_reward,
            MVPRewardComponentType.HORIZONTAL_ALIGN: mvp_horizontal_align_reward,
            MVPRewardComponentType.VERTICAL_ALIGN: mvp_vertical_align_reward,
            MVPRewardComponentType.DEPTH_CLOSER: mvp_depth_closer_reward,
            MVPRewardComponentType.HOOP_PASSAGE: mvp_hoop_passage_reward,
            MVPRewardComponentType.ROUNDTRIP_FINISH: mvp_roundtrip_finish_reward,
            MVPRewardComponentType.COLLISION_PENALTY: mvp_collision_penalty,
            MVPRewardComponentType.MISSED_HOOP_PENALTY: mvp_missed_hoop_penalty,
            MVPRewardComponentType.DRIFT_LOST_PENALTY: mvp_drift_lost_penalty,
            MVPRewardComponentType.TIME_PENALTY: mvp_time_penalty,
        }
        
        # Persistent parameters across timesteps
        self.parameters = {
            "config": self.config,
            "prev_hoop_distance": 1.0,
            "prev_hoop_visible": 0,
            "hoop_passages": 0,
        }
        
        # Component values for debugging/visualization
        self.component_values = {}
    
    def compute_reward(
        self, 
        state: Dict[str, Any], 
        action: Any,
        next_state: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute total reward from all components."""
        if not isinstance(action, dict):
            action_dict = {"raw": action}
        else:
            action_dict = action
        
        total_reward = 0.0
        self.component_values = {}
        
        # Compute each component
        for component_type, component_fn in self.components.items():
            component_reward = component_fn(state, action_dict, next_state, self.parameters, info)
            total_reward += component_reward
            self.component_values[component_type.value] = component_reward
        
        return total_reward
    
    def reset_episode(self):
        """Reset episode-specific parameters."""
        self.parameters["prev_hoop_distance"] = 1.0
        self.parameters["prev_hoop_visible"] = 0
        self.parameters["hoop_passages"] = 0
        self.component_values = {}
    
    def update_config(self, config_dict: Dict[str, float]):
        """Update reward configuration from student UI."""
        self.config.update_from_dict(config_dict)
        self.config.validate_ranges()

# Legacy Reward Functions

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

def create_mvp_hoop_reward(config_dict: Optional[Dict[str, float]] = None) -> MVPRewardFunction:
    """Create MVP hoop navigation reward function."""
    config = MVPRewardConfig()
    if config_dict:
        config.update_from_dict(config_dict)
    return MVPRewardFunction(config)

def register_reward(name: str, reward_fn: Callable) -> None:
    REGISTRY[name] = {
        'fn': reward_fn,
        'description': reward_fn.__doc__ or "No description"
    }

# Register legacy rewards
register_reward("follow_trajectory", follow_trajectory_reward)
register_reward("heading_error", heading_error_reward)

# Register MVP reward
register_reward("mvp_hoop_navigation", create_mvp_hoop_reward)

# User Reward Function

def reward_function(params):
    """
    DeepFlyer MVP Reward Function - Student Edition
    
    This is where YOU design your drone's behavior!
    Modify the values and logic below to train your drone to:
    1. Take off from Point A
    2. Scan 360 degrees to find the hoop
    3. Navigate through the hoop  
    4. Return through the same hoop
    5. Land back at Point A
    
    Just like AWS DeepRacer, but for autonomous drones!
    
    Input Parameters (what your drone can "see"):
    - hoop_x_center_norm: Horizontal position of hoop center (-1.0 = left, +1.0 = right)
    - hoop_y_center_norm: Vertical position of hoop center (-1.0 = down, +1.0 = up)  
    - hoop_visible: Is the hoop visible? (0 = no, 1 = yes)
    - hoop_distance_norm: Distance to hoop (0.0 = very close, 1.0 = far away)
    - drone_vx_norm: Forward/backward velocity (-1.0 to +1.0)
    - drone_vy_norm: Left/right velocity (-1.0 to +1.0)
    - drone_vz_norm: Up/down velocity (-1.0 to +1.0)
    - yaw_rate_norm: Turning rate (-1.0 to +1.0)
    - all_systems_normal: Is everything OK? (True/False)
    - speed: Overall drone speed (0.0 to 2.0 m/s)
    - collision: Did we hit something? (True/False)
    - hoop_passages_completed: How many times through hoop (0, 1, or 2)
    - flight_phase: Current mission phase (TAKEOFF, SCAN_360, NAVIGATE_TO_HOOP, etc.)
    """
    
    # Read input parameters
    hoop_x_center = params.get('hoop_x_center_norm', 0.0)
    hoop_y_center = params.get('hoop_y_center_norm', 0.0)
    hoop_visible = params.get('hoop_visible', 0)
    hoop_distance = params.get('hoop_distance_norm', 1.0)
    
    drone_vx = params.get('drone_vx_norm', 0.0)
    drone_vy = params.get('drone_vy_norm', 0.0)
    drone_vz = params.get('drone_vz_norm', 0.0)
    yaw_rate = params.get('yaw_rate_norm', 0.0)
    
    all_systems_normal = params.get('all_systems_normal', True)
    speed = params.get('speed', 0.0)
    collision = params.get('collision', False)
    hoop_passages = params.get('hoop_passages_completed', 0)
    flight_phase = params.get('flight_phase', 'TAKEOFF')
    
    # Calculate 5 alignment markers farther away from hoop center
    perfect_center = 0.05      # Within 5% = perfect alignment
    marker_1 = 0.10           # Within 10% = excellent alignment  
    marker_2 = 0.20           # Within 20% = good alignment
    marker_3 = 0.30           # Within 30% = okay alignment
    marker_4 = 0.50           # Within 50% = poor alignment
    
    # Distance thresholds for approach rewards
    very_close = 0.20         # Within 20% distance = very close
    close = 0.40              # Within 40% distance = close
    medium_distance = 0.60    # Within 60% distance = medium
    far_distance = 0.80       # Within 80% distance = far
    
    # Speed threshold
    SPEED_THRESHOLD = 1.0
    
    # Calculate total alignment error
    alignment_error = abs(hoop_x_center) + abs(hoop_y_center)
    
    # Give higher reward if the drone is aligned with hoop center
    if hoop_visible and alignment_error <= perfect_center and all_systems_normal:
        reward = 10.0
    elif hoop_visible and alignment_error <= marker_1 and all_systems_normal:
        reward = 7.5
    elif hoop_visible and alignment_error <= marker_2 and all_systems_normal:
        reward = 5.0
    elif hoop_visible and alignment_error <= marker_3 and all_systems_normal:
        reward = 2.5
    elif hoop_visible and alignment_error <= marker_4 and all_systems_normal:
        reward = 1.0
    elif hoop_visible:
        reward = 0.5          # At least hoop is visible
    else:
        reward = 0.1          # Still searching for hoop
    
    # Distance bonus - reward for being close to the hoop
    if hoop_visible and hoop_distance <= very_close:
        reward += 8.0         # Almost there!
    elif hoop_visible and hoop_distance <= close:
        reward += 5.0         # Getting close!
    elif hoop_visible and hoop_distance <= medium_distance:
        reward += 2.0         # I can see it clearly
    elif hoop_visible and hoop_distance <= far_distance:
        reward += 1.0         # Found the target
    
    # Mission progress rewards - the big wins!
    if hoop_passages == 1:
        reward += 100.0       # First passage complete!
    elif hoop_passages == 2:
        reward += 200.0       # Mission accomplished!
    
    # Search behavior rewards when hoop not visible
    if not hoop_visible and flight_phase == "SCAN_360":
        if abs(yaw_rate) > 0.3:
            reward += 1.0     # Good, keep scanning!
    
    # Phase-specific bonuses
    if flight_phase == "THROUGH_HOOP_FIRST" and hoop_visible:
        reward += 5.0         # Lined up for first pass
    elif flight_phase == "THROUGH_HOOP_SECOND" and hoop_visible:
        reward += 8.0         # Lined up for return pass
    elif flight_phase == "LANDING" and hoop_passages >= 2:
        reward += 10.0        # Coming home victorious!
    
    # Speed management
    if speed < SPEED_THRESHOLD:
        reward += 0.5         # Could be faster
    else:
        reward += 2.0         # Good speed!
    
    # Safety penalties
    if collision:
        reward = 1e-3         # Crash penalty - restart learning
    
    if not all_systems_normal:
        reward = reward * 0.5 # System issues penalty
    
    # Small time penalty to encourage efficiency
    reward -= 0.1
    
    # Ensure reward stays within reasonable bounds
    reward = max(1e-3, min(300.0, reward))
    
    return float(reward)

def _calculate_path_reward(distance_from_path, path_width, on_path, HIGH_PATH_REWARD, MEDIUM_PATH_REWARD, LOW_PATH_REWARD, OFF_PATH_PENALTY):
    """Calculate reward based on how close drone is to the path center."""
    # Distance markers from center of path
    marker_1 = 0.1 * path_width
    marker_2 = 0.25 * path_width  
    marker_3 = 0.4 * path_width
    marker_4 = 0.5 * path_width
    
    if distance_from_path <= marker_1 and on_path:
        return HIGH_PATH_REWARD
    elif distance_from_path <= marker_2 and on_path:
        return MEDIUM_PATH_REWARD
    elif distance_from_path <= marker_3 and on_path:
        return LOW_PATH_REWARD
    elif distance_from_path <= marker_4 and on_path:
        return 0.5 * LOW_PATH_REWARD
    else:
        return OFF_PATH_PENALTY

def _calculate_altitude_reward(altitude_error, tolerance, ALTITUDE_REWARD, ALTITUDE_PARTIAL_REWARD):
    """Calculate reward based on altitude accuracy."""
    if abs(altitude_error) < tolerance:
        return ALTITUDE_REWARD
    elif abs(altitude_error) < tolerance * 2:
        return ALTITUDE_PARTIAL_REWARD
    else:
        return 0.0

def _calculate_heading_reward(heading_error, tolerance, HEADING_REWARD, HEADING_PARTIAL_REWARD):
    """Calculate reward based on heading accuracy."""
    if abs(heading_error) < tolerance:
        return HEADING_REWARD
    elif abs(heading_error) < tolerance * 3:
        return HEADING_PARTIAL_REWARD
    else:
        return 0.0 