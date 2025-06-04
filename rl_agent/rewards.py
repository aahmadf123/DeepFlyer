# rl_agent/rewards.py
# Implementations of preset reward functions

import numpy as np

def reach_target_reward(state: dict, action: dict) -> float:
    """
    Reward ∈ [0,1] as the drone approaches the target.
    state["position"]: (x,y,z)
    state["goal"]: (gx,gy,gz)
    state["max_room_diagonal"]: precomputed diagonal
    """
    pos = np.array(state.get("position", [0,0,0]), dtype=float)
    goal = np.array(state.get("goal", [0,0,0]), dtype=float)
    diag = float(state.get("max_room_diagonal", 1.0))
    dist = np.linalg.norm(pos - goal)
    return float(max(0.0, 1.0 - (dist / (diag + 1e-8))))


def avoid_crashes_reward(state: dict, action: dict) -> float:
    """
    Penalty for collisions or very close obstacles.
    state["collision_flag"]: bool
    state["dist_to_obstacle"]: float
    """
    if state.get("collision_flag", False):
        return -1.0
    dist = float(state.get("dist_to_obstacle", np.inf))
    if dist < 0.2:
        return -0.5
    return 0.0


def save_energy_reward(state: dict, action: dict) -> float:
    """
    Encourage low throttle usage (energy efficiency).
    action["throttle"]: float in [0,1]
    """
    throttle = float(action.get("throttle", 0.0))
    return float(1.0 - throttle)


def fly_steady_reward(state: dict, action: dict) -> float:
    """
    Maintain altitude smoothly with minimal vertical speed.
    state["altitude"], state["target_altitude"], state["vertical_velocity"], state["max_altitude_error"]
    """
    z = float(state.get("altitude", 0.0))
    target_z = float(state.get("target_altitude", z))
    max_err = float(state.get("max_altitude_error", 1.0))
    vz = abs(float(state.get("vertical_velocity", 0.0)))
    alt_err = abs(z - target_z)
    altitude_component = max(0.0, 1.0 - (alt_err / (max_err + 1e-8)))
    speed_penalty = 0.5 * vz
    return float(altitude_component - speed_penalty)


def fly_smoothly_reward(state: dict, action: dict) -> float:
    """
    Penalize sudden changes in velocity and angular rates (jerk).
    state["curr_velocity"], state["prev_velocity"], state["curr_angular_velocity"], state["prev_angular_velocity"], state["dt"], state["max_lin_jerk"], state["max_ang_jerk"]
    """
    curr_v = np.array(state.get("curr_velocity", [0,0,0]), dtype=float)
    prev_v = np.array(state.get("prev_velocity", [0,0,0]), dtype=float)
    dt = float(state.get("dt", 1e-2))
    lin_jerk = np.linalg.norm(curr_v - prev_v) / (dt + 1e-8)
    ang_curr = float(state.get("curr_angular_velocity", 0.0))
    ang_prev = float(state.get("prev_angular_velocity", 0.0))
    ang_diff = abs(ang_curr - ang_prev)
    max_lin = float(state.get("max_lin_jerk", 1.0))
    max_ang = float(state.get("max_ang_jerk", 1.0))
    lin_penalty = min(1.0, lin_jerk / (max_lin + 1e-8))
    ang_penalty = min(1.0, ang_diff / (max_ang + 1e-8))
    return float(max(0.0, 1.0 - 0.5 * lin_penalty - 0.5 * ang_penalty))


def be_fast_reward(state: dict, action: dict) -> float:
    """
    Reward forward speed and time-to-goal bonus.
    state["curr_velocity"], state.get("at_goal"), state["time_elapsed"], state["max_time_allowed"], state["max_speed"]
    """
    speed_vec = np.array(state.get("curr_velocity", [0,0,0]), dtype=float)
    speed = np.linalg.norm(speed_vec)
    max_speed = float(state.get("max_speed", 1.0))
    if state.get("at_goal", False):
        t = float(state.get("time_elapsed", 0.0))
        tmax = float(state.get("max_time_allowed", t + 1e-8))
        return float(1.0 + (tmax - t) / (tmax + 1e-8))
    else:
        return float(speed / (max_speed + 1e-8))


def path_efficiency_reward(state: dict, action: dict) -> float:
    """
    Reward based on ratio of straight-line to actual path efficiency.
    state["distance_traveled"], state["straight_line_dist"], state["prev_to_goal_dist"], state["curr_to_goal_dist"], state.get("at_goal")
    """
    dist_traveled = float(state.get("distance_traveled", 0.0))
    straight_line = float(state.get("straight_line_dist", 1e-3))
    if state.get("at_goal", False):
        # Efficiency at goal
        return float(straight_line / max(dist_traveled, 1e-3))
    else:
        prev_d = float(state.get("prev_to_goal_dist", straight_line))
        curr_d = float(state.get("curr_to_goal_dist", straight_line))
        delta = prev_d - curr_d
        return float(delta / (straight_line + 1e-8))


def adaptive_disturbance_reward(state: dict, action: dict) -> float:
    """
    Reward how well actions counter external disturbances.
    state["external_force"], action["thrust_vector"]
    """
    ext = np.array(state.get("external_force", [0.0,0.0,0.0]), dtype=float)
    thrust = np.array(action.get("thrust_vector", [0.0,0.0,0.0]), dtype=float)
    disturbance_mag = np.linalg.norm(ext)
    # projection magnitude of thrust onto external force
    comp_mag = 0.0
    if disturbance_mag > 1e-8:
        comp_mag = abs(np.dot(thrust, ext)) / disturbance_mag
    # normalized compensation minus small penalty for disturbance
    return float((comp_mag / (disturbance_mag + 1e-8)) - 0.1 * disturbance_mag)


def multi_objective_reward(state: dict, action: dict, weights: dict) -> float:
    """
    Weighted sum of multiple preset reward components.
    weights: dict with keys 'reach', 'collision', 'energy', 'speed'
    """
    # retrieve weights or default to 1.0
    w_reach = float(weights.get("reach", 1.0))
    w_coll = float(weights.get("collision", 1.0))
    w_energy = float(weights.get("energy", 1.0))
    w_speed = float(weights.get("speed", 1.0))

    # compute each component
    r_reach = reach_target_reward(state, action)
    r_coll = avoid_crashes_reward(state, action)
    r_energy = save_energy_reward(state, action)
    # step-wise speed towards goal
    straight_line = float(state.get("straight_line_dist", 1e-3))
    prev_d = float(state.get("prev_to_goal_dist", straight_line))
    curr_d = float(state.get("curr_to_goal_dist", straight_line))
    delta = prev_d - curr_d
    r_speed = float(delta / (straight_line + 1e-8)) if straight_line > 0 else 0.0

    return float(w_reach * r_reach + w_coll * r_coll + w_energy * r_energy + w_speed * r_speed) 