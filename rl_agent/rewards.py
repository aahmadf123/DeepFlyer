# rl_agent/rewards.py
# Stub implementations for preset reward functions

def reach_target_reward(state: dict, action: dict) -> float:
    """
    Reward ∈ [0,1] as the drone approaches the target.
    """
    # TODO: implement actual distance-based reward
    return 0.0


def avoid_crashes_reward(state: dict, action: dict) -> float:
    """
    Penalty for collisions or very close obstacles.
    """
    # TODO: implement actual collision penalty logic
    return 0.0


def save_energy_reward(state: dict, action: dict) -> float:
    """
    Encourage low throttle usage (energy efficiency).
    """
    # TODO: implement actual energy-based reward
    return 0.0


def fly_steady_reward(state: dict, action: dict) -> float:
    """
    Maintain altitude smoothly with minimal vertical speed.
    """
    # TODO: implement actual altitude control reward
    return 0.0


def fly_smoothly_reward(state: dict, action: dict) -> float:
    """
    Penalize sudden changes in velocity and angular rates (jerk).
    """
    # TODO: implement actual jerk-based reward
    return 0.0


def be_fast_reward(state: dict, action: dict) -> float:
    """
    Reward forward speed and time-to-goal bonus.
    """
    # TODO: implement actual speed-based reward
    return 0.0 