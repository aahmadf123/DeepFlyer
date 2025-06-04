import gymnasium as gym
from .ros_env import RosEnv


def make_env(env_id: str = "CartPole-v1") -> gym.Env:
    """
    Factory to create environments by ID.
    If env_id starts with 'ros:', attempts to create a RosEnv, else Gymnasium.
    """
    if env_id.startswith("ros:"):
        # ROS 2 environment
        _, name = env_id.split(":", 1)
        try:
            return RosEnv(name)
        except Exception as e:
            print(f"[env] Error initializing ROS env '{name}': {e}. Falling back to Gym '{env_id}'")
    # Default to Gymnasium environment
    try:
        return gym.make(env_id)
    except Exception as e:
        raise ValueError(f"Could not create Gym environment '{env_id}': {e}")
