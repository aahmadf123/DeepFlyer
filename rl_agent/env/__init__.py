import gymnasium as gym
from typing import Optional, Dict, Any, Union, Callable
import logging

from .ros_env import RosEnv, ROS_AVAILABLE
from ..registry import RewardRegistry
from ..env.wrappers import CustomRewardWrapper

# Primary PX4-ROS-COM environments (recommended)
if ROS_AVAILABLE:
    try:
        from .px4_env import (
            PX4BaseEnv,
            PX4ExplorerEnv,
            PX4ResearcherEnv,
            create_explorer_env,
            create_researcher_env
        )
        PX4_ENV_AVAILABLE = True
    except ImportError:
        PX4_ENV_AVAILABLE = False
        logging.warning("PX4 environments not available")
else:
    PX4_ENV_AVAILABLE = False

# PX4 communication utilities (primary PX4-ROS-COM, legacy MAVROS)
if ROS_AVAILABLE:
    try:
        from .px4_comm import (
            PX4Interface,
            MAVROSBridge,
            MessageConverter
        )
        PX4_COMM_AVAILABLE = True
    except ImportError:
        PX4_COMM_AVAILABLE = False

# Other ROS utilities
if ROS_AVAILABLE:
    try:
        from .ros_utils import (
            CoordinateTransform,
            StateProcessor,
            ImageProcessor,
            SafetyMonitor,
            PX4ControlMode,
        )
    except ImportError:
        logging.warning("ROS utilities not fully available")

logger = logging.getLogger(__name__)


def make_env(
    env_id: str = "CartPole-v1",
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    max_episode_steps: int = 1000,
    autoreset: bool = True,
    record_video: bool = False,
    video_folder: Optional[str] = None,
    video_length: int = 0,
    reward_function: str = "follow_trajectory",
    **kwargs
) -> gym.Env:
    """
    Factory to create environments by ID with optional reward wrapper.
    
    Args:
        env_id: Environment ID (Gymnasium ID or 'ros:namespace')
        seed: Random seed for the environment
        render_mode: Render mode for the environment
        max_episode_steps: Maximum steps per episode
        autoreset: Whether to reset the environment after reaching the end of an episode
        record_video: Whether to record video of the environment
        video_folder: Folder to save video recordings
        video_length: Length of video recordings
        reward_function: Reward function ID or callable
        **kwargs: Additional arguments passed to environment constructor
    
    Returns:
        Gym environment, potentially wrapped with custom reward
    """
    # Create base environment
    if env_id.startswith("ros:"):
        # ROS 2 environment
        if not ROS_AVAILABLE:
            logger.warning("ROS2 not available, falling back to CartPole-v1")
            env = gym.make('CartPole-v1')
        else:
            _, namespace = env_id.split(":", 1)
            try:
                # Pass reward configuration to ROS environment
                env = RosEnv(
                    namespace=namespace,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error initializing ROS env '{namespace}': {e}")
                logger.info("Falling back to CartPole-v1")
                env = gym.make('CartPole-v1')
    else:
        # Standard Gymnasium environment
        try:
            env = gym.make(env_id, **kwargs)
        except Exception as e:
            raise ValueError(f"Could not create Gym environment '{env_id}': {e}")
    
    # Apply reward wrapper if specified
    if reward_function is not None:
        if isinstance(reward_function, str):
            # Get from registry
            reward_info = RewardRegistry.get(reward_function)
            if reward_info is None:
                raise ValueError(f"Unknown reward function: {reward_function}")
            reward_fn = reward_info['fn']
        else:
            # Use callable directly
            reward_fn = reward_function
        
        # Wrap environment with custom reward
        env = CustomRewardWrapper(env, reward_fn)
        logger.info(f"Applied reward wrapper: {reward_function}")
    
    return env


# Convenience functions for creating PX4-ROS-COM environments (recommended)
def make_px4_explorer_env(**kwargs) -> gym.Env:
    """Create PX4 Explorer environment with enhanced safety (PX4-ROS-COM)"""
    if not PX4_ENV_AVAILABLE:
        raise RuntimeError("PX4 environments not available. Install required dependencies.")
    return create_explorer_env(**kwargs)


def make_px4_researcher_env(**kwargs) -> gym.Env:
    """Create PX4 Researcher environment with full control (PX4-ROS-COM)"""
    if not PX4_ENV_AVAILABLE:
        raise RuntimeError("PX4 environments not available. Install required dependencies.")
    return create_researcher_env(**kwargs)


# Legacy convenience functions
def make_drone_env(
    namespace: str = "deepflyer",
    reward_function: str = "reach_target",
    **kwargs
) -> gym.Env:
    """Create a drone environment with specified reward function."""
    return make_env(
        env_id=f"ros:{namespace}",
        reward_function=reward_function,
        **kwargs
    )


def make_cartpole_with_reward(reward_function: Union[str, Callable]) -> gym.Env:
    """Create CartPole environment with custom reward for testing."""
    return make_env("CartPole-v1", reward_function=reward_function)


# Export base functionality
__all__ = [
    'make_env',
    'make_drone_env',
    'make_cartpole_with_reward',
    'RosEnv',
    'CustomRewardWrapper',
]

# Add PX4-ROS-COM environments (primary)
if PX4_ENV_AVAILABLE:
    __all__.extend([
        'PX4BaseEnv',
        'PX4ExplorerEnv',
        'PX4ResearcherEnv',
        'create_explorer_env',
        'create_researcher_env',
        'make_px4_explorer_env',
        'make_px4_researcher_env',
    ])

# Add PX4 communication utilities (primary PX4-ROS-COM, legacy MAVROS)
if PX4_COMM_AVAILABLE:
    __all__.extend([
        'PX4Interface',
        'MAVROSBridge',
        'MessageConverter',
    ])

# Add ROS utilities if available
if ROS_AVAILABLE:
    __all__.extend([
        'CoordinateTransform',
        'StateProcessor',
        'ImageProcessor',
        'SafetyMonitor',
        'PX4ControlMode',
    ])

# Export RosEnvV2 if available
if ROS_AVAILABLE:
    try:
        from .ros_env_v2 import RosEnvV2, make_ros_env
        __all__.extend([
            'RosEnvV2',
            'make_ros_env',
        ])
    except ImportError:
        pass
