"""
PX4-ROS-COM Environment Classes for DeepFlyer
Production-ready drone RL environments using direct PX4 communication
"""

# Import all classes from px4_base_env for compatibility
from .px4_base_env import (
    PX4BaseEnv,
    PX4ExplorerEnv, 
    PX4ResearcherEnv,
    create_explorer_env,
    create_researcher_env,
    # Legacy MAVROS aliases (deprecated)
    MAVROSExplorerEnv,
    MAVROSResearcherEnv
)

__all__ = [
    'PX4BaseEnv',
    'PX4ExplorerEnv', 
    'PX4ResearcherEnv',
    'create_explorer_env',
    'create_researcher_env',
    # Legacy aliases for backward compatibility
    'MAVROSExplorerEnv',
    'MAVROSResearcherEnv'
] 