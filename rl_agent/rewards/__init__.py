"""
DeepFlyer Reward Functions Package

This package contains reward functions for hoop navigation training,
following AWS DeepRacer patterns for easy customization.
"""

from .rewards import (
    reward_function,
    HoopNavigationReward,
    RewardConfig,
    get_reward_preset,
    REWARD_PRESETS
)

__all__ = [
    'reward_function',
    'HoopNavigationReward', 
    'RewardConfig',
    'get_reward_preset',
    'REWARD_PRESETS'
]
