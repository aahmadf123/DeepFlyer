"""
Configuration constants for RL & AI module.
Defines available algorithms for Explorer and Researcher modes.
"""

from typing import List

# Explorer mode: beginner-friendly algorithms
EXPLORER_ALGOS: List[str] = [
    "ppo",        # Proximal Policy Optimization
    "sac",        # Soft Actor-Critic
]

# Researcher mode: advanced algorithms for power users
RESEARCHER_ALGOS: List[str] = [
    "ppo",        # Proximal Policy Optimization
    "sac",        # Soft Actor-Critic
    "td3",        # Twin-Delayed DDPG
    "trpo",       # Trust Region Policy Optimization
]
