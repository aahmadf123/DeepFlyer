# rl_agent/distributional.py
"""
Distributional RL extensions for Researcher mode.
Includes a Quantile Regression SAC (QR-SAC) stub that learns a distribution
over returns instead of a point estimate.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict

class QRSACAgent:
    """
    Quantile Regression Soft Actor-Critic (QR-SAC) Agent:
    - Learns quantile-based critics to approximate return distribution.
    - Uses distributional Bellman updates.
    """

    def __init__(self, env, reward_fn: Callable, hyperparams: Dict[str, float]):
        """
        Initialize QR-SAC agent.

        Args:
            env: Gym-style environment.
            reward_fn: Callable reward function.
            hyperparams: Keys like:
                - learning_rate
                - gamma
                - alpha (entropy coefficient)
                - num_quantiles
                - target_update_interval
        """
        self.env = env
        self.reward_fn = reward_fn
        self.hp = hyperparams

        # TODO: Initialize quantile critics and policy network
        # self.actor = ...
        # self.critics = [ ... for _ in range(num_quantiles) ]
        # self.target_critics = [ ... ]

    def train(self):
        """
        Run the QR-SAC training loop:
            1. Sample batch from replay buffer.
            2. Compute distributional TD targets for each quantile.
            3. Update critics using quantile Huber loss.
            4. Update policy to maximize expected quantile values + entropy.
            5. Soft-update target networks.
            6. Log metrics (quantile losses, policy loss).
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """
        Save actor and quantile critic networks.
        """
        raise NotImplementedError

    def load(self, filepath: str):
        """
        Load saved QR-SAC networks from disk.
        """
        raise NotImplementedError 