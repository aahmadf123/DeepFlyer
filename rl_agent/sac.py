# rl_agent/sac.py
"""
Soft Actor-Critic (SAC) algorithm implementation for Explorer mode.
This module provides a SACAgent class with stubs for initialization, training,
model save/load, and key hyperparameters suitable for continuous control tasks.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict

class SACAgent:
    """
    SACAgent implements the Soft Actor-Critic algorithm.
    Explorer mode: off-policy, sample-efficient, with entropy regularization.
    """

    def __init__(self, env, reward_fn: Callable, hyperparams: Dict[str, float]):
        """
        Initialize the SAC agent.

        Args:
            env: Gym-style environment with continuous action space.
            reward_fn: Callable(state, action) -> float for custom reward.
            hyperparams: Dict of SAC hyperparameters, e.g.:
                - learning_rate
                - gamma (discount factor)
                - alpha (entropy coefficient)
                - tau (target smoothing coefficient)
                - buffer_size (replay buffer)
                - batch_size
        """
        self.env = env
        self.reward_fn = reward_fn
        self.hyperparams = hyperparams

        # TODO: Initialize actor, critic, and target networks
        # self.actor = ...
        # self.critic_1 = ...
        # self.critic_2 = ...
        # self.target_critic_1 = ...
        # self.target_critic_2 = ...
        # self.log_alpha = torch.tensor([hyperparams['init_alpha']], requires_grad=True)

    def train(self):
        """
        Run the SAC training loop.

        Steps:
            1. Sample a batch of transitions from replay buffer.
            2. Update critics with TD targets and clipped double Q-learning.
            3. Update actor to maximize expected Q + entropy bonus.
            4. Adjust entropy coefficient alpha via dual gradient descent.
            5. Soft-update target networks.
            6. Log metrics (Q-values, policy loss, alpha) via logger.
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """
        Save all SAC networks and alpha to disk.
        """
        raise NotImplementedError

    def load(self, filepath: str):
        """
        Load saved SAC networks and alpha from disk.
        """
        raise NotImplementedError 