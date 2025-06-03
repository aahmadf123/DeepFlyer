# rl_agent/hierarchical.py
"""
Hierarchical Reinforcement Learning for Researcher mode.
Implements Option-Critic architecture allowing temporal abstractions
(options) over primitive actions.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, List

class OptionCriticAgent:
    """
    OptionCriticAgent implements the option-critic framework:
    - Learns intra-option policies and termination functions.
    - Meta-policy selects options and executes until termination.
    """

    def __init__(self, env, reward_fn: Callable, hyperparams: Dict[str, float]):
        """
        Initialize Option-Critic agent.

        Args:
            env: Gym-style environment.
            reward_fn: Callable reward function.
            hyperparams: Contains keys like:
                - num_options (number of options)
                - intra_option_lr
                - termination_lr
                - critic_lr
        """
        self.env = env
        self.reward_fn = reward_fn
        self.hp = hyperparams

        # TODO: Initialize option policies, termination networks, and critic
        # self.options = [ ... ]
        # self.terminations = [ ... ]
        # self.critic = ...

    def train(self):
        """
        Run Option-Critic training loop:
            1. Sample option according to meta-policy.
            2. Execute intra-option policy until termination.
            3. Update intra-option policy, termination, and critic.
            4. Log metrics for option performance.
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """
        Save all networks (options, termination, critic).
        """
        raise NotImplementedError

    def load(self, filepath: str):
        """
        Load saved Option-Critic networks.
        """
        raise NotImplementedError 