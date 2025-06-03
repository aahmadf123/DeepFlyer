# rl_agent/ppo.py
"""
Proximal Policy Optimization (PPO) algorithm implementation for Explorer mode.
This module defines a PPOAgent class with stubs for initialization, training loop,
model save/load, and key hyperparameters. Any developer can extend these methods
to plug in actual network architecture and training logic.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict

class PPOAgent:
    """
    PPOAgent trains a policy using the Proximal Policy Optimization algorithm.
    Explorer mode: simple on-policy learning, easy to understand and extend.
    """

    def __init__(self, env, reward_fn: Callable, hyperparams: Dict[str, float]):
        """
        Initialize the PPO agent.

        Args:
            env: OpenAI Gym-style environment (must implement reset(), step(), etc.).
            reward_fn: Callable(state, action) -> float to compute custom reward.
            hyperparams: Dictionary of algorithm hyperparameters, e.g.:
                - learning_rate
                - gamma (discount factor)
                - clip_epsilon
                - k_epochs (epochs per update)
                - batch_size
        """
        self.env = env
        self.reward_fn = reward_fn
        self.hyperparams = hyperparams

        # TODO: Initialize policy and value networks here
        # self.policy_net = ...
        # self.value_net = ...
        # self.optimizer = torch.optim.Adam([...], lr=hyperparams['learning_rate'])

    def train(self):
        """
        Run the PPO training loop.

        Steps:
            1. Collect trajectories by running current policy in the environment.
            2. Compute discounted rewards and advantages.
            3. Perform multiple epochs of policy and value network updates,
               using clipped surrogate objective.
            4. Log training metrics (losses, rewards) via logger.
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """
        Save policy and value network weights to disk.

        Args:
            filepath: Path to save the model (prefix); may add suffixes for policy/value.
        """
        # torch.save(self.policy_net.state_dict(), filepath + '_policy.pt')
        # torch.save(self.value_net.state_dict(), filepath + '_value.pt')
        raise NotImplementedError

    def load(self, filepath: str):
        """
        Load policy and value network weights from disk.

        Args:
            filepath: Path prefix where models were saved.
        """
        # self.policy_net.load_state_dict(torch.load(filepath + '_policy.pt'))
        # self.value_net.load_state_dict(torch.load(filepath + '_value.pt'))
        raise NotImplementedError
