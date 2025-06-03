# rl_agent/td3.py
"""
Twin-Delayed DDPG (TD3) algorithm implementation for Researcher mode.
TD3 builds upon DDPG with clipped double Q-learning and delayed policy updates
for more stable off-policy training in continuous action spaces.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict

class TD3Agent:
    """
    TD3Agent implements the TD3 algorithm:
    - Two Q-networks to mitigate overestimation.
    - Delayed policy (actor) updates.
    - Target policy smoothing.
    """

    def __init__(self, env, reward_fn: Callable, hyperparams: Dict[str, float]):
        """
        Initialize TD3 agent.

        Args:
            env: Gym-style environment.
            reward_fn: reward function callable.
            hyperparams: contains keys like:
                - learning_rate
                - gamma
                - tau (for target updates)
                - policy_noise (stddev for target action noise)
                - noise_clip (max abs noise)
                - policy_delay (steps between actor updates)
                - buffer_size
                - batch_size
        """
        self.env = env
        self.reward_fn = reward_fn
        self.hp = hyperparams

        # TODO: Initialize actor, two critics, and their target networks
        # self.actor = ...
        # self.critic_1 = ...
        # self.critic_2 = ...
        # self.target_actor = ...
        # self.target_critic_1 = ...
        # self.target_critic_2 = ...

    def train(self):
        """
        Run the TD3 training loop.

        Steps:
            1. Sample minibatch from replay buffer.
            2. Add clipped noise to target actions for smoothing.
            3. Compute target Q-values with min of target critics.
            4. Update both critic networks.
            5. Every policy_delay steps, update actor network and target networks.
            6. Log metrics (Q losses, policy loss) via logger.
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """
        Save actor and critic models to disk.
        """
        raise NotImplementedError

    def load(self, filepath: str):
        """
        Load actor and critic models from disk.
        """
        raise NotImplementedError 