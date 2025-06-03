# rl_agent/intrinsic.py
"""
Intrinsic motivation modules for Researcher mode:
Provides implementations for curiosity-driven exploration like ICM or RND.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict

class IntrinsicCuriosityAgent:
    """
    Implements an Intrinsic Curiosity Module (ICM):
    - Learns inverse and forward models to compute intrinsic reward.
    - Encourages exploration by rewarding prediction error.
    """

    def __init__(self, env, hyperparams: Dict[str, float]):
        """
        Initialize ICM agent used alongside a base RL algorithm.

        Args:
            env: Gym-style environment.
            hyperparams: Contains keys like:
                - icm_lr (learning rate for ICM networks)
                - beta (trade-off between inverse/forward losses)
                - eta (intrinsic reward scaling)
        """
        self.env = env
        self.hp = hyperparams

        # TODO: Initialize feature encoder, inverse model, forward model
        # self.encoder = ...
        # self.inverse_net = ...
        # self.forward_net = ...

    def compute_intrinsic_reward(self, state, next_state, action):
        """
        Compute intrinsic reward based on forward model prediction error.

        Returns:
            float: Intrinsic reward to add to extrinsic reward.
        """
        # TODO: Encode states, predict next, compute error, scale by eta
        raise NotImplementedError

    def train(self, batch):
        """
        Update ICM networks on a batch of transitions:
            1. Compute inverse prediction loss.
            2. Compute forward prediction loss.
            3. Backpropagate combined loss (beta weighting).
        """
        raise NotImplementedError

class RNDModule:
    """
    Random Network Distillation (RND) for bonus reward:
    - A fixed random target network and a predictor network.
    - Intrinsic reward is the predictor's error in matching target.
    """

    def __init__(self, env, hyperparams: Dict[str, float]):
        """
        Args:
            env: Gym-style environment.
            hyperparams: keys like:
                - rnd_lr
                - rnd_eta (intrinsic reward scale)
        """
        self.env = env
        self.hp = hyperparams

        # TODO: Initialize target and predictor networks
        # self.target = ... (frozen)
        # self.predictor = ...

    def compute_intrinsic_reward(self, state):
        """
        Returns intrinsic reward based on predictor error.
        """
        # TODO: Compute feature, predictor output, target output, error
        raise NotImplementedError 