# rl_agent/trpo.py
"""
Trust Region Policy Optimization (TRPO) algorithm implementation for Researcher mode.
This module wraps SB3-contrib's TRPO and integrates custom reward functions and logging.
"""
import os
from pathlib import Path
from typing import Callable, Dict

from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from rl_agent.env.wrappers import CustomRewardWrapper
from rl_agent.logger import JSONLinesLogger

class _LoggerCallback(BaseCallback):
    """Callback for logging training metrics to JSON lines."""
    def __init__(self, logger: JSONLinesLogger, verbose: int = 0):
        super().__init__(verbose)
        self.logger = logger

    def _on_step(self) -> bool:
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                ep_info = info.get('episode')
                if ep_info:
                    self.logger.log({
                        'episode_reward': ep_info.get('r'),
                        'episode_length': ep_info.get('l'),
                        'timesteps': self.num_timesteps
                    })
        return True

class TRPOAgent:
    """
    TRPOAgent implements Trust Region Policy Optimization.
    Researcher mode: on-policy algorithm with KL constraint.
    """
    def __init__(self, env, reward_fn: Callable, hyperparams: Dict[str, float]):
        """
        Initialize the TRPO agent.

        Args:
            env: Gymnasium environment instance.
            reward_fn: Callable(state, action) -> float for custom reward.
            hyperparams: Dict of TRPO hyperparameters, e.g.:
                - max_kl
                - cg_iterations
                - cg_damping
                - gamma
                - gae_lambda
        """
        self.env = CustomRewardWrapper(env, reward_fn)
        # Map hyperparams to SB3 args
        sb3_kwargs = {
            'max_kl': hyperparams.get('max_kl', 0.01),
            'cg_iters': int(hyperparams.get('cg_iterations', 10)),
            'cg_damping': hyperparams.get('cg_damping', 0.1),
            'gamma': hyperparams.get('gamma', 0.99),
            'gae_lambda': hyperparams.get('gae_lambda', 0.95),
            'verbose': 0,
        }
        # Create TRPO model
        self.model = TRPO(
            policy='MlpPolicy',
            env=self.env,
            **sb3_kwargs
        )

    def train(self, total_timesteps: int, log_dir: str):
        """
        Run the TRPO training loop.

        Args:
            total_timesteps: Total timesteps to collect (approx episodes*steps).
            log_dir: Directory to write JSON logs.
        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / 'trpo_metrics.jsonl'
        json_logger = JSONLinesLogger(log_path)

        tmp_logger = configure(str(log_dir), ['stdout'])
        self.model.set_logger(tmp_logger)

        callback = _LoggerCallback(json_logger)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, filepath: str):
        """
        Save the trained TRPO model to disk.

        Args:
            filepath: Path prefix (appends .zip).
        """
        self.model.save(filepath)

    def load(self, filepath: str):
        """
        Load a trained TRPO model from disk.

        Args:
            filepath: Path prefix where model (.zip) was saved.
        """
        self.model = TRPO.load(filepath, env=self.env)
        return self.model 