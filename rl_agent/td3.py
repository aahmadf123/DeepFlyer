# rl_agent/td3.py
"""
Twin-Delayed DDPG (TD3) algorithm implementation for Researcher mode.
This module wraps SB3's TD3 and integrates custom reward functions and logging.
"""
import os
from pathlib import Path
from typing import Callable, Dict

from stable_baselines3 import TD3
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

class TD3Agent:
    """
    TD3Agent implements the Twin-Delayed DDPG algorithm.
    Researcher mode: off-policy, with double Q-learning and policy delay.
    """

    def __init__(self, env, reward_fn: Callable, hyperparams: Dict[str, float]):
        """
        Initialize the TD3 agent.

        Args:
            env: Gymnasium environment instance.
            reward_fn: Callable(state, action) -> float for custom reward.
            hyperparams: Dict of TD3 hyperparameters, e.g.:
                - learning_rate
                - gamma
                - tau
                - policy_noise
                - noise_clip
                - policy_delay
                - buffer_size
                - batch_size
        """
        self.env = CustomRewardWrapper(env, reward_fn)
        # Map hyperparams to SB3 args
        sb3_kwargs = {
            'learning_rate': hyperparams.get('learning_rate', 1e-3),
            'gamma': hyperparams.get('gamma', 0.99),
            'tau': hyperparams.get('tau', 0.005),
            'policy_noise': hyperparams.get('policy_noise', 0.2),
            'noise_clip': hyperparams.get('noise_clip', 0.5),
            'policy_delay': int(hyperparams.get('policy_delay', 2)),
            'buffer_size': int(hyperparams.get('buffer_size', 1000000)),
            'batch_size': int(hyperparams.get('batch_size', 256)),
            'verbose': 0,
        }
        # Create SB3 TD3 model
        self.model = TD3(
            policy='MlpPolicy',
            env=self.env,
            **sb3_kwargs
        )

    def train(self, total_timesteps: int, log_dir: str):
        """
        Run the TD3 training loop.

        Args:
            total_timesteps: Total timesteps to train.
            log_dir: Directory to write JSON logs.
        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / 'td3_metrics.jsonl'
        json_logger = JSONLinesLogger(log_path)

        tmp_logger = configure(str(log_dir), ['stdout'])
        self.model.set_logger(tmp_logger)

        callback = _LoggerCallback(json_logger)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, filepath: str):
        """
        Save the trained TD3 model to disk.

        Args:
            filepath: Path prefix (appends .zip).
        """
        self.model.save(filepath)

    def load(self, filepath: str):
        """
        Load a trained TD3 model from disk.

        Args:
            filepath: Path prefix where model (.zip) was saved.
        """
        self.model = TD3.load(filepath, env=self.env)
        return self.model 