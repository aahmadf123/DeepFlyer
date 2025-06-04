# rl_agent/ppo.py
"""
Proximal Policy Optimization (PPO) algorithm implementation for Explorer mode.
This module wraps SB3's PPO and integrates custom reward functions and logging.
"""
import os
from pathlib import Path
from typing import Callable, Dict

from stable_baselines3 import PPO
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
        # Log episode info if available
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

class PPOAgent:
    """
    PPOAgent trains a policy using the Proximal Policy Optimization algorithm.
    Explorer mode: on-policy learning, integrates custom rewards via environment wrapper.
    """

    def __init__(self, env, reward_fn: Callable, hyperparams: Dict[str, float]):
        """
        Initialize the PPO agent.

        Args:
            env: Gymnasium environment instance.
            reward_fn: Callable(state, action) -> float for custom reward.
            hyperparams: Dictionary of algorithm hyperparameters, e.g.:
                - learning_rate
                - gamma
                - entropy_coef (ent_coef)
                - clip_epsilon (clip_range)
                - batch_size
                - k_epochs (n_epochs)
        """
        # Wrap environment to use custom reward
        self.env = CustomRewardWrapper(env, reward_fn)

        # Map hyperparams to SB3 arguments
        sb3_kwargs = {
            'learning_rate': hyperparams.get('learning_rate', 3e-4),
            'gamma': hyperparams.get('gamma', 0.99),
            'ent_coef': hyperparams.get('entropy_coef', 0.0),
            'clip_range': hyperparams.get('clip_epsilon', 0.2),
            'batch_size': int(hyperparams.get('batch_size', 64)),
            'n_epochs': int(hyperparams.get('k_epochs', 10)),
            'verbose': 0,
        }
        # Create SB3 PPO model
        self.model = PPO(
            policy='MlpPolicy',
            env=self.env,
            **sb3_kwargs
        )

    def train(self, total_timesteps: int, log_dir: str):
        """
        Run the PPO training loop.

        Args:
            total_timesteps: Total timesteps to train.
            log_dir: Directory to write JSON logs (one file per run).
        """
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / 'ppo_metrics.jsonl'
        json_logger = JSONLinesLogger(log_path)

        # Configure SB3 logger to minimal output
        tmp_sb3_logger = configure(str(log_dir), ['stdout'])
        self.model.set_logger(tmp_sb3_logger)

        callback = _LoggerCallback(json_logger)
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, filepath: str):
        """
        Save the trained model to disk.

        Args:
            filepath: Path prefix to save the model (appends .zip).
        """
        self.model.save(filepath)

    def load(self, filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath: Path prefix where model (.zip) was saved.
        """
        # Reload the model and rewrap env
        self.model = PPO.load(filepath, env=self.env)
        return self.model
