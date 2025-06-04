import subprocess
import sys
import json
import shlex
import uuid
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _serialize_arg(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def start_training_worker(cfg: Dict[str, Any]) -> str:
    """Spawn a background training subprocess.

    Args:
        cfg: Dict parsed from TrainRequest model.

    Returns:
        job_id string (matches status_<job_id>.json written by rl_agent.train)
    """
    # Generate and pass a consistent job_id
    job_id = str(uuid.uuid4())
    python_exe = sys.executable
    cmd = [python_exe, '-m', 'rl_agent.train',
           '--job_id', job_id,
           '--algorithm', cfg['algorithm'],
           '--preset_id', cfg['preset_id'],
           '--hyperparameters', _serialize_arg(cfg['hyperparameters']),
           '--reward_weights', _serialize_arg(cfg['reward_weights']),
           '--max_episodes', str(cfg['max_episodes']),
           '--max_steps_per_episode', str(cfg['max_steps_per_episode'])]

    if cfg.get('scenario_sequence'):
        cmd += ['--scenario_sequence', _serialize_arg(cfg['scenario_sequence'])]
    if cfg.get('randomization'):
        cmd += ['--randomization', _serialize_arg(cfg['randomization'])]
    if cfg.get('base_model_id'):
        cmd += ['--base_model_id', cfg['base_model_id']]
    if cfg.get('time_limit_min'):
        cmd += ['--time_limit_min', str(cfg['time_limit_min'])]

    # Launch the training subprocess in the background
    subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    # Return the job_id so clients can track status
    return job_id
