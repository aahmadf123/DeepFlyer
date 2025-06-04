"""
Command-line entrypoint for launching RL training jobs.
Parses a JSON-based config, selects agent based on algorithm, and stubs job execution.
Writes simple status files for /api/train/status to consume.
"""
import argparse
import json
import uuid
import time
import os
from pathlib import Path

from rl_agent.registry import RewardRegistry
from rl_agent.config import EXPLORER_ALGOS, RESEARCHER_ALGOS
from rl_agent.ppo import PPOAgent
from rl_agent.sac import SACAgent
from rl_agent.td3 import TD3Agent
from rl_agent.trpo import TRPOAgent
from rl_agent.model_based import MBPOAgent
from rl_agent.intrinsic import IntrinsicCuriosityAgent, RNDModule
from rl_agent.meta import MAMLAgent, PEARLAgent
from rl_agent.distributional import QRSACAgent
from rl_agent.hierarchical import OptionCriticAgent
from rl_agent.utils import set_seed
from rl_agent.env import make_env

# Map algorithm names to agent classes
ALGO_MAP = {
    'ppo': PPOAgent,
    'sac': SACAgent,
    'td3': TD3Agent,
    'trpo': TRPOAgent,
}


def main():
    parser = argparse.ArgumentParser(description="Launch an RL training job.")
    parser.add_argument('--algorithm', type=str, default='ppo',
                        help='Algorithm to use: ' + ', '.join(EXPLORER_ALGOS + RESEARCHER_ALGOS))
    parser.add_argument('--preset_id', type=str, required=True,
                        help='Reward preset ID from RewardRegistry')
    parser.add_argument('--hyperparameters', type=str, required=True,
                        help='JSON string of hyperparameters')
    parser.add_argument('--reward_weights', type=str, required=True,
                        help='JSON string of reward component weights')
    parser.add_argument('--max_episodes', type=int, required=True,
                        help='Maximum number of episodes')
    parser.add_argument('--max_steps_per_episode', type=int, required=True,
                        help='Maximum steps per episode')
    parser.add_argument('--scenario_sequence', type=str, default=None,
                        help='JSON list of scenario IDs for curriculum')
    parser.add_argument('--randomization', type=str, default=None,
                        help='JSON dict of randomization ranges')
    parser.add_argument('--base_model_id', type=str, default=None,
                        help='Optional pretrained model ID to fine-tune')
    parser.add_argument('--time_limit_min', type=int, default=None,
                        help='Optional time limit in minutes')
    parser.add_argument('--job_id', type=str, default=None,
                        help='Optional job ID (provided by API)')
    args = parser.parse_args()

    # Parse JSON arguments
    try:
        hyperparams = json.loads(args.hyperparameters)
        reward_weights = json.loads(args.reward_weights)
        scenario_sequence = json.loads(args.scenario_sequence) if args.scenario_sequence else None
        randomization = json.loads(args.randomization) if args.randomization else None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON arguments: {e}")
        return

    # Use provided job_id or generate a new one
    job_id = args.job_id if args.job_id else str(uuid.uuid4())
    status_file = f"status_{job_id}.json"

    # Write initial status
    initial_status = {
        'job_id': job_id,
        'status': 'started',
        'algorithm': args.algorithm,
        'preset_id': args.preset_id,
    }
    with open(status_file, 'w') as f:
        json.dump(initial_status, f)

    print(f"Started training job {job_id} using algorithm '{args.algorithm}' and preset '{args.preset_id}'")

    # ==============================================
    # Orchestrate real training
    # ==============================================
    # Optional seeding if provided in hyperparameters
    if 'seed' in hyperparams:
        try:
            set_seed(int(hyperparams['seed']))
        except Exception:
            print(f"Warning: invalid seed value {hyperparams.get('seed')}")

    # Create environment
    env_id = hyperparams.get('env_id', 'CartPole-v1')
    env = make_env(env_id)

    # Retrieve reward function
    reward_fn = RewardRegistry.get_fn(args.preset_id)

    # Select agent class
    AgentClass = ALGO_MAP.get(args.algorithm)
    if AgentClass is None:
        print(f"Unknown algorithm '{args.algorithm}'. Available: {list(ALGO_MAP.keys())}")
        return

    agent = AgentClass(env, reward_fn, hyperparams)

    # Calculate total timesteps based on episodes and steps
    total_timesteps = args.max_episodes * args.max_steps_per_episode

    # Prepare directories for logs and model checkpoints
    run_dir = Path('runs') / job_id
    log_dir = run_dir / 'logs'
    model_dir = run_dir / 'model'
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Execute training
    try:
        agent.train(total_timesteps=total_timesteps, log_dir=str(log_dir))
    except KeyboardInterrupt:
        print("Training interrupted by user")

    # Save model
    agent.save(str(model_dir / 'model'))

    # Write completion status with model path
    completion_status = {
        'job_id': job_id,
        'status': 'completed',
        'episodes': args.max_episodes,
        'model_path': str(model_dir / 'model.zip')
    }
    with open(status_file, 'w') as f:
        json.dump(completion_status, f)

    print(f"Completed training job {job_id}")


if __name__ == '__main__':
    main()
