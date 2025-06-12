#!/usr/bin/env python3
"""
Training script for the RL supervisor.

This script trains the RL supervisor agent using the SupervisorEnv.
"""

import argparse
import numpy as np
import torch
import gymnasium as gym
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from rl_agent.supervisor_agent import SupervisorAgent
from rl_agent.env.supervisor_env import SupervisorEnv


def train(
    total_timesteps: int = 100000,
    eval_freq: int = 1000,
    n_eval_episodes: int = 5,
    save_freq: int = 10000,
    log_dir: str = "logs",
    model_dir: str = "models",
    with_disturbance: bool = False,
    disturbance_std: float = 0.1,
    device: str = "auto"
):
    """
    Train the RL supervisor agent.
    
    Args:
        total_timesteps: Total number of timesteps to train for
        eval_freq: Frequency of evaluation
        n_eval_episodes: Number of episodes for evaluation
        save_freq: Frequency of saving the model
        log_dir: Directory to save logs
        model_dir: Directory to save models
        with_disturbance: Whether to add wind disturbance
        disturbance_std: Standard deviation of the wind disturbance
        device: Device to run the model on ('cpu', 'cuda', or 'auto')
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Create environment
    env = SupervisorEnv(
        with_disturbance=with_disturbance,
        disturbance_std=disturbance_std
    )
    
    # Create evaluation environment
    eval_env = SupervisorEnv(
        with_disturbance=with_disturbance,
        disturbance_std=disturbance_std
    )
    
    # Create agent
    agent = SupervisorAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    # Training loop
    timestep = 0
    episode = 0
    best_eval_reward = -float('inf')
    
    # Logging
    log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    training_logs = []
    
    # Metrics
    episode_rewards = []
    episode_lengths = []
    episode_pid_gains = []
    eval_rewards = []
    
    # Start training
    print(f"Starting training for {total_timesteps} timesteps...")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_pid_gains = []
    
    while timestep < total_timesteps:
        # Select action
        action, _ = agent.predict(obs)
        
        # Take action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.add_to_buffer(obs, action, next_obs, reward, done)
        
        # Update observation
        obs = next_obs
        
        # Update metrics
        episode_reward += reward
        episode_length += 1
        episode_pid_gains.append(info["pid_gain"])
        
        # Learn
        if len(agent.replay_buffer) > agent.batch_size:
            metrics = agent.learn()
        
        # End of episode
        if done:
            # Log episode results
            print(f"Episode {episode}: {episode_reward:.2f} reward, {episode_length} steps")
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Log episode data
            episode_log = {
                "episode": episode,
                "timestep": timestep,
                "reward": float(episode_reward),
                "length": episode_length,
                "mean_pid_gain": float(np.mean(episode_pid_gains)),
                "final_cross_track_error": float(info["cross_track_error"]),
                "final_heading_error": float(info["heading_error"]),
                "progress": float(info["progress"])
            }
            training_logs.append(episode_log)
            
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_pid_gains = []
            episode += 1
        
        # Evaluation
        if timestep % eval_freq == 0:
            eval_reward = evaluate(agent, eval_env, n_eval_episodes)
            eval_rewards.append(eval_reward)
            print(f"Evaluation at timestep {timestep}: {eval_reward:.2f}")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                save_path = os.path.join(model_dir, "best_model.pt")
                torch.save(agent.get_model_state(), save_path)
                print(f"Saved best model to {save_path}")
        
        # Save model periodically
        if timestep % save_freq == 0 and timestep > 0:
            save_path = os.path.join(model_dir, f"model_{timestep}.pt")
            torch.save(agent.get_model_state(), save_path)
            print(f"Saved model to {save_path}")
            
            # Save logs
            with open(log_file, 'w') as f:
                json.dump(training_logs, f)
        
        timestep += 1
    
    # Save final model
    save_path = os.path.join(model_dir, "final_model.pt")
    torch.save(agent.get_model_state(), save_path)
    print(f"Saved final model to {save_path}")
    
    # Save logs
    with open(log_file, 'w') as f:
        json.dump(training_logs, f)
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_lengths, eval_rewards, log_dir)
    
    return agent


def evaluate(agent, env, n_episodes=5):
    """
    Evaluate the agent.
    
    Args:
        agent: Agent to evaluate
        env: Environment to evaluate in
        n_episodes: Number of episodes to evaluate
        
    Returns:
        mean_reward: Mean reward over episodes
    """
    episode_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards)


def plot_training_curves(episode_rewards, episode_lengths, eval_rewards, log_dir):
    """
    Plot training curves.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        eval_rewards: List of evaluation rewards
        log_dir: Directory to save plots
    """
    # Plot episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(log_dir, 'episode_rewards.png'))
    
    # Plot episode lengths
    plt.figure(figsize=(10, 5))
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.savefig(os.path.join(log_dir, 'episode_lengths.png'))
    
    # Plot evaluation rewards
    plt.figure(figsize=(10, 5))
    plt.plot(eval_rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Evaluation')
    plt.ylabel('Reward')
    plt.savefig(os.path.join(log_dir, 'eval_rewards.png'))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train RL supervisor agent')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps')
    parser.add_argument('--eval-freq', type=int, default=1000, help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--save-freq', type=int, default=10000, help='Save frequency')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory')
    parser.add_argument('--with-disturbance', action='store_true', help='Add wind disturbance')
    parser.add_argument('--disturbance-std', type=float, default=0.1, help='Wind disturbance std')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu, cuda, auto)')
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        with_disturbance=args.with_disturbance,
        disturbance_std=args.disturbance_std,
        device=args.device
    )


if __name__ == '__main__':
    main() 