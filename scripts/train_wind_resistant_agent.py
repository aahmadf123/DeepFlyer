#!/usr/bin/env python3
"""
Train a direct control agent with trajectory following reward function.

This script trains a RL agent to follow trajectories even with wind disturbance.
"""

import os
import sys
import time
import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_agent")

# Import our modules
from rl_agent.env.mavros_env import MAVROSEnv
from rl_agent.direct_control_agent import DirectControlAgent
from rl_agent.rewards import create_cross_track_and_heading_reward


def train_agent(args):
    """
    Train a direct control agent with trajectory following reward.
    
    Args:
        args: Command-line arguments
    """
    logger.info("Starting agent training...")
    
    # Define trajectory from point A to point B
    trajectory = [
        np.array([0.0, 0.0, 1.5]),    # Start point
        np.array([2.5, 2.5, 1.5]),    # Midpoint
        np.array([5.0, 5.0, 1.5]),    # End point
    ]
    
    # Create reward function
    reward_fn = create_cross_track_and_heading_reward(
        cross_track_weight=1.0,
        heading_weight=0.1,
        max_error=2.0,
        max_heading_error=np.pi,
        trajectory=trajectory,  # Use the same trajectory as before
    )
    
    # Create environment
    env = MAVROSEnv(
        namespace=args.namespace,
        goal_position=trajectory[-1],  # End position as goal
        enable_safety_layer=True,
        reward_function=reward_fn,
        auto_arm=True,
        auto_offboard=True,
        max_episode_steps=500
    )
    
    # Create agent
    agent = DirectControlAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_dim=args.hidden_dim,
        lr=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        procrastination_factor=args.procrastination_factor,
        alpha=args.alpha,
        entropy_coef=args.entropy_coef,
        action_smoothness_penalty=args.smoothness_penalty
    )
    
    # Load existing model if specified
    if args.load_model and os.path.exists(args.load_model):
        try:
            logger.info(f"Loading model from {args.load_model}")
            agent_state = np.load(args.load_model, allow_pickle=True).item()
            agent.load_model_state(agent_state)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    # Training loop
    logger.info("Starting training loop...")
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    
    total_steps = 0
    start_time = time.time()
    
    try:
        for episode in range(args.num_episodes):
            # Reset environment
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            # Episode loop
            done = False
            while not done:
                # Agent selects action
                action, _ = agent.predict(obs)
                
                # Environment step
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Add transition to agent's buffer
                agent.add_to_buffer(obs, action, next_obs, reward, terminated)
                
                # Update statistics
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Move to next state
                obs = next_obs
                
                # Check if episode is done
                done = terminated or truncated
                
                # Learn periodically
                if total_steps % args.learn_interval == 0:
                    metrics = agent.learn()
                    if args.verbose:
                        logger.info(f"Step {total_steps}, learn metrics: {metrics}")
            
            # End of episode
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            
            # Log progress
            elapsed = time.time() - start_time
            logger.info(f"Episode {episode+1}/{args.num_episodes}, "
                     f"reward: {episode_reward:.2f}, steps: {episode_steps}, "
                     f"total steps: {total_steps}, time: {elapsed:.1f}s")
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                if args.save_best:
                    save_path = args.save_model.replace(".npy", "_best.npy")
                    agent_state = agent.get_model_state()
                    np.save(save_path, agent_state)
                    logger.info(f"Saved best model to {save_path}, reward: {best_reward:.2f}")
            
            # Save checkpoint periodically
            if (episode + 1) % args.checkpoint_interval == 0:
                checkpoint_path = args.save_model.replace(".npy", f"_ep{episode+1}.npy")
                agent_state = agent.get_model_state()
                np.save(checkpoint_path, agent_state)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Plot learning curve
                if args.plot_progress:
                    plot_learning_curve(episode_rewards, episode_lengths, 
                                      f"progress_ep{episode+1}.png")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    
    finally:
        # Save final model
        agent_state = agent.get_model_state()
        np.save(args.save_model, agent_state)
        logger.info(f"Saved final model to {args.save_model}")
        
        # Plot final learning curve
        if args.plot_progress:
            plot_learning_curve(episode_rewards, episode_lengths, "final_progress.png")
        
        # Clean up
        env.close()
    
    logger.info(f"Training completed. Total steps: {total_steps}")
    return agent


def plot_learning_curve(rewards: List[float], lengths: List[int], filename: str):
    """Plot and save learning curve."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Rewards")
    ax1.grid(True)
    
    # Plot smoothed rewards
    window = min(len(rewards) // 10 + 1, 10)  # Dynamic window size
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window-1, len(rewards)), smoothed, 'r-', linewidth=2)
        
    # Plot episode lengths
    ax2.plot(lengths)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Episode Lengths")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Saved learning curve to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a direct control agent")
    
    # Environment settings
    parser.add_argument('--namespace', type=str, default="deepflyer", 
                      help="ROS namespace")
    parser.add_argument('--mock', action='store_true',
                      help="Use mock ROS implementation")
    
    # Training settings
    parser.add_argument('--num-episodes', type=int, default=100,
                      help="Number of training episodes")
    parser.add_argument('--learn-interval', type=int, default=50,
                      help="Steps between learning updates")
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                      help="Episodes between saving checkpoints")
    
    # Model settings
    parser.add_argument('--hidden-dim', type=int, default=256,
                      help="Hidden dimension size")
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                      help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99,
                      help="Discount factor")
    parser.add_argument('--batch-size', type=int, default=128,
                      help="Batch size for learning")
    parser.add_argument('--procrastination-factor', type=float, default=0.95,
                      help="P3O procrastination factor")
    parser.add_argument('--alpha', type=float, default=0.2,
                      help="P3O blend factor")
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                      help="Entropy coefficient")
    parser.add_argument('--smoothness-penalty', type=float, default=0.01,
                      help="Action smoothness penalty")
    
    # Reward settings
    parser.add_argument('--wind-resistant', action='store_true',
                      help="Use wind-resistant reward function")
    
    # I/O settings
    parser.add_argument('--load-model', type=str, default="",
                      help="Load existing model file")
    parser.add_argument('--save-model', type=str, default="direct_control_agent.npy",
                      help="Save model file")
    parser.add_argument('--save-best', action='store_true',
                      help="Save best model based on reward")
    parser.add_argument('--plot-progress', action='store_true',
                      help="Plot and save learning curves")
    parser.add_argument('--verbose', action='store_true',
                      help="Print verbose output")
    
    args = parser.parse_args()
    
    if args.mock:
        # Force mock ROS environment
        os.environ['USE_MOCK_ROS'] = '1'
        logger.info("Using mock ROS implementation")
    
    try:
        trained_agent = train_agent(args)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True) 