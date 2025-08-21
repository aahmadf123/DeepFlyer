#!/usr/bin/env python3
"""
P3O Training Script for DeepFlyer
Trains the drone to navigate through hoops using direct RL control
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.direct_control_agent import DirectControlAgent, DirectControlConfig
from rl_agent.algorithms.p3o import P3OConfig
from rl_agent.rewards.rewards import HoopNavigationReward, get_reward_preset

# Optional ClearML integration
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("ClearML not available, training without experiment tracking")


class P3OTrainer:
    """Training coordinator for P3O drone control"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup directories
        self.setup_directories()
        
        # Initialize ClearML if available
        self.task = None
        if CLEARML_AVAILABLE and args.use_clearml:
            self.task = Task.init(
                project_name="DeepFlyer",
                task_name=f"P3O_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Load configurations
        self.control_config = self.load_control_config()
        self.p3o_config = self.load_p3o_config()
        self.reward_config = get_reward_preset(args.reward_preset)
        
        # Initialize agent
        self.agent = DirectControlAgent(self.control_config, self.p3o_config)
        
        # Load checkpoint if specified
        if args.checkpoint:
            self.agent.load(args.checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        
        # Initialize reward function
        self.reward_fn = HoopNavigationReward(self.reward_config)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.best_reward = -float('inf')
        
    def setup_directories(self):
        """Create necessary directories"""
        self.model_dir = Path("models/p3o")
        self.log_dir = Path("experiments/logs")
        self.config_dir = Path("config")
        
        for dir_path in [self.model_dir, self.log_dir, self.config_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def load_control_config(self) -> DirectControlConfig:
        """Load or create control configuration"""
        config_path = self.config_dir / "control_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = DirectControlConfig()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            print(f"Loaded control config from {config_path}")
        else:
            config = DirectControlConfig()
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(config.__dict__, f, indent=2)
            print(f"Created default control config at {config_path}")
        
        return config
    
    def load_p3o_config(self) -> P3OConfig:
        """Load or create P3O configuration"""
        config_path = self.config_dir / "p3o_config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = P3OConfig()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            print(f"Loaded P3O config from {config_path}")
        else:
            config = P3OConfig()
            # Override with command line arguments
            if self.args.learning_rate:
                config.learning_rate = self.args.learning_rate
            if self.args.batch_size:
                config.batch_size = self.args.batch_size
            if self.args.procrastination_factor:
                config.procrastination_factor = self.args.procrastination_factor
            
            # Save config
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            print(f"Created P3O config at {config_path}")
        
        return config
    
    def simulate_episode(self, max_steps: int = 500) -> float:
        """
        Simulate one training episode
        This would be replaced with actual drone/simulation in production
        """
        total_reward = 0.0
        self.agent.reset_episode()
        self.reward_fn.reset()
        
        # Generate random initial observation
        obs = self.generate_observation(step=0)
        
        for step in range(max_steps):
            # Get action from agent
            action = self.agent.get_action(obs, training=True)
            
            # Simulate environment step (would be real drone/sim)
            next_obs, reward, done, info = self.simulate_step(obs, action, step)
            
            # Store experience
            self.agent.store_experience(obs, action, next_obs, reward, done)
            
            total_reward += reward
            obs = next_obs
            
            if done:
                break
        
        return total_reward
    
    def generate_observation(self, step: int) -> np.ndarray:
        """Generate simulated observation for training"""
        obs = np.zeros(8, dtype=np.float32)
        
        # Simulate hoop detection
        if np.random.random() > 0.3:  # 70% chance of seeing hoop
            obs[0] = np.random.uniform(-0.5, 0.5)  # Horizontal offset
            obs[1] = np.random.uniform(-0.5, 0.5)  # Vertical offset
            obs[2] = 1.0  # Hoop visible
            obs[3] = max(0.1, 1.0 - step / 500.0)  # Distance decreases over time
        else:
            obs[2] = 0.0  # Hoop not visible
            obs[3] = 1.0
        
        # Simulate drone velocity
        obs[4:7] = np.random.uniform(-0.5, 0.5, 3)  # Velocity
        obs[7] = np.random.uniform(-0.3, 0.3)  # Yaw rate
        
        return obs
    
    def simulate_step(self, obs: np.ndarray, action: np.ndarray, step: int):
        """Simulate environment dynamics"""
        # Generate next observation
        next_obs = obs.copy()
        
        # Update based on action (simplified physics)
        next_obs[4:7] = 0.9 * obs[4:7] + 0.1 * action[:3]  # Velocity update
        next_obs[7] = 0.9 * obs[7] + 0.1 * action[3]  # Yaw rate update
        
        # Update hoop position based on movement
        if obs[2] > 0.5:  # If hoop visible
            next_obs[0] -= action[1] * 0.1  # Lateral movement
            next_obs[1] -= action[2] * 0.1  # Vertical movement
            next_obs[3] = max(0.0, obs[3] - action[0] * 0.05)  # Forward movement
        
        # Check for events
        info = {}
        done = False
        
        # Hoop passage
        if obs[3] > 0.1 and next_obs[3] < 0.1:
            info['hoop_passed'] = True
            done = True
        
        # Collision (random for simulation)
        if np.random.random() < 0.01:
            info['collision'] = True
            done = True
        
        # Calculate reward
        reward, components = self.reward_fn.calculate_reward(next_obs, action, info)
        
        return next_obs, reward, done, info
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting P3O training for {self.args.episodes} episodes")
        print(f"Reward preset: {self.args.reward_preset}")
        print(f"Model will be saved to: {self.model_dir}")
        print("-" * 50)
        
        for episode in range(self.args.episodes):
            # Run episode
            episode_reward = self.simulate_episode(self.args.max_steps)
            
            # Perform training updates
            if len(self.agent.replay_buffer) > self.p3o_config.batch_size:
                for _ in range(self.args.updates_per_episode):
                    stats = self.agent.train_step()
                    if stats:
                        self.training_losses.append(stats)
            
            # Track statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.agent.episode_step)
            
            # Calculate moving averages
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
            else:
                avg_reward = episode_reward
                avg_length = self.agent.episode_step
            
            # Print progress
            if episode % self.args.log_interval == 0:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg(10): {avg_reward:7.2f} | "
                      f"Steps: {self.agent.episode_step:3d} | "
                      f"Buffer: {len(self.agent.replay_buffer):5d}")
                
                # Log to ClearML
                if self.task:
                    self.task.get_logger().report_scalar(
                        "Training", "Episode Reward", 
                        iteration=episode, value=episode_reward
                    )
                    self.task.get_logger().report_scalar(
                        "Training", "Average Reward (10)", 
                        iteration=episode, value=avg_reward
                    )
            
            # Save checkpoint
            if episode % self.args.save_interval == 0 and episode > 0:
                model_path = self.model_dir / f"p3o_checkpoint_{episode}.pt"
                self.agent.save(str(model_path))
                print(f"Saved checkpoint to {model_path}")
            
            # Save best model
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                best_path = self.model_dir / "p3o_best.pt"
                self.agent.save(str(best_path))
                print(f"New best model saved (Avg Reward: {avg_reward:.2f})")
        
        # Final save
        final_path = self.model_dir / "p3o_final.pt"
        self.agent.save(str(final_path))
        print(f"\nTraining completed! Final model saved to {final_path}")
        
        # Print summary
        print("\nTraining Summary:")
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Best Average Reward: {self.best_reward:.2f}")
        print(f"Final Average Reward: {np.mean(self.episode_rewards[-10:]):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Train P3O agent for drone racing")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=500,
                       help="Maximum steps per episode")
    parser.add_argument("--updates_per_episode", type=int, default=10,
                       help="Training updates per episode")
    
    # P3O hyperparameters
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (default: use config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (default: use config)")
    parser.add_argument("--procrastination_factor", type=float, default=None,
                       help="Procrastination factor (default: use config)")
    
    # Reward configuration
    parser.add_argument("--reward_preset", type=str, default="intermediate",
                       choices=["beginner", "intermediate", "advanced", 
                               "speed_focused", "precision_focused"],
                       help="Reward function preset")
    
    # Logging and saving
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Episodes between logging")
    parser.add_argument("--save_interval", type=int, default=100,
                       help="Episodes between checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--use_clearml", action="store_true",
                       help="Use ClearML for experiment tracking")
    
    args = parser.parse_args()
    
    # Create trainer and run training
    trainer = P3OTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()