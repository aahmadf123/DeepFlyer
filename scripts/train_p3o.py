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
from rl_agent.algorithms.replay_buffer import P3OReplayBuffer
from rl_agent.rewards.rewards import HoopNavigationReward, get_reward_preset
from rl_agent.env.safety_layer import SafetyLayer, SafetyBounds
from rl_agent.utils import PerformanceTracker, ClearMLTracker
from rl_agent.config_loader import get_config_manager, TrainingConfig
from rl_agent.flight_phase_integration import integrate_phase_management

# Optional ClearML integration
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("ClearML not available, training without experiment tracking")


class EpisodeManager:
    """Manages episode lifecycle and statistics"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_episode = 0
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_start_time = 0.0
        self.episode_history = []
        
        # Episode statistics
        self.total_reward = 0.0
        self.total_steps = 0
        self.successful_episodes = 0
        self.failed_episodes = 0
        
        # Performance tracking
        self.reward_window = []
        self.window_size = 100
        self.best_reward = -float('inf')
        self.best_episode = 0
        
        # Early stopping
        self.patience_counter = 0
        self.should_stop_early = False
        
    def start_episode(self):
        """Start a new episode"""
        self.current_episode += 1
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
        
        logger.info(f"Starting Episode {self.current_episode}")
    
    def step(self, reward: float):
        """Record a step in the current episode"""
        self.current_step += 1
        self.episode_reward += reward
        self.total_reward += reward
        self.total_steps += 1
    
    def end_episode(self, success: bool = False, reason: str = "completed"):
        """End the current episode"""
        episode_duration = time.time() - self.episode_start_time
        
        # Update statistics
        if success:
            self.successful_episodes += 1
        else:
            self.failed_episodes += 1
        
        # Track reward
        self.reward_window.append(self.episode_reward)
        if len(self.reward_window) > self.window_size:
            self.reward_window.pop(0)
        
        # Update best performance
        if self.episode_reward > self.best_reward:
            self.best_reward = self.episode_reward
            self.best_episode = self.current_episode
            self.patience_counter = 0
            logger.info(f"New best reward: {self.best_reward:.2f} in episode {self.best_episode}")
        else:
            self.patience_counter += 1
        
        # Check early stopping
        if self.patience_counter >= self.config.early_stopping_patience:
            self.should_stop_early = True
            logger.warning(f"Early stopping triggered after {self.patience_counter} episodes without improvement")
        
        # Record episode
        episode_info = {
            'episode': self.current_episode,
            'reward': self.episode_reward,
            'steps': self.current_step,
            'duration': episode_duration,
            'success': success,
            'reason': reason,
            'timestamp': time.time()
        }
        self.episode_history.append(episode_info)
        
        logger.info(f"Episode {self.current_episode} ended: reward={self.episode_reward:.2f}, "
                   f"steps={self.current_step}, duration={episode_duration:.1f}s, reason={reason}")
        
        return episode_info
    
    def get_stats(self) -> Dict[str, float]:
        """Get current episode statistics"""
        avg_reward = np.mean(self.reward_window) if self.reward_window else 0.0
        success_rate = self.successful_episodes / max(1, self.current_episode)
        
        return {
            'current_episode': self.current_episode,
            'total_steps': self.total_steps,
            'avg_reward_100': avg_reward,
            'best_reward': self.best_reward,
            'success_rate': success_rate,
            'patience_counter': self.patience_counter
        }
    
    def should_continue(self) -> bool:
        """Check if training should continue"""
        if self.should_stop_early:
            return False
        if self.current_episode >= self.config.max_episodes:
            return False
        if self.best_reward >= self.config.target_reward:
            logger.info(f"Target reward {self.config.target_reward} achieved!")
            return False
        return True


class P3OTrainer:
    """Enhanced training coordinator for P3O drone control with comprehensive episode management"""
    
    def __init__(self, args):
        self.args = args
        
        # Setup directories
        self.setup_directories()
        
        # Load unified configuration
        self.config_manager = get_config_manager()
        self.training_config = self.config_manager.get_training_config()
        self.p3o_config = self.config_manager.get_p3o_config()
        self.safety_config = self.config_manager.get_safety_config()
        self.camera_config = self.config_manager.get_camera_config()
        self.reward_config = self.config_manager.get_reward_config()
        
        # Initialize ClearML if available
        self.clearml_tracker = None
        if CLEARML_AVAILABLE and args.use_clearml:
            self.clearml_tracker = ClearMLTracker(
                project_name="DeepFlyer",
                task_name=f"P3O_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['p3o', 'training', 'hoop-navigation']
            )
            # Log all configurations to ClearML
            self.clearml_tracker.log_hyperparameters(self.p3o_config.to_dict())
        
        # Initialize agent with unified config
        self.control_config = DirectControlConfig(
            obs_dim=8, action_dim=4,
            max_velocity=2.0, max_yaw_rate=1.0
        )
        self.agent = DirectControlAgent(self.control_config, self.p3o_config)
        
        # Initialize enhanced replay buffer
        self.replay_buffer = P3OReplayBuffer(
            obs_dim=8, action_dim=4,
            buffer_size=10000,
            gamma=self.p3o_config.gamma,
            gae_lambda=self.p3o_config.gae_lambda
        )
        
        # Load checkpoint if specified
        if args.checkpoint:
            self.agent.load(args.checkpoint)
            print(f"Loaded checkpoint from {args.checkpoint}")
        
        # Initialize reward function
        self.reward_fn = HoopNavigationReward(self.reward_config)
        
        # Initialize safety layer with unified config
        safety_bounds = SafetyBounds(
            x_min=-5.0, x_max=5.0,
            y_min=-5.0, y_max=5.0, 
            z_min=0.3, z_max=3.0,
            vel_max_xy=self.control_config.max_velocity,
            max_tilt_angle=30.0,
            min_distance_to_obstacle=0.5
        )
        
        self.safety_layer = SafetyLayer(
            safety_bounds=safety_bounds,
            enable_geofence=self.safety_config.enable_geofence,
            enable_collision_prevention=self.safety_config.enable_collision_avoidance,
            enable_attitude_limits=self.safety_config.enable_attitude_limits,
            enable_velocity_ramping=self.safety_config.enable_velocity_ramping,
            max_acceleration=self.safety_config.max_acceleration,
            log_violations=True
        )
        
        # Initialize episode manager
        self.episode_manager = EpisodeManager(self.training_config)
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # Training statistics  
        self.training_losses = []
        self.safety_interventions = 0
        
        # Checkpointing
        self.last_checkpoint_episode = 0
        self.checkpoint_dir = Path("trained_models/p3o/checkpoints")
        
        # Resume training if specified
        if self.training_config.resume_training and args.checkpoint:
            self.resume_from_checkpoint(args.checkpoint)
        
    def setup_directories(self):
        """Create necessary directories"""
        self.model_dir = Path("trained_models/p3o")
        self.log_dir = Path("experiments/logs")
        self.config_dir = Path("config")
        self.checkpoint_dir = Path("trained_models/p3o/checkpoints")
        
        for dir_path in [self.model_dir, self.log_dir, self.config_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_checkpoint(self, episode: int, metrics: Dict[str, float]) -> str:
        """Create training checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_filename = f"p3o_checkpoint_ep{episode}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Gather training state
        episode_stats = self.episode_manager.get_stats()
        buffer_stats = self.replay_buffer.get_statistics()
        safety_stats = self.safety_layer.get_status()
        
        training_state = {
            'episode_manager': {
                'current_episode': self.episode_manager.current_episode,
                'total_steps': self.episode_manager.total_steps,
                'best_reward': self.episode_manager.best_reward,
                'best_episode': self.episode_manager.best_episode,
                'successful_episodes': self.episode_manager.successful_episodes,
                'patience_counter': self.episode_manager.patience_counter,
                'reward_window': self.episode_manager.reward_window
            },
            'safety_interventions': self.safety_interventions,
            'training_losses': self.training_losses[-100:],  # Keep last 100 losses
            'buffer_stats': buffer_stats,
            'safety_stats': safety_stats
        }
        
        # Save checkpoint using P3O agent
        saved_path = self.agent.p3o.save_checkpoint(
            checkpoint_path=str(checkpoint_path),
            episode=episode,
            metrics=metrics,
            training_state=training_state
        )
        
        # Create "latest" symlink
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        if latest_path.exists():
            latest_path.unlink()
        
        try:
            # Create symlink (works on both Unix and Windows 10+)
            latest_path.symlink_to(checkpoint_path.name)
        except OSError:
            # Fallback: copy file if symlink not supported
            import shutil
            shutil.copy2(checkpoint_path, latest_path)
        
        self.last_checkpoint_episode = episode
        logger.info(f"Training checkpoint created: {saved_path}")
        
        # Log checkpoint to ClearML if available
        if self.clearml_tracker:
            self.clearml_tracker.log_metrics({'checkpoint_saved': 1}, episode)
        
        return saved_path
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint"""
        logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = self.agent.p3o.load_checkpoint(checkpoint_path)
            
            # Restore training state
            if 'training_state' in checkpoint:
                training_state = checkpoint['training_state']
                
                # Restore episode manager state
                if 'episode_manager' in training_state:
                    em_state = training_state['episode_manager']
                    self.episode_manager.current_episode = em_state.get('current_episode', 0)
                    self.episode_manager.total_steps = em_state.get('total_steps', 0)
                    self.episode_manager.best_reward = em_state.get('best_reward', -float('inf'))
                    self.episode_manager.best_episode = em_state.get('best_episode', 0)
                    self.episode_manager.successful_episodes = em_state.get('successful_episodes', 0)
                    self.episode_manager.patience_counter = em_state.get('patience_counter', 0)
                    self.episode_manager.reward_window = em_state.get('reward_window', [])
                
                # Restore other training state
                self.safety_interventions = training_state.get('safety_interventions', 0)
                self.training_losses = training_state.get('training_losses', [])
            
            self.last_checkpoint_episode = checkpoint.get('episode', 0)
            
            logger.info(f"Training resumed from episode {self.episode_manager.current_episode}")
            logger.info(f"Best reward so far: {self.episode_manager.best_reward:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            logger.warning("Starting training from scratch")
    
    def should_save_checkpoint(self, episode: int) -> bool:
        """Check if we should save a checkpoint"""
        # Save at regular intervals
        if episode - self.last_checkpoint_episode >= self.training_config.save_frequency:
            return True
        
        # Save on best performance
        if self.episode_manager.best_episode == episode:
            return True
        
        # Save before potential early stopping
        if self.episode_manager.patience_counter >= self.training_config.early_stopping_patience - 5:
            return True
        
        return False
    
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
        """Simulate environment step with safety layer integration"""
        # Extract position from observation for safety checking
        # In simulation, we'll create a mock position based on step
        mock_position = np.array([
            obs[0] * 2.0,  # Convert normalized to meters
            obs[1] * 2.0,  # Convert normalized to meters 
            1.5 - step * 0.001  # Slowly decreasing altitude
        ])
        
        # Process action through safety layer
        safe_action = self.safety_layer.process_command(
            velocity_command=action[:3] if len(action) >= 3 else action,
            position=mock_position,
            obstacle_distance=obs[3] * 5.0 if obs[2] > 0.5 else None  # Convert to meters
        )
        
        # Count safety interventions
        safety_interventions = 0
        if not np.allclose(action[:3], safe_action, atol=1e-6):
            safety_interventions = 1
        
        # Continue with original simulation logic...
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
            
            # Track safety interventions
            safety_interventions = self.safety_layer.intervention_count
            self.performance_tracker.log_episode(
                reward=episode_reward,
                length=self.agent.episode_step,
                loss=self.training_losses[-1] if self.training_losses else None,
                safety_interventions=safety_interventions
            )
            
            # Reset safety layer for next episode
            self.safety_layer.reset()
            
            # Calculate moving averages
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
            else:
                avg_reward = episode_reward
                avg_length = self.agent.episode_step
            
            # Print progress
            if episode % self.args.log_interval == 0:
                safety_interventions = self.safety_layer.intervention_count
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg(10): {avg_reward:7.2f} | "
                      f"Steps: {self.agent.episode_step:3d} | "
                      f"Safety: {safety_interventions:2d} | "
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
        
        # Save final performance metrics
        self.performance_tracker.save_metrics("final_training_metrics.json")


def main():
    parser = argparse.ArgumentParser(description="Train P3O agent for drone racing")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, required=True,
                       help="Number of training episodes (REQUIRED)")
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
    
    # Integrate phase management for complete system
    trainer = integrate_phase_management(trainer)
    
    trainer.train()


if __name__ == "__main__":
    main()