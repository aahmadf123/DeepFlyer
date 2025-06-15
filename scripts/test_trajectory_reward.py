#!/usr/bin/env python3
"""
Test script for trajectory following reward function.

This script demonstrates how to use the trajectory following reward function
in a direct control reinforcement learning scenario with wind disturbance.
"""

import os
import sys
import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_trajectory_reward")

# Import our reward functions
from rl_agent.rewards import (
    create_trajectory_following_reward,
    create_wind_resistant_reward,
    RewardFunction
)

# Import environment
from rl_agent.env.mavros_env import MAVROSEnv


def test_trajectory_rewards():
    """Test different trajectory following reward functions."""
    logger.info("Testing trajectory following reward functions...")
    
    # Define a simple trajectory from point A to B
    trajectory = [
        np.array([0.0, 0.0, 1.5]),  # Start point
        np.array([2.5, 2.5, 1.5]),  # Mid point
        np.array([5.0, 5.0, 1.5]),  # End point
    ]
    
    # Create a reward function for basic trajectory following
    basic_reward = create_trajectory_following_reward()
    basic_reward.components[0].parameters["trajectory"] = trajectory
    
    # Create a reward function optimized for wind resistance
    wind_reward = create_wind_resistant_reward()
    wind_reward.components[0].parameters["trajectory"] = trajectory
    
    # Create a fully custom reward function
    custom_reward = RewardFunction()
    custom_reward.add_component(
        "follow_trajectory", 
        weight=2.0,  # Very high weight on trajectory following
        parameters={
            "trajectory": trajectory,
            "cross_track_weight": 0.9,  # Strong emphasis on path
            "progress_weight": 0.1      # Little emphasis on speed
        }
    )
    custom_reward.add_component("avoid_crashes", weight=1.5)
    custom_reward.add_component("fly_smoothly", weight=1.2)
    custom_reward.add_component("be_fast", weight=0.1)
    
    # Create environments with each reward function
    envs = []
    names = ["Basic", "Wind-Resistant", "Custom"]
    reward_fns = [basic_reward, wind_reward, custom_reward]
    
    for name, reward_fn in zip(names, reward_fns):
        try:
            env = MAVROSEnv(
                goal_position=trajectory[-1],  # End point
                enable_safety_layer=True,
                reward_function=reward_fn
            )
            envs.append((name, env))
            logger.info(f"Created environment with {name} reward function")
        except Exception as e:
            logger.error(f"Failed to create environment with {name} reward: {e}")
    
    # Test each environment with simulated flight path
    results = []
    for name, env in envs:
        try:
            # Reset environment
            obs, info = env.reset()
            
            # Run test with simulated flight path
            steps = 100
            rewards = []
            positions = []
            
            # Current position (start at the beginning of trajectory)
            position = trajectory[0].copy()
            
            # Add some wind factor (constant drift in one direction)
            wind_direction = np.array([0.2, 0.1, 0.0])
            
            for i in range(steps):
                # Direction to next waypoint
                target_idx = min(int(i / (steps/len(trajectory))), len(trajectory)-1)
                target = trajectory[target_idx]
                direction = target - position
                distance = np.linalg.norm(direction)
                
                # Create velocity command
                if distance > 0.1:
                    velocity = direction / distance * min(distance, 0.5)  # Max speed 0.5 m/s
                else:
                    velocity = np.zeros(3)
                
                # Add wind effect (causing drift)
                position = position + velocity * 0.05 + wind_direction * 0.05
                positions.append(position.copy())
                
                # Create state dictionary
                state = {
                    "position": position,
                    "linear_velocity": velocity,
                    "collision_flag": False,
                    "distance_to_obstacle": 10.0,  # Far from obstacles
                    "prev_velocity": velocity - np.array([0.01, 0.01, 0]),  # Small change
                    "dt": 0.05
                }
                
                # Calculate reward
                reward = env.reward_function.compute_reward(
                    state=state,
                    action=np.zeros(4),  # Dummy action
                    next_state=None,
                    info={}
                )
                rewards.append(reward)
                
                # Record reward components
                if i == 0 or i == steps//2 or i == steps-1:
                    component_values = env.reward_function.component_values
                    logger.info(f"Step {i}, {name} reward: {reward:.3f}, "
                              f"Components: {component_values}")
            
            # Store results
            results.append({
                'name': name,
                'rewards': rewards,
                'positions': positions,
                'trajectory': trajectory
            })
            
        except Exception as e:
            logger.error(f"Error testing {name} reward: {e}")
        
        finally:
            # Clean up
            try:
                env.close()
            except Exception:
                pass
    
    # Plot the results
    plot_reward_comparison(results)
    plot_trajectory_comparison(results)


def plot_reward_comparison(results: List[Dict]):
    """Plot reward comparison between different reward functions."""
    plt.figure(figsize=(10, 6))
    
    for result in results:
        plt.plot(result['rewards'], label=result['name'])
    
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Function Comparison")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig("trajectory_reward_comparison.png")
    logger.info("Saved reward comparison plot to trajectory_reward_comparison.png")


def plot_trajectory_comparison(results: List[Dict]):
    """Plot trajectory comparison between different reward functions."""
    plt.figure(figsize=(10, 8))
    
    # Plot the reference trajectory
    if results:
        traj = results[0]['trajectory']
        traj_x = [p[0] for p in traj]
        traj_y = [p[1] for p in traj]
        plt.plot(traj_x, traj_y, 'k--', linewidth=2, label="Reference Trajectory")
    
    # Plot each result
    for result in results:
        positions = result['positions']
        pos_x = [p[0] for p in positions]
        pos_y = [p[1] for p in positions]
        plt.plot(pos_x, pos_y, linewidth=1.5, label=result['name'])
    
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    
    # Save the figure
    plt.savefig("trajectory_path_comparison.png")
    logger.info("Saved trajectory comparison plot to trajectory_path_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trajectory reward functions")
    parser.add_argument('--mock', action='store_true', help="Use mock ROS implementation")
    
    args = parser.parse_args()
    
    if args.mock:
        # Force mock ROS environment
        os.environ['USE_MOCK_ROS'] = '1'
        logger.info("Using mock ROS implementation")
    
    try:
        test_trajectory_rewards()
    except Exception as e:
        logger.error(f"Error in test: {e}", exc_info=True) 