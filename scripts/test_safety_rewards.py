#!/usr/bin/env python3
"""
Test script for safety layer and reward functions.

This script demonstrates how the safety layer prevents unsafe actions
and how different reward functions affect learning behavior.
"""

import os
import sys
import time
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
logger = logging.getLogger("test_safety_rewards")

# Import our environment and safety/reward classes
from rl_agent.env.px4_base_env import PX4BaseEnv
from rl_agent.env.safety_layer import SafetyLayer, BeginnerSafetyLayer, SafetyBounds
from rl_agent.rewards import RewardFunction, REGISTRY, create_cross_track_and_heading_reward


def test_safety_layer():
    """Test safety layer functionality."""
    logger.info("Testing safety layer...")
    
    # Create environment with safety layer enabled
    env = PX4BaseEnv(
        enable_safety_layer=True,
        goal_position=[5.0, 5.0, 1.5],
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Run test with actions that would violate safety constraints
    steps = 100
    positions = []
    original_actions = []
    safe_actions = []
    rewards = []
    
    for i in range(steps):
        # Create potentially unsafe action (move aggressively toward boundary)
        if i < 20:
            # Start with normal random actions
            action = env.action_space.sample()
        elif i < 40:
            # Then try to move up aggressively (z-axis)
            action = np.array([0.0, 0.0, 1.0, 0.0])
        elif i < 60:
            # Then try to move down aggressively
            action = np.array([0.0, 0.0, -1.0, 0.0])
        elif i < 80:
            # Then try to move horizontally at max speed
            action = np.array([1.0, 1.0, 0.0, 0.0])
        else:
            # Finally try to approach an "obstacle"
            # (we'll simulate this by setting obstacle distance low)
            env.node.state.update(distance_to_obstacle=0.3)
            action = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Step environment (safety layer will modify action if needed)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record data for analysis
        position = obs.get('position', np.zeros(3))
        positions.append(position.copy())
        original_actions.append(action.copy())
        
        # Get the actual safe action that was applied (from safety layer)
        if env.safety_layer:
            safe_action = env.safety_layer.last_safe_cmd
            safe_actions.append(safe_action.copy())
        
        rewards.append(reward)
        
        # Log safety information
        if i % 10 == 0 or info.get('safety_violation', False):
            logger.info(f"Step {i}: Position {position}, Reward: {reward:.3f}")
            
            if 'safety_violation' in info and info['safety_violation']:
                logger.warning(f"Safety violation detected! Interventions: {info.get('safety_interventions', 0)}")
                
            if 'reward_components' in info:
                components = info['reward_components']
                logger.info(f"Reward components: {components}")
        
        # Exit if episode is done
        if terminated or truncated:
            logger.info(f"Episode ended at step {i}")
            break
    
    # Close environment
    env.close()
    
    # Plot results
    plot_safety_results(positions, original_actions, safe_actions, rewards)
    
    logger.info("Safety layer test completed")


def plot_safety_results(
    positions: List[np.ndarray],
    original_actions: List[np.ndarray],
    safe_actions: List[np.ndarray],
    rewards: List[float],
):
    """Plot safety test results."""
    try:
        # Convert to numpy arrays
        positions = np.array(positions)
        original_actions = np.array(original_actions)
        safe_actions = np.array(safe_actions)
        rewards = np.array(rewards)
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot positions
        ax = axs[0]
        ax.plot(positions[:, 0], label='X')
        ax.plot(positions[:, 1], label='Y')
        ax.plot(positions[:, 2], label='Z')
        ax.set_title('Drone Position')
        ax.set_xlabel('Step')
        ax.set_ylabel('Position (m)')
        ax.legend()
        ax.grid(True)
        
        # Plot actions (original vs safe)
        ax = axs[1]
        steps = np.arange(len(original_actions))
        
        # Plot x component of velocity
        ax.plot(steps, original_actions[:, 0], 'b-', label='Original vx')
        ax.plot(steps, safe_actions[:, 0], 'b--', label='Safe vx')
        
        # Plot z component of velocity
        ax.plot(steps, original_actions[:, 2], 'r-', label='Original vz')
        ax.plot(steps, safe_actions[:, 2], 'r--', label='Safe vz')
        
        ax.set_title('Original vs Safe Actions')
        ax.set_xlabel('Step')
        ax.set_ylabel('Action Value')
        ax.legend()
        ax.grid(True)
        
        # Plot rewards
        ax = axs[2]
        ax.plot(rewards)
        ax.set_title('Rewards')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('safety_test_results.png')
        logger.info("Results saved to safety_test_results.png")
        
    except Exception as e:
        logger.error(f"Failed to plot results: {e}")


def test_reward_functions():
    """Test different reward function configurations."""
    logger.info("Testing reward functions...")
    
    # Create environment with default reward function
    env = PX4BaseEnv(
        goal_position=[5.0, 5.0, 1.5],
        enable_safety_layer=True,
    )
    
    # Create a third environment with a custom reward function
    custom_reward = create_cross_track_and_heading_reward(
        cross_track_weight=1.0,
        heading_weight=0.1,
        max_error=2.0,
        max_heading_error=np.pi,
        trajectory=[
            np.array([0.0, 0.0, 1.5]),
            np.array([2.5, 2.5, 1.5]),
            np.array([5.0, 5.0, 1.5]),
        ]
    )
    
    env2 = PX4BaseEnv(
        namespace="deepflyer",
        observation_config={
            'position': True,
            'orientation': True,
            'linear_velocity': True,
            'angular_velocity': True,
            'collision': True,
            'obstacle_distance': True,
            'goal_relative': True,
        },
        action_mode="continuous",
        max_episode_steps=100,
        step_duration=0.05,
        goal_position=[5.0, 5.0, 1.5],
        enable_safety_layer=True,
        reward_function=custom_reward,
    )
    
    # Test each environment with the same actions
    envs = [env, env2]
    env_names = ["Default", "Trajectory-following"]
    
    results = []
    
    for i, (env, name) in enumerate(zip(envs, env_names)):
        logger.info(f"Testing {name} reward configuration...")
        
        # Reset environment
        obs, info = env.reset()
        
        # Run test with fixed action sequence
        steps = 50
        rewards = []
        components = []
        
        for j in range(steps):
            # Use simple proportional controller to move toward goal
            position = obs.get('position', np.zeros(3))
            goal = env.goal_position
            
            # Direction to goal
            direction = goal - position
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                # Normalize and scale
                direction = direction / distance * min(distance, 1.0)
                
                # Create action: [vx, vy, vz, yaw_rate]
                action = np.array([direction[0], direction[1], direction[2], 0.0])
            else:
                # At goal, hover
                action = np.zeros(4)
            
            # Add some noise
            action += np.random.normal(0, 0.1, 4)
            action = np.clip(action, -1.0, 1.0)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record reward
            rewards.append(reward)
            
            # Record reward components if available
            if 'reward_components' in info:
                components.append(info['reward_components'])
            
            # Exit if episode is done
            if terminated or truncated:
                logger.info(f"Episode ended at step {j}")
                break
        
        # Store results
        results.append({
            'name': name,
            'rewards': rewards,
            'components': components,
        })
        
        # Close environment
        env.close()
    
    # Plot comparison
    plot_reward_comparison(results)
    
    logger.info("Reward function test completed")


def plot_reward_comparison(results: List[Dict]):
    """Plot reward function comparison results."""
    try:
        # Create figure with subplots
        fig, axs = plt.subplots(len(results), 1, figsize=(10, 4 * len(results)))
        
        for i, result in enumerate(results):
            name = result['name']
            rewards = result['rewards']
            components = result['components']
            
            ax = axs[i] if len(results) > 1 else axs
            
            # Plot total reward
            ax.plot(rewards, 'k-', linewidth=2, label='Total')
            
            # Plot components if available
            if components and len(components) > 0:
                # Get component names
                component_names = components[0].keys()
                
                # Extract component values
                for comp_name in component_names:
                    comp_values = [c.get(comp_name, 0) for c in components]
                    ax.plot(comp_values, '--', label=comp_name)
            
            ax.set_title(f'{name} Reward')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('reward_comparison.png')
        logger.info("Results saved to reward_comparison.png")
        
    except Exception as e:
        logger.error(f"Failed to plot results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Safety Layer and Reward Functions")
    parser.add_argument(
        "--test", 
        choices=["safety", "rewards", "all"], 
        default="all",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    if args.test in ["safety", "all"]:
        test_safety_layer()
        
    if args.test in ["rewards", "all"]:
        test_reward_functions()


if __name__ == "__main__":
    main() 