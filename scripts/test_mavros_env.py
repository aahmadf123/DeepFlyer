#!/usr/bin/env python3
"""
Test script for MAVROSEnv implementation.

This script demonstrates how to use the MAVROSEnv with both Explorer and 
Researcher modes. It can be run with or without actual ROS/MAVROS installed,
as it will use the mock implementation when needed.
"""

import os
import sys
import time
import numpy as np
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_mavros")

# Import our environment classes
from rl_agent.env.mavros_env import (
    MAVROSEnv, 
    MAVROSExplorerEnv, 
    MAVROSResearcherEnv, 
    MAVROS_AVAILABLE
)


def test_explorer_mode(render: bool = False, steps: int = 100):
    """Test Explorer mode environment."""
    logger.info("Testing Explorer mode environment...")
    
    # Create environment with default settings
    env = MAVROSExplorerEnv()
    
    # Log environment info
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    
    # Reset environment
    obs, info = env.reset()
    logger.info(f"Initial observation keys: {list(obs.keys())}")
    logger.info(f"Initial info: {info}")
    
    # Run simple test loop
    total_reward = 0
    
    for i in range(steps):
        # Take random actions
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Log progress occasionally
        if i % 10 == 0:
            position = obs.get('position', np.zeros(3))
            logger.info(f"Step {i}: Position {position}, Reward: {reward:.3f}")
            
            if 'obstacle_distance' in obs:
                logger.info(f"Obstacle distance: {obs['obstacle_distance'][0]:.2f}m")
        
        # Optional rendering (only relevant if visualization is available)
        if render:
            env.render()
            
        # Exit if episode is done
        if terminated or truncated:
            logger.info(f"Episode ended at step {i}: terminated={terminated}, truncated={truncated}")
            break
    
    logger.info(f"Explorer mode test completed. Total reward: {total_reward:.3f}")
    env.close()


def test_researcher_mode(render: bool = False, steps: int = 100):
    """Test Researcher mode environment."""
    logger.info("Testing Researcher mode environment...")
    
    # Create environment with custom settings
    env = MAVROSResearcherEnv(
        max_episode_steps=steps,
        with_noise=True,
        noise_level=0.02,  # 2% noise
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Run test loop with custom control logic
    total_reward = 0
    
    for i in range(steps):
        # For researcher mode, use a slightly more intelligent policy
        # Move toward goal with some noise
        if 'goal_relative' in obs:
            # Simple proportional control toward goal with normalization
            goal_dir = obs['goal_relative']
            goal_dist = np.linalg.norm(goal_dir)
            
            # Normalize and add noise
            if goal_dist > 0:
                goal_dir = goal_dir / goal_dist
            
            # Linear velocity proportional to distance (capped at 1.0)
            vx = goal_dir[0] * min(goal_dist, 1.0)
            vy = goal_dir[1] * min(goal_dist, 1.0)
            vz = goal_dir[2] * min(goal_dist, 1.0)
            
            # Add some random yaw
            yaw = np.random.uniform(-0.3, 0.3)
            
            # Scale to action space [-1, 1]
            action = np.array([vx, vy, vz, yaw])
            action = np.clip(action, -1.0, 1.0)
        else:
            # Fallback to random actions
            action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Log more detailed information for researcher mode
        if i % 10 == 0:
            position = obs.get('position', np.zeros(3))
            velocity = obs.get('linear_velocity', np.zeros(3))
            logger.info(f"Step {i}: Position {position}, Velocity {velocity}, Reward: {reward:.3f}")
            
            # Log info dict for researchers
            if 'distance_to_goal' in info:
                logger.info(f"Distance to goal: {info['distance_to_goal']:.2f}m")
            
            # Log any safety violations
            if info.get('safety_violation', False):
                logger.warning("Safety boundary violated!")
        
        # Optional rendering
        if render:
            env.render()
            
        # Exit if episode is done
        if terminated or truncated:
            logger.info(f"Episode ended at step {i}: terminated={terminated}, truncated={truncated}")
            break
    
    logger.info(f"Researcher mode test completed. Total reward: {total_reward:.3f}")
    env.close()


def test_mavros_functions():
    """Test MAVROS specific functions."""
    logger.info("Testing MAVROS specific functions...")
    
    # Create basic environment
    env = MAVROSEnv(auto_arm=False, auto_offboard=False)
    
    # Test connection
    logger.info(f"MAVROS connected: {env.is_connected()}")
    
    # Test arming
    logger.info("Testing arm command...")
    armed = env.node.arm(True)
    logger.info(f"Arm command result: {armed}")
    time.sleep(1)
    
    # Test mode setting
    logger.info("Testing mode setting...")
    mode_set = env.node.set_mode("OFFBOARD")
    logger.info(f"Mode set result: {mode_set}")
    time.sleep(1)
    
    # Test disarming
    logger.info("Testing disarm command...")
    disarmed = env.node.arm(False) 
    logger.info(f"Disarm command result: {disarmed}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test MAVROS Environment")
    parser.add_argument(
        "--mode", 
        choices=["explorer", "researcher", "mavros", "all"], 
        default="explorer",
        help="Which mode to test"
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Enable rendering"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=50,
        help="Number of steps to run"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Testing with {'real' if MAVROS_AVAILABLE else 'mock'} MAVROS")
    
    if args.mode in ["explorer", "all"]:
        test_explorer_mode(render=args.render, steps=args.steps)
        
    if args.mode in ["researcher", "all"]:
        test_researcher_mode(render=args.render, steps=args.steps)
    
    if args.mode in ["mavros", "all"]:
        test_mavros_functions()


if __name__ == "__main__":
    main() 