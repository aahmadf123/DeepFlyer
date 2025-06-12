#!/usr/bin/env python3
"""
Test script for DeepFlyer RL models.
"""

import argparse
import gymnasium as gym
import numpy as np
import torch

from rl_agent.supervisor_agent import SupervisorAgent


def test_supervisor(env_name="CartPole-v1", episodes=1):
    """Test the supervisor agent on a simple environment."""
    print(f"Testing SupervisorAgent on {env_name}...")
    
    # Create environment
    env = gym.make(env_name)
    
    # Initialize model
    model = SupervisorAgent(
        observation_space=env.observation_space,
        action_space=env.action_space
    )
    
    # Run episodes
    for episode in range(episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (done or truncated):
            # Select action (PID gain)
            gain, _ = model.predict(observation)
            
            # Get PID control output
            # For testing, we'll use a simple error calculation
            # In practice, this would be the cross-track and heading errors
            cross_track_error = observation[0]  # Position error
            heading_error = observation[2]  # Angular error
            
            # Compute control using PID
            linear_vel, angular_vel = model.pid.compute_control(
                cross_track_error,
                heading_error
            )
            
            # Combine into action
            action = np.array([linear_vel, angular_vel])
            
            # Take action
            next_observation, reward, done, truncated, info = env.step(action)
            
            # Store transition
            model.add_to_buffer(
                observation, 
                gain,  # Store the PID gain as the action
                next_observation, 
                reward, 
                done
            )
            
            # Update observation
            observation = next_observation
            total_reward += reward
            steps += 1
            
            # Learn if enough samples
            if len(model.replay_buffer) > 100:
                model.learn()
        
        print(f"Episode {episode + 1}: {total_reward} reward, {steps} steps")
    
    env.close()
    print("Finished testing SupervisorAgent")


def main():
    parser = argparse.ArgumentParser(description="Test DeepFlyer RL models")
    parser.add_argument(
        "--env", 
        type=str, 
        default="CartPole-v1",
        help="Gym environment to use"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=1,
        help="Number of episodes to run"
    )
    
    args = parser.parse_args()
    test_supervisor(args.env, args.episodes)


if __name__ == "__main__":
    main()
