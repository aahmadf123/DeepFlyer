#!/usr/bin/env python3
"""
Test script for the RL-as-Supervisor approach.

This script runs a simple test of the RL-as-Supervisor approach for
tuning PID controllers on a drone path following task.
"""

import rclpy
import time
import argparse
import numpy as np
from threading import Thread
import os
import logging
import json
from datetime import datetime

from rl_agent.rl_pid_supervisor import RLPIDSupervisor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("test_rl_supervisor")


def run_test_sequence(supervisor: RLPIDSupervisor, args):
    """Run a test sequence."""
    # Wait for ROS2 nodes to initialize
    logger.info("Waiting for initialization...")
    time.sleep(3.0)
    
    # Define test path
    origin = [0.0, 0.0, 1.5]  # Start at 1.5m height
    target = [10.0, 0.0, 1.5]  # Go 10m forward
    
    # Set path
    supervisor.set_path(origin, target)
    logger.info(f"Set path from {origin} to {target}")
    
    # Training phase
    if args.train:
        logger.info("Starting data collection...")
        supervisor.enable_logging(True)
        
        # Create metrics tracker
        metrics = {
            "timesteps": [],
            "rewards": [],
            "on_policy_losses": [],
            "off_policy_losses": [],
            "entropy": [],
            "cross_track_errors": [],
            "heading_errors": [],
            "pid_gains": []
        }
        
        start_time = time.time()
        last_learn_time = start_time
        last_print_time = start_time
        
        # Collect data and periodically trigger P3O learning
        logger.info(f"Collecting data for {args.collect_time} seconds...")
        while time.time() - start_time < args.collect_time:
            # Periodically trigger learning
            current_time = time.time()
            if current_time - last_learn_time >= args.learn_interval:
                # Train the agent
                train_metrics = supervisor.learn(batch_size=args.batch_size)
                
                # Record metrics
                if "on_policy_loss" in train_metrics:
                    metrics["timesteps"].append(current_time - start_time)
                    metrics["on_policy_losses"].append(train_metrics.get("on_policy_loss", float('nan')))
                    metrics["off_policy_losses"].append(train_metrics.get("off_policy_value_loss", float('nan')))
                    metrics["entropy"].append(train_metrics.get("entropy", float('nan')))
                    metrics["cross_track_errors"].append(float(supervisor.cross_track_error))
                    metrics["heading_errors"].append(float(supervisor.heading_error))
                    metrics["pid_gains"].append(float(supervisor.pid_controller.kp))
                    metrics["rewards"].append(-(abs(supervisor.cross_track_error) + 0.1 * abs(supervisor.heading_error)))
                
                last_learn_time = current_time
                
                # Print progress periodically
                if current_time - last_print_time >= 5.0:
                    logger.info(
                        f"Training progress: {current_time - start_time:.1f}/{args.collect_time}s | "
                        f"P-gain: {supervisor.pid_controller.kp:.3f} | "
                        f"Cross-track error: {supervisor.cross_track_error:.3f}"
                    )
                    last_print_time = current_time
            
            time.sleep(0.1)  # Small sleep to avoid high CPU usage
        
        # Final training
        logger.info(f"Performing final training with {args.train_steps} steps...")
        for _ in range(args.train_steps):
            supervisor.learn(batch_size=args.batch_size)
        
        # Save the model if path specified
        if args.save_model:
            model_dir = os.path.dirname(args.save_model)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
            supervisor.save_model(args.save_model)
            logger.info(f"Model saved to {args.save_model}")
            
            # Save metrics alongside model
            metrics_path = os.path.splitext(args.save_model)[0] + "_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            logger.info(f"Training metrics saved to {metrics_path}")
    
    # Test phase
    if args.test:
        # Load model if path specified
        if args.load_model:
            supervisor.load_model(args.load_model)
            logger.info(f"Model loaded from {args.load_model}")
        
        # Follow path with trained agent
        logger.info("Following path with trained agent...")
        supervisor.enable_logging(True)  # Enable logging for metrics
        
        # Wait for test to complete
        logger.info(f"Testing for {args.test_time} seconds...")
        
        start_time = time.time()
        last_print_time = start_time
        
        while time.time() - start_time < args.test_time:
            # Print progress periodically
            current_time = time.time()
            if current_time - last_print_time >= 2.0:
                logger.info(
                    f"Test progress: {current_time - start_time:.1f}/{args.test_time}s | "
                    f"P-gain: {supervisor.pid_controller.kp:.3f} | "
                    f"Cross-track error: {supervisor.cross_track_error:.3f} | "
                    f"Heading error: {supervisor.heading_error:.3f}"
                )
                last_print_time = current_time
            
            time.sleep(0.1)  # Small sleep to avoid high CPU usage
    
    logger.info("Test sequence completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test RL-as-Supervisor for PID tuning.')
    parser.add_argument('--train', action='store_true', help='Perform training')
    parser.add_argument('--test', action='store_true', help='Perform testing')
    parser.add_argument('--collect_time', type=float, default=60.0, help='Data collection time (seconds)')
    parser.add_argument('--test_time', type=float, default=30.0, help='Test duration (seconds)')
    parser.add_argument('--train_steps', type=int, default=1000, help='Number of final training steps')
    parser.add_argument('--save_model', type=str, default='', help='Path to save the model')
    parser.add_argument('--load_model', type=str, default='', help='Path to load the model')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for learning')
    parser.add_argument('--learn_interval', type=float, default=1.0, help='Interval between learning updates (seconds)')
    parser.add_argument('--procrastination_factor', type=float, default=0.95, help='P3O procrastination factor (0-1)')
    parser.add_argument('--alpha', type=float, default=0.2, help='Blend factor for P3O (0-1)')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient')
    args = parser.parse_args()
    
    if not args.train and not args.test:
        logger.error("Error: Please specify at least one of --train or --test")
        return
    
    # Initialize ROS2
    rclpy.init()
    
    # Create supervisor with P3O specific parameters
    supervisor = RLPIDSupervisor(
        procrastination_factor=args.procrastination_factor,
        alpha=args.alpha,
        entropy_coef=args.entropy_coef
    )
    
    # Create thread for ROS2 spinning
    ros_thread = Thread(target=lambda: rclpy.spin(supervisor))
    ros_thread.daemon = True
    ros_thread.start()
    
    try:
        # Run test sequence
        run_test_sequence(supervisor, args)
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    except Exception as e:
        logger.error(f"Error during test: {str(e)}", exc_info=True)
    finally:
        # Clean up
        supervisor.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == '__main__':
    main() 