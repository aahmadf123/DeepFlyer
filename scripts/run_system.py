#!/usr/bin/env python3
"""
System runner script.

This script provides a simplified interface to run the complete RL-as-Supervisor system.
"""

import argparse
import subprocess
import time
import os
import signal
import sys
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("run_system")


class ProcessManager:
    """Process manager for handling multiple processes."""
    
    def __init__(self):
        """Initialize process manager."""
        self.processes = []
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.kill_all()
        sys.exit(0)
    
    def add_process(self, process):
        """Add process to manager."""
        self.processes.append(process)
    
    def kill_all(self):
        """Kill all processes."""
        logger.info(f"Terminating {len(self.processes)} processes...")
        for process in self.processes:
            if process and process.poll() is None:
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Process {process.pid} did not terminate gracefully, killing...")
                        process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process: {str(e)}")
        
        # Clear process list
        self.processes = []


def run_command(cmd, process_manager=None, background=False, check_exists=None):
    """
    Run a command.
    
    Args:
        cmd: Command to run
        process_manager: Process manager to register process with
        background: Whether to run in background
        check_exists: Path to check exists before running command
    
    Returns:
        Process object if background, otherwise None
    """
    # Check if file exists
    if check_exists and not os.path.exists(check_exists):
        logger.error(f"Required file {check_exists} does not exist")
        return None
        
    try:
        logger.debug(f"Running command: {cmd}")
        process = subprocess.Popen(cmd, shell=True)
        
        # Register process with manager if provided
        if process_manager and background:
            process_manager.add_process(process)
        
        if background:
            return process
        else:
            process.wait()
            if process.returncode != 0:
                logger.error(f"Command failed with error code {process.returncode}: {cmd}")
            return None
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return None


def create_directory(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path to directory
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {path}: {str(e)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run the RL-as-Supervisor system.')
    
    # Mode selection
    parser.add_argument('mode', choices=['train', 'test', 'demo', 'wind-test'], help='Operation mode')
    
    # Common options
    parser.add_argument('--viz', action='store_true', help='Enable visualization')
    parser.add_argument('--wind', action='store_true', help='Enable wind simulation')
    parser.add_argument('--wind-speed', type=float, default=1.0, help='Wind speed (m/s)')
    parser.add_argument('--wind-dir', type=float, default=1.57, help='Wind direction (radians)')
    parser.add_argument('--wind-gust', type=float, default=0.1, help='Wind gust frequency (Hz)')
    parser.add_argument('--wind-var', type=float, default=0.3, help='Wind variability (0-1)')
    
    # Training options
    parser.add_argument('--collect-time', type=float, default=60.0, help='Data collection time (seconds)')
    parser.add_argument('--train-steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save/load models')
    parser.add_argument('--model-name', type=str, default=None, help='Model filename (without extension)')
    
    # Test options
    parser.add_argument('--test-time', type=float, default=30.0, help='Test duration (seconds)')
    
    # Path options
    parser.add_argument('--origin', type=float, nargs=3, default=[0, 0, 1.5], help='Path origin [x y z]')
    parser.add_argument('--target', type=float, nargs=3, default=[10, 0, 1.5], help='Path target [x y z]')
    
    # Logging
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create model directory if it doesn't exist
    create_directory(args.model_dir)
    
    # Generate model name if not provided
    if args.model_name is None:
        wind_str = f"wind{args.wind_speed}" if args.wind else "nowind"
        args.model_name = f"rl_pid_{args.mode}_{wind_str}"
    
    # Full model path
    model_path = os.path.join(args.model_dir, f"{args.model_name}.pkl")
    
    # Create process manager
    process_manager = ProcessManager()
    
    try:
        # Start wind simulation if enabled
        if args.wind:
            logger.info(f"Starting wind simulation (speed={args.wind_speed} m/s)...")
            wind_cmd = (
                f"python3 -m scripts.simulate_wind "
                f"--direction {args.wind_dir} "
                f"--speed {args.wind_speed} "
                f"--gust-freq {args.wind_gust} "
                f"--variability {args.wind_var}"
            )
            wind_process = run_command(wind_cmd, process_manager, background=True)
            if not wind_process:
                logger.error("Failed to start wind simulation")
            else:
                time.sleep(1)  # Give time for wind simulator to start
        
        # Start visualization if enabled
        if args.viz:
            logger.info("Starting path visualization...")
            viz_cmd = (
                f"python3 -m scripts.visualize_path "
                f"--origin {args.origin[0]} {args.origin[1]} {args.origin[2]} "
                f"--target {args.target[0]} {args.target[1]} {args.target[2]}"
            )
            viz_process = run_command(viz_cmd, process_manager, background=True)
            if not viz_process:
                logger.error("Failed to start visualization")
            else:
                time.sleep(1)  # Give time for visualization to start
        
        # Execute mode-specific commands
        if args.mode == 'train':
            logger.info(f"Training model (collecting data for {args.collect_time}s)...")
            train_cmd = (
                f"python3 -m scripts.test_rl_supervisor "
                f"--train "
                f"--collect_time {args.collect_time} "
                f"--train_steps {args.train_steps} "
                f"--save_model {model_path}"
            )
            if run_command(train_cmd, process_manager) is None:
                logger.info(f"Model saved to {model_path}")
        
        elif args.mode == 'test':
            if not os.path.exists(model_path):
                logger.error(f"Error: Model file {model_path} does not exist.")
                return
            
            logger.info(f"Testing model (duration: {args.test_time}s)...")
            test_cmd = (
                f"python3 -m scripts.test_rl_supervisor "
                f"--test "
                f"--test_time {args.test_time} "
                f"--load_model {model_path}"
            )
            run_command(test_cmd, process_manager, check_exists=model_path)
        
        elif args.mode == 'demo':
            logger.info("Running demo with standard PID controller...")
            demo_cmd = (
                f"python3 -m scripts.test_rl_supervisor "
                f"--test "
                f"--test_time {args.test_time}"
            )
            run_command(demo_cmd, process_manager)
        
        elif args.mode == 'wind-test':
            # Test without wind first
            logger.info("Testing with standard PID (no wind)...")
            no_wind_cmd = (
                f"python3 -m scripts.test_rl_supervisor "
                f"--test "
                f"--test_time {args.test_time}"
            )
            run_command(no_wind_cmd, process_manager)
            
            # Then test with wind
            if not args.wind:
                logger.info("Starting wind simulation for comparison...")
                wind_cmd = (
                    f"python3 -m scripts.simulate_wind "
                    f"--direction {args.wind_dir} "
                    f"--speed {args.wind_speed} "
                    f"--gust-freq {args.wind_gust} "
                    f"--variability {args.wind_var}"
                )
                wind_process = run_command(wind_cmd, process_manager, background=True)
                if wind_process:
                    time.sleep(1)  # Give time for wind simulator to start
            
            logger.info("Testing with standard PID (with wind)...")
            with_wind_cmd = (
                f"python3 -m scripts.test_rl_supervisor "
                f"--test "
                f"--test_time {args.test_time}"
            )
            run_command(with_wind_cmd, process_manager)
            
            # Finally test with RL-tuned PID if model exists
            if os.path.exists(model_path):
                logger.info(f"Testing with RL-tuned PID (with wind)...")
                rl_cmd = (
                    f"python3 -m scripts.test_rl_supervisor "
                    f"--test "
                    f"--test_time {args.test_time} "
                    f"--load_model {model_path}"
                )
                run_command(rl_cmd, process_manager, check_exists=model_path)
            else:
                logger.warning(f"Model file {model_path} does not exist. Skipping RL-tuned PID test.")
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
    finally:
        # Clean up processes
        logger.info("Cleaning up...")
        process_manager.kill_all()


if __name__ == '__main__':
    main() 