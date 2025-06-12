#!/usr/bin/env python3
"""
Test script for the PID controller.

This script tests the PID controller with different gains and visualizes the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from rl_agent.pid_controller import PIDController
from rl_agent.error_calculator import ErrorCalculator


def test_pid(
    kp: float = 1.0,
    ki: float = 0.0,
    kd: float = 0.0,
    steps: int = 1000,
    dt: float = 0.1,
    with_disturbance: bool = False,
    disturbance_std: float = 0.1,
    output_dir: str = "output"
):
    """
    Test the PID controller.
    
    Args:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        steps: Number of steps
        dt: Time step
        with_disturbance: Whether to add wind disturbance
        disturbance_std: Standard deviation of the wind disturbance
        output_dir: Directory to save output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create controllers and calculators
    pid = PIDController(kp, ki, kd)
    error_calculator = ErrorCalculator()
    
    # Path definition
    origin = np.array([0.0, 0.0, 1.0])
    target = np.array([10.0, 0.0, 1.0])
    error_calculator.set_path(origin.tolist(), target.tolist())
    
    # Initialize state
    position = origin.copy()
    velocity = np.zeros(3)
    orientation = np.zeros(3)
    
    # Initialize history
    positions = [position.copy()]
    velocities = [velocity.copy()]
    orientations = [orientation.copy()]
    cross_track_errors = []
    heading_errors = []
    linear_velocities = []
    angular_velocities = []
    
    # Wind disturbance
    wind_direction = np.random.uniform(0, 2 * np.pi)
    wind_speed = 0.0
    if with_disturbance:
        wind_speed = np.random.uniform(0.0, 1.0)
    
    # Drone dynamics parameters
    mass = 1.0  # kg
    max_vel = 2.0  # m/s
    max_accel = 1.0  # m/s^2
    drag_coeff = 0.1
    
    # Simulation loop
    for step in range(steps):
        # Compute errors
        cross_track_error, heading_error = error_calculator.compute_errors(
            position,
            orientation[2]
        )
        
        cross_track_errors.append(cross_track_error)
        heading_errors.append(heading_error)
        
        # Compute control using PID
        linear_vel, angular_vel = pid.compute_control(
            cross_track_error,
            heading_error,
            dt
        )
        
        linear_velocities.append(linear_vel)
        angular_velocities.append(angular_vel)
        
        # Update state
        # Update orientation
        orientation[2] += angular_vel * dt
        # Normalize yaw to [-pi, pi]
        orientation[2] = np.arctan2(
            np.sin(orientation[2]),
            np.cos(orientation[2])
        )
        
        # Compute acceleration from control
        heading_vector = np.array([
            np.cos(orientation[2]),
            np.sin(orientation[2]),
            0.0
        ])
        
        # Apply linear velocity command
        target_velocity = heading_vector * linear_vel
        
        # Add wind disturbance
        if with_disturbance:
            wind_vector = np.array([
                np.cos(wind_direction),
                np.sin(wind_direction),
                0.0
            ]) * wind_speed
            
            # Add random gusts
            if np.random.random() < 0.05:  # 5% chance of a gust
                gust = np.random.normal(0, disturbance_std, 3)
                wind_vector += gust
            
            # Apply wind to velocity
            target_velocity += wind_vector
        
        # Simple dynamics: velocity approaches target with some lag
        accel = (target_velocity - velocity) * 2.0  # Gain for responsiveness
        
        # Limit acceleration
        accel_norm = np.linalg.norm(accel)
        if accel_norm > max_accel:
            accel = accel * (max_accel / accel_norm)
        
        # Update velocity
        velocity += accel * dt
        
        # Apply drag
        velocity -= velocity * drag_coeff * dt
        
        # Limit velocity
        vel_norm = np.linalg.norm(velocity)
        if vel_norm > max_vel:
            velocity = velocity * (max_vel / vel_norm)
        
        # Update position
        position += velocity * dt
        
        # Store history
        positions.append(position.copy())
        velocities.append(velocity.copy())
        orientations.append(orientation.copy())
        
        # Check if reached the target
        progress = error_calculator.compute_progress(position)
        if progress >= 0.99:
            print(f"Reached target in {step + 1} steps")
            break
    
    # Convert to numpy arrays
    positions = np.array(positions)
    velocities = np.array(velocities)
    orientations = np.array(orientations)
    cross_track_errors = np.array(cross_track_errors)
    heading_errors = np.array(heading_errors)
    linear_velocities = np.array(linear_velocities)
    angular_velocities = np.array(angular_velocities)
    
    # Plot results
    # Path
    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Drone Path')
    plt.plot([origin[0], target[0]], [origin[1], target[1]], 'r--', label='Target Path')
    plt.scatter(origin[0], origin[1], c='g', marker='o', label='Origin')
    plt.scatter(target[0], target[1], c='r', marker='x', label='Target')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(f'Drone Path (kp={kp}, ki={ki}, kd={kd})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(output_dir, 'path.png'))
    
    # Errors
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(cross_track_errors)
    plt.grid(True)
    plt.title('Cross-Track Error')
    plt.ylabel('Error')
    
    plt.subplot(2, 1, 2)
    plt.plot(heading_errors)
    plt.grid(True)
    plt.title('Heading Error')
    plt.xlabel('Step')
    plt.ylabel('Error (rad)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'errors.png'))
    
    # Controls
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(linear_velocities)
    plt.grid(True)
    plt.title('Linear Velocity')
    plt.ylabel('Velocity')
    
    plt.subplot(2, 1, 2)
    plt.plot(angular_velocities)
    plt.grid(True)
    plt.title('Angular Velocity')
    plt.xlabel('Step')
    plt.ylabel('Velocity (rad/s)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'controls.png'))
    
    print(f"Results saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test PID controller')
    parser.add_argument('--kp', type=float, default=1.0, help='Proportional gain')
    parser.add_argument('--ki', type=float, default=0.0, help='Integral gain')
    parser.add_argument('--kd', type=float, default=0.0, help='Derivative gain')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--with-disturbance', action='store_true', help='Add wind disturbance')
    parser.add_argument('--disturbance-std', type=float, default=0.1, help='Wind disturbance std')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    
    args = parser.parse_args()
    
    test_pid(
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        steps=args.steps,
        dt=args.dt,
        with_disturbance=args.with_disturbance,
        disturbance_std=args.disturbance_std,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main() 