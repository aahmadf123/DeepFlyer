"""
PID Controller for drone path following.
"""

import numpy as np
from typing import Tuple


class PIDController:
    """PID controller for drone path following."""
    
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Initialize error history
        self.prev_error = 0.0
        self.integral = 0.0
    
    def compute_control(
        self,
        cross_track_error: float,
        heading_error: float,
        dt: float = 0.1
    ) -> Tuple[float, float]:
        """
        Compute control output using PID.
        
        Args:
            cross_track_error: Perpendicular distance from path
            heading_error: Difference between current and desired heading
            dt: Time step
            
        Returns:
            velocity_command: Tuple of (linear_velocity, angular_velocity)
        """
        # Update integral
        self.integral += cross_track_error * dt
        
        # Compute derivative
        derivative = (cross_track_error - self.prev_error) / dt
        
        # Compute control output
        control = (
            self.kp * cross_track_error +
            self.ki * self.integral +
            self.kd * derivative
        )
        
        # Update previous error
        self.prev_error = cross_track_error
        
        # Convert to velocity command
        # Scale control to reasonable velocity range
        linear_velocity = np.clip(control, -1.0, 1.0)
        
        # Heading control is proportional to heading error
        angular_velocity = -self.kp * heading_error
        
        return linear_velocity, angular_velocity
    
    def update_gains(self, kp: float, ki: float = None, kd: float = None):
        """
        Update PID gains.
        
        Args:
            kp: New proportional gain
            ki: New integral gain (optional)
            kd: New derivative gain (optional)
        """
        self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd 