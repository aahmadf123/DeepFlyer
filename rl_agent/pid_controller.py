"""
PID Controller for path following with adaptive gains.
"""

import numpy as np


class PIDController:
    """
    PID Controller for path following with adaptive gains.
    
    This controller computes control outputs based on cross-track and heading errors
    using PID control, with gains that can be adaptively tuned by the RL agent.
    """
    
    def __init__(
        self,
        kp: float = 0.8,
        ki: float = 0.0,
        kd: float = 0.2,
        max_linear_velocity: float = 1.0,
        max_angular_velocity: float = 1.0
    ):
        """
        Initialize the PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            max_linear_velocity: Maximum linear velocity
            max_angular_velocity: Maximum angular velocity
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_linear_vel = max_linear_velocity
        self.max_angular_vel = max_angular_velocity
        
        # Initialize error integrals
        self.cross_track_integral = 0.0
        self.heading_integral = 0.0
        
        # Initialize error derivatives (previous errors)
        self.prev_cross_track_error = 0.0
        self.prev_heading_error = 0.0
        
        # Timestamp for dt calculation
        self.prev_time = None
    
    def update_gains(
        self,
        kp: float = None,
        ki: float = None,
        kd: float = None
    ) -> None:
        """
        Update PID gains.
        
        Args:
            kp: New proportional gain (if None, keep current)
            ki: New integral gain (if None, keep current)
            kd: New derivative gain (if None, keep current)
        """
        if kp is not None:
            self.kp = kp
        
        if ki is not None:
            self.ki = ki
        
        if kd is not None:
            self.kd = kd
    
    def reset(self) -> None:
        """Reset controller state."""
        self.cross_track_integral = 0.0
        self.heading_integral = 0.0
        self.prev_cross_track_error = 0.0
        self.prev_heading_error = 0.0
        self.prev_time = None
    
    def compute_control(
        self,
        cross_track_error: float,
        heading_error: float,
        timestamp: float = None
    ) -> tuple:
        """
        Compute control outputs based on errors.
        
        Args:
            cross_track_error: Cross-track error (m)
            heading_error: Heading error (rad)
            timestamp: Current timestamp for dt calculation
            
        Returns:
            linear_velocity: Linear velocity command (m/s)
            angular_velocity: Angular velocity command (rad/s)
        """
        # Calculate dt
        if timestamp is None:
            timestamp = float(np.datetime64('now', 'ms'))
            
        if self.prev_time is None:
            dt = 0.1  # Default dt if first call
        else:
            dt = max(0.001, timestamp - self.prev_time)  # Ensure positive dt
        
        self.prev_time = timestamp
        
        # Update integral terms
        self.cross_track_integral += cross_track_error * dt
        self.heading_integral += heading_error * dt
        
        # Calculate derivative terms
        cross_track_derivative = (cross_track_error - self.prev_cross_track_error) / dt
        heading_derivative = (heading_error - self.prev_heading_error) / dt
        
        # Store current errors for next derivative calculation
        self.prev_cross_track_error = cross_track_error
        self.prev_heading_error = heading_error
        
        # Calculate control outputs using PID
        # Linear velocity inversely proportional to heading error
        linear_velocity = self.max_linear_vel * (1.0 - min(abs(heading_error) / np.pi, 0.8))
        
        # Angular velocity using PID control
        angular_velocity = (
            self.kp * heading_error +
            self.ki * self.heading_integral +
            self.kd * heading_derivative
        )
        
        # Cross-track correction term
        # Use cross-track error to adjust angular velocity
        cross_track_correction = self.kp * 0.5 * cross_track_error
        
        # Apply correction with sign based on which side of the path we're on
        angular_velocity += cross_track_correction
        
        # Clamp controls to limits
        linear_velocity = max(0.0, min(linear_velocity, self.max_linear_vel))
        angular_velocity = max(-self.max_angular_vel, min(angular_velocity, self.max_angular_vel))
        
        return linear_velocity, angular_velocity 