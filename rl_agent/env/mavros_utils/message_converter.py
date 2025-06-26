"""
Message Converter Utilities
Handles conversion between different message formats and coordinate systems
"""

import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MessageConverter:
    """Utility class for message format conversions"""
    
    @staticmethod
    def enu_to_ned(position: np.ndarray) -> np.ndarray:
        """Convert ENU coordinates to NED (for PX4)"""
        return np.array([position[1], position[0], -position[2]])
    
    @staticmethod  
    def ned_to_enu(position: np.ndarray) -> np.ndarray:
        """Convert NED coordinates to ENU (from PX4)"""
        return np.array([position[1], position[0], -position[2]])
    
    @staticmethod
    def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w,x,y,z] to Euler angles [roll,pitch,yaw]"""
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    @staticmethod
    def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles [roll,pitch,yaw] to quaternion [w,x,y,z]"""
        roll, pitch, yaw = euler
        
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    @staticmethod
    def calculate_bearing(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
        """Calculate bearing from one position to another"""
        diff = to_pos[:2] - from_pos[:2]  # Only x,y components
        return np.arctan2(diff[1], diff[0])
    
    @staticmethod
    def px4_timestamp() -> int:
        """Generate PX4-compatible timestamp (microseconds)"""
        import time
        return int(time.time() * 1e6)
    
    @staticmethod
    def validate_position(position: np.ndarray, bounds: Dict[str, float]) -> bool:
        """Validate position is within safe bounds"""
        if len(position) != 3:
            return False
        
        x, y, z = position
        
        if 'x_min' in bounds and x < bounds['x_min']:
            return False
        if 'x_max' in bounds and x > bounds['x_max']:
            return False
        if 'y_min' in bounds and y < bounds['y_min']:
            return False
        if 'y_max' in bounds and y > bounds['y_max']:
            return False
        if 'z_min' in bounds and z < bounds['z_min']:
            return False
        if 'z_max' in bounds and z > bounds['z_max']:
            return False
        
        return True
    
    @staticmethod
    def clamp_velocity(velocity: np.ndarray, max_velocity: float) -> np.ndarray:
        """Clamp velocity magnitude to maximum value"""
        magnitude = np.linalg.norm(velocity)
        if magnitude > max_velocity:
            return velocity * (max_velocity / magnitude)
        return velocity 