"""ROS2 message conversion utilities and helpers for DeepFlyer."""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import cv2
from dataclasses import dataclass
import quaternion  # numpy-quaternion for rotation handling

try:
    from geometry_msgs.msg import Quaternion, Vector3, Point
    from sensor_msgs.msg import Image, CompressedImage
    from std_msgs.msg import Header
    import tf_transformations
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@dataclass
class PX4ControlMode:
    """PX4 control mode configuration."""
    position_control: bool = False
    velocity_control: bool = True
    acceleration_control: bool = False
    attitude_control: bool = False
    body_rate_control: bool = False
    manual_control: bool = False


class MessageConverter:
    """Utility class for converting between ROS messages and numpy arrays."""
    
    @staticmethod
    def quaternion_to_array(q: 'Quaternion') -> np.ndarray:
        """Convert ROS Quaternion to numpy array [x, y, z, w]."""
        return np.array([q.x, q.y, q.z, q.w])
    
    @staticmethod
    def array_to_quaternion(arr: np.ndarray) -> 'Quaternion':
        """Convert numpy array [x, y, z, w] to ROS Quaternion."""
        q = Quaternion()
        q.x, q.y, q.z, q.w = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        return q
    
    @staticmethod
    def vector3_to_array(v: 'Vector3') -> np.ndarray:
        """Convert ROS Vector3 to numpy array."""
        return np.array([v.x, v.y, v.z])
    
    @staticmethod
    def array_to_vector3(arr: np.ndarray) -> 'Vector3':
        """Convert numpy array to ROS Vector3."""
        v = Vector3()
        v.x, v.y, v.z = float(arr[0]), float(arr[1]), float(arr[2])
        return v
    
    @staticmethod
    def point_to_array(p: 'Point') -> np.ndarray:
        """Convert ROS Point to numpy array."""
        return np.array([p.x, p.y, p.z])
    
    @staticmethod
    def array_to_point(arr: np.ndarray) -> 'Point':
        """Convert numpy array to ROS Point."""
        p = Point()
        p.x, p.y, p.z = float(arr[0]), float(arr[1]), float(arr[2])
        return p


class CoordinateTransform:
    """Handle coordinate transformations between different frames."""
    
    @staticmethod
    def ned_to_enu(position: np.ndarray) -> np.ndarray:
        """Convert North-East-Down to East-North-Up coordinates."""
        return np.array([position[1], position[0], -position[2]])
    
    @staticmethod
    def enu_to_ned(position: np.ndarray) -> np.ndarray:
        """Convert East-North-Up to North-East-Down coordinates."""
        return np.array([position[1], position[0], -position[2]])
    
    @staticmethod
    def body_to_world_velocity(
        body_velocity: np.ndarray, 
        orientation: np.ndarray
    ) -> np.ndarray:
        """Transform velocity from body frame to world frame.
        
        Args:
            body_velocity: Velocity in body frame [vx, vy, vz]
            orientation: Quaternion [x, y, z, w]
        
        Returns:
            Velocity in world frame
        """
        if not ROS_AVAILABLE:
            # Simple rotation without tf_transformations
            # This is a simplified version - production should use proper quaternion math
            return body_velocity
        
        # Create rotation matrix from quaternion
        rotation_matrix = tf_transformations.quaternion_matrix(orientation)[:3, :3]
        return rotation_matrix @ body_velocity
    
    @staticmethod
    def world_to_body_velocity(
        world_velocity: np.ndarray,
        orientation: np.ndarray
    ) -> np.ndarray:
        """Transform velocity from world frame to body frame."""
        if not ROS_AVAILABLE:
            return world_velocity
        
        # Create inverse rotation matrix
        rotation_matrix = tf_transformations.quaternion_matrix(orientation)[:3, :3]
        return rotation_matrix.T @ world_velocity


class StateProcessor:
    """Process and augment state information for reward functions."""
    
    def __init__(self, max_room_diagonal: float = 14.14):  # sqrt(10^2 + 10^2) for 10x10 room
        self.max_room_diagonal = max_room_diagonal
        self.prev_position = None
        self.prev_velocity = None
        self.prev_angular_velocity = None
        self.prev_timestamp = None
    
    def process_state(self, raw_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw state into format expected by reward functions.
        
        Adds derived quantities like:
        - Previous states for jerk calculation
        - Goal-relative distances
        - Normalized values
        """
        processed = raw_state.copy()
        
        # Add room diagonal for normalization
        processed['max_room_diagonal'] = self.max_room_diagonal
        
        # Calculate time delta
        current_time = raw_state.get('timestamp', 0.0)
        if self.prev_timestamp is not None:
            processed['dt'] = current_time - self.prev_timestamp
        else:
            processed['dt'] = 0.05  # default step duration
        
        # Add previous states for smoothness calculations
        if self.prev_velocity is not None:
            processed['prev_velocity'] = self.prev_velocity.copy()
        else:
            processed['prev_velocity'] = raw_state.get('linear_velocity', np.zeros(3)).copy()
        
        if self.prev_angular_velocity is not None:
            processed['prev_angular_velocity'] = self.prev_angular_velocity
        else:
            processed['prev_angular_velocity'] = 0.0
        
        # Extract current values
        position = raw_state.get('position', np.zeros(3))
        velocity = raw_state.get('linear_velocity', np.zeros(3))
        angular_velocity = raw_state.get('angular_velocity', np.zeros(3))
        
        # Add derived values
        processed['altitude'] = position[2]
        processed['vertical_velocity'] = velocity[2]
        processed['curr_velocity'] = velocity
        processed['curr_angular_velocity'] = angular_velocity[2]  # yaw rate
        
        # Goal-related calculations (if goal is set)
        if 'goal' in processed:
            goal = np.array(processed['goal'])
            processed['straight_line_dist'] = np.linalg.norm(goal - position)
            
            if self.prev_position is not None:
                processed['prev_to_goal_dist'] = np.linalg.norm(goal - self.prev_position)
            else:
                processed['prev_to_goal_dist'] = processed['straight_line_dist']
            
            processed['curr_to_goal_dist'] = processed['straight_line_dist']
        
        # Collision and obstacle info
        processed['collision_flag'] = raw_state.get('collision_flag', False)
        processed['dist_to_obstacle'] = raw_state.get('distance_to_obstacle', float('inf'))
        
        # Update previous states
        self.prev_position = position.copy()
        self.prev_velocity = velocity.copy()
        self.prev_angular_velocity = angular_velocity[2]
        self.prev_timestamp = current_time
        
        return processed
    
    def process_action(self, raw_action: np.ndarray, action_mode: str = "continuous") -> Dict[str, Any]:
        """Process raw action into format expected by reward functions."""
        action_dict = {'raw': raw_action}
        
        if action_mode == "continuous":
            # For continuous actions, estimate throttle from velocity commands
            # This is simplified - actual throttle depends on drone dynamics
            velocity_magnitude = np.linalg.norm(raw_action[:3])
            max_velocity = 1.5
            action_dict['throttle'] = min(1.0, velocity_magnitude / max_velocity)
            
            # Thrust vector (simplified)
            action_dict['thrust_vector'] = raw_action[:3] * 0.5  # scaling factor
        else:
            # For discrete actions, map to approximate throttle
            action_to_throttle = {
                0: 0.0,   # hover
                1: 0.3,   # forward
                2: 0.3,   # backward
                3: 0.3,   # left
                4: 0.3,   # right
                5: 0.5,   # up
                6: 0.2,   # down
                7: 0.2,   # rotate left
                8: 0.2,   # rotate right
            }
            action_dict['throttle'] = action_to_throttle.get(int(raw_action), 0.0)
            action_dict['thrust_vector'] = np.zeros(3)
        
        return action_dict


class ImageProcessor:
    """Process camera images for RL observations."""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size using cv2."""
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def process_depth_image(depth_image: np.ndarray, max_depth: float = 10.0) -> np.ndarray:
        """Process depth image for observations.
        
        Args:
            depth_image: Raw depth image
            max_depth: Maximum depth value for normalization
        
        Returns:
            Normalized depth image
        """
        # Clip to max depth
        depth_clipped = np.clip(depth_image, 0, max_depth)
        # Normalize to [0, 1]
        return depth_clipped / max_depth
    
    @staticmethod
    def extract_depth_features(depth_image: np.ndarray, num_sectors: int = 8) -> np.ndarray:
        """Extract depth features from depth image.
        
        Divides image into sectors and computes min distance per sector.
        """
        h, w = depth_image.shape
        cx, cy = w // 2, h // 2
        
        features = []
        for i in range(num_sectors):
            angle_start = i * 2 * np.pi / num_sectors
            angle_end = (i + 1) * 2 * np.pi / num_sectors
            
            # Create sector mask
            y, x = np.ogrid[:h, :w]
            angles = np.arctan2(y - cy, x - cx)
            mask = (angles >= angle_start) & (angles < angle_end)
            
            # Get minimum distance in sector
            sector_depths = depth_image[mask]
            if len(sector_depths) > 0:
                min_dist = np.min(sector_depths[sector_depths > 0.1])  # ignore very close
                features.append(min_dist)
            else:
                features.append(10.0)  # max distance
        
        return np.array(features)


class SafetyMonitor:
    """Monitor safety constraints and limits."""
    
    def __init__(
        self,
        max_velocity: float = 2.0,
        max_acceleration: float = 5.0,
        min_altitude: float = 0.3,
        max_altitude: float = 2.8,
        geofence_bounds: Optional[Tuple[float, float, float, float]] = None
    ):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.geofence_bounds = geofence_bounds or (-5, -5, 15, 15)  # x_min, y_min, x_max, y_max
    
    def check_velocity_limits(self, velocity: np.ndarray) -> Tuple[bool, str]:
        """Check if velocity is within safe limits."""
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude > self.max_velocity:
            return False, f"Velocity {vel_magnitude:.2f} exceeds limit {self.max_velocity}"
        return True, "OK"
    
    def check_altitude_limits(self, altitude: float) -> Tuple[bool, str]:
        """Check if altitude is within safe limits."""
        if altitude < self.min_altitude:
            return False, f"Altitude {altitude:.2f} below minimum {self.min_altitude}"
        if altitude > self.max_altitude:
            return False, f"Altitude {altitude:.2f} above maximum {self.max_altitude}"
        return True, "OK"
    
    def check_geofence(self, position: np.ndarray) -> Tuple[bool, str]:
        """Check if position is within geofence bounds."""
        x, y = position[0], position[1]
        x_min, y_min, x_max, y_max = self.geofence_bounds
        
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return False, f"Position ({x:.2f}, {y:.2f}) outside geofence"
        return True, "OK"
    
    def apply_safety_limits(self, command: np.ndarray, current_state: Dict[str, Any]) -> np.ndarray:
        """Apply safety limits to velocity command."""
        limited_command = command.copy()
        
        # Limit velocity magnitude
        vel_magnitude = np.linalg.norm(command[:3])
        if vel_magnitude > self.max_velocity:
            limited_command[:3] = command[:3] * (self.max_velocity / vel_magnitude)
        
        # Check altitude and limit vertical velocity if needed
        altitude = current_state.get('position', np.zeros(3))[2]
        if altitude <= self.min_altitude and command[2] < 0:
            limited_command[2] = max(0, command[2])  # prevent going lower
        elif altitude >= self.max_altitude and command[2] > 0:
            limited_command[2] = min(0, command[2])  # prevent going higher
        
        return limited_command


# PX4/MAVROS specific utilities
class PX4Interface:
    """Interface utilities for PX4/MAVROS integration."""
    
    @staticmethod
    def create_offboard_control_mode() -> Dict[str, bool]:
        """Create offboard control mode configuration."""
        return {
            'position': False,
            'velocity': True,
            'acceleration': False,
            'attitude': False,
            'body_rate': False,
        }
    
    @staticmethod
    def velocity_setpoint_to_mavros(
        velocity_setpoint: np.ndarray,
        yaw_rate: float = 0.0
    ) -> Dict[str, Any]:
        """Convert velocity setpoint to MAVROS format.
        
        Args:
            velocity_setpoint: [vx, vy, vz] in m/s
            yaw_rate: Yaw rate in rad/s
        
        Returns:
            Dictionary with MAVROS setpoint fields
        """
        return {
            'coordinate_frame': 'FRAME_LOCAL_NED',
            'type_mask': 0b0000011111000111,  # Ignore all except velocities
            'velocity': {
                'x': float(velocity_setpoint[0]),
                'y': float(velocity_setpoint[1]), 
                'z': float(velocity_setpoint[2]),
            },
            'yaw_rate': float(yaw_rate),
        }
    
    @staticmethod
    def check_px4_status(status_msg: Dict[str, Any]) -> Tuple[bool, str]:
        """Check PX4 status for readiness.
        
        Args:
            status_msg: Status message from PX4
        
        Returns:
            (is_ready, status_string)
        """
        if not status_msg.get('armed', False):
            return False, "Vehicle not armed"
        
        if not status_msg.get('offboard_control_active', False):
            return False, "Offboard control not active"
        
        if status_msg.get('failsafe_active', False):
            return False, "Failsafe active"
        
        return True, "Ready"


# Export utilities
__all__ = [
    'MessageConverter',
    'CoordinateTransform', 
    'StateProcessor',
    'ImageProcessor',
    'SafetyMonitor',
    'PX4Interface',
    'PX4ControlMode',
] 