"""
Error calculator for path following.

This module computes cross-track and heading errors for path following.
"""

import numpy as np
from typing import Tuple, List


class ErrorCalculator:
    """Error calculator for path following."""
    
    def __init__(
        self,
        origin: List[float] = None,
        target: List[float] = None
    ):
        """
        Initialize error calculator.
        
        Args:
            origin: Origin point of the path [x, y, z]
            target: Target point of the path [x, y, z]
        """
        if origin is None:
            origin = [0.0, 0.0, 0.0]
        if target is None:
            target = [10.0, 0.0, 0.0]
            
        self.origin = np.array(origin)
        self.target = np.array(target)
        
        # Compute path direction vector
        self.path_vector = self.target - self.origin
        self.path_length = np.linalg.norm(self.path_vector)
        self.path_direction = self.path_vector / self.path_length
        
        # Compute path angle in the XY plane
        self.path_angle = np.arctan2(self.path_direction[1], self.path_direction[0])
    
    def compute_errors(
        self,
        position: np.ndarray,
        heading: float
    ) -> Tuple[float, float]:
        """
        Compute cross-track and heading errors.
        
        Args:
            position: Current position [x, y, z]
            heading: Current heading angle (yaw) in radians
            
        Returns:
            cross_track_error: Perpendicular distance from the path
            heading_error: Difference between current heading and path direction
        """
        # Convert position to numpy array if it's not already
        position = np.array(position)
        
        # Vector from origin to current position
        position_vector = position - self.origin
        
        # Project position vector onto path vector
        projection = np.dot(position_vector, self.path_direction)
        
        # Compute closest point on path
        closest_point = self.origin + projection * self.path_direction
        
        # Compute cross-track error (perpendicular distance to path)
        cross_track_vector = position - closest_point
        cross_track_error = np.linalg.norm(cross_track_vector)
        
        # Determine sign of cross-track error (left/right of path)
        # For a 2D path in the XY plane, we can use the z-component of the cross product
        cross_product = np.cross(
            np.append(self.path_direction[:2], 0),
            np.append(position_vector[:2], 0)
        )
        if cross_product[2] < 0:
            cross_track_error = -cross_track_error
        
        # Compute heading error
        heading_error = heading - self.path_angle
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        return cross_track_error, heading_error
    
    def set_path(self, origin: List[float], target: List[float]):
        """
        Set a new path.
        
        Args:
            origin: Origin point of the path [x, y, z]
            target: Target point of the path [x, y, z]
        """
        self.origin = np.array(origin)
        self.target = np.array(target)
        
        # Recompute path direction vector
        self.path_vector = self.target - self.origin
        self.path_length = np.linalg.norm(self.path_vector)
        self.path_direction = self.path_vector / self.path_length
        
        # Recompute path angle in the XY plane
        self.path_angle = np.arctan2(self.path_direction[1], self.path_direction[0])
    
    def compute_progress(self, position: np.ndarray) -> float:
        """
        Compute progress along the path (0.0 to 1.0).
        
        Args:
            position: Current position [x, y, z]
            
        Returns:
            progress: Progress along the path (0.0 to 1.0)
        """
        # Convert position to numpy array if it's not already
        position = np.array(position)
        
        # Vector from origin to current position
        position_vector = position - self.origin
        
        # Project position vector onto path vector
        projection = np.dot(position_vector, self.path_direction)
        
        # Compute progress (0.0 to 1.0)
        progress = np.clip(projection / self.path_length, 0.0, 1.0)
        
        return progress 