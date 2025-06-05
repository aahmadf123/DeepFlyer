import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import time
import threading

logger = logging.getLogger(__name__)


@dataclass
class SafetyBounds:
    """Safety boundaries for drone operation."""
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = 0.1  # Minimum height (m)
    z_max: float = 5.0  # Maximum height (m)
    
    vel_max_xy: float = 2.0  # Maximum horizontal velocity (m/s)
    vel_max_z_up: float = 1.0  # Maximum ascent velocity (m/s)
    vel_max_z_down: float = 0.5  # Maximum descent velocity (m/s)
    
    max_tilt_angle: float = 30.0  # Maximum tilt angle in degrees
    
    min_distance_to_obstacle: float = 0.5  # Minimum distance to obstacles (m)
    
    def is_position_in_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within bounds."""
        x, y, z = position
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max and
                self.z_min <= z <= self.z_max)
    
    def get_position_violation(self, position: np.ndarray) -> Optional[str]:
        """
        Get description of position bound violation, or None if within bounds.
        
        Returns None if position is within bounds, otherwise a string describing 
        the violation.
        """
        x, y, z = position
        
        if x < self.x_min:
            return f"X position ({x:.2f}) below minimum ({self.x_min})"
        if x > self.x_max:
            return f"X position ({x:.2f}) above maximum ({self.x_max})"
        
        if y < self.y_min:
            return f"Y position ({y:.2f}) below minimum ({self.y_min})"
        if y > self.y_max:
            return f"Y position ({y:.2f}) above maximum ({self.y_max})"
        
        if z < self.z_min:
            return f"Z position ({z:.2f}) below minimum altitude ({self.z_min})"
        if z > self.z_max:
            return f"Z position ({z:.2f}) above maximum altitude ({self.z_max})"
        
        return None
    
    def clip_position(self, position: np.ndarray) -> np.ndarray:
        """Clip position to stay within bounds."""
        x, y, z = position
        x = np.clip(x, self.x_min, self.x_max)
        y = np.clip(y, self.y_min, self.y_max)
        z = np.clip(z, self.z_min, self.z_max)
        return np.array([x, y, z])
    
    def clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Clip velocity to be within safe limits.
        
        Args:
            velocity: 3D velocity vector [vx, vy, vz]
            
        Returns:
            Clipped velocity vector
        """
        vx, vy, vz = velocity
        
        # Clip horizontal velocity
        vxy_magnitude = np.sqrt(vx**2 + vy**2)
        if vxy_magnitude > self.vel_max_xy:
            scale = self.vel_max_xy / vxy_magnitude
            vx *= scale
            vy *= scale
        
        # Clip vertical velocity (different limits for ascent vs descent)
        if vz > 0:  # Going up
            vz = min(vz, self.vel_max_z_up)
        else:  # Going down
            vz = max(vz, -self.vel_max_z_down)
            
        return np.array([vx, vy, vz])


class SafetyMonitor:
    """
    Safety monitoring system for drone operation.
    
    This class monitors the drone state and detects unsafe conditions,
    allowing preventative actions to be taken before accidents occur.
    """
    
    def __init__(
        self,
        safety_bounds: Optional[SafetyBounds] = None,
        check_interval: float = 0.1,  # How often to check safety (seconds)
        enable_geofence: bool = True,
        enable_collision_prevention: bool = True,
        enable_attitude_limits: bool = True,
        log_violations: bool = True,
    ):
        """
        Initialize safety monitor.
        
        Args:
            safety_bounds: Safety boundaries for drone operation
            check_interval: How frequently to run safety checks
            enable_geofence: Enable geofencing based on position
            enable_collision_prevention: Enable collision prevention based on depth
            enable_attitude_limits: Enable attitude angle limits
            log_violations: Log safety violations to logger
        """
        self.safety_bounds = safety_bounds or SafetyBounds()
        self.check_interval = check_interval
        
        self.enable_geofence = enable_geofence
        self.enable_collision_prevention = enable_collision_prevention
        self.enable_attitude_limits = enable_attitude_limits
        self.log_violations = log_violations
        
        # Safety status
        self.is_safe = True
        self.violations = []
        self.safety_override_active = False
        self.emergency_stop_requested = False
        
        # Last check time
        self.last_check_time = 0.0
        
        # For emergency stop callback
        self.emergency_stop_callback = None
        
        # For automated monitoring
        self._monitoring = False
        self._monitor_thread = None
    
    def check_safety(
        self,
        position: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        orientation_euler: Optional[np.ndarray] = None,  # roll, pitch, yaw in degrees
        obstacle_distance: Optional[float] = None,
        check_all: bool = True,
    ) -> bool:
        """
        Check if current state is safe.
        
        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            orientation_euler: Current orientation in Euler angles [roll, pitch, yaw] (degrees)
            obstacle_distance: Distance to nearest obstacle (m)
            check_all: Whether to check all conditions or return on first violation
            
        Returns:
            True if safe, False otherwise
        """
        self.violations = []
        
        # Geofence check
        if self.enable_geofence and position is not None:
            violation = self.safety_bounds.get_position_violation(position)
            if violation:
                if self.log_violations:
                    logger.warning(f"Geofence violation: {violation}")
                self.violations.append(f"Geofence: {violation}")
                if not check_all:
                    return False
        
        # Collision prevention
        if (self.enable_collision_prevention and 
            obstacle_distance is not None and 
            obstacle_distance < self.safety_bounds.min_distance_to_obstacle):
            
            violation = f"Collision risk: distance to obstacle ({obstacle_distance:.2f}m) < minimum safe distance ({self.safety_bounds.min_distance_to_obstacle}m)"
            if self.log_violations:
                logger.warning(violation)
            self.violations.append(violation)
            if not check_all:
                return False
        
        # Attitude limits
        if self.enable_attitude_limits and orientation_euler is not None:
            roll, pitch, _ = orientation_euler  # in degrees
            max_angle = self.safety_bounds.max_tilt_angle
            
            if abs(roll) > max_angle:
                violation = f"Excessive roll: {roll:.2f}° > {max_angle}°"
                if self.log_violations:
                    logger.warning(violation)
                self.violations.append(violation)
                if not check_all:
                    return False
            
            if abs(pitch) > max_angle:
                violation = f"Excessive pitch: {pitch:.2f}° > {max_angle}°"
                if self.log_violations:
                    logger.warning(violation)
                self.violations.append(violation)
                if not check_all:
                    return False
        
        self.is_safe = len(self.violations) == 0
        return self.is_safe
    
    def compute_safe_action(
        self, 
        original_velocity: np.ndarray, 
        position: np.ndarray,
        obstacle_distance: Optional[float] = None,
        obstacle_direction: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute safe velocity command based on original command and safety constraints.
        
        Args:
            original_velocity: Original velocity command [vx, vy, vz]
            position: Current position [x, y, z]
            obstacle_distance: Distance to nearest obstacle (m)
            obstacle_direction: Direction to nearest obstacle [x, y, z], normalized
            
        Returns:
            Safe velocity command
        """
        safe_velocity = original_velocity.copy()
        
        # First, clip to maximum velocity limits
        safe_velocity = self.safety_bounds.clip_velocity(safe_velocity)
        
        # Handle geofence - prevent moving closer to boundary
        if self.enable_geofence:
            x, y, z = position
            vx, vy, vz = safe_velocity
            
            # X boundaries
            if x <= self.safety_bounds.x_min and vx < 0:
                vx = 0.0
            elif x >= self.safety_bounds.x_max and vx > 0:
                vx = 0.0
                
            # Y boundaries
            if y <= self.safety_bounds.y_min and vy < 0:
                vy = 0.0
            elif y >= self.safety_bounds.y_max and vy > 0:
                vy = 0.0
                
            # Z boundaries
            if z <= self.safety_bounds.z_min and vz < 0:
                vz = 0.0
            elif z >= self.safety_bounds.z_max and vz > 0:
                vz = 0.0
                
            safe_velocity = np.array([vx, vy, vz])
        
        # Handle obstacle avoidance
        if (self.enable_collision_prevention and 
            obstacle_distance is not None and 
            obstacle_direction is not None):
            
            # Determine safety margin - as we get closer, reduce allowed velocity toward obstacle
            margin = self.safety_bounds.min_distance_to_obstacle
            
            if obstacle_distance < margin:
                # Scale is 0 at or below minimum distance, 1 at margin
                scale = min(1.0, max(0.0, (obstacle_distance - margin/2) / (margin/2)))
                
                # Project velocity onto obstacle direction
                v_toward_obstacle = np.dot(safe_velocity, obstacle_direction)
                
                # If moving toward obstacle, reduce velocity
                if v_toward_obstacle > 0:
                    # Compute component to remove
                    v_cancel = v_toward_obstacle * obstacle_direction * (1.0 - scale)
                    safe_velocity = safe_velocity - v_cancel
        
        # If emergency stop is active, override with zero velocity
        if self.emergency_stop_requested:
            safe_velocity = np.zeros(3)
            
        return safe_velocity
    
    def request_emergency_stop(self):
        """
        Request an emergency stop of the drone.
        
        This sets the emergency stop flag and calls any registered callback.
        """
        logger.warning("Emergency stop requested!")
        self.emergency_stop_requested = True
        
        if self.emergency_stop_callback:
            try:
                self.emergency_stop_callback()
            except Exception as e:
                logger.error(f"Error in emergency stop callback: {e}")
    
    def clear_emergency_stop(self):
        """Clear emergency stop flag."""
        logger.info("Emergency stop cleared")
        self.emergency_stop_requested = False
    
    def register_emergency_callback(self, callback: Callable):
        """
        Register function to call when emergency stop is requested.
        
        Args:
            callback: Function to call with no arguments
        """
        self.emergency_stop_callback = callback
    
    def start_monitoring(self, state_getter: Callable):
        """
        Start background thread for continuous safety monitoring.
        
        Args:
            state_getter: Function that returns dict with state variables
                         (position, velocity, orientation_euler, obstacle_distance)
        """
        if self._monitoring:
            logger.warning("Monitoring already active")
            return
            
        def monitor_loop():
            while self._monitoring:
                try:
                    current_time = time.time()
                    if current_time - self.last_check_time >= self.check_interval:
                        state = state_getter()
                        
                        # Check safety
                        is_safe = self.check_safety(
                            position=state.get('position'),
                            velocity=state.get('velocity'),
                            orientation_euler=state.get('orientation_euler'),
                            obstacle_distance=state.get('obstacle_distance'),
                        )
                        
                        # If unsafe, consider emergency stop
                        if not is_safe and not self.emergency_stop_requested:
                            # Criteria for emergency stop could be more specific
                            # For now, just check if violations exist
                            if any("Collision risk" in v for v in self.violations):
                                self.request_emergency_stop()
                        
                        self.last_check_time = current_time
                    
                    time.sleep(self.check_interval / 2)  # Check at twice the rate
                except Exception as e:
                    logger.error(f"Error in safety monitor: {e}")
                    time.sleep(self.check_interval)
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Safety monitoring started")
    
    def stop_monitoring(self):
        """Stop background safety monitoring."""
        if not self._monitoring:
            return
            
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Safety monitoring stopped")


class VelocityRamper:
    """
    Implements gradual velocity ramping for smoother control.
    
    This class limits the rate of change of velocity commands to prevent
    sudden movements that could destabilize the drone.
    """
    
    def __init__(
        self,
        max_acceleration: float = 1.0,  # m/s²
        max_jerk: float = 2.0,  # m/s³
        dt: float = 0.05,  # Control interval in seconds
    ):
        """
        Initialize velocity ramper.
        
        Args:
            max_acceleration: Maximum allowed acceleration (m/s²)
            max_jerk: Maximum allowed jerk (m/s³)
            dt: Control interval in seconds
        """
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.dt = dt
        
        # Current state
        self.current_velocity = np.zeros(3)
        self.current_acceleration = np.zeros(3)
        self.last_update = time.time()
    
    def reset(self, velocity: Optional[np.ndarray] = None):
        """
        Reset ramper state.
        
        Args:
            velocity: Initial velocity [vx, vy, vz]
        """
        self.current_velocity = np.zeros(3) if velocity is None else velocity
        self.current_acceleration = np.zeros(3)
        self.last_update = time.time()
    
    def update(self, target_velocity: np.ndarray) -> np.ndarray:
        """
        Update and get ramped velocity command.
        
        Args:
            target_velocity: Target velocity [vx, vy, vz]
            
        Returns:
            Ramped velocity command [vx, vy, vz]
        """
        # Calculate actual dt since last update
        current_time = time.time()
        dt = min(0.2, current_time - self.last_update)  # Cap at 200ms to avoid huge jumps
        self.last_update = current_time
        
        if dt <= 0:
            return self.current_velocity
        
        # For each axis, compute smooth velocity transition
        ramped_velocity = np.zeros(3)
        
        for i in range(3):
            # Calculate error
            error = target_velocity[i] - self.current_velocity[i]
            
            # Update acceleration (limited by max jerk)
            target_accel = error / dt
            accel_error = target_accel - self.current_acceleration[i]
            jerk = np.clip(accel_error / dt, -self.max_jerk, self.max_jerk)
            
            # Apply jerk to get new acceleration
            self.current_acceleration[i] = self.current_acceleration[i] + jerk * dt
            
            # Clip acceleration
            self.current_acceleration[i] = np.clip(
                self.current_acceleration[i], 
                -self.max_acceleration,
                self.max_acceleration
            )
            
            # Update velocity
            self.current_velocity[i] = self.current_velocity[i] + self.current_acceleration[i] * dt
        
        return self.current_velocity


class SafetyLayer:
    """
    Complete safety layer for drone control.
    
    Combines geofencing, collision prevention, emergency stop,
    and velocity ramping into a comprehensive safety system.
    """
    
    def __init__(
        self,
        safety_bounds: Optional[SafetyBounds] = None,
        enable_geofence: bool = True,
        enable_collision_prevention: bool = True,
        enable_attitude_limits: bool = True,
        enable_velocity_ramping: bool = True,
        max_acceleration: float = 0.5,  # m/s²
        log_violations: bool = True,
    ):
        """
        Initialize safety layer.
        
        Args:
            safety_bounds: Safety boundaries for drone operation
            enable_geofence: Enable geofencing based on position
            enable_collision_prevention: Enable collision prevention based on depth
            enable_attitude_limits: Enable attitude angle limits
            enable_velocity_ramping: Enable smooth velocity transitions
            max_acceleration: Maximum acceleration (m/s²) for velocity ramping
            log_violations: Log safety violations to logger
        """
        self.safety_bounds = safety_bounds or SafetyBounds()
        self.enable_geofence = enable_geofence
        self.enable_collision_prevention = enable_collision_prevention
        self.enable_attitude_limits = enable_attitude_limits
        self.enable_velocity_ramping = enable_velocity_ramping
        
        # Initialize components
        self.monitor = SafetyMonitor(
            safety_bounds=safety_bounds,
            enable_geofence=enable_geofence,
            enable_collision_prevention=enable_collision_prevention,
            enable_attitude_limits=enable_attitude_limits,
            log_violations=log_violations
        )
        
        self.ramper = VelocityRamper(
            max_acceleration=max_acceleration,
            dt=0.05  # Assume 20Hz control rate by default
        )
        
        # Statistics
        self.intervention_count = 0
        self.last_original_cmd = np.zeros(3)
        self.last_safe_cmd = np.zeros(3)
    
    def process_command(
        self,
        velocity_command: np.ndarray,
        position: np.ndarray,
        obstacle_distance: Optional[float] = None,
        obstacle_direction: Optional[np.ndarray] = None,
        orientation_euler: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process velocity command to ensure safety.
        
        Args:
            velocity_command: Original velocity command [vx, vy, vz]
            position: Current position [x, y, z]
            obstacle_distance: Distance to nearest obstacle (m)
            obstacle_direction: Direction to nearest obstacle [x, y, z]
            orientation_euler: Current orientation in Euler angles [roll, pitch, yaw] (degrees)
            
        Returns:
            Safe velocity command
        """
        # Keep track of original command
        self.last_original_cmd = velocity_command.copy()
        
        # Check if emergency stop is active
        if self.monitor.emergency_stop_requested:
            logger.warning("Emergency stop active, returning zero velocity")
            self.last_safe_cmd = np.zeros(3)
            return self.last_safe_cmd
        
        # Check safety and get safe velocity
        safe_velocity = self.monitor.compute_safe_action(
            original_velocity=velocity_command,
            position=position,
            obstacle_distance=obstacle_distance,
            obstacle_direction=obstacle_direction
        )
        
        # Record intervention if command was modified
        if not np.array_equal(velocity_command, safe_velocity):
            self.intervention_count += 1
        
        # Apply velocity ramping if enabled
        if self.enable_velocity_ramping:
            ramped_velocity = self.ramper.update(safe_velocity)
            self.last_safe_cmd = ramped_velocity
            return ramped_velocity
        else:
            self.last_safe_cmd = safe_velocity
            return safe_velocity
    
    def reset(self):
        """Reset safety layer state."""
        self.ramper.reset()
        self.monitor.clear_emergency_stop()
        self.intervention_count = 0
    
    def request_emergency_stop(self):
        """Request emergency stop."""
        self.monitor.request_emergency_stop()
    
    def get_status(self) -> Dict:
        """Get safety layer status."""
        return {
            'is_safe': self.monitor.is_safe,
            'violations': self.monitor.violations.copy(),
            'emergency_stop_active': self.monitor.emergency_stop_requested,
            'interventions': self.intervention_count,
            'last_original_cmd': self.last_original_cmd.tolist(),
            'last_safe_cmd': self.last_safe_cmd.tolist(),
        }
    
    def start_monitoring(self, state_getter: Callable):
        """
        Start background safety monitoring.
        
        Args:
            state_getter: Function that returns state variables
        """
        self.monitor.start_monitoring(state_getter)
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitor.stop_monitoring()


class BeginnerSafetyLayer(SafetyLayer):
    """
    Safety layer with tighter constraints for beginners.
    
    This version has reduced flight envelope and increased safety margins.
    """
    
    def __init__(self):
        """Initialize beginner safety layer with conservative settings."""
        # Define tighter safety bounds for beginners
        beginner_bounds = SafetyBounds(
            x_min=-5.0,
            x_max=5.0,
            y_min=-5.0, 
            y_max=5.0,
            z_min=0.3,  # Higher minimum altitude
            z_max=3.0,  # Lower maximum altitude
            
            vel_max_xy=0.5,  # Slower horizontal speed
            vel_max_z_up=0.3,  # Slower ascent
            vel_max_z_down=0.2,  # Slower descent
            
            max_tilt_angle=15.0,  # More limited tilt
            
            min_distance_to_obstacle=1.0  # Larger obstacle margin
        )
        
        # Initialize with conservative settings
        super().__init__(
            safety_bounds=beginner_bounds,
            enable_geofence=True,
            enable_collision_prevention=True,
            enable_attitude_limits=True,
            enable_velocity_ramping=True,
            max_acceleration=0.2,  # Very gentle acceleration
            log_violations=True
        ) 