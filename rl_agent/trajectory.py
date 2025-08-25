"""
Trajectory Implementation

This module implements the complete flight trajectory:
1. Takeoff to target altitude
2. 360-degree scan to detect hoops
3. Navigate toward detected hoop
4. Fly through hoop (first passage)
5. Return through same hoop (second passage)
6. Return to origin and land

This serves as the production trajectory implementation for the DeepFlyer system.
"""

import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging

# RL agent imports
from rl_agent.config import DeepFlyerConfig
from rl_agent.rewards.rewards import HoopRewardFunction, HoopRewardConfig

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory execution"""
    
    # Takeoff parameters
    takeoff_altitude: float = 1.5       # meters
    takeoff_speed: float = 0.5         # m/s
    
    # Scanning parameters
    scan_yaw_rate: float = 0.3         # rad/s (about 17 deg/s)
    scan_hover_time: float = 0.5       # seconds to hover at each detection
    scan_altitude_min: float = 0.8     # meters - lower bound for scan sweep
    scan_altitude_max: float = 3.0     # meters - upper bound for scan sweep
    scan_altitude_rate: float = 0.25   # m/s while sweeping when hoop not visible
    scan_selection_align_weight: float = 0.7  # weight for vertical alignment in scan score
    scan_selection_dist_weight: float = 0.3   # weight for distance in scan score
    
    # Navigation parameters
    approach_speed: float = 0.8        # m/s when approaching hoop
    alignment_threshold: float = 0.1   # normalized center offset for alignment
    passage_distance: float = 0.3      # normalized distance threshold for passage
    
    # Landing parameters
    landing_speed: float = 0.3         # m/s descent rate
    landing_threshold: float = 0.2     # altitude threshold for landing complete
    
    # Safety parameters
    max_flight_time: float = 300.0     # 5 minutes maximum flight time
    emergency_land_altitude: float = 0.1  # Emergency landing altitude
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'takeoff_altitude': self.takeoff_altitude,
            'takeoff_speed': self.takeoff_speed,
            'scan_yaw_rate': self.scan_yaw_rate,
            'scan_hover_time': self.scan_hover_time,
            'approach_speed': self.approach_speed,
            'alignment_threshold': self.alignment_threshold,
            'passage_distance': self.passage_distance,
            'landing_speed': self.landing_speed,
            'landing_threshold': self.landing_threshold,
            'max_flight_time': self.max_flight_time,
            'emergency_land_altitude': self.emergency_land_altitude
        }


class FlightPhase(Enum):
    TAKEOFF = "TAKEOFF"
    SCAN_360 = "SCAN_360"
    NAVIGATE_TO_HOOP = "NAVIGATE_TO_HOOP"
    THROUGH_HOOP_FIRST = "THROUGH_HOOP_FIRST"
    RETURN_TO_HOOP = "RETURN_TO_HOOP"
    THROUGH_HOOP_SECOND = "THROUGH_HOOP_SECOND"
    RETURN_TO_ORIGIN = "RETURN_TO_ORIGIN"
    LANDING = "LANDING"
    COMPLETED = "COMPLETED"


class PhaseController:
    """Controller for managing flight phases"""
    
    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.phase = FlightPhase.TAKEOFF
        self.phase_start_time = time.time()
        self.flight_start_time = time.time()
        
        # Phase-specific state
        self.scan_start_yaw = 0.0
        self.scan_complete_yaw = 0.0
        self.detected_hoops = []
        self.target_hoop = None
        self.hoop_passages = 0
        # Altitude selection during scan
        self.best_scan_score: float = -1e9
        self.best_scan_altitude: Optional[float] = None
        self.best_scan_yaw: Optional[float] = None
        self.desired_target_altitude: Optional[float] = None
        
        logger.info("Phase Controller initialized")
    
    def update_phase(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> FlightPhase:
        """
        Update current flight phase based on observation and drone state
        
        Args:
            observation: 8D observation [hoop_x, hoop_y, hoop_visible, hoop_distance, 
                                        drone_vx, drone_vy, drone_vz, yaw_rate]
            drone_state: Complete drone state from environment
            
        Returns:
            Current flight phase
        """
        current_time = time.time()
        phase_duration = current_time - self.phase_start_time
        flight_duration = current_time - self.flight_start_time
        
        # Safety check - emergency landing if flight too long
        if flight_duration > self.config.max_flight_time:
            logger.warning("Maximum flight time exceeded, emergency landing")
            self._transition_to_phase(FlightPhase.LANDING)
            return self.phase
        
        # Extract observation components
        hoop_x_norm = observation[0]
        hoop_y_norm = observation[1]
        hoop_visible = observation[2] > 0.5
        hoop_distance_norm = observation[3]
        
        # Get drone position and orientation
        position = drone_state.get('position', np.zeros(3))
        altitude = position[2]
        yaw = drone_state.get('yaw', 0.0)
        
        # Phase transition logic
        if self.phase == FlightPhase.TAKEOFF:
            if altitude >= self.config.takeoff_altitude - 0.2:
                self.scan_start_yaw = yaw
                self._transition_to_phase(FlightPhase.SCAN_360)
                logger.info(f"Takeoff complete at {altitude:.1f}m, starting 360° scan")
        
        elif self.phase == FlightPhase.SCAN_360:
            # While scanning, evaluate vertical alignment and distance when hoop is visible
            if hoop_visible:
                align_score = 1.0 - min(1.0, abs(hoop_y_norm))
                dist_score = 1.0 - min(1.0, hoop_distance_norm)
                score = (
                    self.config.scan_selection_align_weight * align_score
                    + self.config.scan_selection_dist_weight * dist_score
                )
                if score > self.best_scan_score:
                    self.best_scan_score = score
                    self.best_scan_altitude = altitude
                    self.best_scan_yaw = yaw
            # Check if we've completed a (nearly) full rotation
            yaw_progress = abs(yaw - self.scan_start_yaw)
            if yaw_progress >= 2 * np.pi * 0.95:  # 95% of full rotation
                # Select target altitude if a good alignment was seen
                if self.best_scan_altitude is not None:
                    self.desired_target_altitude = self.best_scan_altitude
                else:
                    # Default to current altitude if nothing detected
                    self.desired_target_altitude = altitude
                # Transition if any hoop was seen or continue scanning with new sweep
                if self.best_scan_altitude is not None or self.detected_hoops:
                    if self.detected_hoops and self.target_hoop is None:
                        self.target_hoop = self.detected_hoops[0]
                    self._transition_to_phase(FlightPhase.NAVIGATE_TO_HOOP)
                    logger.info(
                        f"Scan complete. Selected altitude {self.desired_target_altitude:.2f} m"
                    )
                else:
                    # Reset scan parameters for another sweep (possibly at different altitude)
                    self.scan_start_yaw = yaw
                    self.best_scan_score = -1e9
                    self.best_scan_altitude = None
                    self.best_scan_yaw = None
                    logger.warning("No hoops detected in sweep; repeating scan...")
        
        elif self.phase == FlightPhase.NAVIGATE_TO_HOOP:
            if hoop_visible and hoop_distance_norm < self.config.passage_distance + 0.2:
                self._transition_to_phase(FlightPhase.THROUGH_HOOP_FIRST)
                logger.info("Approaching hoop for first passage")
        
        elif self.phase == FlightPhase.THROUGH_HOOP_FIRST:
            if self._check_hoop_passage(observation):
                self.hoop_passages += 1
                self._transition_to_phase(FlightPhase.RETURN_TO_HOOP)
                logger.info("First hoop passage complete!")
        
        elif self.phase == FlightPhase.RETURN_TO_HOOP:
            if hoop_visible and hoop_distance_norm < self.config.passage_distance + 0.2:
                self._transition_to_phase(FlightPhase.THROUGH_HOOP_SECOND)
                logger.info("Approaching hoop for return passage")
        
        elif self.phase == FlightPhase.THROUGH_HOOP_SECOND:
            if self._check_hoop_passage(observation):
                self.hoop_passages += 1
                self._transition_to_phase(FlightPhase.RETURN_TO_ORIGIN)
                logger.info("Second hoop passage complete! Returning to origin")
        
        elif self.phase == FlightPhase.RETURN_TO_ORIGIN:
            # Check distance to spawn point (Point A)
            spawn_position = drone_state.get('spawn_position', np.zeros(3))
            distance_to_origin = np.linalg.norm(position - spawn_position)
            if distance_to_origin < 1.0:  # Within 1 meter of origin
                self._transition_to_phase(FlightPhase.LANDING)
                logger.info("Arrived at origin, starting landing")
        
        elif self.phase == FlightPhase.LANDING:
            if altitude < self.config.landing_threshold:
                self._transition_to_phase(FlightPhase.COMPLETED)
                logger.info("Landing complete! Trajectory finished")
        
        return self.phase
    
    def _transition_to_phase(self, new_phase: FlightPhase) -> None:
        """Transition to a new flight phase"""
        self.phase = new_phase
        self.phase_start_time = time.time()
        logger.info(f"Phase transition: {new_phase.value}")
    
    def _check_hoop_passage(self, observation: np.ndarray) -> bool:
        """Check if drone has successfully passed through the hoop"""
        hoop_x_norm = observation[0]
        hoop_y_norm = observation[1]
        hoop_distance_norm = observation[3]
        
        # Check alignment (centered on hoop)
        aligned = (abs(hoop_x_norm) < self.config.alignment_threshold and
                  abs(hoop_y_norm) < self.config.alignment_threshold)
        
        # Check distance (very close to hoop)
        very_close = hoop_distance_norm < self.config.passage_distance
        
        return aligned and very_close
    
    def add_detected_hoop(self, position: np.ndarray, confidence: float = 0.9) -> None:
        """Add a detected hoop during scanning phase"""
        # Avoid duplicates
        for existing_hoop in self.detected_hoops:
            if np.linalg.norm(existing_hoop['position'] - position) < 1.0:
                return
        
        hoop_info = {
            'position': position.copy(),
            'confidence': confidence,
            'detection_time': time.time()
        }
        
        self.detected_hoops.append(hoop_info)
        logger.info(f"Detected hoop #{len(self.detected_hoops)} at {position}")
    
    def get_phase_info(self) -> Dict[str, Any]:
        """Get information about current phase"""
        current_time = time.time()
        return {
            'current_phase': self.phase.value,
            'phase_duration': current_time - self.phase_start_time,
            'flight_duration': current_time - self.flight_start_time,
            'detected_hoops': len(self.detected_hoops),
            'hoop_passages': self.hoop_passages,
            'target_hoop': self.target_hoop,
            'desired_target_altitude': self.desired_target_altitude,
            'best_scan_altitude': self.best_scan_altitude,
            'best_scan_score': self.best_scan_score
        }


class ActionGenerator:
    """Generates actions for each phase of the trajectory"""
    
    def __init__(self, config: TrajectoryConfig):
        self.config = config
        # Internal state for altitude sweeping during SCAN_360
        self._scan_alt_direction: float = 1.0  # 1 for up, -1 for down
    
    def generate_action(self, phase: FlightPhase, observation: np.ndarray, 
                       drone_state: Dict[str, Any]) -> np.ndarray:
        """
        Generate action for current phase
        
        Args:
            phase: Current flight phase
            observation: 8D observation vector
            drone_state: Complete drone state
            
        Returns:
            4D action vector [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
        """
        if phase == FlightPhase.TAKEOFF:
            return self._takeoff_action(observation, drone_state)
        elif phase == FlightPhase.SCAN_360:
            return self._scan_action(observation, drone_state)
        elif phase == FlightPhase.NAVIGATE_TO_HOOP:
            return self._navigate_action(observation, drone_state)
        elif phase == FlightPhase.THROUGH_HOOP_FIRST:
            return self._passage_action(observation, drone_state, forward=True)
        elif phase == FlightPhase.RETURN_TO_HOOP:
            return self._return_action(observation, drone_state)
        elif phase == FlightPhase.THROUGH_HOOP_SECOND:
            return self._passage_action(observation, drone_state, forward=False)
        elif phase == FlightPhase.RETURN_TO_ORIGIN:
            return self._return_to_origin_action(observation, drone_state)
        elif phase == FlightPhase.LANDING:
            return self._landing_action(observation, drone_state)
        else:
            return np.zeros(4)  # COMPLETED or unknown phase
    
    def _takeoff_action(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> np.ndarray:
        """Generate takeoff action - ascend vertically"""
        position = drone_state.get('position', np.zeros(3))
        altitude = position[2]
        
        if altitude < self.config.takeoff_altitude:
            vz_cmd = 0.5  # Ascend
        else:
            vz_cmd = 0.0  # Hover
        
        return np.array([0.0, 0.0, vz_cmd, 0.0])
    
    def _scan_action(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> np.ndarray:
        """Generate scanning action - rotate in place and sweep altitude to find hoop center height"""
        hoop_y_norm = observation[1]
        hoop_visible = observation[2] > 0.5
        altitude = float(drone_state.get('position', np.zeros(3))[2])
        
        # Yaw behavior: rotate, slower if hoop visible for stabilization
        yaw_rate_cmd = self.config.scan_yaw_rate * (0.33 if hoop_visible else 1.0)
        
        # Altitude behavior:
        # - If hoop visible, directly converge vertically to center (y_norm -> 0)
        # - If not visible, perform a gentle altitude sweep between min and max
        if hoop_visible:
            # Map y_center_norm to vertical velocity (negative sign to drive toward center)
            vz_cmd = float(np.clip(-0.6 * hoop_y_norm, -self.config.scan_altitude_rate, self.config.scan_altitude_rate))
        else:
            # Sweep altitude up and down within bounds
            if altitude >= self.config.scan_altitude_max:
                self._scan_alt_direction = -1.0
            elif altitude <= self.config.scan_altitude_min:
                self._scan_alt_direction = 1.0
            vz_cmd = self._scan_alt_direction * self.config.scan_altitude_rate
        
        return np.array([0.0, 0.0, vz_cmd, yaw_rate_cmd])
    
    def _navigate_action(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> np.ndarray:
        """Generate navigation action - approach detected hoop"""
        hoop_x_norm = observation[0]
        hoop_y_norm = observation[1]
        hoop_visible = observation[2] > 0.5
        hoop_distance_norm = observation[3]
        
        if not hoop_visible:
            # Search for hoop
            return np.array([0.0, 0.0, 0.0, 0.2])
        
        # Calculate approach commands
        vx_cmd = 0.3 * (1.0 - hoop_distance_norm)  # Approach hoop
        vy_cmd = -0.5 * hoop_x_norm                # Center horizontally
        vz_cmd = -0.3 * hoop_y_norm                # Center vertically
        yaw_rate_cmd = -0.2 * hoop_x_norm          # Align yaw
        
        return np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd])
    
    def _passage_action(self, observation: np.ndarray, drone_state: Dict[str, Any], 
                       forward: bool = True) -> np.ndarray:
        """Generate hoop passage action"""
        hoop_x_norm = observation[0]
        hoop_y_norm = observation[1]
        hoop_visible = observation[2] > 0.5
        
        if not hoop_visible:
            # Lost hoop, search
            return np.array([0.0, 0.0, 0.0, 0.1])
        
        # Precise alignment and forward motion
        direction = 1.0 if forward else -1.0
        
        vx_cmd = direction * 0.4      # Move through hoop
        vy_cmd = -0.8 * hoop_x_norm   # Precise horizontal alignment
        vz_cmd = -0.6 * hoop_y_norm   # Precise vertical alignment
        yaw_rate_cmd = -0.1 * hoop_x_norm  # Fine yaw adjustment
        
        return np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd])
    
    def _return_action(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> np.ndarray:
        """Generate return action - turn around and approach hoop again"""
        hoop_visible = observation[2] > 0.5
        
        if hoop_visible:
            # Hoop is visible, approach from other side
            return self._navigate_action(observation, drone_state)
        else:
            # Turn around to find hoop again
            return np.array([0.0, 0.0, 0.0, -0.4])
    
    def _return_to_origin_action(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> np.ndarray:
        """Generate return to origin action"""
        position = drone_state.get('position', np.zeros(3))
        spawn_position = drone_state.get('spawn_position', np.zeros(3))
        
        # Vector to origin
        to_origin = spawn_position - position
        distance_to_origin = np.linalg.norm(to_origin[:2])  # Horizontal distance
        
        if distance_to_origin > 0.1:
            # Normalize and scale
            direction = to_origin[:2] / distance_to_origin
            vx_cmd = direction[0] * 0.5
            vy_cmd = direction[1] * 0.5
        else:
            vx_cmd = 0.0
            vy_cmd = 0.0
        
        # Maintain altitude
        vz_cmd = 0.0
        yaw_rate_cmd = 0.0
        
        return np.array([vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd])
    
    def _landing_action(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> np.ndarray:
        """Generate landing action - descend at origin"""
        position = drone_state.get('position', np.zeros(3))
        altitude = position[2]
        
        if altitude > self.config.landing_threshold:
            vz_cmd = -0.3  # Descend
        else:
            vz_cmd = 0.0   # Landed
        
        return np.array([0.0, 0.0, vz_cmd, 0.0]) 