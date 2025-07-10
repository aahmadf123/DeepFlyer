"""
MVP Trajectory Implementation for DeepFlyer

This module demonstrates the complete MVP flight trajectory:
1. Takeoff from Point A
2. Perform 360° yaw rotation to scan and detect hoops using YOLO11m + depth camera
3. Navigate toward and fly through the single detected hoop
4. Turn around after flying through to return through the same hoop
5. Land at the original Point A

This serves as both a working implementation and educational reference.
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from .env.mvp_trajectory import MVPTrajectoryEnv, MVPFlightPhase
from .algorithms.p3o import P3O, P3OConfig, MVPTrainingConfig
from .rewards import MVPRewardFunction, MVPRewardConfig

logger = logging.getLogger(__name__)


@dataclass
class MVPTrajectoryConfig:
    """Configuration for MVP trajectory execution"""
    
    # Takeoff parameters
    takeoff_altitude: float = 1.5       # meters
    takeoff_speed: float = 0.5         # m/s
    
    # Scanning parameters
    scan_yaw_rate: float = 0.3         # rad/s (about 17 deg/s)
    scan_hover_time: float = 0.5       # seconds to hover at each detection
    
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


class MVPPhaseController:
    """Controller for managing MVP flight phases"""
    
    def __init__(self, config: MVPTrajectoryConfig):
        self.config = config
        self.phase = MVPFlightPhase.TAKEOFF
        self.phase_start_time = time.time()
        self.flight_start_time = time.time()
        
        # Phase-specific state
        self.scan_start_yaw = 0.0
        self.scan_complete_yaw = 0.0
        self.detected_hoops = []
        self.target_hoop = None
        self.hoop_passages = 0
        
        logger.info("MVP Phase Controller initialized")
    
    def update_phase(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> MVPFlightPhase:
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
            self._transition_to_phase(MVPFlightPhase.LANDING)
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
        if self.phase == MVPFlightPhase.TAKEOFF:
            if altitude >= self.config.takeoff_altitude - 0.2:
                self.scan_start_yaw = yaw
                self._transition_to_phase(MVPFlightPhase.SCAN_360)
                logger.info(f"Takeoff complete at {altitude:.1f}m, starting 360° scan")
        
        elif self.phase == MVPFlightPhase.SCAN_360:
            # Check if we've completed a full rotation
            yaw_progress = abs(yaw - self.scan_start_yaw)
            if yaw_progress >= 2 * np.pi * 0.95:  # 95% of full rotation
                if self.detected_hoops:
                    self.target_hoop = self.detected_hoops[0]  # Choose first detected hoop
                    self._transition_to_phase(MVPFlightPhase.NAVIGATE_TO_HOOP)
                    logger.info(f"Scan complete, found {len(self.detected_hoops)} hoop(s)")
                else:
                    # Continue scanning if no hoops found
                    logger.warning("No hoops detected, continuing scan...")
        
        elif self.phase == MVPFlightPhase.NAVIGATE_TO_HOOP:
            if hoop_visible and hoop_distance_norm < self.config.passage_distance + 0.2:
                self._transition_to_phase(MVPFlightPhase.THROUGH_HOOP_FIRST)
                logger.info("Approaching hoop for first passage")
        
        elif self.phase == MVPFlightPhase.THROUGH_HOOP_FIRST:
            if self._check_hoop_passage(observation):
                self.hoop_passages += 1
                self._transition_to_phase(MVPFlightPhase.RETURN_TO_HOOP)
                logger.info("First hoop passage complete!")
        
        elif self.phase == MVPFlightPhase.RETURN_TO_HOOP:
            if hoop_visible and hoop_distance_norm < self.config.passage_distance + 0.2:
                self._transition_to_phase(MVPFlightPhase.THROUGH_HOOP_SECOND)
                logger.info("Approaching hoop for return passage")
        
        elif self.phase == MVPFlightPhase.THROUGH_HOOP_SECOND:
            if self._check_hoop_passage(observation):
                self.hoop_passages += 1
                self._transition_to_phase(MVPFlightPhase.RETURN_TO_ORIGIN)
                logger.info("Second hoop passage complete! Returning to origin")
        
        elif self.phase == MVPFlightPhase.RETURN_TO_ORIGIN:
            # Check distance to spawn point (Point A)
            spawn_position = drone_state.get('spawn_position', np.zeros(3))
            distance_to_origin = np.linalg.norm(position - spawn_position)
            if distance_to_origin < 1.0:  # Within 1 meter of origin
                self._transition_to_phase(MVPFlightPhase.LANDING)
                logger.info("Arrived at origin, starting landing")
        
        elif self.phase == MVPFlightPhase.LANDING:
            if altitude < self.config.landing_threshold:
                self._transition_to_phase(MVPFlightPhase.COMPLETED)
                logger.info("Landing complete! MVP trajectory finished")
        
        return self.phase
    
    def _transition_to_phase(self, new_phase: MVPFlightPhase) -> None:
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
            'target_hoop': self.target_hoop
        }


class MVPActionGenerator:
    """Generates actions for each phase of the MVP trajectory"""
    
    def __init__(self, config: MVPTrajectoryConfig):
        self.config = config
    
    def generate_action(self, phase: MVPFlightPhase, observation: np.ndarray, 
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
        if phase == MVPFlightPhase.TAKEOFF:
            return self._takeoff_action(observation, drone_state)
        elif phase == MVPFlightPhase.SCAN_360:
            return self._scan_action(observation, drone_state)
        elif phase == MVPFlightPhase.NAVIGATE_TO_HOOP:
            return self._navigate_action(observation, drone_state)
        elif phase == MVPFlightPhase.THROUGH_HOOP_FIRST:
            return self._passage_action(observation, drone_state, forward=True)
        elif phase == MVPFlightPhase.RETURN_TO_HOOP:
            return self._return_action(observation, drone_state)
        elif phase == MVPFlightPhase.THROUGH_HOOP_SECOND:
            return self._passage_action(observation, drone_state, forward=False)
        elif phase == MVPFlightPhase.RETURN_TO_ORIGIN:
            return self._return_to_origin_action(observation, drone_state)
        elif phase == MVPFlightPhase.LANDING:
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
        """Generate scanning action - rotate in place"""
        hoop_visible = observation[2] > 0.5
        
        if hoop_visible:
            # Slow down rotation when hoop is visible
            yaw_rate_cmd = 0.1
        else:
            # Normal scan rate
            yaw_rate_cmd = 0.3
        
        return np.array([0.0, 0.0, 0.0, yaw_rate_cmd])
    
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


class MVPTrajectoryDemo:
    """Complete MVP trajectory demonstration"""
    
    def __init__(self, reward_config: Optional[Dict[str, float]] = None,
                 trajectory_config: Optional[MVPTrajectoryConfig] = None):
        """
        Initialize MVP trajectory demo
        
        Args:
            reward_config: Student-tunable reward configuration
            trajectory_config: Trajectory execution configuration
        """
        self.trajectory_config = trajectory_config or MVPTrajectoryConfig()
        
        # Initialize environment
        self.env = MVPTrajectoryEnv(reward_config=reward_config)
        
        # Initialize controllers
        self.phase_controller = MVPPhaseController(self.trajectory_config)
        self.action_generator = MVPActionGenerator(self.trajectory_config)
        
        # Trajectory state
        self.episode_count = 0
        self.total_reward = 0.0
        self.best_reward = float('-inf')
        
        logger.info("MVP Trajectory Demo initialized")
    
    def run_episode(self, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Run a complete MVP trajectory episode
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Episode statistics
        """
        observation, info = self.env.reset()
        self.phase_controller = MVPPhaseController(self.trajectory_config)
        
        episode_reward = 0.0
        episode_steps = 0
        phase_history = []
        
        for step in range(max_steps):
            # Get drone state
            drone_state = self.env.state.get_all()
            drone_state['spawn_position'] = self.env.spawn_position
            
            # Update flight phase
            current_phase = self.phase_controller.update_phase(observation, drone_state)
            
            # Generate action based on current phase
            action = self.action_generator.generate_action(current_phase, observation, drone_state)
            
            # Execute action
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Record phase
            phase_info = self.phase_controller.get_phase_info()
            phase_history.append({
                'step': step,
                'phase': current_phase.value,
                'reward': reward,
                'observation': observation.copy(),
                'action': action.copy(),
                **phase_info
            })
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # Episode statistics
        episode_stats = {
            'episode': self.episode_count,
            'total_reward': episode_reward,
            'steps': episode_steps,
            'completed': current_phase == MVPFlightPhase.COMPLETED,
            'final_phase': current_phase.value,
            'hoop_passages': self.phase_controller.hoop_passages,
            'detected_hoops': len(self.phase_controller.detected_hoops),
            'phase_history': phase_history
        }
        
        self.episode_count += 1
        self.total_reward += episode_reward
        self.best_reward = max(self.best_reward, episode_reward)
        
        logger.info(f"Episode {self.episode_count}: Reward={episode_reward:.1f}, "
                   f"Phase={current_phase.value}, Steps={episode_steps}")
        
        return episode_stats
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of trajectory performance"""
        return {
            'episodes_run': self.episode_count,
            'total_reward': self.total_reward,
            'average_reward': self.total_reward / max(self.episode_count, 1),
            'best_reward': self.best_reward,
            'trajectory_config': self.trajectory_config.to_dict()
        }


# Convenience functions
def create_mvp_demo(reward_config: Optional[Dict[str, float]] = None) -> MVPTrajectoryDemo:
    """Create MVP trajectory demonstration"""
    return MVPTrajectoryDemo(reward_config=reward_config)


def run_mvp_demo_episode() -> Dict[str, Any]:
    """Run a single MVP trajectory episode with default settings"""
    demo = create_mvp_demo()
    return demo.run_episode()


def demonstrate_mvp_phases():
    """Demonstrate each phase of the MVP trajectory"""
    phases = [
        "TAKEOFF: Ascend to 1.5m altitude",
        "SCAN_360: Rotate 360° to detect hoops using YOLO11+depth",
        "NAVIGATE_TO_HOOP: Approach the detected hoop",
        "THROUGH_HOOP_FIRST: Fly through hoop (first passage)",
        "RETURN_TO_HOOP: Turn around and approach hoop again",
        "THROUGH_HOOP_SECOND: Fly through hoop (return passage)",
        "RETURN_TO_ORIGIN: Navigate back to Point A",
        "LANDING: Descend and land at Point A",
        "COMPLETED: Mission accomplished!"
    ]
    
    print("MVP Flight Trajectory Phases:")
    print("=" * 50)
    for i, phase in enumerate(phases, 1):
        print(f"{i}. {phase}")
    print("=" * 50)


if __name__ == "__main__":
    # Demonstrate MVP trajectory phases
    demonstrate_mvp_phases()
    
    # Run a demo episode
    print("\nRunning MVP trajectory demo...")
    episode_stats = run_mvp_demo_episode()
    print(f"Demo complete: {episode_stats['final_phase']} in {episode_stats['steps']} steps") 