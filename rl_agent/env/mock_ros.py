"""
Mock ROS Environment for Development without ROS
Simulates ROS functionality for testing and development
"""

import numpy as np
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging

from .ros_env import RosEnv, RosEnvState
from ..config import DeepFlyerConfig, get_course_layout

logger = logging.getLogger(__name__)


@dataclass
class MockDronePhysics:
    """Simple drone physics simulation for mock environment"""
    position: np.ndarray = None
    velocity: np.ndarray = None
    acceleration: np.ndarray = None
    orientation: np.ndarray = None  # quaternion
    angular_velocity: np.ndarray = None
    
    # Physics parameters
    mass: float = 1.5  # kg
    drag_coefficient: float = 0.1
    max_thrust: float = 20.0  # N
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 0.8])
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.acceleration is None:
            self.acceleration = np.zeros(3)
        if self.orientation is None:
            self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3)


class MockROSMessages:
    """Mock ROS message types for testing"""
    
    @dataclass
    class Vector3:
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
    
    @dataclass
    class Quaternion:
        w: float = 1.0
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
    
    @dataclass
    class Point:
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
    
    @dataclass
    class Header:
        stamp: float = 0.0
        frame_id: str = "map"
    
    @dataclass
    class Twist:
        linear: 'MockROSMessages.Vector3' = None
        angular: 'MockROSMessages.Vector3' = None
        
        def __post_init__(self):
            if self.linear is None:
                self.linear = MockROSMessages.Vector3()
            if self.angular is None:
                self.angular = MockROSMessages.Vector3()


class MockRosEnv(RosEnv):
    """Mock ROS environment for development and testing without ROS"""
    
    def __init__(self,
                 spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.8),
                 physics_dt: float = 0.02,  # 50Hz physics
                 enable_physics: bool = True,
                 add_noise: bool = True,
                 simulate_latency: bool = True,
                 **kwargs):
        """
        Initialize mock ROS environment
        
        Args:
            spawn_position: Initial drone spawn position
            physics_dt: Physics simulation timestep
            enable_physics: Enable realistic physics simulation
            add_noise: Add realistic sensor noise
            simulate_latency: Simulate communication latency
            **kwargs: Additional arguments for RosEnv
        """
        self.spawn_position = np.array(spawn_position)
        self.physics_dt = physics_dt
        self.enable_physics = enable_physics
        self.add_noise = add_noise
        self.simulate_latency = simulate_latency
        
        # Initialize mock physics
        self.physics = MockDronePhysics()
        self.physics.position = self.spawn_position.copy()
        
        # Generate course layout
        self.course_hoops = get_course_layout(spawn_position)
        self.current_target_hoop = 0
        self.hoops_completed = 0
        self.lap_number = 1
        
        # Control commands
        self.last_velocity_command = np.zeros(3)
        self.command_timestamp = 0.0
        
        # Physics simulation thread
        self.physics_thread = None
        self.physics_running = False
        
        # Mock sensors
        self.mock_sensors = {
            'battery_level': 1.0,
            'armed': True,
            'connected': True,
            'gps_fix': True
        }
        
        # Initialize base environment (will call our _init_mock_interface)
        super().__init__(use_ros=False, **kwargs)
        
        logger.info("MockRosEnv initialized with physics simulation")
    
    def _init_mock_interface(self) -> None:
        """Initialize enhanced mock interface with physics"""
        super()._init_mock_interface()
        
        # Start physics simulation
        if self.enable_physics:
            self.physics_running = True
            self.physics_thread = threading.Thread(target=self._physics_simulation_thread, daemon=True)
            self.physics_thread.start()
    
    def _physics_simulation_thread(self) -> None:
        """Physics simulation thread"""
        while self.physics_running:
            try:
                self._update_physics()
                time.sleep(self.physics_dt)
            except Exception as e:
                logger.error(f"Physics simulation error: {e}")
                break
    
    def _update_physics(self) -> None:
        """Update drone physics simulation"""
        dt = self.physics_dt
        
        # Apply velocity command with some lag
        command_age = time.time() - self.command_timestamp
        if command_age < 0.5:  # Command is recent
            target_velocity = self.last_velocity_command.copy()
        else:
            target_velocity = np.zeros(3)  # Command timeout
        
        # Simple first-order velocity tracking
        velocity_error = target_velocity - self.physics.velocity
        self.physics.acceleration = velocity_error * 5.0  # Proportional control
        
        # Add gravity
        self.physics.acceleration[2] -= 9.81  # Gravity in Z
        
        # Add drag
        drag_force = -self.physics.drag_coefficient * self.physics.velocity * np.linalg.norm(self.physics.velocity)
        self.physics.acceleration += drag_force / self.physics.mass
        
        # Integrate velocity
        self.physics.velocity += self.physics.acceleration * dt
        
        # Integrate position
        self.physics.position += self.physics.velocity * dt
        
        # Ground collision
        if self.physics.position[2] < 0.0:
            self.physics.position[2] = 0.0
            self.physics.velocity[2] = max(0.0, self.physics.velocity[2])
        
        # Add noise if enabled
        if self.add_noise:
            position_noise = np.random.normal(0, 0.01, 3)  # 1cm noise
            velocity_noise = np.random.normal(0, 0.05, 3)  # 5cm/s noise
            
            self.physics.position += position_noise
            self.physics.velocity += velocity_noise
        
        # Update state
        self.state.update('position', self.physics.position.copy())
        self.state.update('velocity', self.physics.velocity.copy())
        self.state.update('orientation', self.physics.orientation.copy())
        
        # Update mock sensors
        self._update_mock_sensors()
        
        # Check hoop passage
        self._check_hoop_passage()
    
    def _update_mock_sensors(self) -> None:
        """Update mock sensor readings"""
        # Simulate battery drain
        current_battery = self.mock_sensors['battery_level']
        self.mock_sensors['battery_level'] = max(0.0, current_battery - 0.0001)  # Slow drain
        
        # Update state with sensor data
        self.state.update('battery_level', self.mock_sensors['battery_level'])
        self.state.update('armed', self.mock_sensors['armed'])
        self.state.update('connected', self.mock_sensors['connected'])
    
    def _check_hoop_passage(self) -> None:
        """Check if drone has passed through target hoop"""
        if self.current_target_hoop >= len(self.course_hoops):
            return
        
        target_hoop = self.course_hoops[self.current_target_hoop]
        hoop_pos = np.array(target_hoop['position'])
        hoop_radius = target_hoop['diameter'] / 2.0
        
        # Check if drone is close to hoop center
        distance_to_hoop = np.linalg.norm(self.physics.position - hoop_pos)
        
        if distance_to_hoop < hoop_radius:
            logger.info(f"Passed through hoop {self.current_target_hoop + 1}")
            self.hoops_completed += 1
            self.current_target_hoop += 1
            
            # Check lap completion
            if self.current_target_hoop >= len(self.course_hoops):
                self.lap_number += 1
                self.current_target_hoop = 0
                logger.info(f"Lap {self.lap_number - 1} completed")
    
    def _process_action(self, action: np.ndarray) -> None:
        """Process action and update velocity command"""
        # Convert action to velocity command
        lateral_cmd = np.clip(action[0], -1.0, 1.0)
        vertical_cmd = np.clip(action[1], -1.0, 1.0)
        speed_cmd = np.clip(action[2], -1.0, 1.0)
        
        # Scale to velocity
        action_config = self.config.ACTION_CONFIG
        
        lateral_velocity = lateral_cmd * action_config['components']['lateral_cmd']['max_speed']
        vertical_velocity = vertical_cmd * action_config['components']['vertical_cmd']['max_speed']
        forward_velocity = action_config['components']['speed_cmd']['base_speed'] * (1.0 + 0.5 * speed_cmd)
        
        # Store velocity command
        self.last_velocity_command = np.array([forward_velocity, lateral_velocity, vertical_velocity])
        self.command_timestamp = time.time()
        
        # Simulate latency if enabled
        if self.simulate_latency:
            time.sleep(0.01)  # 10ms latency
    
    def _calculate_reward(self, observation: np.ndarray, action: np.ndarray, info: Dict[str, Any]) -> float:
        """Calculate reward for mock environment"""
        # Simple distance-based reward
        if self.current_target_hoop >= len(self.course_hoops):
            return 0.0
        
        target_hoop = self.course_hoops[self.current_target_hoop]
        target_pos = np.array(target_hoop['position'])
        distance = np.linalg.norm(self.physics.position - target_pos)
        
        # Distance reward (negative distance)
        distance_reward = -distance
        
        # Hoop passage bonus
        passage_bonus = 0.0
        if hasattr(self, '_last_target_hoop'):
            if self.current_target_hoop > self._last_target_hoop:
                passage_bonus = 50.0
        self._last_target_hoop = self.current_target_hoop
        
        # Speed penalty for being too slow
        speed = np.linalg.norm(self.physics.velocity)
        speed_penalty = -0.1 if speed < 0.2 else 0.0
        
        return distance_reward + passage_bonus + speed_penalty
    
    def _check_episode_done(self, observation: np.ndarray, info: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check episode termination for mock environment"""
        # Course completion
        total_hoops_needed = len(self.course_hoops) * self.config.HOOP_CONFIG['num_laps']
        if self.hoops_completed >= total_hoops_needed:
            return True, False
        
        # Crash (ground impact with high speed)
        if self.physics.position[2] <= 0.1 and np.linalg.norm(self.physics.velocity) > 1.0:
            return True, False
        
        # Out of bounds
        bounds = self.config.COURSE_DIMENSIONS
        pos = self.physics.position
        if (abs(pos[0]) > bounds['length']/2 or 
            abs(pos[1]) > bounds['width']/2 or 
            pos[2] > bounds['height']):
            return True, False
        
        # Max steps
        max_steps = self.config.TRAINING_CONFIG['max_steps_per_episode']
        if info.get('episode_step', 0) >= max_steps:
            return False, True
        
        return False, False
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset mock environment"""
        # Reset physics
        self.physics.position = self.spawn_position.copy()
        self.physics.velocity = np.zeros(3)
        self.physics.acceleration = np.zeros(3)
        
        # Reset course progress
        self.current_target_hoop = 0
        self.hoops_completed = 0
        self.lap_number = 1
        
        # Reset sensors
        self.mock_sensors['battery_level'] = 1.0
        self.mock_sensors['armed'] = True
        
        # Reset velocity command
        self.last_velocity_command = np.zeros(3)
        self.command_timestamp = time.time()
        
        return super().reset(seed=seed, options=options)
    
    def close(self) -> None:
        """Clean up mock environment"""
        self.physics_running = False
        if self.physics_thread and self.physics_thread.is_alive():
            self.physics_thread.join(timeout=1.0)
        
        super().close()
    
    def get_physics_state(self) -> Dict[str, Any]:
        """Get current physics state for debugging"""
        return {
            'position': self.physics.position.copy(),
            'velocity': self.physics.velocity.copy(),
            'acceleration': self.physics.acceleration.copy(),
            'target_hoop': self.current_target_hoop,
            'hoops_completed': self.hoops_completed,
            'lap_number': self.lap_number
        }


# Convenience function
def create_mock_env(**kwargs) -> MockRosEnv:
    """Create mock ROS environment"""
    return MockRosEnv(**kwargs)


__all__ = ['MockRosEnv', 'MockDronePhysics', 'MockROSMessages', 'create_mock_env'] 