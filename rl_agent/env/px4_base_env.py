"""
PX4 Environment Classes for DeepFlyer
Primary: PX4-ROS-COM communication (recommended)
Fallback: MAVROS communication (legacy support)
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Tuple, Union
from gymnasium import spaces

try:
    import rclpy
    from rclpy.node import Node
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from .ros_env import RosEnv, RosEnvState
from .px4_comm import PX4Interface, MAVROSBridge, MessageConverter
from .safety_layer import SafetyLayer, BeginnerSafetyLayer
from .zed_integration import create_zed_interface, ZEDInterface
from ..config import DeepFlyerConfig, get_course_layout

logger = logging.getLogger(__name__)


class PX4BaseEnv(RosEnv):
    """Base PX4 environment with PX4-ROS-COM communication (recommended)"""
    
    def __init__(self, 
                 use_px4_com: bool = True,  # Default to PX4-ROS-COM
                 use_zed: bool = False,
                 spawn_position: Tuple[float, float, float] = (0.0, 0.0, 0.8),
                 **kwargs):
        """
        Initialize PX4 base environment
        
        Args:
            use_px4_com: Use PX4-ROS-COM (True, recommended) vs MAVROS (False, legacy)
            use_zed: Enable ZED Mini camera integration
            spawn_position: Initial drone spawn position
            **kwargs: Additional arguments for RosEnv
        """
        self.use_px4_com = use_px4_com
        self.use_zed = use_zed
        self.spawn_position = np.array(spawn_position)
        
        # Log communication method
        comm_method = "PX4-ROS-COM (recommended)" if use_px4_com else "MAVROS (legacy)"
        logger.info(f"Initializing DeepFlyer with {comm_method}")
        
        # Generate course layout
        self.course_hoops = get_course_layout(spawn_position)
        self.current_target_hoop = 0
        self.lap_number = 1
        self.hoops_completed = 0
        
        # Initialize safety layer (will be set by subclasses)
        self.safety_layer = None
        
        # Initialize ZED interface if requested
        self.zed_interface: Optional[ZEDInterface] = None
        if use_zed:
            try:
                self.zed_interface = create_zed_interface("auto")
                logger.info("ZED interface initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ZED interface: {e}")
                self.zed_interface = None
        
        super().__init__(**kwargs)
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces"""
        obs_config = self.config.OBSERVATION_CONFIG
        action_config = self.config.ACTION_CONFIG
        
        # 12-dimensional observation space
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_config['dimension'],),
            dtype=np.float32
        )
        
        # 3-dimensional action space [lateral, vertical, speed]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_config['dimension'],),
            dtype=np.float32
        )
    
    def _setup_ros_interface(self) -> None:
        """Setup ROS interface with PX4-ROS-COM (primary) or MAVROS (fallback)"""
        if self.use_px4_com:
            self.px4_interface = PX4Interface(self.node)
            logger.info("Using PX4-ROS-COM interface (recommended)")
        else:
            self.mavros_interface = MAVROSBridge(self.node)
            logger.warning("Using MAVROS interface (legacy - consider switching to PX4-ROS-COM)")
        
        # Start ZED interface if available
        if self.zed_interface is not None:
            if not self.zed_interface.start():
                logger.warning("Failed to start ZED interface")
                self.zed_interface = None
    
    def _get_observation(self) -> np.ndarray:
        """Get current 12-dimensional observation"""
        # Get current state
        current_pos = self.state.get('position', np.zeros(3))
        current_vel = self.state.get('velocity', np.zeros(3))
        
        # Get current target hoop
        if self.current_target_hoop < len(self.course_hoops):
            target_hoop = self.course_hoops[self.current_target_hoop]
            target_pos = np.array(target_hoop['position'])
        else:
            target_pos = current_pos  # Fallback
        
        # Calculate direction to hoop (normalized)
        direction_to_hoop = target_pos - current_pos
        distance_to_hoop = np.linalg.norm(direction_to_hoop)
        direction_to_hoop = direction_to_hoop / max(distance_to_hoop, 0.01)
        
        # Velocity alignment with target direction
        velocity_magnitude = np.linalg.norm(current_vel)
        velocity_alignment = np.dot(current_vel, direction_to_hoop) / max(velocity_magnitude, 0.01)
        
        # Vision features (if ZED available)
        visual_alignment = 0.0
        visual_distance = 1.0
        hoop_visible = 0.0
        
        if self.zed_interface and self.zed_interface.is_connected():
            frame = self.zed_interface.get_frame()
            if frame and frame.rgb_image is not None:
                # Process with YOLO11 (simplified for now)
                visual_alignment = 0.0  # Would be computed by vision processor
                visual_distance = min(distance_to_hoop, 5.0) / 5.0
                hoop_visible = 1.0
        
        # Course progress
        progress_in_lap = self.current_target_hoop / max(len(self.course_hoops), 1)
        overall_progress = (self.lap_number - 1) / self.config.HOOP_CONFIG['num_laps']
        
        # Construct 12-dimensional observation
        observation = np.array([
            # Direction to hoop (3 dimensions)
            np.clip(direction_to_hoop[0], -1.0, 1.0),  # X direction
            np.clip(direction_to_hoop[1], -1.0, 1.0),  # Y direction
            np.clip(direction_to_hoop[2], -1.0, 1.0),  # Z direction
            
            # Current velocity (2 dimensions)
            np.clip(current_vel[0] / 2.0, -1.0, 1.0),  # Forward velocity
            np.clip(current_vel[1] / 2.0, -1.0, 1.0),  # Lateral velocity
            
            # Navigation metrics (2 dimensions)
            np.clip(distance_to_hoop / 5.0, 0.0, 1.0),  # Distance to target
            np.clip(velocity_alignment, -1.0, 1.0),      # Velocity alignment
            
            # Vision features (3 dimensions)
            np.clip(visual_alignment, -1.0, 1.0),        # Visual alignment
            visual_distance,                             # Visual distance
            hoop_visible,                               # Hoop visibility
            
            # Course progress (2 dimensions)
            progress_in_lap,                            # Lap progress
            overall_progress                            # Overall progress
        ], dtype=np.float32)
        
        return observation
    
    def _process_action(self, action: np.ndarray) -> None:
        """Process and execute action"""
        # Apply safety constraints if enabled
        if self.safety_layer:
            action = self.safety_layer.apply_safety_constraints(action, self.state.get_all())
        
        # Extract action components
        lateral_cmd = np.clip(action[0], -1.0, 1.0)
        vertical_cmd = np.clip(action[1], -1.0, 1.0)
        speed_cmd = np.clip(action[2], -1.0, 1.0)
        
        # Convert to velocity commands
        action_config = self.config.ACTION_CONFIG
        
        lateral_velocity = lateral_cmd * action_config['components']['lateral_cmd']['max_speed']
        vertical_velocity = vertical_cmd * action_config['components']['vertical_cmd']['max_speed']
        forward_velocity = action_config['components']['speed_cmd']['base_speed'] * (1.0 + 0.5 * speed_cmd)
        
        # Create velocity command
        velocity_cmd = np.array([forward_velocity, lateral_velocity, vertical_velocity])
        
        # Send command to flight controller
        if self.use_ros:
            if self.use_px4_com:
                self.px4_interface.send_velocity_command(velocity_cmd)
            else:
                self.mavros_interface.send_velocity_command(velocity_cmd)
        else:
            # Update mock physics
            if hasattr(self, 'mock_physics'):
                self.mock_physics['velocity'] = velocity_cmd.copy()
    
    def _calculate_reward(self, observation: np.ndarray, action: np.ndarray, info: Dict[str, Any]) -> float:
        """Calculate reward using the configurable reward function"""
        try:
            from ..rewards.rewards import reward_function
            
            # Prepare parameters for reward function
            current_pos = self.state.get('position', np.zeros(3))
            target_hoop = self.course_hoops[self.current_target_hoop] if self.current_target_hoop < len(self.course_hoops) else None
            
            if target_hoop is None:
                return 0.0
            
            target_pos = np.array(target_hoop['position'])
            distance_to_hoop = np.linalg.norm(target_pos - current_pos)
            
            params = {
                'distance_to_target_hoop': distance_to_hoop,
                'current_position': current_pos,
                'target_position': target_pos,
                'current_velocity': self.state.get('velocity', np.zeros(3)),
                'hoop_visible': observation[9] > 0.5,
                'hoop_alignment': observation[7],
                'episode_step': info.get('episode_step', 0),
                'episode_time': info.get('episode_time', 0.0)
            }
            
            return reward_function(params)
            
        except Exception as e:
            logger.warning(f"Error in reward calculation: {e}")
            return 0.0
    
    def _check_episode_done(self, observation: np.ndarray, info: Dict[str, Any]) -> Tuple[bool, bool]:
        """Check if episode is terminated or truncated"""
        # Check for termination conditions
        terminated = False
        
        # Check if course completed
        if self.hoops_completed >= len(self.course_hoops) * self.config.HOOP_CONFIG['num_laps']:
            terminated = True
        
        # Check for safety violations
        current_pos = self.state.get('position', np.zeros(3))
        if self.safety_layer and not self.safety_layer.is_position_safe(current_pos):
            terminated = True
        
        # Check for truncation (max steps)
        truncated = info.get('episode_step', 0) >= self.config.TRAINING_CONFIG['max_steps_per_episode']
        
        return terminated, truncated
    
    def close(self) -> None:
        """Clean up environment resources"""
        if self.zed_interface:
            self.zed_interface.stop()
        
        super().close()


class PX4ExplorerEnv(PX4BaseEnv):
    """Explorer-level environment with PX4-ROS-COM and enhanced safety"""
    
    def __init__(self, **kwargs):
        # Force PX4-ROS-COM for Explorer environment
        kwargs['use_px4_com'] = True
        super().__init__(**kwargs)
        
        # Use beginner safety layer
        self.safety_layer = BeginnerSafetyLayer(self.config)
        
        # Override some config values for beginners
        self.config.ACTION_CONFIG['components']['lateral_cmd']['max_speed'] = 0.5  # Slower
        self.config.ACTION_CONFIG['components']['vertical_cmd']['max_speed'] = 0.3  # Slower
        self.config.ACTION_CONFIG['components']['speed_cmd']['base_speed'] = 0.4   # Slower
        
        logger.info("PX4ExplorerEnv initialized with enhanced safety (PX4-ROS-COM)")


class PX4ResearcherEnv(PX4BaseEnv):
    """Researcher-level environment with PX4-ROS-COM and full control"""
    
    def __init__(self, **kwargs):
        # Default to PX4-ROS-COM but allow override
        kwargs.setdefault('use_px4_com', True)
        super().__init__(**kwargs)
        
        # Use full safety layer
        self.safety_layer = SafetyLayer(self.config)
        
        logger.info("PX4ResearcherEnv initialized with full capabilities (PX4-ROS-COM)")


# Legacy compatibility aliases (deprecated - use PX4 versions)
MAVROSExplorerEnv = PX4ExplorerEnv
MAVROSResearcherEnv = PX4ResearcherEnv


# Convenience functions for creating environments
def create_explorer_env(**kwargs) -> PX4ExplorerEnv:
    """Create Explorer-level environment with PX4-ROS-COM"""
    return PX4ExplorerEnv(**kwargs)


def create_researcher_env(**kwargs) -> PX4ResearcherEnv:
    """Create Researcher-level environment with PX4-ROS-COM"""
    return PX4ResearcherEnv(**kwargs)


__all__ = ['PX4BaseEnv', 'PX4ExplorerEnv', 'PX4ResearcherEnv', 
           'MAVROSExplorerEnv', 'MAVROSResearcherEnv',  # Legacy aliases
           'create_explorer_env', 'create_researcher_env'] 