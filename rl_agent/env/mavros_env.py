import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import time
import logging

from .ros_env import RosEnv, MAVROS_AVAILABLE
from .safety_layer import SafetyLayer, BeginnerSafetyLayer, SafetyBounds
from ..rewards import RewardFunction, create_default_reward_function, REGISTRY

logger = logging.getLogger(__name__)

# Flight modes supported by PX4 (subset most relevant to education)
FLIGHT_MODES = [
    "MANUAL",        # Manual control
    "STABILIZED",    # Simple attitude stabilization  
    "ALTCTL",        # Altitude control
    "POSCTL",        # Position control
    "OFFBOARD",      # External control via MAVROS
    "AUTO.TAKEOFF",  # Automatic takeoff
    "AUTO.LAND",     # Automatic landing
    "AUTO.MISSION",  # Mission execution
    "AUTO.RTL",      # Return to launch
    "AUTO.HOLD"      # Hold position
]

class MAVROSEnv(RosEnv):
    """
    Specialized environment for Pixhawk/PX4 drones using MAVROS.
    
    This simplified interface makes it easier to work with the Pixhawk 6c
    flight controller used in the DeepFlyer educational platform.
    """
    
    def __init__(
        self,
        namespace: str = "deepflyer",
        observation_config: Optional[Dict[str, bool]] = None,
        action_mode: str = "continuous",
        max_episode_steps: int = 500,
        step_duration: float = 0.05,
        timeout: float = 5.0,
        goal_position: Optional[List[float]] = None,
        target_altitude: Optional[float] = None,
        camera_resolution: Tuple[int, int] = (84, 84),
        use_zed: bool = True,
        auto_arm: bool = False,
        auto_offboard: bool = False,
        safety_boundaries: Optional[Dict[str, float]] = None,
        enable_safety_layer: bool = True,
        reward_function: Optional[RewardFunction] = None,
        custom_reward_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Pixhawk/PX4 drone environment.
        
        Args:
            namespace: ROS2 namespace for topics
            observation_config: Dict specifying which observations to include
            action_mode: "continuous" or "discrete" action space
            max_episode_steps: Maximum steps per episode
            step_duration: Duration of each environment step
            timeout: Timeout for waiting for sensor data
            goal_position: Goal position [x, y, z] for navigation tasks
            target_altitude: Target altitude for altitude hold tasks
            camera_resolution: Resolution to downsample camera images to
            use_zed: Whether to use ZED Mini stereo camera topics
            auto_arm: Automatically arm the drone during reset
            auto_offboard: Automatically set OFFBOARD mode during reset
            safety_boundaries: Safety boundaries for the drone (e.g., {'x_min': -5, 'x_max': 5})
            enable_safety_layer: Whether to enable the safety layer
            reward_function: Custom reward function (if None, default will be created)
            custom_reward_weights: Custom weights for default reward components
        """
        if not MAVROS_AVAILABLE:
            logger.warning("MAVROS is not available. Using mock implementation.")
            
        # Call parent constructor with MAVROS enabled
        super().__init__(
            namespace=namespace,
            observation_config=observation_config,
            action_mode=action_mode,
            max_episode_steps=max_episode_steps,
            step_duration=step_duration,
            timeout=timeout,
            goal_position=goal_position,
            target_altitude=target_altitude,
            camera_resolution=camera_resolution,
            use_zed=use_zed,
            use_mavros=True,  # Always use MAVROS
            auto_arm=auto_arm,
            auto_offboard=auto_offboard
        )
        
        # Safety boundaries
        self.safety_bounds = SafetyBounds()
        if safety_boundaries:
            for key, value in safety_boundaries.items():
                if hasattr(self.safety_bounds, key):
                    setattr(self.safety_bounds, key, value)
        
        # Initialize safety layer
        self.enable_safety_layer = enable_safety_layer
        if enable_safety_layer:
            self.safety_layer = SafetyLayer(safety_bounds=self.safety_bounds)
            
            # Register emergency stop callback
            self.safety_layer.monitor.register_emergency_callback(self._handle_emergency)
            
            # Start safety monitoring
            self.safety_layer.start_monitoring(self._get_safety_state)
        else:
            self.safety_layer = None
            
        # Initialize reward function
        if reward_function is None:
            self.reward_function = create_default_reward_function()
            
            # Apply custom weights if provided
            if custom_reward_weights:
                for i, component in enumerate(self.reward_function.components):
                    component_name = component.component_type.value
                    if component_name in custom_reward_weights:
                        self.reward_function.components[i].weight = custom_reward_weights[component_name]
        else:
            self.reward_function = reward_function
            
        # Store previous action for reward calculation
        self.prev_action = None
        
        logger.info("MAVROSEnv initialized with safety layer and reward function")
    
    def _get_safety_state(self) -> Dict[str, Any]:
        """
        Get current state for safety monitoring.
        
        Returns:
            Dictionary with state variables for safety checks
        """
        state = self.node.state.get_snapshot()
        
        # Convert quaternion to Euler angles if available
        orientation_euler = None
        if state['orientation'] is not None:
            try:
                from scipy.spatial.transform import Rotation
                quat = state['orientation']
                rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]])
                orientation_euler = rot.as_euler('xyz', degrees=True)
            except ImportError:
                # Fallback without scipy
                orientation_euler = np.zeros(3)
        
        return {
            'position': state['position'],
            'velocity': state['linear_velocity'],
            'orientation_euler': orientation_euler,
            'obstacle_distance': state.get('distance_to_obstacle', float('inf')),
        }
    
    def _handle_emergency(self):
        """Handle emergency stop request."""
        logger.warning("Emergency stop triggered! Sending zero velocity.")
        # Send zero velocity command
        self.node.send_velocity_command(np.zeros(3), np.zeros(3))
        
        # Try to switch to position hold mode if possible
        try:
            self.node.set_mode("AUTO.HOLD")
        except Exception as e:
            logger.error(f"Failed to set HOLD mode during emergency: {e}")
    
    def arm(self) -> bool:
        """Arm the drone."""
        return self.node.arm(True)
    
    def disarm(self) -> bool:
        """Disarm the drone."""
        return self.node.arm(False)
    
    def set_mode(self, mode: str) -> bool:
        """
        Set the flight mode of the drone.
        
        Args:
            mode: One of the PX4 flight modes, e.g. "OFFBOARD", "POSCTL"
        """
        if mode not in FLIGHT_MODES:
            logger.warning(f"Unknown flight mode: {mode}. Expected one of {FLIGHT_MODES}")
        
        return self.node.set_mode(mode)
    
    def is_armed(self) -> bool:
        """Check if the drone is armed."""
        return self.node.state.armed
    
    def is_connected(self) -> bool:
        """Check if MAVROS is connected to the flight controller."""
        return self.node.state.connected
    
    def get_flight_mode(self) -> str:
        """Get the current flight mode."""
        return self.node.state.flight_mode
    
    def takeoff(self, target_altitude: Optional[float] = None) -> bool:
        """
        Initiate an automatic takeoff sequence.
        
        Args:
            target_altitude: Target altitude in meters
        """
        alt = target_altitude or self.target_altitude or 1.5
        
        # Set to takeoff mode
        success = self.node.set_mode("AUTO.TAKEOFF")
        
        if success:
            logger.info(f"Takeoff initiated to altitude: {alt}m")
            # Wait for takeoff to begin
            time.sleep(1.0)
        else:
            logger.error("Failed to initiate takeoff")
            
        return success
    
    def land(self) -> bool:
        """Initiate the landing sequence."""
        # Set to land mode
        success = self.node.set_mode("AUTO.LAND")
        
        if success:
            logger.info("Landing initiated")
        else:
            logger.error("Failed to initiate landing")
            
        return success
    
    def return_to_launch(self) -> bool:
        """Initiate return to launch (RTL) sequence."""
        # Set to RTL mode
        success = self.node.set_mode("AUTO.RTL")
        
        if success:
            logger.info("Return to launch initiated")
        else:
            logger.error("Failed to initiate return to launch")
            
        return success
    
    def set_offboard_mode(self) -> bool:
        """Set the drone to OFFBOARD mode (required for direct control)."""
        # First make sure the drone is armed
        if not self.is_armed():
            logger.warning("Drone is not armed, arming before setting OFFBOARD mode")
            self.arm()
            time.sleep(0.5)  # Give it time to arm
        
        # Set to OFFBOARD mode
        success = self.node.set_mode("OFFBOARD")
        
        if success:
            logger.info("OFFBOARD mode activated")
        else:
            logger.error("Failed to set OFFBOARD mode")
            
        return success
    
    def is_in_safety_bounds(self) -> bool:
        """Check if the drone is within the defined safety boundaries."""
        state = self.node.state.get_snapshot()
        position = state['position']
        
        return self.safety_bounds.is_position_in_bounds(position)
    
    def set_safety_boundaries(self, boundaries: Dict[str, float]):
        """Update safety boundaries."""
        for key, value in boundaries.items():
            if hasattr(self.safety_bounds, key):
                setattr(self.safety_bounds, key, value)
    
    def prepare_for_learning(self) -> bool:
        """Setup drone for reinforcement learning (arm and set to OFFBOARD mode)."""
        # Ensure connected
        if not self.is_connected():
            logger.error("Not connected to flight controller")
            return False
        
        # Arm the drone
        if not self.is_armed():
            success = self.arm()
            if not success:
                logger.error("Failed to arm the drone")
                return False
            time.sleep(0.5)  # Give it time to arm
        
        # Set to OFFBOARD mode
        success = self.set_offboard_mode()
        if not success:
            logger.error("Failed to set OFFBOARD mode")
            return False
        
        # Wait for OFFBOARD mode to activate
        time.sleep(0.5)
        
        return True
    
    def _calculate_reward(self, state: Dict[str, Any], action: np.ndarray) -> float:
        """
        Calculate reward using the reward function.
        
        Args:
            state: Current state dictionary
            action: Current action
            
        Returns:
            Calculated reward
        """
        # Get current state snapshot for reward calculation
        reward_state = state.copy()
        
        # Add goal position if not in state
        if 'goal_position' not in reward_state and self.goal_position is not None:
            reward_state['goal_position'] = self.goal_position
            
        # Add target altitude if not in state
        if 'target_altitude' not in reward_state and self.target_altitude is not None:
            reward_state['target_altitude'] = self.target_altitude
        
        # Calculate reward using the reward function
        reward = self.reward_function.compute_reward(
            state=reward_state,
            action=action,
            next_state=None,  # Not used in our reward functions
            info={'prev_action': self.prev_action}
        )
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step with safety layer and custom rewards.
        
        This extends the base step method with safety measures and custom rewards.
        """
        # Store original action for reward calculation
        original_action = action.copy()
        
        # Apply safety layer if enabled
        if self.enable_safety_layer and self.safety_layer is not None:
            # Get current state for safety processing
            state = self.node.state.get_snapshot()
            position = state['position']
            
            # Get obstacle information if available
            obstacle_distance = state.get('distance_to_obstacle', None)
            obstacle_direction = None  # Would need to be calculated from depth image
            
            # Process action through safety layer
            linear_vel, angular_vel = self._process_action(action)
            safe_linear_vel = self.safety_layer.process_command(
                velocity_command=linear_vel,
                position=position,
                obstacle_distance=obstacle_distance,
                obstacle_direction=obstacle_direction
            )
            
            # Override linear velocity with safe version
            self.node.send_velocity_command(safe_linear_vel, angular_vel)
        else:
            # Use normal action processing
            linear_vel, angular_vel = self._process_action(action)
            self.node.send_velocity_command(linear_vel, angular_vel)
        
        # Wait for step duration
        time.sleep(self.step_duration)
        
        # Get new observation
        observation = self._get_observation()
        state = self.node.state.get_snapshot()
        
        # Calculate reward using reward function
        reward = self._calculate_reward(state, original_action)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Check collision
        if state['collision_flag']:
            terminated = True
            # Collision penalty is handled by reward function
        
        # Check if goal reached
        distance_to_goal = np.linalg.norm(state['position'] - self.goal_position)
        goal_reached = distance_to_goal < 0.2  # within 20cm of goal
        
        if goal_reached:
            terminated = True
            # Goal bonus is handled by reward function
        
        # Check safety boundary violation
        safety_violation = False
        if self.enable_safety_layer and self.safety_layer is not None:
            safety_status = self.safety_layer.get_status()
            safety_violation = not safety_status['is_safe']
            
            # Terminate if emergency stop is active
            if safety_status['emergency_stop_active']:
                terminated = True
        
        # Check truncation
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # Update episode reward
        self.episode_reward += reward
        
        # Store action for next reward calculation
        self.prev_action = original_action
        
        # Build info dict
        info = {
            'episode_step': self.current_step,
            'episode_reward': self.episode_reward,
            'distance_to_goal': distance_to_goal,
            'goal_reached': goal_reached,
            'collision': state['collision_flag'],
            'position': state['position'].tolist(),
            'velocity': state['linear_velocity'].tolist(),
        }
        
        # Add safety information if enabled
        if self.enable_safety_layer and self.safety_layer is not None:
            info.update({
                'safety_violation': safety_violation,
                'safety_interventions': self.safety_layer.intervention_count,
            })
        
        # Add MAVROS specific info
        if self.use_mavros:
            info.update({
                'armed': state['armed'],
                'flight_mode': state['flight_mode'],
            })
        
        # Add reward components for analysis
        if hasattr(self.reward_function, 'component_values'):
            info['reward_components'] = self.reward_function.component_values
        
        return observation, float(reward), terminated, truncated, info
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment with safety layer and reward function reset."""
        # Call parent reset
        observation, info = super().reset(seed=seed, options=options)
        
        # Reset safety layer if enabled
        if self.enable_safety_layer and self.safety_layer is not None:
            self.safety_layer.reset()
        
        # Reset reward function statistics
        self.reward_function.reset_stats()
        
        # Reset previous action
        self.prev_action = None
        
        # If auto_offboard is disabled but we still want to prepare for learning
        if not self.auto_offboard and not self.auto_arm:
            if options and options.get('prepare_for_learning', False):
                self.prepare_for_learning()
        
        return observation, info
    
    def close(self):
        """Clean up resources."""
        # Stop safety monitoring if active
        if self.enable_safety_layer and self.safety_layer is not None:
            self.safety_layer.stop_monitoring()
        
        # Call parent close method
        super().close()


class MAVROSExplorerEnv(MAVROSEnv):
    """
    Explorer mode version of the MAVROS environment.
    
    This version simplifies the environment for educational purposes.
    Designed for beginners (ages 11-22) to learn reinforcement learning.
    """
    
    def __init__(
        self,
        namespace: str = "deepflyer",
        max_episode_steps: int = 500,
        step_duration: float = 0.1,  # Slower control update (10Hz)
        camera_resolution: Tuple[int, int] = (64, 64),  # Smaller images
        use_zed: bool = True,
    ):
        """
        Initialize simplified Explorer mode environment.
        """
        # Simplified observation space for beginners
        observation_config = {
            'position': True,
            'orientation': True,
            'linear_velocity': True,
            'angular_velocity': False,  # Simplified
            'front_camera': True,
            'down_camera': False,  # Just use one camera
            'collision': True,
            'obstacle_distance': True,
            'goal_relative': True,
        }
        
        # Create beginner-friendly reward function
        reward_function = create_default_reward_function()
        
        # Adjust weights to make it more beginner-friendly
        custom_reward_weights = {
            'reach_target': 1.0,    # Primary goal
            'avoid_crashes': 1.5,   # Safety first for beginners
            'fly_steady': 0.5,      # Encourage smooth flight
            'minimize_time': 0.05,  # Less time pressure
        }
        
        # Use beginner safety layer with more restrictive bounds
        beginner_safety = BeginnerSafetyLayer()
            
        # Call parent constructor with explorer-friendly defaults
        super().__init__(
            namespace=namespace,
            observation_config=observation_config,
            action_mode="continuous",  # Simpler to start with continuous
            max_episode_steps=max_episode_steps,
            step_duration=step_duration,
            goal_position=[3.0, 3.0, 1.5],  # Simple goal
            target_altitude=1.5,
            camera_resolution=camera_resolution,
            use_zed=use_zed,
            auto_arm=False,  # Safety first for beginners
            auto_offboard=False,
            enable_safety_layer=True,
            reward_function=reward_function,
            custom_reward_weights=custom_reward_weights,
        )
        
        # Override safety layer with beginner version
        self.safety_layer = beginner_safety
        self.safety_layer.monitor.register_emergency_callback(self._handle_emergency)
        self.safety_layer.start_monitoring(self._get_safety_state)
        
        logger.info("Explorer mode initialized with simplified configuration")


class MAVROSResearcherEnv(MAVROSEnv):
    """
    Researcher mode version of the MAVROS environment.
    
    This version provides full access to all features for advanced users.
    Designed for university students and researchers to experiment with
    advanced reinforcement learning algorithms.
    """
    
    def __init__(
        self,
        namespace: str = "deepflyer",
        observation_config: Optional[Dict[str, bool]] = None,
        action_mode: str = "continuous",
        max_episode_steps: int = 1000,  # Longer episodes
        step_duration: float = 0.05,  # Faster control (20Hz)
        use_zed: bool = True,
        auto_arm: bool = True,  # Auto-arm for convenience
        auto_offboard: bool = True,  # Auto-offboard for convenience
        with_noise: bool = True,  # Add noise to observations
        noise_level: float = 0.05,  # 5% noise
        enable_safety_layer: bool = True,  # Still have safety but configurable
        custom_reward_function: Optional[RewardFunction] = None,
    ):
        """
        Initialize Researcher mode environment with advanced features.
        """
        # Use full observation space by default
        full_obs_config = {
            'position': True,
            'orientation': True,
            'linear_velocity': True,
            'angular_velocity': True,
            'linear_acceleration': True,
            'front_camera': True,
            'down_camera': False,  # Use ZED by default
            'collision': True,
            'obstacle_distance': True,
            'goal_relative': True,
        }
        
        # Override with user config if provided
        if observation_config:
            full_obs_config.update(observation_config)
            
        # Call parent constructor with researcher-oriented defaults
        super().__init__(
            namespace=namespace,
            observation_config=full_obs_config,
            action_mode=action_mode,
            max_episode_steps=max_episode_steps,
            step_duration=step_duration,
            goal_position=None,  # Will be set randomly
            camera_resolution=(128, 128),  # Higher resolution
            use_zed=use_zed,
            auto_arm=auto_arm,
            auto_offboard=auto_offboard,
            enable_safety_layer=enable_safety_layer,
            reward_function=custom_reward_function,
        )
        
        # Additional researcher-specific configuration
        self.with_noise = with_noise
        self.noise_level = noise_level
        
        # Random goals within safety boundaries
        self._set_random_goal()
        
        logger.info("Researcher mode initialized with advanced configuration")
    
    def _set_random_goal(self):
        """Set a random goal within safety boundaries."""
        # Get bounds from safety layer
        bounds = self.safety_bounds
        
        # Random position within bounds (but not too close to boundaries)
        margin = 1.0
        x = np.random.uniform(bounds.x_min + margin, bounds.x_max - margin)
        y = np.random.uniform(bounds.y_min + margin, bounds.y_max - margin)
        z = np.random.uniform(bounds.z_min + margin, bounds.z_max - margin)
        
        self.goal_position = np.array([x, y, z])
        logger.info(f"New random goal set: {self.goal_position}")
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get observation with optional noise added."""
        obs = super()._get_observation()
        
        # Add noise to numerical observations
        if self.with_noise:
            for key, value in obs.items():
                # Only add noise to numerical data, not images or discrete values
                if key in ['position', 'orientation', 'linear_velocity', 
                          'angular_velocity', 'goal_relative']:
                    # Add Gaussian noise proportional to value range
                    noise = np.random.normal(0, self.noise_level * np.abs(value).mean(), value.shape)
                    obs[key] = value + noise.astype(value.dtype)
        
        return obs
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset with random goal unless specified in options."""
        # Set new random goal if not provided in options
        if options is None or 'goal_position' not in options:
            self._set_random_goal()
            
        # Call parent reset
        return super().reset(seed=seed, options=options) 