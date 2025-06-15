#!/usr/bin/env python3
"""
Direct RL Control Node for drone control.

This module implements a ROS2 node that uses P3O to directly control a drone
without an intermediate PID controller.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Vector3, TwistStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32, Bool
from mavros_msgs.msg import State, AttitudeTarget
import numpy as np
import torch
import gymnasium as gym
from typing import List, Dict, Tuple, Optional, Any, Union
import time
import threading

from rl_agent.direct_control_agent import DirectControlAgent


# Observation and action spaces
MAX_ERROR = 10.0  # Maximum cross-track error (meters)
MAX_HEADING_ERROR = np.pi  # Maximum heading error (radians)
MAX_THRUST = 1.0  # Maximum thrust (normalized 0-1)
MAX_RATE = 1.0  # Maximum angular rate (rad/s)


class DirectControlNode(Node):
    """
    ROS2 node for direct RL-based drone control using P3O.
    
    This node uses P3O (Procrastinated Policy-based Observer) to
    directly control a drone without an intermediate PID controller.
    """
    
    def __init__(
        self,
        node_name: str = 'direct_control_node',
        observation_dim: int = 12,  # State + path info
        action_dim: int = 4,  # thrust, roll, pitch, yaw
        hidden_dim: int = 256,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        lr: float = 3e-4,
        procrastination_factor: float = 0.95,
        alpha: float = 0.2,
        entropy_coef: float = 0.01,
        clip_ratio: float = 0.2,
        n_updates: int = 10,
        update_freq: float = 20.0,  # Hz
        safety_layer: bool = True
    ):
        """
        Initialize the Direct Control Node.
        
        Args:
            node_name: ROS2 node name
            observation_dim: Dimension of the observation space
            action_dim: Dimension of the action space (control commands)
            hidden_dim: Hidden dimension of the neural networks
            gamma: Discount factor
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            lr: Learning rate
            procrastination_factor: P3O procrastination factor (0-1)
            alpha: Blend factor for P3O (mixing on-policy and off-policy)
            entropy_coef: Entropy coefficient to encourage exploration
            clip_ratio: PPO clip parameter
            n_updates: Number of policy updates per learning iteration
            update_freq: Frequency of control updates (Hz)
            safety_layer: Whether to use a safety layer
        """
        super().__init__(node_name)
        
        # Create observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -MAX_RATE, -MAX_RATE, -MAX_RATE]),
            high=np.array([MAX_THRUST, MAX_RATE, MAX_RATE, MAX_RATE]),
            dtype=np.float32
        )
        
        # Create direct control agent
        self.agent = DirectControlAgent(
            observation_space=self.observation_space,
            action_space=self.action_space,
            device="auto",
            hidden_dim=hidden_dim,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            lr=lr,
            procrastination_factor=procrastination_factor,
            alpha=alpha,
            entropy_coef=entropy_coef,
            clip_ratio=clip_ratio,
            n_updates=n_updates
        )
        
        # Initialize state variables
        self.drone_pose = None
        self.drone_velocity = None
        self.path_start = np.zeros(3)
        self.path_end = np.zeros(3)
        self.path_vector = np.zeros(3)
        self.path_length = 0.0
        self.cross_track_error = 0.0
        self.heading_error = 0.0
        self.last_timestamp = None
        self.last_observation = None
        self.last_action = None
        self.logging_enabled = False
        self.is_armed = False
        self.safety_layer = safety_layer
        
        # Mutex for thread safety
        self.state_lock = threading.Lock()
        
        # Create subscribers
        self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            10
        )
        
        self.create_subscription(
            TwistStamped,
            '/mavros/local_position/velocity_local',
            self.velocity_callback,
            10
        )
        
        self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            10
        )
        
        self.create_subscription(
            Bool,
            '/deepflyer/logging_enabled',
            self.logging_callback,
            10
        )
        
        # Create publishers
        self.error_pub = self.create_publisher(
            Float32,
            '/deepflyer/cross_track_error',
            10
        )
        
        self.heading_error_pub = self.create_publisher(
            Float32,
            '/deepflyer/heading_error',
            10
        )
        
        self.path_pub = self.create_publisher(
            Path,
            '/deepflyer/path',
            10
        )
        
        # Publisher for direct control commands
        self.attitude_pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10
        )
        
        # Create timer for control loop
        self.timer = self.create_timer(1.0 / update_freq, self.control_loop)
        
        self.get_logger().info('DirectControlNode has been initialized')
    
    def pose_callback(self, msg: PoseStamped) -> None:
        """Process drone pose."""
        with self.state_lock:
            position = msg.pose.position
            self.drone_pose = np.array([position.x, position.y, position.z])
    
    def velocity_callback(self, msg: TwistStamped) -> None:
        """Process drone velocity."""
        with self.state_lock:
            linear = msg.twist.linear
            self.drone_velocity = np.array([linear.x, linear.y, linear.z])
    
    def state_callback(self, msg: State) -> None:
        """Process drone state."""
        with self.state_lock:
            self.is_armed = msg.armed
    
    def logging_callback(self, msg: Bool) -> None:
        """Process logging enabled flag."""
        self.logging_enabled = msg.data
    
    def set_path(self, start: List[float], end: List[float]) -> None:
        """
        Set a path from start to end.
        
        Args:
            start: Start position [x, y, z]
            end: End position [x, y, z]
        """
        with self.state_lock:
            self.path_start = np.array(start)
            self.path_end = np.array(end)
            self.path_vector = self.path_end - self.path_start
            self.path_length = np.linalg.norm(self.path_vector)
            
            # Publish path for visualization
            self._publish_path()
            
            self.get_logger().info(f"Path set from {start} to {end}")
    
    def control_loop(self) -> None:
        """
        Main control loop that runs at the update frequency.
        
        Computes errors, gets control commands from the agent, and sends them to the drone.
        """
        with self.state_lock:
            if self.drone_pose is None or self.drone_velocity is None:
                return
            
            # Compute errors
            self._compute_errors()
            
            # Get observation
            observation = self._get_observation()
            
            # Use agent to get control commands
            action, _ = self.agent.predict(observation)
            
            # Apply safety layer if enabled
            if self.safety_layer:
                action = self._apply_safety_layer(action, observation)
            
            # Send control commands to drone
            self._send_control_commands(action)
            
            # Publish errors
            self._publish_errors()
            
            # Store data for learning if logging is enabled
            if self.logging_enabled and self.last_observation is not None and self.last_action is not None:
                # Compute reward based on errors (lower error is better)
                reward = -(abs(self.cross_track_error) + 0.1 * abs(self.heading_error))
                
                # Add action smoothness penalty if we have previous actions
                if self.agent.last_action is not None:
                    action_diff = np.linalg.norm(action - self.agent.last_action)
                    reward -= self.agent.action_smoothness_penalty * action_diff
                
                # Add to buffer
                self.agent.add_to_buffer(
                    self.last_observation,
                    self.last_action,
                    observation,
                    reward,
                    False
                )
            
            # Update for next iteration
            self.last_observation = observation.copy()
            self.last_action = action.copy()
    
    def _compute_errors(self) -> None:
        """Compute cross-track and heading errors."""
        if self.path_length == 0:
            return
        
        # Compute projection of drone position onto path
        drone_rel_pos = self.drone_pose - self.path_start
        path_dir = self.path_vector / self.path_length
        
        # Projection distance along the path
        proj_dist = np.dot(drone_rel_pos, path_dir)
        proj_dist = min(max(0, proj_dist), self.path_length)
        
        # Projected point on path
        proj_point = self.path_start + proj_dist * path_dir
        
        # Cross-track error is distance from drone to projected point
        self.cross_track_error = np.linalg.norm(self.drone_pose - proj_point)
        
        # Compute heading error if drone is moving
        if np.linalg.norm(self.drone_velocity) > 0.1:
            vel_dir = self.drone_velocity / np.linalg.norm(self.drone_velocity)
            heading_dot = np.dot(vel_dir, path_dir)
            self.heading_error = np.arccos(np.clip(heading_dot, -1.0, 1.0))
        else:
            self.heading_error = 0.0
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            observation: Observation vector for the agent
        """
        # Path information
        path_progress = np.dot(self.drone_pose - self.path_start, 
                              self.path_vector) / max(self.path_length, 1e-6)
        path_progress = np.clip(path_progress, 0, 1)
        
        # Normalized errors for better learning
        norm_cross_error = self.cross_track_error / MAX_ERROR
        norm_heading_error = self.heading_error / MAX_HEADING_ERROR
        
        # Path direction (normalized)
        path_dir = self.path_vector / max(self.path_length, 1e-6)
        
        # Vector from drone to path end
        to_goal = self.path_end - self.drone_pose
        dist_to_goal = np.linalg.norm(to_goal)
        to_goal_dir = to_goal / max(dist_to_goal, 1e-6)
        
        observation = np.array([
            # Drone state
            self.drone_pose[0], self.drone_pose[1], self.drone_pose[2],
            self.drone_velocity[0], self.drone_velocity[1], self.drone_velocity[2],
            
            # Path and error information
            path_dir[0], path_dir[1], path_dir[2],
            norm_cross_error,
            norm_heading_error,
            path_progress,
        ], dtype=np.float32)
        
        return observation
    
    def _apply_safety_layer(self, action: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """
        Apply safety constraints to the control action.
        
        Args:
            action: Raw control action [thrust, roll_rate, pitch_rate, yaw_rate]
            observation: Current observation
            
        Returns:
            safe_action: Safety-constrained action
        """
        # Extract relevant information from observation
        drone_pos = observation[0:3]
        drone_vel = observation[3:6]
        
        # Create a copy of the action to modify
        safe_action = action.copy()
        
        # 1. Ensure minimum thrust to prevent free-fall
        safe_action[0] = max(safe_action[0], 0.2)
        
        # 2. Limit maximum thrust
        safe_action[0] = min(safe_action[0], 0.9)
        
        # 3. Limit angular rates based on altitude (more conservative at low altitude)
        altitude = drone_pos[2]
        if altitude < 1.0:
            # More conservative at low altitude
            rate_scale = 0.5 * altitude
            safe_action[1:] = safe_action[1:] * rate_scale
        
        # 4. Prevent actions that would cause collision with ground
        if altitude < 0.5 and drone_vel[2] < 0:
            # If descending at low altitude, reduce descent rate
            safe_action[0] = max(safe_action[0], 0.6)  # Increase thrust
        
        # 5. Limit maximum angular rates overall
        max_rate = 0.7  # rad/s
        for i in range(1, 4):
            safe_action[i] = np.clip(safe_action[i], -max_rate, max_rate)
        
        return safe_action
    
    def _send_control_commands(self, action: np.ndarray) -> None:
        """
        Send control commands to the drone.
        
        Args:
            action: Control action [thrust, roll_rate, pitch_rate, yaw_rate]
        """
        # Create attitude target message
        attitude_msg = AttitudeTarget()
        attitude_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set the type mask for rate control
        # Ignore attitude and use body rate
        attitude_msg.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        
        # Set thrust (normalized 0-1)
        attitude_msg.thrust = float(action[0])
        
        # Set body rates (rad/s)
        attitude_msg.body_rate.x = float(action[1])  # Roll rate
        attitude_msg.body_rate.y = float(action[2])  # Pitch rate
        attitude_msg.body_rate.z = float(action[3])  # Yaw rate
        
        # Publish the message
        self.attitude_pub.publish(attitude_msg)
    
    def _publish_errors(self) -> None:
        """Publish current errors."""
        cross_track_msg = Float32()
        cross_track_msg.data = float(self.cross_track_error)
        self.error_pub.publish(cross_track_msg)
        
        heading_error_msg = Float32()
        heading_error_msg.data = float(self.heading_error)
        self.heading_error_pub.publish(heading_error_msg)
    
    def _publish_path(self) -> None:
        """Publish current path for visualization."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        
        # Add start and end points
        start_pose = PoseStamped()
        start_pose.header = path_msg.header
        start_pose.pose.position.x = float(self.path_start[0])
        start_pose.pose.position.y = float(self.path_start[1])
        start_pose.pose.position.z = float(self.path_start[2])
        
        end_pose = PoseStamped()
        end_pose.header = path_msg.header
        end_pose.pose.position.x = float(self.path_end[0])
        end_pose.pose.position.y = float(self.path_end[1])
        end_pose.pose.position.z = float(self.path_end[2])
        
        path_msg.poses = [start_pose, end_pose]
        self.path_pub.publish(path_msg)
    
    def enable_logging(self, enabled: bool) -> None:
        """
        Enable or disable logging.
        
        Args:
            enabled: Whether to enable logging
        """
        self.logging_enabled = enabled
        self.get_logger().info(f"Logging {'enabled' if enabled else 'disabled'}")
    
    def learn(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Perform one iteration of P3O learning.
        
        Args:
            batch_size: Batch size for learning (if None, use default)
            
        Returns:
            metrics: Dictionary of learning metrics
        """
        return self.agent.learn(batch_size)
    
    def save_model(self, path: str) -> None:
        """
        Save agent model to disk.
        
        Args:
            path: Path to save the model
        """
        state_dict = self.agent.get_model_state()
        torch.save(state_dict, path)
        self.get_logger().info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load agent model from disk.
        
        Args:
            path: Path to load the model from
        """
        state_dict = torch.load(path)
        self.agent.load_model_state(state_dict)
        self.get_logger().info(f"Model loaded from {path}")


def main(args=None):
    """Run the DirectControlNode."""
    rclpy.init(args=args)
    node = DirectControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
