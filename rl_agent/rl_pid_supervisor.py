#!/usr/bin/env python3
"""
RL PID Supervisor using P3O.

This module implements a ROS2 node that uses RL to tune PID controllers
for drone path following using the P3O algorithm.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Vector3
from nav_msgs.msg import Path
from std_msgs.msg import Float32, Bool
from mavros_msgs.msg import State
import numpy as np
import torch
import gymnasium as gym
from typing import List, Dict, Tuple, Optional, Any, Union
import time
import threading

from rl_agent.supervisor_agent import SupervisorAgent
from rl_agent.pid_controller import PIDController

# Observation and action spaces
MAX_ERROR = 10.0  # Maximum cross-track error (meters)
MAX_HEADING_ERROR = np.pi  # Maximum heading error (radians)
MAX_GAIN = 5.0  # Maximum PID gain


class RLPIDSupervisor(Node):
    """
    ROS2 node for RL-based PID tuning using P3O algorithm.
    
    This node uses P3O (Procrastinated Policy-based Observer) to 
    adaptively tune PID gains for improved path following.
    """
    
    def __init__(
        self,
        node_name: str = 'rl_pid_supervisor',
        observation_dim: int = 8,
        action_dim: int = 1,
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
        update_freq: float = 10.0  # Hz
    ):
        """
        Initialize the RL PID Supervisor.
        
        Args:
            node_name: ROS2 node name
            observation_dim: Dimension of the observation space
            action_dim: Dimension of the action space (PID gains)
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
            low=0.0,
            high=MAX_GAIN,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        # Create supervisor agent
        self.agent = SupervisorAgent(
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
        
        # Create PID controller for path following
        self.pid_controller = self.agent.pid
        
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
            Vector3,
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
        
        self.p_gain_pub = self.create_publisher(
            Float32,
            '/deepflyer/p_gain',
            10
        )
        
        self.path_pub = self.create_publisher(
            Path,
            '/deepflyer/path',
            10
        )
        
        # Create timer for control loop
        self.timer = self.create_timer(1.0 / update_freq, self.control_loop)
        
        self.get_logger().info('RLPIDSupervisor has been initialized')
    
    def pose_callback(self, msg: PoseStamped) -> None:
        """Process drone pose."""
        with self.state_lock:
            position = msg.pose.position
            self.drone_pose = np.array([position.x, position.y, position.z])
    
    def velocity_callback(self, msg: Vector3) -> None:
        """Process drone velocity."""
        with self.state_lock:
            self.drone_velocity = np.array([msg.x, msg.y, msg.z])
    
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
        
        Computes errors, updates the agent, and collects data for learning.
        """
        with self.state_lock:
            if self.drone_pose is None or self.drone_velocity is None:
                return
            
            # Compute errors
            self._compute_errors()
            
            # Get observation
            observation = self._get_observation()
            
            # Use agent to adjust PID gains
            action, _ = self.agent.predict(observation)
            
            # Publish current gain
            self._publish_gain()
            
            # Store data for learning if logging is enabled
            if self.logging_enabled and self.last_observation is not None:
                # Compute reward based on errors (lower error is better)
                reward = -(abs(self.cross_track_error) + 0.1 * abs(self.heading_error))
                
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
            self.last_action = action
    
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
        
        # Publish errors
        self._publish_errors()
    
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
        
        # Drone velocity
        norm_velocity = np.linalg.norm(self.drone_velocity)
        
        # Current PID gains
        current_p_gain = self.pid_controller.kp / MAX_GAIN
        current_i_gain = self.pid_controller.ki / MAX_GAIN
        current_d_gain = self.pid_controller.kd / MAX_GAIN
        
        observation = np.array([
            norm_cross_error,
            norm_heading_error,
            path_progress,
            norm_velocity,
            current_p_gain,
            current_i_gain,
            current_d_gain,
            1.0 if self.is_armed else 0.0  # Drone armed state
        ], dtype=np.float32)
        
        return observation
    
    def _publish_errors(self) -> None:
        """Publish current errors."""
        cross_track_msg = Float32()
        cross_track_msg.data = float(self.cross_track_error)
        self.error_pub.publish(cross_track_msg)
        
        heading_error_msg = Float32()
        heading_error_msg.data = float(self.heading_error)
        self.heading_error_pub.publish(heading_error_msg)
    
    def _publish_gain(self) -> None:
        """Publish current PID gain."""
        p_gain_msg = Float32()
        p_gain_msg.data = float(self.pid_controller.kp)
        self.p_gain_pub.publish(p_gain_msg)
    
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
    """Run the RLPIDSupervisor node."""
    rclpy.init(args=args)
    supervisor = RLPIDSupervisor()
    rclpy.spin(supervisor)
    supervisor.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 