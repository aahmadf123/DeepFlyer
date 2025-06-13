"""
MAVROS Interface for ROS2 communication with PX4.

This module provides utilities for communicating with the PX4 flight controller
through MAVROS in a ROS2 environment.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from typing import Tuple, Dict, Any, Callable
import math
from threading import Lock

# ROS2 message types
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


class MAVROSInterface:
    """Interface for communicating with MAVROS."""
    
    def __init__(self, node: Node):
        """
        Initialize the MAVROS interface.
        
        Args:
            node: ROS2 node
        """
        self.node = node
        
        # Create QoS profile for reliable communication
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # State variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.angular_velocity = np.zeros(3)
        
        # Mutex for thread safety
        self.state_lock = Lock()
        
        # Create subscribers
        self._setup_subscribers()
        
        # Create publishers
        self._setup_publishers()
        
        self.node.get_logger().info("MAVROS Interface initialized")
    
    def _setup_subscribers(self):
        """Set up all subscribers."""
        self.pose_sub = self.node.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self._pose_callback,
            self.qos_profile
        )
        
        self.velocity_sub = self.node.create_subscription(
            TwistStamped,
            '/mavros/local_position/velocity_local',
            self._velocity_callback,
            self.qos_profile
        )
        
        self.imu_sub = self.node.create_subscription(
            Imu,
            '/mavros/imu/data',
            self._imu_callback,
            self.qos_profile
        )
    
    def _setup_publishers(self):
        """Set up all publishers."""
        self.vel_pub = self.node.create_publisher(
            Twist,
            '/mavros/setpoint_velocity/cmd_vel',
            self.qos_profile
        )
    
    def _pose_callback(self, msg: PoseStamped):
        """
        Process pose messages.
        
        Args:
            msg: PoseStamped message
        """
        with self.state_lock:
            # Extract position
            self.position[0] = msg.pose.position.x
            self.position[1] = msg.pose.position.y
            self.position[2] = msg.pose.position.z
            
            # Extract orientation (quaternion to Euler)
            q = msg.pose.orientation
            # Convert quaternion to Euler angles
            sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
            
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            self.orientation = np.array([roll, pitch, yaw])
    
    def _velocity_callback(self, msg: TwistStamped):
        """
        Process velocity messages.
        
        Args:
            msg: TwistStamped message
        """
        with self.state_lock:
            self.velocity[0] = msg.twist.linear.x
            self.velocity[1] = msg.twist.linear.y
            self.velocity[2] = msg.twist.linear.z
            
            # Also store angular velocity
            self.angular_velocity[0] = msg.twist.angular.x
            self.angular_velocity[1] = msg.twist.angular.y
            self.angular_velocity[2] = msg.twist.angular.z
    
    def _imu_callback(self, msg: Imu):
        """
        Process IMU messages.
        
        Args:
            msg: Imu message
        """
        # We're primarily getting orientation from the pose,
        # but could use IMU for more accurate angular rates if needed
        pass
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state.
        
        Returns:
            state: Dictionary containing position, velocity, and orientation
        """
        with self.state_lock:
            return {
                'position': self.position.copy(),
                'velocity': self.velocity.copy(),
                'orientation': self.orientation.copy(),
                'angular_velocity': self.angular_velocity.copy()
            }
    
    def send_velocity_command(self, linear_vel: float, angular_vel: float):
        """
        Send velocity command to the drone.
        
        Args:
            linear_vel: Linear velocity (forward)
            angular_vel: Angular velocity (yaw)
        """
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = angular_vel
        
        self.vel_pub.publish(vel_cmd)
        
        self.node.get_logger().debug(f"Sent velocity command: [{linear_vel:.2f}, {angular_vel:.2f}]") 