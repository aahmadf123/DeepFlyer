"""
ROS2 node for integration with PX4.

This node subscribes to position, velocity, and orientation topics,
computes errors, and publishes velocity commands.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from typing import List, Tuple
import time
import math
from threading import Lock

# ROS2 message types
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

# Local imports
from rl_agent.supervisor_agent import SupervisorAgent
from rl_agent.error_calculator import ErrorCalculator
from rl_agent.pid_controller import PIDController


class RL_PID_Node(Node):
    """ROS2 node for RL-supervised PID control."""
    
    def __init__(self):
        """Initialize the ROS2 node."""
        super().__init__('rl_pid_node')
        
        # Create QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create publishers and subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            qos_profile
        )
        
        self.velocity_sub = self.create_subscription(
            TwistStamped,
            '/mavros/local_position/velocity_local',
            self.velocity_callback,
            qos_profile
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            qos_profile
        )
        
        self.vel_pub = self.create_publisher(
            Twist,
            '/mavros/setpoint_velocity/cmd_vel',
            qos_profile
        )
        
        # Create error calculator
        self.error_calculator = ErrorCalculator()
        
        # Create PID controller
        self.pid_controller = PIDController()
        
        # Create RL supervisor agent
        # This would normally require proper gym spaces
        # For now, we'll initialize it later
        self.supervisor = None
        
        # State variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)  # [roll, pitch, yaw]
        
        # Error variables
        self.cross_track_error = 0.0
        self.heading_error = 0.0
        
        # Control variables
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Path definition
        self.origin = np.array([0.0, 0.0, 1.0])  # Start at 1m height
        self.target = np.array([10.0, 0.0, 1.0])  # Go 10m forward
        self.error_calculator.set_path(self.origin.tolist(), self.target.tolist())
        
        # Mutex for thread safety
        self.state_lock = Lock()
        
        # Control loop timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('RL PID node initialized')
    
    def pose_callback(self, msg: PoseStamped):
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
            # This is a simplified conversion, might need a more robust one
            sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
            
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            self.orientation = np.array([roll, pitch, yaw])
    
    def velocity_callback(self, msg: TwistStamped):
        """
        Process velocity messages.
        
        Args:
            msg: TwistStamped message
        """
        with self.state_lock:
            self.velocity[0] = msg.twist.linear.x
            self.velocity[1] = msg.twist.linear.y
            self.velocity[2] = msg.twist.linear.z
    
    def imu_callback(self, msg: Imu):
        """
        Process IMU messages.
        
        Args:
            msg: Imu message
        """
        # We're already getting orientation from the pose,
        # but could use IMU for more accurate angular rates if needed
        pass
    
    def control_loop(self):
        """Main control loop."""
        with self.state_lock:
            # Compute errors
            cross_track_error, heading_error = self.error_calculator.compute_errors(
                self.position,
                self.orientation[2]  # Yaw
            )
            
            self.cross_track_error = cross_track_error
            self.heading_error = heading_error
            
            # If supervisor is initialized, use it to adjust PID gains
            if self.supervisor is not None:
                # Create observation (state + errors)
                observation = np.concatenate([
                    self.position,
                    self.velocity,
                    self.orientation,
                    np.array([cross_track_error, heading_error])
                ])
                
                # Get PID gain from supervisor
                gain, _ = self.supervisor.predict(observation)
                
                # Update PID controller with new gain
                self.pid_controller.update_gains(gain.item())
            
            # Compute control using PID
            linear_vel, angular_vel = self.pid_controller.compute_control(
                cross_track_error,
                heading_error
            )
            
            self.linear_velocity = linear_vel
            self.angular_velocity = angular_vel
            
            # Create and publish velocity command
            vel_cmd = Twist()
            vel_cmd.linear.x = linear_vel
            vel_cmd.angular.z = angular_vel
            
            self.vel_pub.publish(vel_cmd)
            
            # Log state (for debugging)
            if self.supervisor is not None:
                self.get_logger().debug(
                    f"Pos: {self.position}, "
                    f"Errors: [{cross_track_error:.2f}, {heading_error:.2f}], "
                    f"Gain: {self.pid_controller.kp:.2f}, "
                    f"Cmd: [{linear_vel:.2f}, {angular_vel:.2f}]"
                )
            else:
                self.get_logger().debug(
                    f"Pos: {self.position}, "
                    f"Errors: [{cross_track_error:.2f}, {heading_error:.2f}], "
                    f"Cmd: [{linear_vel:.2f}, {angular_vel:.2f}]"
                )
    
    def initialize_supervisor(self, model_path: str = None):
        """
        Initialize the RL supervisor agent.
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        import gymnasium as gym
        from gymnasium import spaces
        
        # Create observation space (position, velocity, orientation, errors)
        obs_dim = 3 + 3 + 3 + 2  # pos + vel + orient + errors
        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Create action space (PID gains)
        action_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(1,),  # Just P gain for now
            dtype=np.float32
        )
        
        # Create supervisor agent
        self.supervisor = SupervisorAgent(
            observation_space=observation_space,
            action_space=action_space
        )
        
        # Load pre-trained model if provided
        if model_path is not None:
            import torch
            self.supervisor.load_model_state(torch.load(model_path))
            self.get_logger().info(f"Loaded supervisor model from {model_path}")
        
        self.get_logger().info("Supervisor agent initialized")


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    node = RL_PID_Node()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 