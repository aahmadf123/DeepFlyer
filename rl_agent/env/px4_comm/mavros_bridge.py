"""
MAVROS Bridge for Traditional PX4 Communication
Provides MAVROS-based interface as alternative to PX4-ROS-COM
Backup for PX4-ROS-COM
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from mavros_msgs.msg import State
    from mavros_msgs.srv import CommandBool, SetMode
    MAVROS_AVAILABLE = True
except ImportError:
    MAVROS_AVAILABLE = False
    logging.warning("MAVROS not available")

logger = logging.getLogger(__name__)


class MAVROSBridge:
    """MAVROS-based communication bridge for PX4"""
    
    def __init__(self, node: Node):
        self.node = node
        self.current_state = None
        self.current_pose = None
        self.connected = False
        
        if not MAVROS_AVAILABLE:
            raise RuntimeError("MAVROS not available")
        
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()
        
        logger.info("MAVROSBridge initialized")
    
    def _setup_publishers(self) -> None:
        """Setup MAVROS publishers"""
        self.setpoint_pub = self.node.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        
        self.pose_pub = self.node.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 10)
    
    def _setup_subscribers(self) -> None:
        """Setup MAVROS subscribers"""
        self.node.create_subscription(
            State, '/mavros/state', self._state_callback, 10)
        
        self.node.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._pose_callback, 10)
    
    def _setup_services(self) -> None:
        """Setup MAVROS service clients"""
        self.arm_client = self.node.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_client = self.node.create_client(SetMode, '/mavros/set_mode')
    
    def _state_callback(self, msg: State) -> None:
        """Process MAVROS state updates"""
        self.current_state = msg
        self.connected = msg.connected
    
    def _pose_callback(self, msg: PoseStamped) -> None:
        """Process pose updates"""
        self.current_pose = msg
    
    def send_velocity_command(self, velocity: np.ndarray, yaw_rate: float = 0.0) -> None:
        """Send velocity command via MAVROS"""
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        
        msg.twist.linear.x = velocity[0]
        msg.twist.linear.y = velocity[1]
        msg.twist.linear.z = velocity[2]
        msg.twist.angular.z = yaw_rate
        
        self.setpoint_pub.publish(msg)
    
    def arm(self) -> bool:
        """Arm the vehicle"""
        if not self.arm_client.wait_for_service(timeout_sec=2.0):
            logger.error("Arm service not available")
            return False
        
        request = CommandBool.Request()
        request.value = True
        
        future = self.arm_client.call_async(request)
        # Note: In production, should use proper async handling
        return True
    
    def set_mode(self, mode: str) -> bool:
        """Set flight mode"""
        if not self.mode_client.wait_for_service(timeout_sec=2.0):
            logger.error("Mode service not available")
            return False
        
        request = SetMode.Request()
        request.custom_mode = mode
        
        future = self.mode_client.call_async(request)
        return True
    
    def get_position(self) -> Optional[np.ndarray]:
        """Get current position"""
        if self.current_pose is None:
            return None
        
        return np.array([
            self.current_pose.pose.position.x,
            self.current_pose.pose.position.y,
            self.current_pose.pose.position.z
        ])
    
    def is_armed(self) -> bool:
        """Check if vehicle is armed"""
        return self.current_state is not None and self.current_state.armed
    
    def is_connected(self) -> bool:
        """Check if connected to flight controller"""
        return self.connected 