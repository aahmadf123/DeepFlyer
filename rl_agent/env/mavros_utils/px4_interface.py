"""
PX4 Interface for Direct Communication
Handles PX4-ROS-COM messages and flight controller interaction
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

try:
    import rclpy
    from rclpy.node import Node
    from px4_msgs.msg import (
        VehicleLocalPosition, VehicleAttitude, VehicleStatus,
        TrajectorySetpoint, OffboardControlMode, VehicleCommand
    )
    PX4_MSGS_AVAILABLE = True
except ImportError:
    PX4_MSGS_AVAILABLE = False
    logging.warning("PX4 messages not available")

logger = logging.getLogger(__name__)


@dataclass
class PX4State:
    """Container for PX4 flight controller state"""
    position: np.ndarray = None
    velocity: np.ndarray = None
    attitude: np.ndarray = None
    armed: bool = False
    offboard_mode: bool = False
    battery_level: float = 1.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.velocity is None:
            self.velocity = np.zeros(3)
        if self.attitude is None:
            self.attitude = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion


class PX4Interface:
    """Direct interface to PX4 flight controller via PX4-ROS-COM"""
    
    def __init__(self, node: Node):
        self.node = node
        self.state = PX4State()
        self.connected = False
        
        if not PX4_MSGS_AVAILABLE:
            raise RuntimeError("PX4 messages not available. Install px4_msgs package.")
        
        self._setup_publishers()
        self._setup_subscribers()
        
        logger.info("PX4Interface initialized")
    
    def _setup_publishers(self) -> None:
        """Setup PX4 command publishers"""
        self.trajectory_pub = self.node.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        
        self.offboard_mode_pub = self.node.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        
        self.vehicle_command_pub = self.node.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
    
    def _setup_subscribers(self) -> None:
        """Setup PX4 state subscribers"""
        self.node.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position',
            self._position_callback, 10)
        
        self.node.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude',
            self._attitude_callback, 10)
        
        self.node.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1',
            self._status_callback, 10)
    
    def _position_callback(self, msg: VehicleLocalPosition) -> None:
        """Process position updates from PX4"""
        self.state.position = np.array([msg.x, msg.y, -msg.z])  # Convert NED to ENU
        self.state.velocity = np.array([msg.vx, msg.vy, -msg.vz])
        self.state.timestamp = time.time()
        self.connected = True
    
    def _attitude_callback(self, msg: VehicleAttitude) -> None:
        """Process attitude updates from PX4"""
        self.state.attitude = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]])
    
    def _status_callback(self, msg: VehicleStatus) -> None:
        """Process status updates from PX4"""
        self.state.armed = (msg.arming_state == 2)  # ARMING_STATE_ARMED
        self.state.offboard_mode = (msg.nav_state == 14)  # NAVIGATION_STATE_OFFBOARD
    
    def send_velocity_command(self, velocity: np.ndarray, yaw_rate: float = 0.0) -> None:
        """Send velocity command to PX4"""
        msg = TrajectorySetpoint()
        msg.timestamp = int(time.time() * 1e6)  # PX4 timestamp format
        
        # Set velocity (convert ENU to NED)
        msg.velocity[0] = velocity[0]    # North
        msg.velocity[1] = velocity[1]    # East  
        msg.velocity[2] = -velocity[2]   # Down (negative in NED)
        
        # Set position to NaN (velocity control)
        msg.position[0] = float('nan')
        msg.position[1] = float('nan')
        msg.position[2] = float('nan')
        
        msg.yawspeed = yaw_rate
        
        self.trajectory_pub.publish(msg)
    
    def send_offboard_mode(self) -> None:
        """Enable offboard control mode"""
        msg = OffboardControlMode()
        msg.timestamp = int(time.time() * 1e6)
        msg.position = True
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        
        self.offboard_mode_pub.publish(msg)
    
    def arm(self) -> None:
        """Send arm command"""
        self._send_vehicle_command(400, 1.0)  # MAV_CMD_COMPONENT_ARM_DISARM
    
    def disarm(self) -> None:
        """Send disarm command"""
        self._send_vehicle_command(400, 0.0)  # MAV_CMD_COMPONENT_ARM_DISARM
    
    def _send_vehicle_command(self, command: int, param1: float = 0.0) -> None:
        """Send vehicle command"""
        msg = VehicleCommand()
        msg.timestamp = int(time.time() * 1e6)
        msg.command = command
        msg.param1 = param1
        msg.target_system = 1
        msg.target_component = 1
        
        self.vehicle_command_pub.publish(msg)
    
    def get_state(self) -> PX4State:
        """Get current PX4 state"""
        return self.state
    
    def is_connected(self) -> bool:
        """Check if connected to PX4"""
        return self.connected and (time.time() - self.state.timestamp < 1.0) 