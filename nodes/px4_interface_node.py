#!/usr/bin/env python3
"""
PX4 Interface Node for MVP Hoop Navigation

This ROS2 node provides the interface between the RL agent and PX4 flight controller
using PX4-ROS-COM for low-latency communication.

Subscribes to:
- /deepflyer/rl_action (MVP 4D actions from RL agent)
- /fmu/out/vehicle_local_position (drone position/velocity)
- /fmu/out/vehicle_status (flight controller status)

Publishes to:
- /fmu/in/vehicle_command (PX4 commands)
- /fmu/in/offboard_control_mode (control mode)
- /fmu/in/trajectory_setpoint (position/velocity commands)
- /deepflyer/course_state (MVP trajectory state)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import time
import logging
from typing import Optional, Dict, Any
from enum import Enum

# PX4 message imports
from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode, 
    TrajectorySetpoint,
    VehicleLocalPosition,
    VehicleStatus
)

# Standard ROS2 messages
from std_msgs.msg import Header
from geometry_msgs.msg import Point

# Custom message imports
from deepflyer_msgs.msg import RLAction, CourseState

# Import MVP trajectory components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_agent.mvp_trajectory import MVPFlightPhase

logger = logging.getLogger(__name__)


class PX4FlightMode(Enum):
    """PX4 flight modes"""
    MANUAL = 0
    STABILIZED = 1
    ACRO = 2
    RATTITUDE = 3
    ALTITUDE = 4
    POSITION = 5
    OFFBOARD = 6
    READY = 7
    AUTO_TAKEOFF = 8
    AUTO_LAND = 9
    AUTO_RTL = 10


class MVPSafetyLimits:
    """Safety limits for MVP flight"""
    
    def __init__(self):
        # Velocity limits (m/s)
        self.max_velocity_xy = 2.0
        self.max_velocity_z = 1.5
        self.max_yaw_rate = 1.0  # rad/s
        
        # Position limits (relative to takeoff point)
        self.max_horizontal_distance = 10.0  # meters
        self.min_altitude = 0.2              # meters
        self.max_altitude = 5.0              # meters
        
        # Emergency thresholds
        self.emergency_battery_level = 20.0  # percent
        self.max_flight_time = 300.0         # seconds (5 minutes)


class PX4InterfaceNode(Node):
    """
    ROS2 node for PX4-ROS-COM communication with MVP trajectory control
    """
    
    def __init__(self):
        super().__init__('px4_interface_node')
        
        # Initialize parameters
        self.declare_parameter('enable_safety_limits', True)
        self.declare_parameter('takeoff_altitude', 1.5)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('auto_arm', False)
        
        # Get parameters
        self.enable_safety = self.get_parameter('enable_safety_limits').get_parameter_value().bool_value
        self.takeoff_altitude = self.get_parameter('takeoff_altitude').get_parameter_value().double_value
        self.control_freq = self.get_parameter('control_frequency').get_parameter_value().double_value
        self.auto_arm = self.get_parameter('auto_arm').get_parameter_value().bool_value
        
        # Initialize safety limits
        self.safety_limits = MVPSafetyLimits()
        
        # Flight state
        self.is_armed = False
        self.is_offboard = False
        self.current_flight_mode = PX4FlightMode.MANUAL
        self.takeoff_position: Optional[np.ndarray] = None
        self.current_position: Optional[np.ndarray] = None
        self.current_velocity: Optional[np.ndarray] = None
        self.current_yaw = 0.0
        
        # MVP trajectory state
        self.current_phase = MVPFlightPhase.TAKEOFF
        self.phase_start_time = time.time()
        self.flight_start_time = time.time()
        self.hoop_passages = 0
        self.episode_active = False
        
        # Action tracking
        self.latest_rl_action: Optional[RLAction] = None
        self.last_command_time = time.time()
        self.command_timeout = 1.0  # seconds
        
        # Performance tracking
        self.command_count = 0
        self.safety_overrides = 0
        
        # QoS profiles for PX4 communication
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create publishers (to PX4)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            px4_qos
        )
        
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            px4_qos
        )
        
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            px4_qos
        )
        
        # Create publisher for course state
        self.course_state_pub = self.create_publisher(
            CourseState,
            '/deepflyer/course_state',
            10
        )
        
        # Create subscribers (from PX4)
        self.vehicle_local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            px4_qos
        )
        
        self.vehicle_status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            px4_qos
        )
        
        # Create subscriber for RL actions
        self.rl_action_sub = self.create_subscription(
            RLAction,
            '/deepflyer/rl_action',
            self.rl_action_callback,
            10
        )
        
        # Create timers
        self.control_timer = self.create_timer(
            1.0 / self.control_freq,
            self.control_loop
        )
        
        self.offboard_mode_timer = self.create_timer(
            0.1,  # 10Hz offboard mode heartbeat
            self.publish_offboard_control_mode
        )
        
        self.course_state_timer = self.create_timer(
            1.0,  # 1Hz course state updates
            self.publish_course_state
        )
        
        self.get_logger().info("PX4 Interface Node initialized")
        self.get_logger().info(f"Safety limits: {'enabled' if self.enable_safety else 'disabled'}")
        self.get_logger().info(f"Control frequency: {self.control_freq} Hz")
        self.get_logger().info(f"Takeoff altitude: {self.takeoff_altitude} m")
    
    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        """Update drone position and velocity from PX4"""
        self.current_position = np.array([msg.x, msg.y, msg.z])
        self.current_velocity = np.array([msg.vx, msg.vy, msg.vz])
        self.current_yaw = msg.heading
        
        # Set takeoff position on first position update
        if self.takeoff_position is None and msg.z > 0.1:
            self.takeoff_position = self.current_position.copy()
            self.get_logger().info(f"Takeoff position set: {self.takeoff_position}")
    
    def vehicle_status_callback(self, msg: VehicleStatus):
        """Update flight controller status from PX4"""
        self.is_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        
        # Update flight mode
        if msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self.current_flight_mode = PX4FlightMode.OFFBOARD
            self.is_offboard = True
        else:
            self.is_offboard = False
            
        # Log status changes
        if hasattr(self, '_last_armed_state') and self._last_armed_state != self.is_armed:
            self.get_logger().info(f"Armed state changed: {self.is_armed}")
        self._last_armed_state = self.is_armed
    
    def rl_action_callback(self, msg: RLAction):
        """Receive actions from RL agent"""
        self.latest_rl_action = msg
        self.last_command_time = time.time()
        
        # Start episode if not active
        if not self.episode_active:
            self.start_episode()
    
    def start_episode(self):
        """Start a new MVP episode"""
        self.episode_active = True
        self.current_phase = MVPFlightPhase.TAKEOFF
        self.phase_start_time = time.time()
        self.flight_start_time = time.time()
        self.hoop_passages = 0
        
        self.get_logger().info("Starting new MVP episode")
        
        # Arm and switch to offboard if enabled
        if self.auto_arm and not self.is_armed:
            self.arm_vehicle()
    
    def control_loop(self):
        """Main control loop - execute RL actions"""
        if not self.episode_active or self.latest_rl_action is None:
            return
        
        # Check for command timeout
        if time.time() - self.last_command_time > self.command_timeout:
            self.get_logger().warn("RL action timeout - stopping")
            self.send_stop_command()
            return
        
        # Process RL action
        self.process_rl_action(self.latest_rl_action)
        
        # Update MVP trajectory phase
        self.update_mvp_phase()
        
        self.command_count += 1
    
    def process_rl_action(self, action_msg: RLAction):
        """Process and execute RL action with safety checks"""
        try:
            # Extract 4D action
            vx_cmd = action_msg.vx_cmd
            vy_cmd = action_msg.vy_cmd  
            vz_cmd = action_msg.vz_cmd
            yaw_rate_cmd = action_msg.yaw_rate_cmd
            
            # Apply safety constraints if enabled
            if self.enable_safety:
                vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd = self.apply_safety_constraints(
                    vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd
                )
                
                # Check if safety override was applied
                if (abs(vx_cmd - action_msg.vx_cmd) > 0.01 or
                    abs(vy_cmd - action_msg.vy_cmd) > 0.01 or 
                    abs(vz_cmd - action_msg.vz_cmd) > 0.01 or
                    abs(yaw_rate_cmd - action_msg.yaw_rate_cmd) > 0.01):
                    self.safety_overrides += 1
            
            # Send trajectory setpoint to PX4
            self.send_velocity_command(vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd)
            
        except Exception as e:
            self.get_logger().error(f"Error processing RL action: {e}")
    
    def apply_safety_constraints(self, vx: float, vy: float, vz: float, yaw_rate: float) -> tuple:
        """Apply safety constraints to velocity commands"""
        # Velocity magnitude limits
        vx = np.clip(vx, -self.safety_limits.max_velocity_xy, self.safety_limits.max_velocity_xy)
        vy = np.clip(vy, -self.safety_limits.max_velocity_xy, self.safety_limits.max_velocity_xy)
        vz = np.clip(vz, -self.safety_limits.max_velocity_z, self.safety_limits.max_velocity_z)
        yaw_rate = np.clip(yaw_rate, -self.safety_limits.max_yaw_rate, self.safety_limits.max_yaw_rate)
        
        # Position-based constraints
        if self.current_position is not None and self.takeoff_position is not None:
            # Horizontal distance limit
            horizontal_distance = np.linalg.norm(self.current_position[:2] - self.takeoff_position[:2])
            if horizontal_distance > self.safety_limits.max_horizontal_distance:
                # Force movement back toward origin
                direction_home = self.takeoff_position[:2] - self.current_position[:2]
                direction_home = direction_home / (np.linalg.norm(direction_home) + 1e-6)
                vx = direction_home[0] * 0.5
                vy = direction_home[1] * 0.5
            
            # Altitude limits
            altitude = self.current_position[2]
            if altitude < self.safety_limits.min_altitude and vz < 0:
                vz = max(vz, 0.0)  # Prevent further descent
            elif altitude > self.safety_limits.max_altitude and vz > 0:
                vz = min(vz, 0.0)  # Prevent further ascent
        
        return vx, vy, vz, yaw_rate
    
    def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """Send velocity command to PX4"""
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # Set velocity commands (NED frame)
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.yawspeed = float(yaw_rate)
        
        # Use NaN for position (velocity control mode)
        msg.position = [float('nan')] * 3
        msg.acceleration = [float('nan')] * 3
        msg.jerk = [float('nan')] * 3
        msg.yaw = float('nan')
        
        self.trajectory_setpoint_pub.publish(msg)
    
    def send_stop_command(self):
        """Send stop command (zero velocity)"""
        self.send_velocity_command(0.0, 0.0, 0.0, 0.0)
    
    def publish_offboard_control_mode(self):
        """Publish offboard control mode (required for PX4 offboard)"""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # Enable velocity control
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        
        self.offboard_control_mode_pub.publish(msg)
    
    def arm_vehicle(self):
        """Send arm command to PX4"""
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0  # 1.0 to arm, 0.0 to disarm
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        
        self.vehicle_command_pub.publish(msg)
        self.get_logger().info("Arm command sent")
    
    def disarm_vehicle(self):
        """Send disarm command to PX4"""
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 0.0  # 0.0 to disarm
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        
        self.vehicle_command_pub.publish(msg)
        self.get_logger().info("Disarm command sent")
    
    def update_mvp_phase(self):
        """Update MVP flight phase based on current state"""
        if self.current_position is None:
            return
        
        current_time = time.time()
        altitude = self.current_position[2]
        
        # Simple phase progression logic
        if self.current_phase == MVPFlightPhase.TAKEOFF:
            if altitude >= self.takeoff_altitude - 0.2:
                self.current_phase = MVPFlightPhase.SCAN_360
                self.phase_start_time = current_time
                self.get_logger().info("Phase: TAKEOFF -> SCAN_360")
        
        # Additional phase transitions would be handled by the RL agent
        # or vision system based on hoop detection and navigation progress
    
    def publish_course_state(self):
        """Publish MVP course state information"""
        msg = CourseState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        # MVP Flight Phase Information
        msg.current_phase = self.current_phase.value
        phase_duration = time.time() - self.phase_start_time
        msg.phase_duration = phase_duration
        msg.total_flight_time = time.time() - self.flight_start_time
        
        # Hoop Information
        msg.hoop_detected = False  # Would be updated by vision system
        msg.hoop_passages_completed = self.hoop_passages
        
        # Navigation State
        if self.current_position is not None:
            msg.drone_position = Point(
                x=self.current_position[0],
                y=self.current_position[1], 
                z=self.current_position[2]
            )
            msg.altitude = self.current_position[2]
        
        if self.takeoff_position is not None:
            msg.takeoff_position = Point(
                x=self.takeoff_position[0],
                y=self.takeoff_position[1],
                z=self.takeoff_position[2]
            )
            
            if self.current_position is not None:
                msg.distance_to_origin = float(np.linalg.norm(
                    self.current_position - self.takeoff_position
                ))
        
        # Episode Management
        msg.episode_active = self.episode_active
        msg.episode_time = time.time() - self.flight_start_time if self.episode_active else 0.0
        
        # Safety and Status
        msg.safety_override_active = self.enable_safety
        msg.status_message = f"Armed: {self.is_armed}, Offboard: {self.is_offboard}"
        msg.error_code = 0 if self.is_armed and self.is_offboard else 1
        
        self.course_state_pub.publish(msg)
    
    def emergency_land(self):
        """Trigger emergency landing"""
        self.get_logger().warn("Emergency landing triggered")
        
        # Send land command
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = VehicleCommand.VEHICLE_CMD_NAV_LAND
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        
        self.vehicle_command_pub.publish(msg)
        
        # End episode
        self.episode_active = False


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        px4_interface = PX4InterfaceNode()
        
        px4_interface.get_logger().info("PX4 Interface Node started")
        
        # Spin the node
        rclpy.spin(px4_interface)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in PX4 interface node: {e}")
    finally:
        if 'px4_interface' in locals():
            px4_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 