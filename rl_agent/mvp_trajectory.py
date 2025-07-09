#!/usr/bin/env python3
"""
MVP Flight Trajectory Implementation
Takeoff -> 360° Scan -> Through Hoop -> Return -> Land
"""

import numpy as np
import time
from enum import Enum
from typing import Optional, Tuple
import logging

# ROS2 imports
import rclpy
from rclpy.node import Node
from px4_msgs.msg import TrajectorySetpoint, VehicleCommand, OffboardControlMode
from deepflyer_msgs.msg import VisionFeatures, DroneState

logger = logging.getLogger(__name__)


class FlightPhase(Enum):
    """MVP flight phases"""
    IDLE = 0
    TAKEOFF = 1
    SCANNING = 2
    APPROACH = 3
    THROUGH_HOOP = 4
    RETURN = 5
    LANDING = 6
    COMPLETE = 7


class MVPTrajectoryController(Node):
    """
    Implements the MVP flight trajectory:
    1. Takeoff to 0.8m
    2. 360° yaw scan to detect hoops
    3. Navigate through detected hoop
    4. Turn around and return through same hoop
    5. Land at origin
    """
    
    def __init__(self):
        super().__init__('mvp_trajectory_controller')
        
        # Flight parameters
        self.takeoff_altitude = 0.8  # meters
        self.scan_yaw_rate = 0.5  # rad/s
        self.approach_speed = 0.3  # m/s
        self.through_speed = 0.5  # m/s
        
        # State tracking
        self.phase = FlightPhase.IDLE
        self.start_time = None
        self.phase_start_time = None
        self.initial_yaw = 0.0
        self.hoop_detected = False
        self.hoop_position: Optional[Tuple[float, float, float]] = None
        self.passed_through = False
        
        # Current state
        self.current_position = np.zeros(3)
        self.current_yaw = 0.0
        self.vision_features: Optional[VisionFeatures] = None
        
        # Setup publishers
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        
        # Setup subscribers
        self.create_subscription(
            DroneState, '/deepflyer/drone_state',
            self._drone_state_callback, 10)
        self.create_subscription(
            VisionFeatures, '/deepflyer/vision_features',
            self._vision_callback, 10)
        
        # Control timer (20Hz)
        self.control_timer = self.create_timer(0.05, self._control_loop)
        
        # Offboard mode timer (2Hz)
        self.offboard_timer = self.create_timer(0.5, self._send_offboard_mode)
        
        self.get_logger().info("MVP Trajectory Controller initialized")
    
    def start_mission(self):
        """Start the MVP mission"""
        self.phase = FlightPhase.TAKEOFF
        self.start_time = time.time()
        self.phase_start_time = time.time()
        self.get_logger().info("Starting MVP mission")
        
        # Send arm command
        self._send_arm_command(True)
    
    def _drone_state_callback(self, msg: DroneState):
        """Update current drone state"""
        self.current_position = np.array([
            msg.position.x,
            msg.position.y, 
            msg.position.z
        ])
        self.current_yaw = msg.euler_angles.z
    
    def _vision_callback(self, msg: VisionFeatures):
        """Process vision features"""
        self.vision_features = msg
        
        # Detect hoop during scanning phase
        if self.phase == FlightPhase.SCANNING and msg.hoop_detected:
            if not self.hoop_detected:
                self.hoop_detected = True
                # Estimate hoop position based on current drone position and vision
                distance = msg.hoop_distance
                angle = self.current_yaw + np.arctan2(msg.hoop_center_x, 1.0)
                self.hoop_position = (
                    self.current_position[0] + distance * np.cos(angle),
                    self.current_position[1] + distance * np.sin(angle),
                    self.takeoff_altitude  # Assume hoop at same altitude
                )
                self.get_logger().info(f"Hoop detected at position: {self.hoop_position}")
    
    def _control_loop(self):
        """Main control loop implementing MVP trajectory"""
        if self.phase == FlightPhase.IDLE:
            return
        
        # Get time in current phase
        phase_time = time.time() - self.phase_start_time
        
        # Phase-specific control
        if self.phase == FlightPhase.TAKEOFF:
            self._handle_takeoff(phase_time)
        
        elif self.phase == FlightPhase.SCANNING:
            self._handle_scanning(phase_time)
        
        elif self.phase == FlightPhase.APPROACH:
            self._handle_approach()
        
        elif self.phase == FlightPhase.THROUGH_HOOP:
            self._handle_through_hoop()
        
        elif self.phase == FlightPhase.RETURN:
            self._handle_return()
        
        elif self.phase == FlightPhase.LANDING:
            self._handle_landing()
    
    def _handle_takeoff(self, phase_time: float):
        """Handle takeoff phase"""
        # Vertical velocity to reach target altitude
        if self.current_position[2] < self.takeoff_altitude - 0.1:
            # Ascend
            self._send_velocity_command(0, 0, 0.3, 0)
        else:
            # Reached altitude, start scanning
            self.get_logger().info("Takeoff complete, starting 360° scan")
            self._transition_to_phase(FlightPhase.SCANNING)
            self.initial_yaw = self.current_yaw
    
    def _handle_scanning(self, phase_time: float):
        """Handle 360° scanning phase"""
        # Rotate in place
        self._send_velocity_command(0, 0, 0, self.scan_yaw_rate)
        
        # Check if we've completed 360°
        yaw_diff = self.current_yaw - self.initial_yaw
        if abs(yaw_diff) > 2 * np.pi or (self.hoop_detected and phase_time > 2.0):
            if self.hoop_detected:
                self.get_logger().info("Scan complete, hoop found, starting approach")
                self._transition_to_phase(FlightPhase.APPROACH)
            else:
                self.get_logger().warning("Scan complete, no hoop detected, landing")
                self._transition_to_phase(FlightPhase.LANDING)
    
    def _handle_approach(self):
        """Handle approach to hoop phase"""
        if not self.hoop_position or not self.vision_features:
            return
        
        # Use vision feedback for approach
        if self.vision_features.hoop_detected:
            # Visual servoing approach
            lateral_error = self.vision_features.hoop_center_x
            vertical_error = self.vision_features.hoop_center_y
            
            # Proportional control
            lateral_vel = -lateral_error * 0.5
            vertical_vel = -vertical_error * 0.3
            forward_vel = self.approach_speed
            
            # Reduce speed when close
            if self.vision_features.hoop_distance < 1.0:
                forward_vel *= 0.5
            
            self._send_velocity_command(forward_vel, lateral_vel, vertical_vel, 0)
            
            # Check if aligned and close enough to go through
            if (abs(lateral_error) < 0.1 and 
                abs(vertical_error) < 0.1 and 
                self.vision_features.hoop_distance < 0.5):
                self.get_logger().info("Aligned with hoop, going through")
                self._transition_to_phase(FlightPhase.THROUGH_HOOP)
        else:
            # Lost visual, stop
            self._send_velocity_command(0, 0, 0, 0)
    
    def _handle_through_hoop(self):
        """Handle flying through the hoop"""
        # Maintain forward velocity
        self._send_velocity_command(self.through_speed, 0, 0, 0)
        
        # Check if we've passed through (hoop no longer visible or behind us)
        if not self.vision_features.hoop_detected or time.time() - self.phase_start_time > 3.0:
            self.get_logger().info("Passed through hoop, turning around")
            self.passed_through = True
            self._transition_to_phase(FlightPhase.RETURN)
    
    def _handle_return(self):
        """Handle return through hoop phase"""
        phase_time = time.time() - self.phase_start_time
        
        if phase_time < 2.0:
            # Turn around 180°
            self._send_velocity_command(0, 0, 0, 1.0)
        else:
            # Approach hoop again
            if self.vision_features and self.vision_features.hoop_detected:
                # Same approach logic
                lateral_error = self.vision_features.hoop_center_x
                forward_vel = self.approach_speed
                lateral_vel = -lateral_error * 0.5
                
                self._send_velocity_command(forward_vel, lateral_vel, 0, 0)
                
                # Check if passed through again
                if phase_time > 5.0:
                    self.get_logger().info("Return complete, landing")
                    self._transition_to_phase(FlightPhase.LANDING)
            else:
                # Search for hoop
                self._send_velocity_command(0, 0, 0, 0.3)
    
    def _handle_landing(self):
        """Handle landing phase"""
        # Descend at origin
        distance_to_origin = np.linalg.norm(self.current_position[:2])
        
        if distance_to_origin > 0.3:
            # Navigate back to origin first
            direction = -self.current_position[:2] / (distance_to_origin + 0.001)
            self._send_velocity_command(
                direction[0] * 0.3,
                direction[1] * 0.3,
                0, 0
            )
        else:
            # Descend
            if self.current_position[2] > 0.2:
                self._send_velocity_command(0, 0, -0.2, 0)
            else:
                # Landed
                self._send_velocity_command(0, 0, 0, 0)
                self._send_arm_command(False)
                self.get_logger().info("Landing complete, mission finished")
                self._transition_to_phase(FlightPhase.COMPLETE)
    
    def _send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """Send velocity command to PX4"""
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.velocity = [vx, vy, vz]
        msg.yawspeed = yaw_rate
        
        # NaN for position (velocity control)
        msg.position = [float('nan')] * 3
        
        self.trajectory_pub.publish(msg)
    
    def _send_offboard_mode(self):
        """Send offboard control mode"""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        
        self.offboard_mode_pub.publish(msg)
    
    def _send_arm_command(self, arm: bool):
        """Send arm/disarm command"""
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = 400  # VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = float(arm)
        msg.target_system = 1
        msg.target_component = 1
        
        self.vehicle_command_pub.publish(msg)
    
    def _transition_to_phase(self, new_phase: FlightPhase):
        """Transition to new flight phase"""
        self.phase = new_phase
        self.phase_start_time = time.time()
        self.get_logger().info(f"Transitioning to phase: {new_phase.name}")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    controller = MVPTrajectoryController()
    
    # Start mission after 2 seconds
    timer = controller.create_timer(2.0, lambda: controller.start_mission())
    timer.cancel()  # One-shot timer
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 