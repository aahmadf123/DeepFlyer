#!/usr/bin/env python3
"""
Wind disturbance simulator.

This script simulates wind disturbances by publishing forces to the drone.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time
import argparse
from geometry_msgs.msg import Vector3, WrenchStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class WindDisturbanceNode(Node):
    """Node to simulate wind disturbances."""
    
    def __init__(self):
        """Initialize the node."""
        super().__init__('wind_disturbance')
        
        # Create QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create publisher for external forces
        self.force_pub = self.create_publisher(
            WrenchStamped,
            '/mavros/external_force',
            qos_profile
        )
        
        # Wind parameters
        self.wind_direction = 0.0  # radians
        self.wind_speed = 0.0  # m/s
        self.wind_gust_frequency = 0.1  # Hz
        self.wind_variability = 0.2  # Fraction of wind speed
        
        # Create timer for wind updates
        self.wind_timer = self.create_timer(0.1, self.publish_wind)
        
        self.get_logger().info('Wind disturbance node initialized')
    
    def set_wind_params(
        self,
        wind_direction: float,
        wind_speed: float,
        gust_frequency: float = 0.1,
        variability: float = 0.2
    ):
        """
        Set wind parameters.
        
        Args:
            wind_direction: Wind direction in radians (0 = east, π/2 = north)
            wind_speed: Wind speed in m/s
            gust_frequency: Frequency of wind gusts in Hz
            variability: Wind variability as fraction of wind speed
        """
        self.wind_direction = wind_direction
        self.wind_speed = wind_speed
        self.wind_gust_frequency = gust_frequency
        self.wind_variability = variability
        
        self.get_logger().info(
            f"Wind params set: direction={wind_direction:.2f} rad, "
            f"speed={wind_speed:.2f} m/s, "
            f"gust_freq={gust_frequency:.2f} Hz, "
            f"variability={variability:.2f}"
        )
    
    def publish_wind(self):
        """Publish wind disturbance force."""
        # Add some variability to wind speed
        wind_variability = np.sin(time.time() * 2 * np.pi * self.wind_gust_frequency)
        current_wind = self.wind_speed * (1.0 + self.wind_variability * wind_variability)
        
        # Compute wind force components
        wind_x = current_wind * np.cos(self.wind_direction)
        wind_y = current_wind * np.sin(self.wind_direction)
        
        # Create force message
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        # Set force components (negative because force is applied to the drone)
        msg.wrench.force.x = -wind_x
        msg.wrench.force.y = -wind_y
        msg.wrench.force.z = 0.0
        
        # Publish force
        self.force_pub.publish(msg)
        
        if abs(wind_variability) > 0.8:  # Log only significant gusts
            self.get_logger().debug(
                f"Wind gust: [{wind_x:.2f}, {wind_y:.2f}, 0.00] N"
            )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Simulate wind disturbances.')
    parser.add_argument('--direction', type=float, default=np.pi/2, help='Wind direction in radians')
    parser.add_argument('--speed', type=float, default=1.0, help='Wind speed in m/s')
    parser.add_argument('--gust-freq', type=float, default=0.1, help='Gust frequency in Hz')
    parser.add_argument('--variability', type=float, default=0.3, help='Wind variability (0-1)')
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    wind_node = WindDisturbanceNode()
    
    # Set wind parameters
    wind_node.set_wind_params(
        args.direction,
        args.speed,
        args.gust_freq,
        args.variability
    )
    
    try:
        # Spin node
        rclpy.spin(wind_node)
    except KeyboardInterrupt:
        pass
    finally:
        wind_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 