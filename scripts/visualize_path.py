#!/usr/bin/env python3
"""
Path visualization script.

This script visualizes the drone's path and errors during path following.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from threading import Thread, Lock
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import argparse
from typing import List, Tuple


class PathVisualizerNode(Node):
    """Node for visualizing drone path and errors."""
    
    def __init__(self, path_origin: List[float] = [0, 0, 1], path_target: List[float] = [10, 0, 1]):
        """
        Initialize the node.
        
        Args:
            path_origin: Origin point of the path [x, y, z]
            path_target: Target point of the path [x, y, z]
        """
        super().__init__('path_visualizer')
        
        # Create QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_callback,
            qos_profile
        )
        
        # For custom PID gain and error messages (will be published by our controller)
        self.metrics_sub = self.create_subscription(
            Float32MultiArray,
            '/rl_pid_supervisor/metrics',
            self.metrics_callback,
            qos_profile
        )
        
        # Set up data structures
        self.path_origin = np.array(path_origin)
        self.path_target = np.array(path_target)
        self.position_history = []
        self.error_history = []
        self.pid_gain_history = []
        self.timestamp_history = []
        
        # Current position
        self.current_position = np.zeros(3)
        
        # Current metrics
        self.current_cross_track_error = 0.0
        self.current_heading_error = 0.0
        self.current_pid_gain = 1.0
        
        # Thread synchronization
        self.data_lock = Lock()
        
        # Data collection timestamp
        self.start_time = time.time()
        
        # Visualization
        self.fig = None
        self.ani = None
        
        self.get_logger().info('Path visualizer initialized')
    
    def pose_callback(self, msg: PoseStamped):
        """
        Process pose messages.
        
        Args:
            msg: PoseStamped message
        """
        with self.data_lock:
            # Update current position
            self.current_position[0] = msg.pose.position.x
            self.current_position[1] = msg.pose.position.y
            self.current_position[2] = msg.pose.position.z
            
            # Add to history
            self.position_history.append(self.current_position.copy())
            self.timestamp_history.append(time.time() - self.start_time)
            
            # Add current metrics to history
            self.error_history.append([self.current_cross_track_error, self.current_heading_error])
            self.pid_gain_history.append(self.current_pid_gain)
    
    def metrics_callback(self, msg: Float32MultiArray):
        """
        Process metrics messages.
        
        Args:
            msg: Float32MultiArray message with [cross_track_error, heading_error, pid_gain]
        """
        if len(msg.data) >= 3:
            with self.data_lock:
                self.current_cross_track_error = msg.data[0]
                self.current_heading_error = msg.data[1]
                self.current_pid_gain = msg.data[2]
    
    def visualize(self):
        """Visualize the path and errors."""
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.canvas.manager.set_window_title('Drone Path Following Visualization')
        
        # Create layout
        gs = self.fig.add_gridspec(3, 2)
        
        # Create subplots
        self.ax_path = self.fig.add_subplot(gs[0:2, 0])
        self.ax_error = self.fig.add_subplot(gs[0, 1])
        self.ax_heading = self.fig.add_subplot(gs[1, 1])
        self.ax_pid = self.fig.add_subplot(gs[2, :])
        
        # Set titles
        self.ax_path.set_title('Drone Path')
        self.ax_error.set_title('Cross-Track Error')
        self.ax_heading.set_title('Heading Error')
        self.ax_pid.set_title('PID Gain (P)')
        
        # Set labels
        self.ax_path.set_xlabel('X position (m)')
        self.ax_path.set_ylabel('Y position (m)')
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Error (m)')
        self.ax_heading.set_xlabel('Time (s)')
        self.ax_heading.set_ylabel('Error (rad)')
        self.ax_pid.set_xlabel('Time (s)')
        self.ax_pid.set_ylabel('Gain')
        
        # Plot initial path line
        path_xs = [self.path_origin[0], self.path_target[0]]
        path_ys = [self.path_origin[1], self.path_target[1]]
        self.path_line, = self.ax_path.plot(path_xs, path_ys, 'k--', label='Target Path')
        
        # Initial empty plots
        self.pos_line, = self.ax_path.plot([], [], 'b-', label='Drone Path')
        self.current_pos, = self.ax_path.plot([], [], 'ro')
        
        self.cross_track_line, = self.ax_error.plot([], [], 'r-')
        self.heading_line, = self.ax_heading.plot([], [], 'g-')
        self.pid_line, = self.ax_pid.plot([], [], 'b-')
        
        # Add legends
        self.ax_path.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Create animation
        self.ani = FuncAnimation(
            self.fig, self.update_plot, interval=100,
            blit=False, cache_frame_data=False
        )
        
        plt.show()
    
    def update_plot(self, frame):
        """
        Update the plot with new data.
        
        Args:
            frame: Animation frame number
        """
        with self.data_lock:
            if not self.position_history:
                return self.pos_line, self.current_pos, self.cross_track_line, self.heading_line, self.pid_line
            
            # Extract positions
            positions = np.array(self.position_history)
            timestamps = np.array(self.timestamp_history)
            errors = np.array(self.error_history)
            pid_gains = np.array(self.pid_gain_history)
            
            # Update path plot
            self.pos_line.set_data(positions[:, 0], positions[:, 1])
            self.current_pos.set_data([positions[-1, 0]], [positions[-1, 1]])
            
            # Update error plots
            self.cross_track_line.set_data(timestamps, errors[:, 0])
            self.heading_line.set_data(timestamps, errors[:, 1])
            
            # Update PID gain plot
            self.pid_line.set_data(timestamps, pid_gains)
            
            # Adjust axis limits
            self.ax_path.relim()
            self.ax_path.autoscale_view()
            self.ax_error.relim()
            self.ax_error.autoscale_view()
            self.ax_heading.relim()
            self.ax_heading.autoscale_view()
            self.ax_pid.relim()
            self.ax_pid.autoscale_view()
            
            return self.pos_line, self.current_pos, self.cross_track_line, self.heading_line, self.pid_line


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize drone path following.')
    parser.add_argument('--origin', type=float, nargs=3, default=[0, 0, 1.5], help='Path origin [x y z]')
    parser.add_argument('--target', type=float, nargs=3, default=[10, 0, 1.5], help='Path target [x y z]')
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    visualizer = PathVisualizerNode(args.origin, args.target)
    
    # Create thread for ROS2 spinning
    ros_thread = Thread(target=lambda: rclpy.spin(visualizer))
    ros_thread.daemon = True
    ros_thread.start()
    
    try:
        # Start visualization
        visualizer.visualize()
    except KeyboardInterrupt:
        pass
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == '__main__':
    main() 