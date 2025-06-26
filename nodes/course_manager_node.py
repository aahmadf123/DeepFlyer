#!/usr/bin/env python3
"""
Course Manager ROS2 Node
Manages course navigation state and hoop completion tracking
"""

import numpy as np
import time
from typing import Dict, Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

try:
    from deepflyer_msgs.msg import CourseState, DroneState
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False

from rl_agent.config import DeepFlyerConfig, get_course_layout


class CourseManagerNode(Node):
    """ROS2 node for course management and hoop tracking"""
    
    def __init__(self):
        super().__init__('course_manager_node')
        
        # Parameters
        self.declare_parameter('spawn_x', 0.0)
        self.declare_parameter('spawn_y', 0.0)
        self.declare_parameter('spawn_z', 0.8)
        self.declare_parameter('hoop_completion_radius', 0.4)
        self.declare_parameter('publish_frequency', 10.0)
        
        # Get parameters
        spawn_x = self.get_parameter('spawn_x').value
        spawn_y = self.get_parameter('spawn_y').value
        spawn_z = self.get_parameter('spawn_z').value
        self.spawn_position = (spawn_x, spawn_y, spawn_z)
        self.hoop_completion_radius = self.get_parameter('hoop_completion_radius').value
        self.publish_frequency = self.get_parameter('publish_frequency').value
        
        # Load configuration
        self.config = DeepFlyerConfig()
        
        # Generate course layout
        self.course_hoops = get_course_layout(self.spawn_position)
        
        # Course state
        self.current_target_hoop = 0
        self.current_lap = 1
        self.hoops_completed_this_lap = 0
        self.total_hoops_completed = 0
        self.episode_id = int(time.time())
        self.episode_start_time = time.time()
        self.episode_step_count = 0
        
        # Tracking state
        self.current_drone_state: Optional[DroneState] = None
        self.last_hoop_completion_time = 0.0
        self.hoop_completion_times = []
        
        # Setup ROS interface
        self._setup_ros_interface()
        
        # Start publishing timer
        pub_period = 1.0 / self.publish_frequency
        self.pub_timer = self.create_timer(pub_period, self.publish_course_state)
        
        self.get_logger().info(f"Course manager started with {len(self.course_hoops)} hoops")
    
    def _setup_ros_interface(self):
        """Setup ROS publishers and subscribers"""
        if not CUSTOM_MSGS_AVAILABLE:
            self.get_logger().error("Custom messages not available")
            return
        
        # QoS profile
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers
        self.course_state_pub = self.create_publisher(
            CourseState, '/deepflyer/course_state', reliable_qos)
        
        # Subscribers
        self.drone_state_sub = self.create_subscription(
            DroneState, '/deepflyer/drone_state', self.drone_state_callback, reliable_qos)
    
    def drone_state_callback(self, msg: DroneState):
        """Update drone state and check hoop completion"""
        self.current_drone_state = msg
        self.episode_step_count += 1
        
        # Check for hoop completion
        self._check_hoop_completion()
    
    def _check_hoop_completion(self):
        """Check if drone has completed current target hoop"""
        if not self.current_drone_state or self.current_target_hoop >= len(self.course_hoops):
            return
        
        # Get current drone position
        drone_pos = np.array([
            self.current_drone_state.position.x,
            self.current_drone_state.position.y,
            self.current_drone_state.position.z
        ])
        
        # Get target hoop position
        target_hoop = self.course_hoops[self.current_target_hoop]
        hoop_pos = np.array(target_hoop['position'])
        
        # Check distance to hoop center
        distance_to_hoop = np.linalg.norm(drone_pos - hoop_pos)
        
        if distance_to_hoop < self.hoop_completion_radius:
            self._complete_hoop()
    
    def _complete_hoop(self):
        """Mark current hoop as completed and advance to next"""
        completion_time = time.time()
        
        self.get_logger().info(f"Completed hoop {self.current_target_hoop + 1}")
        
        # Update completion tracking
        self.hoops_completed_this_lap += 1
        self.total_hoops_completed += 1
        self.hoop_completion_times.append(completion_time - self.last_hoop_completion_time)
        self.last_hoop_completion_time = completion_time
        
        # Advance to next hoop
        self.current_target_hoop += 1
        
        # Check lap completion
        if self.current_target_hoop >= len(self.course_hoops):
            self._complete_lap()
    
    def _complete_lap(self):
        """Complete current lap and check course completion"""
        self.get_logger().info(f"Completed lap {self.current_lap}")
        
        self.current_lap += 1
        self.current_target_hoop = 0  # Reset to first hoop
        self.hoops_completed_this_lap = 0
        
        # Check course completion
        if self.current_lap > self.config.HOOP_CONFIG['num_laps']:
            self._complete_course()
    
    def _complete_course(self):
        """Complete the entire course"""
        course_time = time.time() - self.episode_start_time
        self.get_logger().info(f"Course completed in {course_time:.2f} seconds!")
        
        # Reset for next episode (could be triggered externally)
        self._reset_episode()
    
    def _reset_episode(self):
        """Reset episode state"""
        self.current_target_hoop = 0
        self.current_lap = 1
        self.hoops_completed_this_lap = 0
        self.total_hoops_completed = 0
        self.episode_id = int(time.time())
        self.episode_start_time = time.time()
        self.episode_step_count = 0
        self.hoop_completion_times.clear()
        
        self.get_logger().info("Episode reset")
    
    def publish_course_state(self):
        """Publish current course state"""
        if not CUSTOM_MSGS_AVAILABLE or not self.current_drone_state:
            return
        
        msg = CourseState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        
        # Current progress
        msg.current_target_hoop = self.current_target_hoop
        msg.current_lap = self.current_lap
        msg.total_laps = self.config.HOOP_CONFIG['num_laps']
        msg.hoops_completed_this_lap = self.hoops_completed_this_lap
        msg.total_hoops_completed = self.total_hoops_completed
        
        # Course layout
        msg.num_hoops_in_course = len(self.course_hoops)
        
        # Convert hoop positions to ROS messages
        for hoop in self.course_hoops:
            pos = msg.hoop_positions.add()
            pos.x = hoop['position'][0]
            pos.y = hoop['position'][1] 
            pos.z = hoop['position'][2]
            msg.hoop_diameters.append(hoop['diameter'])
        
        # Target hoop details
        if self.current_target_hoop < len(self.course_hoops):
            target_hoop = self.course_hoops[self.current_target_hoop]
            msg.target_position.x = target_hoop['position'][0]
            msg.target_position.y = target_hoop['position'][1]
            msg.target_position.z = target_hoop['position'][2]
            msg.target_diameter = target_hoop['diameter']
            
            # Calculate distance and bearing
            drone_pos = np.array([
                self.current_drone_state.position.x,
                self.current_drone_state.position.y,
                self.current_drone_state.position.z
            ])
            target_pos = np.array(target_hoop['position'])
            
            msg.distance_to_target = float(np.linalg.norm(target_pos - drone_pos))
            
            # Bearing calculation (2D)
            diff = target_pos[:2] - drone_pos[:2]
            msg.bearing_to_target = float(np.arctan2(diff[1], diff[0]))
        
        # Progress tracking
        msg.lap_progress = self.hoops_completed_this_lap / len(self.course_hoops)
        total_hoops_needed = len(self.course_hoops) * self.config.HOOP_CONFIG['num_laps']
        msg.overall_progress = self.total_hoops_completed / total_hoops_needed
        
        # Estimate time remaining
        if self.hoop_completion_times:
            avg_hoop_time = np.mean(self.hoop_completion_times)
            remaining_hoops = total_hoops_needed - self.total_hoops_completed
            msg.estimated_time_remaining = avg_hoop_time * remaining_hoops
        else:
            msg.estimated_time_remaining = 0.0
        
        # Course events (reset each publish cycle)
        msg.hoop_passed = False  # Would be set by completion detection
        msg.lap_completed = False
        msg.course_completed = (self.current_lap > self.config.HOOP_CONFIG['num_laps'])
        msg.new_target_hoop = False
        
        # Episode management
        msg.episode_id = self.episode_id
        msg.episode_time_elapsed = time.time() - self.episode_start_time
        msg.episode_reset_requested = False
        msg.episode_step_count = self.episode_step_count
        
        # Performance metrics
        if self.hoop_completion_times:
            msg.average_hoop_time = np.mean(self.hoop_completion_times)
        else:
            msg.average_hoop_time = 0.0
        
        msg.navigation_efficiency = msg.overall_progress / max(msg.episode_time_elapsed, 1.0)
        msg.collision_count = 0  # TODO: Implement collision detection
        msg.boundary_violations = 0  # TODO: Implement boundary violation detection
        
        self.course_state_pub.publish(msg)


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = CourseManagerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Course manager node error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main() 