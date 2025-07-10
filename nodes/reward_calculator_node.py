#!/usr/bin/env python3
"""
Reward Calculator ROS2 Node
Standalone node for computing and publishing detailed reward feedback
"""

import numpy as np
import time
import yaml
import os
from typing import Dict, Any, Optional

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Custom messages
try:
    from deepflyer_msgs.msg import RewardFeedback, DroneState, CourseState, VisionFeatures, RLAction
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False
    print("Custom messages not available")

# DeepFlyer imports
from rl_agent.config import DeepFlyerConfig


class RewardCalculatorNode(Node):
    """ROS2 node for reward calculation and feedback"""
    
    def __init__(self):
        super().__init__('reward_calculator_node')
        
        # Parameters
        self.declare_parameter('reward_config_file', 'config/reward_params.yaml')
        self.declare_parameter('publish_frequency', 20.0)
        self.declare_parameter('enable_detailed_breakdown', True)
        
        # Get parameters
        self.reward_config_file = self.get_parameter('reward_config_file').value
        self.publish_frequency = self.get_parameter('publish_frequency').value
        self.enable_detailed_breakdown = self.get_parameter('enable_detailed_breakdown').value
        
        # Load configuration
        self.config = DeepFlyerConfig()
        self.reward_params = self._load_reward_parameters()
        
        # State tracking
        self.current_drone_state: Optional[DroneState] = None
        self.current_course_state: Optional[CourseState] = None
        self.current_vision_features: Optional[VisionFeatures] = None
        self.last_action: Optional[RLAction] = None
        
        # Episode tracking
        self.episode_start_time = time.time()
        self.cumulative_reward = 0.0
        self.episode_step = 0
        self.reward_history = []
        
        # Previous state for delta calculations
        self.prev_drone_state: Optional[DroneState] = None
        self.prev_distance_to_target = None
        
        # Setup ROS interface
        self._setup_ros_interface()
        
        # Start reward calculation timer
        calc_period = 1.0 / self.publish_frequency
        self.calc_timer = self.create_timer(calc_period, self.calculate_and_publish_reward)
        
        self.get_logger().info("Reward calculator node started")
    
    def _setup_ros_interface(self):
        """Setup ROS publishers and subscribers"""
        if not CUSTOM_MSGS_AVAILABLE:
            self.get_logger().error("Custom messages not available")
            return
        
        # QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publishers
        self.reward_feedback_pub = self.create_publisher(
            RewardFeedback, '/deepflyer/reward_feedback', reliable_qos)
        
        # Subscribers
        self.drone_state_sub = self.create_subscription(
            DroneState, '/deepflyer/drone_state', self.drone_state_callback, reliable_qos)
        
        self.course_state_sub = self.create_subscription(
            CourseState, '/deepflyer/course_state', self.course_state_callback, reliable_qos)
        
        self.vision_features_sub = self.create_subscription(
            VisionFeatures, '/deepflyer/vision_features', self.vision_features_callback, reliable_qos)
        
        self.rl_action_sub = self.create_subscription(
            RLAction, '/deepflyer/rl_action', self.rl_action_callback, reliable_qos)
    
    def _load_reward_parameters(self) -> Dict[str, float]:
        """Load reward parameters from configuration file"""
        reward_params = self.config.REWARD_CONFIG.copy()
        
        # Try to load custom parameters from file
        if os.path.exists(self.reward_config_file):
            try:
                with open(self.reward_config_file, 'r') as f:
                    custom_params = yaml.safe_load(f)
                    reward_params.update(custom_params)
                self.get_logger().info(f"Loaded custom reward parameters from {self.reward_config_file}")
            except Exception as e:
                self.get_logger().warning(f"Failed to load reward config: {e}")
        
        return reward_params
    
    def drone_state_callback(self, msg: DroneState):
        """Update drone state"""
        self.prev_drone_state = self.current_drone_state
        self.current_drone_state = msg
    
    def course_state_callback(self, msg: CourseState):
        """Update course state"""
        self.current_course_state = msg
        
        # Reset episode if new episode detected
        if hasattr(self, '_last_episode_id'):
            if msg.episode_id != self._last_episode_id:
                self._reset_episode()
        self._last_episode_id = msg.episode_id
    
    def vision_features_callback(self, msg: VisionFeatures):
        """Update vision features"""
        self.current_vision_features = msg
    
    def rl_action_callback(self, msg: RLAction):
        """Update last RL action"""
        self.last_action = msg
        self.episode_step += 1
    
    def calculate_and_publish_reward(self):
        """Calculate reward and publish feedback"""
        if not self._has_required_data():
            return
        
        try:
            # Calculate reward components
            reward_components = self._calculate_reward_components()
            
            # Compute total reward
            total_reward = sum(reward_components.values())
            self.cumulative_reward += total_reward
            self.reward_history.append(total_reward)
            
            # Create and publish reward feedback message
            feedback_msg = self._create_reward_feedback_message(reward_components, total_reward)
            self.reward_feedback_pub.publish(feedback_msg)
            
            # Log periodically
            if self.episode_step % 50 == 0:
                avg_reward = np.mean(self.reward_history[-50:]) if self.reward_history else 0.0
                self.get_logger().info(
                    f"Step {self.episode_step}: Total reward: {total_reward:.2f}, "
                    f"Cumulative: {self.cumulative_reward:.1f}, Avg(50): {avg_reward:.2f}"
                )
        
        except Exception as e:
            self.get_logger().error(f"Reward calculation error: {e}")
    
    def _has_required_data(self) -> bool:
        """Check if we have all required data for reward calculation"""
        return (self.current_drone_state is not None and 
                self.current_course_state is not None and
                self.current_vision_features is not None)
    
    def _calculate_reward_components(self) -> Dict[str, float]:
        """Calculate individual reward components"""
        components = {}
        
        # Get current state data
        drone_pos = np.array([
            self.current_drone_state.position.x,
            self.current_drone_state.position.y,
            self.current_drone_state.position.z
        ])
        
        target_pos = np.array([
            self.current_course_state.target_position.x,
            self.current_course_state.target_position.y,
            self.current_course_state.target_position.z
        ])
        
        distance_to_target = np.linalg.norm(target_pos - drone_pos)
        
        # 1. Hoop approach reward
        if self.prev_distance_to_target is not None:
            distance_delta = self.prev_distance_to_target - distance_to_target
            if distance_delta > 0:  # Getting closer
                components['hoop_approach_reward'] = (
                    self.reward_params['hoop_approach_reward'] * distance_delta
                )
            else:
                components['hoop_approach_reward'] = 0.0
        else:
            components['hoop_approach_reward'] = 0.0
        
        self.prev_distance_to_target = distance_to_target
        
        # 2. Hoop passage reward
        if self.current_course_state.hoop_passed:
            components['hoop_passage_reward'] = self.reward_params['hoop_passage_reward']
            
            # Center bonus if well aligned
            if abs(self.current_vision_features.alignment_error) < 0.2:
                components['hoop_center_bonus'] = self.reward_params['hoop_center_bonus']
            else:
                components['hoop_center_bonus'] = 0.0
        else:
            components['hoop_passage_reward'] = 0.0
            components['hoop_center_bonus'] = 0.0
        
        # 3. Visual alignment reward
        if self.current_vision_features.hoop_detected:
            alignment_quality = 1.0 - abs(self.current_vision_features.alignment_error)
            components['visual_alignment_reward'] = (
                self.reward_params['visual_alignment_reward'] * alignment_quality
            )
        else:
            components['visual_alignment_reward'] = 0.0
        
        # 4. Forward progress reward
        drone_vel = np.array([
            self.current_drone_state.linear_velocity.x,
            self.current_drone_state.linear_velocity.y,
            self.current_drone_state.linear_velocity.z
        ])
        
        forward_speed = drone_vel[0]  # Assuming x is forward
        if forward_speed > 0.1:
            components['forward_progress_reward'] = self.reward_params['forward_progress_reward']
        else:
            components['forward_progress_reward'] = 0.0
        
        # 5. Speed efficiency bonus
        speed = np.linalg.norm(drone_vel)
        if 0.4 <= speed <= 0.8:  # Optimal speed range
            components['speed_efficiency_bonus'] = self.reward_params['speed_efficiency_bonus']
        else:
            components['speed_efficiency_bonus'] = 0.0
        
        # 6. Lap completion bonus
        if self.current_course_state.lap_completed:
            components['lap_completion_bonus'] = self.reward_params['lap_completion_bonus']
        else:
            components['lap_completion_bonus'] = 0.0
        
        # 7. Course completion bonus
        if self.current_course_state.course_completed:
            components['course_completion_bonus'] = self.reward_params['course_completion_bonus']
        else:
            components['course_completion_bonus'] = 0.0
        
        # 8. Smooth flight bonus
        if self.last_action:
            # Simple smoothness metric based on action magnitude
            action_magnitude = np.sqrt(
                self.last_action.lateral_cmd**2 + 
                self.last_action.vertical_cmd**2 + 
                self.last_action.speed_cmd**2
            )
            if action_magnitude < 0.5:  # Gentle actions
                components['smooth_flight_bonus'] = self.reward_params['smooth_flight_bonus']
            else:
                components['smooth_flight_bonus'] = 0.0
        else:
            components['smooth_flight_bonus'] = 0.0
        
        # 9. Penalties
        
        # Wrong direction penalty
        if distance_to_target > 5.0:  # Too far from target
            components['wrong_direction_penalty'] = self.reward_params['wrong_direction_penalty']
        else:
            components['wrong_direction_penalty'] = 0.0
        
        # Slow progress penalty
        if speed < 0.1:
            components['slow_progress_penalty'] = self.reward_params['slow_progress_penalty']
        else:
            components['slow_progress_penalty'] = 0.0
        
        # Boundary violations (safety penalties - non-configurable)
        if self.current_course_state.boundary_violations > 0:
            components['boundary_violation_penalty'] = self.reward_params['boundary_violation_penalty']
        else:
            components['boundary_violation_penalty'] = 0.0
        
        return components
    
    def _create_reward_feedback_message(self, components: Dict[str, float], total_reward: float) -> RewardFeedback:
        """Create reward feedback message"""
        msg = RewardFeedback()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Total reward information
        msg.total_reward = total_reward
        msg.cumulative_reward = self.cumulative_reward
        msg.episode_step = self.episode_step
        
        # Reward component breakdown
        msg.hoop_approach_reward = components.get('hoop_approach_reward', 0.0)
        msg.hoop_passage_reward = components.get('hoop_passage_reward', 0.0)
        msg.hoop_center_bonus = components.get('hoop_center_bonus', 0.0)
        msg.visual_alignment_reward = components.get('visual_alignment_reward', 0.0)
        msg.forward_progress_reward = components.get('forward_progress_reward', 0.0)
        msg.speed_efficiency_bonus = components.get('speed_efficiency_bonus', 0.0)
        msg.lap_completion_bonus = components.get('lap_completion_bonus', 0.0)
        msg.course_completion_bonus = components.get('course_completion_bonus', 0.0)
        msg.smooth_flight_bonus = components.get('smooth_flight_bonus', 0.0)
        msg.precision_bonus = 0.0
        
        # Penalty breakdown
        msg.wrong_direction_penalty = components.get('wrong_direction_penalty', 0.0)
        msg.hoop_miss_penalty = 0.0
        msg.collision_penalty = 0.0
        msg.slow_progress_penalty = components.get('slow_progress_penalty', 0.0)
        msg.erratic_flight_penalty = 0.0
        
        # Safety penalties
        msg.boundary_violation_penalty = components.get('boundary_violation_penalty', 0.0)
        msg.emergency_landing_penalty = 0.0
        
        # Context information
        if self.current_course_state:
            msg.distance_to_target = self.current_course_state.distance_to_target
            msg.current_hoop_id = self.current_course_state.current_target_hoop
            msg.current_lap = self.current_course_state.current_lap
        
        if self.current_vision_features:
            msg.hoop_alignment_error = self.current_vision_features.alignment_error
            msg.hoop_visible = self.current_vision_features.hoop_detected
        
        # Performance metrics
        episode_time = time.time() - self.episode_start_time
        msg.episode_efficiency = self.cumulative_reward / max(episode_time, 1.0)
        msg.learning_progress = min(1.0, max(0.0, self.cumulative_reward / 1000.0))
        msg.goal_achieved = self.current_course_state.course_completed if self.current_course_state else False
        
        return msg
    
    def _reset_episode(self):
        """Reset episode tracking"""
        self.episode_start_time = time.time()
        self.cumulative_reward = 0.0
        self.episode_step = 0
        self.reward_history.clear()
        self.prev_distance_to_target = None
        
        self.get_logger().info("Episode reset detected")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        node = RewardCalculatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Reward calculator node error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main() 