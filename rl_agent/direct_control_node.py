#!/usr/bin/env python3
"""
Direct Control Node for DeepFlyer
ROS2 node implementing P3O direct control for drone racing
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
from typing import Optional, Dict, Any

# ROS2 message imports
from geometry_msgs.msg import Twist, TwistStamped
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus
)
from std_msgs.msg import Bool, Float32

# Custom messages
from deepflyer_msgs.msg import (
    VisionFeatures,
    DroneState,
    RLAction,
    RewardFeedback
)

# DeepFlyer imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_agent.direct_control_agent import DirectControlAgent, DirectControlConfig
from rl_agent.algorithms.p3o import P3OConfig
from rl_agent.rewards.rewards import HoopNavigationReward, RewardConfig


class DirectControlNode(Node):
    """ROS2 node for P3O direct control"""
    
    def __init__(self):
        super().__init__('direct_control_node')
        
        # Declare parameters
        self.declare_parameter('control_frequency', 20.0)  # Hz
        self.declare_parameter('training_mode', True)
        self.declare_parameter('model_path', 'models/p3o_direct.pt')
        self.declare_parameter('use_safety', True)
        self.declare_parameter('max_velocity', 2.0)  # m/s
        self.declare_parameter('max_yaw_rate', 1.0)  # rad/s
        
        # Get parameters
        self.control_freq = self.get_parameter('control_frequency').value
        self.training_mode = self.get_parameter('training_mode').value
        self.model_path = self.get_parameter('model_path').value
        self.use_safety = self.get_parameter('use_safety').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_yaw_rate = self.get_parameter('max_yaw_rate').value
        
        # Initialize agent
        control_config = DirectControlConfig(
            max_velocity=self.max_velocity,
            max_yaw_rate=self.max_yaw_rate
        )
        
        p3o_config = P3OConfig()
        
        self.agent = DirectControlAgent(control_config, p3o_config)
        
        # Load model if exists
        if os.path.exists(self.model_path):
            self.agent.load(self.model_path)
            self.get_logger().info(f"Loaded model from {self.model_path}")
        
        # Initialize reward function
        self.reward_fn = HoopNavigationReward()
        
        # State tracking
        self.vision_data = {}
        self.drone_state = {}
        self.vehicle_status = None
        self.last_obs = None
        self.last_action = None
        self.episode_active = False
        
        # Setup subscribers
        self._setup_subscribers()
        
        # Setup publishers
        self._setup_publishers()
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_freq,
            self.control_loop
        )
        
        # Training timer (slower)
        if self.training_mode:
            self.training_timer = self.create_timer(
                1.0,  # Train every second
                self.training_step
            )
        
        self.get_logger().info("Direct Control Node initialized")
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Vision features from YOLO + ZED
        self.vision_sub = self.create_subscription(
            VisionFeatures,
            '/deepflyer/vision_features',
            self.vision_callback,
            qos
        )
        
        # Vehicle position (PX4)
        self.position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.position_callback,
            qos
        )
        
        # Vehicle status (PX4)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            qos
        )
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Trajectory setpoint (PX4 direct control)
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            qos
        )
        
        # Offboard control mode
        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos
        )
        
        # RL action (for monitoring)
        self.action_pub = self.create_publisher(
            RLAction,
            '/deepflyer/rl_action',
            qos
        )
        
        # Reward feedback
        self.reward_pub = self.create_publisher(
            RewardFeedback,
            '/deepflyer/reward_feedback',
            qos
        )
    
    def vision_callback(self, msg: VisionFeatures):
        """Process vision features"""
        self.vision_data = {
            'hoop_detected': msg.hoop_detected,
            'hoop_center_x': msg.hoop_center_x,
            'hoop_center_y': msg.hoop_center_y,
            'hoop_distance': msg.hoop_distance
        }
    
    def position_callback(self, msg: VehicleLocalPosition):
        """Process vehicle position"""
        self.drone_state['position'] = [msg.x, msg.y, msg.z]
        self.drone_state['velocity'] = [msg.vx, msg.vy, msg.vz]
        self.drone_state['yaw'] = msg.heading
        self.drone_state['yaw_rate'] = msg.heading_rate if hasattr(msg, 'heading_rate') else 0.0
    
    def status_callback(self, msg: VehicleStatus):
        """Process vehicle status"""
        self.vehicle_status = msg
        
        # Check if we should start/stop episode
        if msg.arming_state == 2 and not self.episode_active:  # Armed
            self.start_episode()
        elif msg.arming_state != 2 and self.episode_active:  # Disarmed
            self.end_episode()
    
    def control_loop(self):
        """Main control loop"""
        if not self.episode_active:
            return
        
        if not self.vision_data or not self.drone_state:
            return
        
        # Build observation
        obs = self.agent.process_observation(self.vision_data, self.drone_state)
        
        # Get action from agent
        action = self.agent.get_action(obs, training=self.training_mode)
        
        # Apply safety constraints if enabled
        if self.use_safety:
            action = self.agent.apply_safety_constraints(action, self.drone_state)
        
        # Send control commands
        self.send_control_commands(action)
        
        # Calculate reward if we have previous observation
        if self.last_obs is not None and self.last_action is not None:
            # Check for special events
            info = self.check_events()
            
            # Calculate reward
            reward, components = self.reward_fn.calculate_reward(obs, action, info)
            
            # Store experience
            if self.training_mode:
                self.agent.store_experience(
                    self.last_obs,
                    self.last_action,
                    obs,
                    reward,
                    info.get('episode_done', False)
                )
            
            # Publish reward feedback
            self.publish_reward_feedback(reward, components)
        
        self.last_obs = obs
        self.last_action = action
    
    def send_control_commands(self, action: np.ndarray):
        """Send velocity commands to PX4"""
        # Publish offboard mode
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(time.time() * 1e6)
        offboard_msg.position = False
        offboard_msg.velocity = True
        offboard_msg.acceleration = False
        self.offboard_pub.publish(offboard_msg)
        
        # Publish trajectory setpoint
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(time.time() * 1e6)
        traj_msg.velocity[0] = float(action[0])  # vx
        traj_msg.velocity[1] = float(action[1])  # vy
        traj_msg.velocity[2] = float(action[2])  # vz
        traj_msg.yaw_rate = float(action[3])     # yaw_rate
        self.trajectory_pub.publish(traj_msg)
        
        # Publish RL action for monitoring
        action_msg = RLAction()
        action_msg.header.stamp = self.get_clock().now().to_msg()
        action_msg.velocity_x = float(action[0])
        action_msg.velocity_y = float(action[1])
        action_msg.velocity_z = float(action[2])
        action_msg.yaw_rate = float(action[3])
        self.action_pub.publish(action_msg)
    
    def check_events(self) -> Dict[str, Any]:
        """Check for special events (collision, hoop passage, etc.)"""
        info = {}
        
        # Check collision (simplified - would use actual sensor)
        position = self.drone_state.get('position', [0, 0, 0])
        if position[2] < 0.1:  # Hit ground
            info['collision'] = True
        else:
            info['collision'] = False
        
        # Check out of bounds
        horizontal_dist = np.sqrt(position[0]**2 + position[1]**2)
        if horizontal_dist > 5.0 or position[2] > 3.0:
            info['out_of_bounds'] = True
        else:
            info['out_of_bounds'] = False
        
        # Check hoop passage (simplified)
        if self.vision_data.get('hoop_distance', 1.0) < 0.1:
            # Would need more sophisticated detection
            info['hoop_passed'] = True
        else:
            info['hoop_passed'] = False
        
        info['position'] = position
        info['time_elapsed'] = time.time()
        
        return info
    
    def publish_reward_feedback(self, reward: float, components: Dict[str, float]):
        """Publish reward feedback"""
        msg = RewardFeedback()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.total_reward = float(reward)
        msg.episode_reward = float(sum(self.agent.episode_rewards))
        
        # Add main components
        msg.hoop_detection_reward = components.get('visual_tracking', 0.0)
        msg.alignment_reward = components.get('horizontal_alignment', 0.0) + components.get('vertical_alignment', 0.0)
        msg.approach_reward = components.get('approach', 0.0)
        msg.passage_reward = components.get('hoop_passage', 0.0)
        msg.collision_penalty = components.get('collision_penalty', 0.0)
        
        self.reward_pub.publish(msg)
    
    def training_step(self):
        """Perform training update"""
        if not self.training_mode or not self.episode_active:
            return
        
        stats = self.agent.train_step()
        if stats:
            self.get_logger().info(
                f"Training: Loss={stats['policy_loss']:.3f}, "
                f"Value={stats['value_loss']:.3f}, "
                f"KL={stats['kl_divergence']:.3f}"
            )
    
    def start_episode(self):
        """Start new episode"""
        self.episode_active = True
        self.agent.reset_episode()
        self.reward_fn.reset()
        self.last_obs = None
        self.last_action = None
        self.get_logger().info("Episode started")
    
    def end_episode(self):
        """End current episode"""
        if not self.episode_active:
            return
        
        self.episode_active = False
        stats = self.agent.get_episode_stats()
        
        self.get_logger().info(
            f"Episode ended: Reward={stats.get('total_reward', 0):.1f}, "
            f"Steps={stats.get('episode_length', 0)}"
        )
        
        # Save model periodically
        if self.training_mode and stats.get('episode_length', 0) > 0:
            self.agent.save(self.model_path)


def main(args=None):
    rclpy.init(args=args)
    node = DirectControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()