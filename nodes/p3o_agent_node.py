#!/usr/bin/env python3
"""
P3O Agent Node for DeepFlyer
Main RL agent that processes vision features and outputs control commands
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from typing import Dict, Optional
import time

# Custom messages
from deepflyer_msgs.msg import (
    VisionFeatures, DroneState, CourseState,
    RLAction, RewardFeedback
)

# PX4 messages
from px4_msgs.msg import TrajectorySetpoint, OffboardControlMode

# DeepFlyer RL components
from rl_agent.algorithms.p3o import P3O
from rl_agent.config import DeepFlyerConfig
from rl_agent.utils import ClearMLTracker, LiveDashboardStream


class P3OAgentNode(Node):
    """
    P3O Reinforcement Learning Agent Node
    
    Subscribes to:
    - /deepflyer/vision_features: Processed vision data
    - /deepflyer/drone_state: Current drone state
    - /deepflyer/course_state: Course progress
    
    Publishes:
    - /deepflyer/rl_action: 3D action commands
    - /deepflyer/reward_feedback: Reward breakdown
    - /fmu/in/trajectory_setpoint: PX4 velocity commands
    """
    
    def __init__(self):
        super().__init__('p3o_agent_node')
        
        # Load configuration
        self.config = DeepFlyerConfig()
        
        # Initialize P3O agent
        self.agent = P3O(
            observation_dim=self.config.OBSERVATION_CONFIG['dimension'],
            action_dim=self.config.ACTION_CONFIG['dimension'],
            hidden_dim=256,
            learning_rate=self.config.P3O_CONFIG['learning_rate'],
            gamma=self.config.P3O_CONFIG['gamma'],
            clip_ratio=self.config.P3O_CONFIG['clip_epsilon'],
            entropy_coef=self.config.P3O_CONFIG['entropy_coef']
        )
        
        # ClearML tracking
        self.clearml = ClearMLTracker(
            project_name="DeepFlyer",
            task_name="P3O Hoop Navigation"
        )
        self.dashboard = LiveDashboardStream(self.clearml)
        
        # Log hyperparameters
        self.clearml.log_hyperparameters(self.config.P3O_CONFIG)
        
        # State tracking
        self.vision_features: Optional[VisionFeatures] = None
        self.drone_state: Optional[DroneState] = None
        self.course_state: Optional[CourseState] = None
        self.last_observation: Optional[np.ndarray] = None
        self.last_action: Optional[np.ndarray] = None
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.total_steps = 0
        
        # Setup ROS2 communication
        self._setup_subscribers()
        self._setup_publishers()
        
        # Control timer (20Hz)
        self.control_timer = self.create_timer(0.05, self._control_loop)
        
        self.get_logger().info("P3O Agent Node initialized")
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Vision features
        self.vision_sub = self.create_subscription(
            VisionFeatures,
            '/deepflyer/vision_features',
            self._vision_callback,
            qos
        )
        
        # Drone state
        self.state_sub = self.create_subscription(
            DroneState,
            '/deepflyer/drone_state',
            self._state_callback,
            qos
        )
        
        # Course state
        self.course_sub = self.create_subscription(
            CourseState,
            '/deepflyer/course_state',
            self._course_callback,
            qos
        )
    
    def _setup_publishers(self):
        """Setup ROS2 publishers"""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # RL action
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
        
        # PX4 trajectory setpoint
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
    
    def _vision_callback(self, msg: VisionFeatures):
        """Process vision features"""
        self.vision_features = msg
    
    def _state_callback(self, msg: DroneState):
        """Process drone state"""
        self.drone_state = msg
    
    def _course_callback(self, msg: CourseState):
        """Process course state"""
        self.course_state = msg
    
    def _control_loop(self):
        """Main control loop at 20Hz"""
        # Check if we have all required data
        if not all([self.vision_features, self.drone_state, self.course_state]):
            return
        
        # Build observation vector (12D as per architecture)
        observation = self._build_observation()
        
        # Get action from P3O agent
        action = self.agent.predict(observation)
        
        # Apply safety constraints
        action = self._apply_safety_constraints(action)
        
        # Publish RL action
        self._publish_rl_action(action)
        
        # Convert to PX4 commands and publish
        self._publish_px4_commands(action)
        
        # Calculate reward if we have previous observation
        if self.last_observation is not None and self.last_action is not None:
            reward, reward_components = self._calculate_reward()
            
            # Update agent
            self.agent.add_to_buffer(
                self.last_observation,
                self.last_action,
                observation,
                reward,
                False  # Episode not done
            )
            
            # Publish reward feedback
            self._publish_reward_feedback(reward, reward_components)
            
            # Update tracking
            self.episode_reward += reward
            self.episode_steps += 1
            self.total_steps += 1
            
            # Log to dashboard
            self.dashboard.log_step(
                state={"observation": observation.tolist()},
                action=action.tolist(),
                reward=reward,
                reward_components=reward_components
            )
            
            # Learn every 10 steps
            if self.total_steps % 10 == 0:
                metrics = self.agent.learn()
                if metrics:
                    self.clearml.log_metrics(metrics, self.total_steps)
        
        # Update for next iteration
        self.last_observation = observation.copy()
        self.last_action = action.copy()
    
    def _build_observation(self) -> np.ndarray:
        """Build 12D observation vector as per architecture"""
        obs = np.zeros(12)
        
        # Direction to hoop (3D)
        if self.vision_features.hoop_detected:
            # Calculate normalized direction
            obs[0] = self.vision_features.hoop_center_x  # Already normalized
            obs[1] = self.vision_features.hoop_center_y
            obs[2] = np.clip(self.vision_features.hoop_distance / 5.0, 0, 1)  # Normalize distance
        
        # Current velocity (2D)
        obs[3] = np.clip(self.drone_state.linear_velocity.x / 2.0, -1, 1)  # Forward velocity
        obs[4] = np.clip(self.drone_state.linear_velocity.y / 2.0, -1, 1)  # Lateral velocity
        
        # Navigation metrics (2D)
        obs[5] = np.clip(self.vision_features.hoop_distance / 5.0, 0, 1) if self.vision_features.hoop_detected else 1.0
        obs[6] = self.vision_features.alignment_error if self.vision_features.hoop_detected else 0.0
        
        # Vision features (3D)
        obs[7] = self.vision_features.hoop_center_x if self.vision_features.hoop_detected else 0.0
        obs[8] = np.clip(self.vision_features.hoop_distance / 5.0, 0, 1) if self.vision_features.hoop_detected else 1.0
        obs[9] = 1.0 if self.vision_features.hoop_detected else 0.0
        
        # Course progress (2D)
        obs[10] = self.course_state.course_progress
        obs[11] = self.course_state.lap_number / 3.0  # Normalize by total laps
        
        return obs
    
    def _apply_safety_constraints(self, action: np.ndarray) -> np.ndarray:
        """Apply safety constraints to actions"""
        # Limit maximum velocities
        MAX_LATERAL = 0.8  # m/s
        MAX_VERTICAL = 0.4  # m/s
        MAX_FORWARD = 0.6  # m/s
        
        # Scale actions to velocity limits
        safe_action = action.copy()
        safe_action[0] *= MAX_LATERAL   # lateral_cmd
        safe_action[1] *= MAX_VERTICAL  # vertical_cmd
        safe_action[2] *= MAX_FORWARD   # speed_cmd
        
        return safe_action
    
    def _publish_rl_action(self, action: np.ndarray):
        """Publish RL action for monitoring"""
        msg = RLAction()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.lateral_cmd = float(action[0])
        msg.vertical_cmd = float(action[1])
        msg.speed_cmd = float(action[2])
        
        self.action_pub.publish(msg)
    
    def _publish_px4_commands(self, action: np.ndarray):
        """Convert RL action to PX4 trajectory setpoint"""
        # Offboard control mode
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = True
        offboard_msg.acceleration = False
        
        self.offboard_pub.publish(offboard_msg)
        
        # Trajectory setpoint
        traj_msg = TrajectorySetpoint()
        traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        
        # Convert action to velocity commands
        traj_msg.velocity[0] = action[2]  # Forward (speed_cmd)
        traj_msg.velocity[1] = action[0]  # Lateral (lateral_cmd)
        traj_msg.velocity[2] = -action[1]  # Vertical (NED frame)
        
        traj_msg.yaw = float('nan')  # Let flight controller handle yaw
        
        self.trajectory_pub.publish(traj_msg)
    
    def _calculate_reward(self) -> tuple:
        """Calculate reward using student-configurable parameters"""
        reward_config = self.config.REWARD_CONFIG
        components = {}
        
        # Hoop approach reward
        if self.vision_features.hoop_detected:
            distance_reward = reward_config['hoop_approach_reward'] * np.exp(
                -self.vision_features.hoop_distance / reward_config['normalization_ranges']['distance_decay_factor']
            )
            components['hoop_approach'] = distance_reward
        else:
            components['hoop_approach'] = 0.0
        
        # Visual alignment reward
        if self.vision_features.hoop_detected:
            alignment_reward = reward_config['visual_alignment_reward'] * (
                1.0 - abs(self.vision_features.alignment_error)
            )
            components['visual_alignment'] = alignment_reward
        else:
            components['visual_alignment'] = 0.0
        
        # Forward progress reward
        forward_velocity = self.drone_state.linear_velocity.x
        if forward_velocity > 0.1:
            components['forward_progress'] = reward_config['forward_progress_reward']
        else:
            components['forward_progress'] = 0.0
        
        # Wrong direction penalty
        if not self.vision_features.hoop_detected and forward_velocity < -0.1:
            components['wrong_direction'] = reward_config['wrong_direction_penalty']
        else:
            components['wrong_direction'] = 0.0
        
        # Total reward
        total_reward = sum(components.values())
        
        return total_reward, components
    
    def _publish_reward_feedback(self, total_reward: float, components: Dict[str, float]):
        """Publish reward feedback for educational monitoring"""
        msg = RewardFeedback()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.total_reward = total_reward
        msg.hoop_progress_reward = components.get('hoop_approach', 0.0)
        msg.alignment_reward = components.get('visual_alignment', 0.0)
        msg.collision_penalty = components.get('collision', 0.0)
        msg.episode_time = float(self.episode_steps * 0.05)  # 20Hz control
        msg.lap_completed = self.course_state.lap_number > 0
        
        # Additional fields for student monitoring
        msg.reward_components = list(components.keys())
        msg.component_values = list(components.values())
        
        self.reward_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = P3OAgentNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.clearml.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 