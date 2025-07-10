#!/usr/bin/env python3
"""
RL Agent Node for MVP Hoop Navigation

This ROS2 node integrates the P3O RL agent with the MVP system, handling:
- 8D observation space construction from ROS topics
- P3O action selection and training
- Episode management and reward calculation
- Real-time training with student-tunable parameters

Subscribes to:
- /deepflyer/vision_features (hoop detection)
- /deepflyer/course_state (trajectory progress)
- /fmu/out/vehicle_local_position (drone velocity)

Publishes to:
- /deepflyer/rl_action (4D actions to PX4)
- /deepflyer/reward_feedback (reward breakdown)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
import threading
import json

# ROS2 message imports
from std_msgs.msg import Header
from px4_msgs.msg import VehicleLocalPosition

# Custom message imports
from deepflyer_msgs.msg import VisionFeatures, CourseState, RLAction, RewardFeedback

# Import RL components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_agent.algorithms.p3o import P3O, P3OConfig, MVPTrainingConfig
from rl_agent.algorithms.replay_buffer import ReplayBuffer
from rl_agent.env.mvp_trajectory import MVPFlightPhase

logger = logging.getLogger(__name__)


class ObservationBuffer:
    """Buffer for constructing 8D observation space from multiple ROS topics"""
    
    def __init__(self):
        # Vision components
        self.hoop_x_center_norm = 0.0
        self.hoop_y_center_norm = 0.0
        self.hoop_visible = 0
        self.hoop_distance_norm = 1.0
        
        # Drone velocity components
        self.drone_vx_norm = 0.0
        self.drone_vy_norm = 0.0
        self.drone_vz_norm = 0.0
        self.yaw_rate_norm = 0.0
        
        # Timestamps for data freshness
        self.vision_timestamp = 0.0
        self.velocity_timestamp = 0.0
        
        # Data validity
        self.vision_timeout = 1.0  # seconds
        self.velocity_timeout = 1.0  # seconds
    
    def update_vision(self, vision_msg: VisionFeatures):
        """Update vision components from VisionFeatures message"""
        if vision_msg.hoop_detected:
            self.hoop_x_center_norm = vision_msg.hoop_center_x_norm
            self.hoop_y_center_norm = vision_msg.hoop_center_y_norm
            self.hoop_visible = 1
            self.hoop_distance_norm = vision_msg.hoop_distance_norm
        else:
            self.hoop_x_center_norm = 0.0
            self.hoop_y_center_norm = 0.0
            self.hoop_visible = 0
            self.hoop_distance_norm = 1.0
        
        self.vision_timestamp = time.time()
    
    def update_velocity(self, velocity_msg: VehicleLocalPosition):
        """Update velocity components from VehicleLocalPosition message"""
        # Normalize velocities to [-1, 1] range
        max_linear_vel = 2.0  # m/s
        max_angular_vel = 1.0  # rad/s
        
        self.drone_vx_norm = np.clip(velocity_msg.vx / max_linear_vel, -1.0, 1.0)
        self.drone_vy_norm = np.clip(velocity_msg.vy / max_linear_vel, -1.0, 1.0)  
        self.drone_vz_norm = np.clip(velocity_msg.vz / max_linear_vel, -1.0, 1.0)
        
        # Extract yaw rate from heading derivative (approximation)
        if hasattr(self, '_prev_heading') and hasattr(self, '_prev_time'):
            dt = time.time() - self._prev_time
            if dt > 0:
                yaw_rate = (velocity_msg.heading - self._prev_heading) / dt
                self.yaw_rate_norm = np.clip(yaw_rate / max_angular_vel, -1.0, 1.0)
        
        self._prev_heading = velocity_msg.heading
        self._prev_time = time.time()
        self.velocity_timestamp = time.time()
    
    def get_observation(self) -> np.ndarray:
        """Get 8D observation vector"""
        current_time = time.time()
        
        # Check data freshness
        vision_valid = (current_time - self.vision_timestamp) < self.vision_timeout
        velocity_valid = (current_time - self.velocity_timestamp) < self.velocity_timeout
        
        # Use zeros for stale data
        if not vision_valid:
            hoop_x, hoop_y, hoop_vis, hoop_dist = 0.0, 0.0, 0, 1.0
        else:
            hoop_x, hoop_y, hoop_vis, hoop_dist = (
                self.hoop_x_center_norm, self.hoop_y_center_norm, 
                self.hoop_visible, self.hoop_distance_norm
            )
        
        if not velocity_valid:
            drone_vx, drone_vy, drone_vz, yaw_rate = 0.0, 0.0, 0.0, 0.0
        else:
            drone_vx, drone_vy, drone_vz, yaw_rate = (
                self.drone_vx_norm, self.drone_vy_norm,
                self.drone_vz_norm, self.yaw_rate_norm
            )
        
        return np.array([
            hoop_x, hoop_y, float(hoop_vis), hoop_dist,
            drone_vx, drone_vy, drone_vz, yaw_rate
        ], dtype=np.float32)
    
    def is_valid(self) -> bool:
        """Check if observation data is valid"""
        current_time = time.time()
        vision_valid = (current_time - self.vision_timestamp) < self.vision_timeout
        velocity_valid = (current_time - self.velocity_timestamp) < self.velocity_timeout
        return vision_valid and velocity_valid


class RLAgentNode(Node):
    """
    Main RL agent node for MVP hoop navigation with P3O algorithm
    """
    
    def __init__(self):
        super().__init__('rl_agent_node')
        
        # Initialize parameters
        self.declare_parameter('training_mode', True)
        self.declare_parameter('model_save_path', 'models/mvp_p3o_model.pt')
        self.declare_parameter('training_config_file', 'config/mvp_training.json')
        self.declare_parameter('action_frequency', 20.0)
        
        # Get parameters
        self.training_mode = self.get_parameter('training_mode').get_parameter_value().bool_value
        self.model_save_path = self.get_parameter('model_save_path').get_parameter_value().string_value
        self.training_config_file = self.get_parameter('training_config_file').get_parameter_value().string_value
        self.action_freq = self.get_parameter('action_frequency').get_parameter_value().double_value
        
        # Load configurations
        self.p3o_config = self._load_p3o_config()
        self.training_config = self._load_training_config()
        
        # Initialize RL components
        self.p3o_agent = P3O(
            obs_dim=8,  # MVP 8D observation space
            action_dim=4,  # MVP 4D action space
            config=self.p3o_config,
            device="cpu"  # Use CPU for real-time inference
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            obs_dim=8,
            action_dim=4,
            max_size=10000
        )
        
        # Initialize observation buffer
        self.obs_buffer = ObservationBuffer()
        
        # Episode management
        self.episode_active = False
        self.episode_step = 0
        self.episode_start_time = 0.0
        self.episode_reward = 0.0
        self.episode_count = 0
        
        # Data storage for training
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_infos = []
        
        # Performance tracking
        self.total_episodes = 0
        self.successful_episodes = 0
        self.best_episode_reward = float('-inf')
        self.recent_rewards = []
        
        # Threading for training
        self.training_thread = None
        self.training_active = False
        
        # Create subscribers
        self.vision_features_sub = self.create_subscription(
            VisionFeatures,
            '/deepflyer/vision_features',
            self.vision_features_callback,
            10
        )
        
        self.course_state_sub = self.create_subscription(
            CourseState,
            '/deepflyer/course_state',
            self.course_state_callback,
            10
        )
        
        self.vehicle_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_position_callback,
            10
        )
        
        # Create publishers
        self.rl_action_pub = self.create_publisher(
            RLAction,
            '/deepflyer/rl_action',
            10
        )
        
        self.reward_feedback_pub = self.create_publisher(
            RewardFeedback,
            '/deepflyer/reward_feedback',
            10
        )
        
        # Create timers
        self.action_timer = self.create_timer(
            1.0 / self.action_freq,
            self.action_loop
        )
        
        self.reward_timer = self.create_timer(
            1.0 / self.action_freq,  # Same frequency as actions
            self.reward_loop
        )
        
        # Training timer (lower frequency)
        if self.training_mode:
            self.training_timer = self.create_timer(
                1.0,  # 1Hz training updates
                self.training_loop
            )
        
        # Statistics timer
        self.stats_timer = self.create_timer(5.0, self.publish_statistics)
        
        self.get_logger().info("RL Agent Node initialized")
        self.get_logger().info(f"Training mode: {self.training_mode}")
        self.get_logger().info(f"Action frequency: {self.action_freq} Hz")
        self.get_logger().info(f"Observation space: 8D, Action space: 4D")
    
    def _load_p3o_config(self) -> P3OConfig:
        """Load P3O configuration from file or use defaults"""
        try:
            with open(self.training_config_file, 'r') as f:
                config_dict = json.load(f)
            
            config = P3OConfig()
            config.update_from_dict(config_dict.get('p3o', {}))
            
            self.get_logger().info(f"Loaded P3O config from {self.training_config_file}")
            return config
            
        except Exception as e:
            self.get_logger().warn(f"Failed to load P3O config: {e}, using defaults")
            return P3OConfig()
    
    def _load_training_config(self) -> MVPTrainingConfig:
        """Load training configuration"""
        try:
            with open(self.training_config_file, 'r') as f:
                config_dict = json.load(f)
            
            training_minutes = config_dict.get('training_time_minutes', 60)
            config = MVPTrainingConfig()
            config.set_training_time(training_minutes)
            
            return config
            
        except Exception as e:
            self.get_logger().warn(f"Failed to load training config: {e}, using defaults")
            return MVPTrainingConfig()
    

    
    def vision_features_callback(self, msg: VisionFeatures):
        """Update observation buffer with vision data"""
        self.obs_buffer.update_vision(msg)
    
    def course_state_callback(self, msg: CourseState):
        """Handle course state updates"""
        # Start episode if trajectory becomes active
        if msg.episode_active and not self.episode_active:
            self.start_episode()
        
        # End episode if trajectory completes or fails
        elif self.episode_active and not msg.episode_active:
            self.end_episode(trajectory_completed=msg.trajectory_completed)
    
    def vehicle_position_callback(self, msg: VehicleLocalPosition):
        """Update observation buffer with velocity data"""
        self.obs_buffer.update_velocity(msg)
    
    def action_loop(self):
        """Main action selection loop"""
        if not self.episode_active or not self.obs_buffer.is_valid():
            return
        
        # Get current observation
        observation = self.obs_buffer.get_observation()
        
        # Select action using P3O agent
        action, log_prob, value = self.p3o_agent.select_action(
            observation, 
            deterministic=not self.training_mode
        )
        
        # Store for training
        if self.training_mode:
            self.episode_observations.append(observation.copy())
            self.episode_actions.append(action.copy())
        
        # Create and publish RLAction message
        action_msg = RLAction()
        action_msg.header = Header()
        action_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Set 4D action
        action_msg.vx_cmd = float(action[0])
        action_msg.vy_cmd = float(action[1])
        action_msg.vz_cmd = float(action[2])
        action_msg.yaw_rate_cmd = float(action[3])
        
        # Set metadata
        action_msg.action_confidence = float(np.exp(log_prob))  # Convert log prob to confidence
        action_msg.episode_step = self.episode_step
        action_msg.episode_time = time.time() - self.episode_start_time
        action_msg.safety_override = False
        action_msg.emergency_stop = False
        action_msg.flight_phase = "ACTIVE"
        
        # Store raw action
        action_msg.raw_vx_cmd = float(action[0])
        action_msg.raw_vy_cmd = float(action[1])
        action_msg.raw_vz_cmd = float(action[2])
        action_msg.raw_yaw_rate_cmd = float(action[3])
        
        self.rl_action_pub.publish(action_msg)
        
        self.episode_step += 1
    
    def reward_loop(self):
        """Calculate and publish reward feedback"""
        if not self.episode_active or len(self.episode_observations) == 0:
            return
        
        # Get current observation and last action
        observation = self.obs_buffer.get_observation()
        
        if len(self.episode_actions) == 0:
            return
        
        last_action = self.episode_actions[-1]
        
        # Get additional context from course state
        current_velocity = np.linalg.norm([observation[4], observation[5], observation[6]])
        
        # Construct parameters for DeepRacer-style reward function
        params = {
            'hoop_x_center_norm': observation[0],
            'hoop_y_center_norm': observation[1],
            'hoop_visible': int(observation[2]),
            'hoop_distance_norm': observation[3],
            'drone_vx_norm': observation[4],
            'drone_vy_norm': observation[5],
            'drone_vz_norm': observation[6],
            'yaw_rate_norm': observation[7],
            'all_systems_normal': True,  # Would be set by safety system
            'speed': current_velocity * 2.0,  # Convert normalized to m/s
            'collision': False,  # Would be set by safety system
            'hoop_passages_completed': 0,  # Would be updated by course manager
            'flight_phase': 'NAVIGATE_TO_HOOP'  # Would be updated by course manager
        }
        
        # Import and call DeepRacer-style reward function
        from rl_agent.rewards import reward_function
        reward = reward_function(params)
        
        # Store for training
        if self.training_mode:
            self.episode_rewards.append(reward)
            self.episode_infos.append({
                'episode_step': self.episode_step,
                'episode_time': time.time() - self.episode_start_time
            })
        
        self.episode_reward += reward
        
        # Publish reward feedback
        self.publish_reward_feedback(reward, params)
    
    def publish_reward_feedback(self, reward: float, params: Dict[str, Any]):
        """Publish reward feedback for DeepRacer-style reward function"""
        msg = RewardFeedback()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Total reward (single value from DeepRacer-style function)
        msg.total_reward = float(reward)
        
        # For DeepRacer-style function, we don't have component breakdown
        # Set all component rewards to 0 (students see total reward instead)
        msg.hoop_detected_reward = 0.0
        msg.horizontal_align_reward = 0.0
        msg.vertical_align_reward = 0.0
        msg.depth_closer_reward = 0.0
        msg.hoop_passage_reward = 0.0
        msg.roundtrip_finish_reward = 0.0
        
        # Penalties set to 0 (handled within the main reward function)
        msg.collision_penalty = 0.0
        msg.missed_hoop_penalty = 0.0
        msg.drift_lost_penalty = 0.0
        msg.time_penalty = 0.0
        
        # Episode information
        msg.episode_step = self.episode_step
        msg.episode_time = time.time() - self.episode_start_time if self.episode_active else 0.0
        msg.total_episode_reward = float(self.episode_reward)
        msg.hoop_passages_completed = params.get('hoop_passages_completed', 0)
        msg.current_flight_phase = params.get('flight_phase', 'UNKNOWN')
        
        # Performance metrics
        msg.best_episode_reward = float(self.best_episode_reward)
        msg.average_reward_last_10 = float(np.mean(self.recent_rewards[-10:])) if self.recent_rewards else 0.0
        msg.successful_trajectories = self.successful_episodes
        msg.total_episodes = self.total_episodes
        
        # Current reward settings (default values for DeepRacer-style function)
        msg.hoop_detected_setting = 1.0
        msg.horizontal_align_setting = 5.0
        msg.vertical_align_setting = 5.0
        msg.depth_closer_setting = 10.0
        msg.hoop_passage_setting = 100.0
        msg.roundtrip_finish_setting = 200.0
        msg.collision_penalty_setting = -25.0
        msg.missed_hoop_penalty_setting = -25.0
        msg.drift_lost_penalty_setting = -10.0
        msg.time_penalty_setting = -1.0
        
        self.reward_feedback_pub.publish(msg)
    
    def start_episode(self):
        """Start a new training episode"""
        self.episode_active = True
        self.episode_step = 0
        self.episode_start_time = time.time()
        self.episode_reward = 0.0
        
        # Reset episode data
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_infos = []
        
        self.get_logger().info(f"Started episode {self.episode_count + 1}")
    
    def end_episode(self, trajectory_completed: bool = False):
        """End current episode and add to replay buffer"""
        if not self.episode_active:
            return
        
        self.episode_active = False
        self.total_episodes += 1
        
        if trajectory_completed:
            self.successful_episodes += 1
        
        # Update performance tracking
        self.recent_rewards.append(self.episode_reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        self.best_episode_reward = max(self.best_episode_reward, self.episode_reward)
        
        # Add episode to replay buffer for training
        if (self.training_mode and len(self.episode_observations) > 0 and 
            len(self.episode_actions) > 0 and len(self.episode_rewards) > 0):
            
            self._add_episode_to_buffer()
        
        episode_time = time.time() - self.episode_start_time
        self.get_logger().info(
            f"Episode {self.total_episodes} ended: "
            f"Reward={self.episode_reward:.1f}, "
            f"Steps={self.episode_step}, "
            f"Time={episode_time:.1f}s, "
            f"Success={trajectory_completed}"
        )
        
        self.episode_count += 1
    
    def _add_episode_to_buffer(self):
        """Add episode experience to replay buffer"""
        # Add transitions to replay buffer
        for i in range(len(self.episode_observations) - 1):
            obs = self.episode_observations[i]
            action = self.episode_actions[i]
            reward = self.episode_rewards[i]
            next_obs = self.episode_observations[i + 1]
            done = (i == len(self.episode_observations) - 2)  # Last transition
            
            self.replay_buffer.add(obs, action, reward, next_obs, done)
    
    def training_loop(self):
        """Training loop for P3O agent"""
        if not self.training_mode or len(self.replay_buffer) < self.p3o_config.batch_size:
            return
        
        # Train P3O agent
        try:
            metrics = self.p3o_agent.update(self.replay_buffer)
            
            if metrics:
                self.get_logger().info(
                    f"Training: Policy Loss={metrics.get('policy_loss', 0):.4f}, "
                    f"Value Loss={metrics.get('value_loss', 0):.4f}, "
                    f"Buffer Size={len(self.replay_buffer)}"
                )
        
        except Exception as e:
            self.get_logger().error(f"Training error: {e}")
    
    def publish_statistics(self):
        """Publish training statistics"""
        if self.total_episodes > 0:
            success_rate = self.successful_episodes / self.total_episodes
            avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0
            
            self.get_logger().info(
                f"Stats: Episodes={self.total_episodes}, "
                f"Success Rate={success_rate:.1%}, "
                f"Avg Reward={avg_reward:.1f}, "
                f"Best Reward={self.best_episode_reward:.1f}"
            )
    
    def save_model(self):
        """Save trained model"""
        if self.training_mode:
            try:
                self.p3o_agent.save_model(self.model_save_path)
                self.get_logger().info(f"Model saved to {self.model_save_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            self.p3o_agent.load_model(self.model_save_path)
            self.get_logger().info(f"Model loaded from {self.model_save_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to load model: {e}")


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    
    try:
        rl_agent = RLAgentNode()
        
        rl_agent.get_logger().info("RL Agent Node started")
        
        # Try to load existing model
        rl_agent.load_model()
        
        # Spin the node
        rclpy.spin(rl_agent)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in RL agent node: {e}")
    finally:
        if 'rl_agent' in locals():
            # Save model before shutdown
            rl_agent.save_model()
            rl_agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 