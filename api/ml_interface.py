#!/usr/bin/env python3
"""
DeepFlyer ML Interface
Simple interface for Jay's backend to interact with ML components
"""

from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict

# Import your ML components
from rl_agent.config import DeepFlyerConfig
from rl_agent.utils import ClearMLTracker


@dataclass
class RewardConfig:
    """Simplified reward configuration for backend integration"""
    # Positive rewards (student tunable)
    hoop_approach_reward: float = 10.0
    hoop_passage_reward: float = 50.0
    visual_alignment_reward: float = 5.0
    forward_progress_reward: float = 3.0
    
    # Penalties (student tunable)
    wrong_direction_penalty: float = -2.0
    hoop_miss_penalty: float = -25.0
    collision_penalty: float = -100.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'RewardConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TrainingMetrics:
    """Training metrics for frontend display"""
    episode: int = 0
    total_steps: int = 0
    episode_reward: float = 0.0
    average_reward: float = 0.0
    success_rate: float = 0.0
    is_training: bool = False


class DeepFlyerMLInterface:
    """
    Simple interface for backend to interact with DeepFlyer ML components
    
    Usage for Jay:
    ```python
    ml = DeepFlyerMLInterface()
    
    # Get current metrics
    metrics = ml.get_training_metrics()
    
    # Update reward parameters
    ml.update_reward_config(RewardConfig(hoop_approach_reward=15.0))
    
    # Start/stop training
    ml.start_training(minutes=60)
    ml.stop_training()
    ```
    """
    
    def __init__(self):
        self.config = DeepFlyerConfig()
        self.clearml_tracker: Optional[ClearMLTracker] = None
        self.current_metrics = TrainingMetrics()
        
        # Initialize ClearML if available
        try:
            self.clearml_tracker = ClearMLTracker(
                project_name="DeepFlyer",
                task_name="Backend Integration"
            )
        except Exception:
            self.clearml_tracker = None
    
    def get_reward_config(self) -> RewardConfig:
        """Get current reward configuration"""
        reward_cfg = self.config.REWARD_CONFIG
        return RewardConfig(
            hoop_approach_reward=reward_cfg.get('hoop_approach_reward', 10.0),
            hoop_passage_reward=reward_cfg.get('hoop_passage_reward', 50.0),
            visual_alignment_reward=reward_cfg.get('visual_alignment_reward', 5.0),
            forward_progress_reward=reward_cfg.get('forward_progress_reward', 3.0),
            wrong_direction_penalty=reward_cfg.get('wrong_direction_penalty', -2.0),
            hoop_miss_penalty=reward_cfg.get('hoop_miss_penalty', -25.0),
            collision_penalty=reward_cfg.get('collision_penalty', -100.0)
        )
    
    def update_reward_config(self, new_config: RewardConfig) -> bool:
        """
        Update reward configuration
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update internal config
            self.config.REWARD_CONFIG.update(new_config.to_dict())
            
            # Update running ROS nodes in real-time
            try:
                from .ros_bridge import update_reward_parameters
                success = update_reward_parameters(new_config.to_dict())
                if not success:
                    print("Warning: Could not update live ROS nodes")
            except ImportError:
                print("ROS bridge not available - config updated locally only")
            
            # Log to ClearML if available
            if self.clearml_tracker:
                self.clearml_tracker.log_hyperparameters(new_config.to_dict())
            
            return True
        except Exception:
            return False
    
    def get_training_metrics(self) -> TrainingMetrics:
        """Get current training metrics"""
        return self.current_metrics
    
    def start_training(self, minutes: int = 60) -> bool:
        """
        Start training session
        
        Args:
            minutes: Training duration in minutes
            
        Returns:
            True if started successfully
        """
        try:
            self.current_metrics.is_training = True
            # Training logic is handled by ROS2 nodes
            return True
        except Exception:
            return False
    
    def stop_training(self) -> bool:
        """Stop training session"""
        try:
            self.current_metrics.is_training = False
            return True
        except Exception:
            return False
    
    def get_live_data(self) -> Dict[str, Any]:
        """Get live training data for frontend"""
        base_data = {
            "metrics": asdict(self.current_metrics),
            "reward_config": self.get_reward_config().to_dict(),
            "timestamp": __import__('time').time()
        }
        
        # Try to get real-time data from ROS bridge
        try:
            from .ros_bridge import get_realtime_data
            ros_data = get_realtime_data()
            if ros_data:
                base_data.update(ros_data)
        except ImportError:
            pass
        
        return base_data


# Simple usage example for Jay
if __name__ == "__main__":
    # Example usage
    ml_interface = DeepFlyerMLInterface()
    
    # Get current config
    config = ml_interface.get_reward_config()
    print(f"Current config: {config.to_dict()}")
    
    # Update rewards
    new_config = RewardConfig(hoop_approach_reward=15.0)
    success = ml_interface.update_reward_config(new_config)
    print(f"Update successful: {success}")
    
    # Get metrics
    metrics = ml_interface.get_training_metrics()
    print(f"Training metrics: {asdict(metrics)}") 