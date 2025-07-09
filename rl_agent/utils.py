import numpy as np
import torch
import random

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ClearML Integration
try:
    from clearml import Task, Logger as ClearMLLogger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    logger.warning("ClearML not available. Install with: pip install clearml")


class ClearMLTracker:
    """ClearML integration for live training monitoring like AWS DeepRacer"""
    
    def __init__(self, project_name: str = "DeepFlyer", 
                 task_name: str = "Hoop Navigation Training",
                 tags: list = None):
        """
        Initialize ClearML tracking
        
        Args:
            project_name: ClearML project name
            task_name: Task name for this training run
            tags: Optional tags for the task
        """
        self.enabled = CLEARML_AVAILABLE
        self.task = None
        self.logger = None
        
        if self.enabled:
            try:
                # Initialize ClearML task
                self.task = Task.init(
                    project_name=project_name,
                    task_name=task_name,
                    tags=tags or ["drone", "rl", "p3o", "hoop-navigation"]
                )
                
                # Get ClearML logger
                self.logger = self.task.get_logger()
                
                logger.info(f"ClearML tracking initialized: {project_name}/{task_name}")
            except Exception as e:
                logger.error(f"Failed to initialize ClearML: {e}")
                self.enabled = False
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """Log hyperparameters to ClearML"""
        if not self.enabled:
            return
        
        try:
            self.task.connect(config)
        except Exception as e:
            logger.error(f"Failed to log hyperparameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics to ClearML"""
        if not self.enabled:
            return
        
        try:
            for name, value in metrics.items():
                self.logger.report_scalar(
                    title=name.replace("_", " ").title(),
                    series=name,
                    value=value,
                    iteration=step
                )
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_reward_components(self, reward_components: Dict[str, float], step: int):
        """Log reward component breakdown for educational insight"""
        if not self.enabled:
            return
        
        try:
            # Log each reward component
            for component, value in reward_components.items():
                self.logger.report_scalar(
                    title="Reward Components",
                    series=component,
                    value=value,
                    iteration=step
                )
            
            # Log total reward
            total_reward = sum(reward_components.values())
            self.logger.report_scalar(
                title="Reward Components",
                series="total_reward",
                value=total_reward,
                iteration=step
            )
        except Exception as e:
            logger.error(f"Failed to log reward components: {e}")
    
    def log_episode_video(self, video_path: str, episode: int):
        """Upload episode video to ClearML"""
        if not self.enabled or not video_path:
            return
        
        try:
            self.logger.report_media(
                title="Episode Recording",
                series=f"Episode {episode}",
                local_path=video_path,
                iteration=episode
            )
        except Exception as e:
            logger.error(f"Failed to upload video: {e}")
    
    def log_model_checkpoint(self, model_path: str, episode: int):
        """Log model checkpoint to ClearML"""
        if not self.enabled:
            return
        
        try:
            self.task.upload_artifact(
                name=f"model_episode_{episode}",
                artifact_object=model_path
            )
        except Exception as e:
            logger.error(f"Failed to log model checkpoint: {e}")
    
    def close(self):
        """Close ClearML task"""
        if self.enabled and self.task:
            self.task.close()


# Live Dashboard Data Stream
class LiveDashboardStream:
    """Stream training data for live dashboard visualization"""
    
    def __init__(self, clearml_tracker: Optional[ClearMLTracker] = None):
        self.clearml = clearml_tracker
        self.current_episode = 0
        self.episode_data = {
            "rewards": [],
            "actions": [],
            "states": [],
            "reward_components": []
        }
    
    def start_episode(self, episode: int):
        """Start tracking a new episode"""
        self.current_episode = episode
        self.episode_data = {
            "rewards": [],
            "actions": [],
            "states": [],
            "reward_components": []
        }
    
    def log_step(self, state: Dict, action: list, reward: float, 
                 reward_components: Dict[str, float]):
        """Log a single step within an episode"""
        self.episode_data["states"].append(state)
        self.episode_data["actions"].append(action)
        self.episode_data["rewards"].append(reward)
        self.episode_data["reward_components"].append(reward_components)
    
    def end_episode(self, success: bool):
        """End current episode and send summary to dashboard"""
        if self.clearml:
            # Log episode metrics
            metrics = {
                "episode_reward": sum(self.episode_data["rewards"]),
                "episode_length": len(self.episode_data["rewards"]),
                "episode_success": float(success),
                "average_reward": np.mean(self.episode_data["rewards"]) if self.episode_data["rewards"] else 0
            }
            
            self.clearml.log_metrics(metrics, self.current_episode)
            
            # Log final reward component breakdown
            if self.episode_data["reward_components"]:
                final_components = self.episode_data["reward_components"][-1]
                self.clearml.log_reward_components(final_components, self.current_episode)
