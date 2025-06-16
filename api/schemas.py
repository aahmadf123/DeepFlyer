from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Hyperparams(BaseModel):
    learning_rate: float = Field(..., description="Learning rate for the RL algorithm")
    gamma: float = Field(..., description="Discount factor for future rewards")
    entropy_coef: Optional[float] = Field(None, description="Entropy regularization coefficient (if applicable)")

class Randomization(BaseModel):
    imu_noise: List[float] = Field(..., description="Uniform range for IMU noise sigma [min, max]")
    camera_noise: List[float] = Field(..., description="Uniform range for camera noise [min, max]")

class TrainRequest(BaseModel):
    algorithm: str = Field(..., description="The RL algorithm to use (e.g. ppo, sac, td3)")
    preset_id: str = Field(..., description="Reward preset identifier")
    hyperparameters: Hyperparams
    cross_track_weight: float = Field(1.0, description="Weight for cross-track error (path following)")
    heading_weight: float = Field(0.1, description="Weight for heading error")
    max_episodes: int = Field(..., description="Maximum number of training episodes")
    max_steps_per_episode: int = Field(..., description="Maximum steps per episode")
    scenario_sequence: Optional[List[str]] = Field(None, description="List of scenario IDs for curriculum mode")
    randomization: Optional[Randomization] = Field(None, description="Domain randomization parameters")
    base_model_id: Optional[str] = Field(None, description="ID of a pretrained base model to fine-tune")
    time_limit_min: Optional[int] = Field(None, description="Time limit in minutes for training")
