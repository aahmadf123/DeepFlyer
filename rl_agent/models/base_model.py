import os
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Any, Optional
import gymnasium as gym


class BaseModel(ABC):
    """
    Base class for all reinforcement learning models.
    
    This abstract class defines the common interface that all model implementations
    should follow. It provides basic functionality for saving/loading models and
    standardizing the interface for training and inference.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize the base model.
        
        Args:
            observation_space: The environment's observation space
            action_space: The environment's action space
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
            **kwargs: Additional model-specific arguments
        """
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Training metadata
        self.total_timesteps = 0
        self.training_started = False
        
    @abstractmethod
    def predict(
        self, 
        observation: Union[np.ndarray, Dict[str, np.ndarray]], 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Predict action based on observation.
        
        Args:
            observation: The current observation from the environment
            deterministic: Whether to return deterministic actions (for evaluation)
            
        Returns:
            actions: The predicted actions
            states: Additional state information (if any)
        """
        pass
    
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            **kwargs: Additional training arguments
            
        Returns:
            training_info: Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def learn(
        self, 
        batch_size: int
    ) -> Dict[str, float]:
        """
        Perform one iteration of learning on a batch of data.
        
        Args:
            batch_size: Size of the batch to learn from
            
        Returns:
            metrics: Dictionary of learning metrics (losses, etc.)
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create state dict with model parameters and metadata
        state_dict = {
            "model_state": self.get_model_state(),
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "total_timesteps": self.total_timesteps
        }
        
        torch.save(state_dict, path)
        
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        state_dict = torch.load(path, map_location=self.device)
        
        # Verify spaces match
        if not self._is_space_equivalent(state_dict["observation_space"], self.observation_space):
            raise ValueError("Loaded model has incompatible observation space")
        
        if not self._is_space_equivalent(state_dict["action_space"], self.action_space):
            raise ValueError("Loaded model has incompatible action space")
        
        # Load model state and metadata
        self.load_model_state(state_dict["model_state"])
        self.total_timesteps = state_dict.get("total_timesteps", 0)
        
    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        """
        Get model state for saving.
        
        Returns:
            state_dict: Dictionary containing model state
        """
        pass
    
    @abstractmethod
    def load_model_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Load model state from state dictionary.
        
        Args:
            state_dict: Dictionary containing model state
        """
        pass
    
    def _is_space_equivalent(self, space1: gym.Space, space2: gym.Space) -> bool:
        """
        Check if two gym spaces are equivalent.
        
        Args:
            space1: First gym space
            space2: Second gym space
            
        Returns:
            bool: True if spaces are equivalent, False otherwise
        """
        if type(space1) != type(space2):
            return False
        
        if isinstance(space1, gym.spaces.Box):
            return (
                np.allclose(space1.low, space2.low) and
                np.allclose(space1.high, space2.high) and
                space1.shape == space2.shape and
                space1.dtype == space2.dtype
            )
        elif isinstance(space1, gym.spaces.Discrete):
            return space1.n == space2.n
        elif isinstance(space1, gym.spaces.Dict):
            if set(space1.spaces.keys()) != set(space2.spaces.keys()):
                return False
            return all(
                self._is_space_equivalent(space1.spaces[key], space2.spaces[key])
                for key in space1.spaces
            )
        
        # For other space types, just compare the spaces directly
        return space1 == space2
