#!/usr/bin/env python3
"""
Domain Randomization for DeepFlyer Sim-to-Real Transfer
Implements AWS DeepRacer-style domain randomization for robust drone RL training
"""

import numpy as np
import cv2
import random
from typing import Dict, Tuple, Optional, List, Any, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RandomizationLevel(Enum):
    """Domain randomization intensity levels"""
    MINIMAL = "minimal"      # Light randomization for testing
    MODERATE = "moderate"    # Balanced randomization for training
    AGGRESSIVE = "aggressive"  # Heavy randomization for robustness
    STUDENT = "student"      # Student-configurable level


@dataclass
class VisualRandomizationParams:
    """Parameters for visual domain randomization"""
    # Lighting variations
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.7, 1.3)
    hue_shift_range: Tuple[float, float] = (-0.1, 0.1)
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    
    # Noise and artifacts
    gaussian_noise_std: Tuple[float, float] = (0.0, 0.02)
    motion_blur_kernel_range: Tuple[int, int] = (0, 7)
    
    # Color space variations
    channel_dropout_prob: float = 0.1
    random_grayscale_prob: float = 0.05
    
    # Weather and environmental effects
    fog_density_range: Tuple[float, float] = (0.0, 0.3)
    rain_intensity_range: Tuple[float, float] = (0.0, 0.2)


@dataclass 
class PhysicsRandomizationParams:
    """Parameters for physics domain randomization"""
    # Wind disturbances
    wind_velocity_range: Tuple[float, float] = (0.0, 2.0)  # m/s
    wind_direction_range: Tuple[float, float] = (0.0, 360.0)  # degrees
    wind_turbulence_scale: Tuple[float, float] = (0.0, 0.5)
    
    # Drone dynamics variations
    mass_variation_range: Tuple[float, float] = (0.8, 1.2)  # multiplier
    inertia_variation_range: Tuple[float, float] = (0.9, 1.1)  # multiplier
    thrust_efficiency_range: Tuple[float, float] = (0.9, 1.1)  # multiplier
    
    # Sensor noise and delays
    imu_noise_scale: Tuple[float, float] = (1.0, 1.5)  # multiplier
    gps_accuracy_range: Tuple[float, float] = (0.1, 2.0)  # meters
    sensor_delay_range: Tuple[float, float] = (0.0, 0.05)  # seconds


@dataclass
class EnvironmentRandomizationParams:
    """Parameters for environment domain randomization"""
    # Hoop variations
    hoop_position_noise: Tuple[float, float] = (0.0, 0.2)  # meters
    hoop_rotation_range: Tuple[float, float] = (-15.0, 15.0)  # degrees
    hoop_scale_range: Tuple[float, float] = (0.9, 1.1)  # multiplier
    hoop_color_variations: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 165, 0),  # Orange
    ])
    
    # Background variations
    background_brightness_range: Tuple[float, float] = (0.3, 1.0)
    sky_color_variations: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (135, 206, 235),  # Sky blue
        (128, 128, 128),  # Cloudy gray
        (255, 165, 0),    # Sunset orange
        (70, 130, 180),   # Steel blue
    ])
    
    # Obstacle placement randomization
    obstacle_density_range: Tuple[float, float] = (0.0, 0.3)  # probability
    obstacle_size_range: Tuple[float, float] = (0.1, 0.8)  # meters


class DomainRandomizer(ABC):
    """Abstract base class for domain randomizers"""
    
    @abstractmethod
    def randomize(self, data: Any, params: Any) -> Any:
        """Apply randomization to input data"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset randomizer state"""
        pass


class VisualRandomizer(DomainRandomizer):
    """Randomizes visual observations for robustness"""
    
    def __init__(self, params: VisualRandomizationParams):
        self.params = params
        self.rng = np.random.RandomState()
    
    def randomize(self, image: np.ndarray, params: Optional[VisualRandomizationParams] = None) -> np.ndarray:
        """
        Apply visual randomization to image
        
        Args:
            image: Input RGB image
            params: Optional override parameters
            
        Returns:
            Randomized image
        """
        if params is None:
            params = self.params
        
        # Work on copy to avoid modifying original
        img = image.astype(np.float32) / 255.0
        
        # Brightness adjustment
        brightness = self.rng.uniform(*params.brightness_range)
        img = np.clip(img * brightness, 0, 1)
        
        # Contrast adjustment
        contrast = self.rng.uniform(*params.contrast_range)
        img = np.clip((img - 0.5) * contrast + 0.5, 0, 1)
        
        # Convert to HSV for hue/saturation adjustments
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Saturation adjustment
        saturation = self.rng.uniform(*params.saturation_range)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * saturation, 0, 1)
        
        # Hue shift
        hue_shift = self.rng.uniform(*params.hue_shift_range)
        img_hsv[:, :, 0] = np.fmod(img_hsv[:, :, 0] + hue_shift, 1.0)
        
        # Convert back to RGB
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        
        # Gamma correction
        gamma = self.rng.uniform(*params.gamma_range)
        img = np.power(img, gamma)
        
        # Add Gaussian noise
        if params.gaussian_noise_std[1] > 0:
            noise_std = self.rng.uniform(*params.gaussian_noise_std)
            noise = self.rng.normal(0, noise_std, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        # Motion blur
        if params.motion_blur_kernel_range[1] > 0:
            kernel_size = self.rng.randint(*params.motion_blur_kernel_range)
            if kernel_size > 0 and kernel_size % 2 == 1:  # Must be odd
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size//2, :] = 1.0 / kernel_size
                img = cv2.filter2D(img, -1, kernel)
        
        # Channel dropout
        if self.rng.random() < params.channel_dropout_prob:
            channel = self.rng.randint(0, 3)
            img[:, :, channel] = 0
        
        # Random grayscale
        if self.rng.random() < params.random_grayscale_prob:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.stack([gray, gray, gray], axis=2)
        
        # Environmental effects
        img = self._add_fog(img, params.fog_density_range)
        img = self._add_rain(img, params.rain_intensity_range)
        
        return (img * 255).astype(np.uint8)
    
    def _add_fog(self, img: np.ndarray, density_range: Tuple[float, float]) -> np.ndarray:
        """Add fog effect to image"""
        if density_range[1] <= 0:
            return img
        
        density = self.rng.uniform(*density_range)
        if density <= 0:
            return img
        
        # Create fog mask
        fog = np.ones_like(img) * 0.8  # Light gray fog
        return img * (1 - density) + fog * density
    
    def _add_rain(self, img: np.ndarray, intensity_range: Tuple[float, float]) -> np.ndarray:
        """Add rain effect to image"""
        if intensity_range[1] <= 0:
            return img
        
        intensity = self.rng.uniform(*intensity_range)
        if intensity <= 0:
            return img
        
        h, w = img.shape[:2]
        
        # Generate random rain drops
        num_drops = int(intensity * h * w * 0.01)  # Density based on intensity
        for _ in range(num_drops):
            x = self.rng.randint(0, w)
            y = self.rng.randint(0, h)
            length = self.rng.randint(3, 15)
            
            # Draw rain drop as a line
            y_end = min(h - 1, y + length)
            if y_end > y:
                cv2.line(img, (x, y), (x, y_end), (0.9, 0.9, 0.9), 1)
        
        return img
    
    def reset(self):
        """Reset randomizer state"""
        # Re-seed with random state to maintain randomness
        self.rng = np.random.RandomState()


class PhysicsRandomizer(DomainRandomizer):
    """Randomizes physics parameters for robustness"""
    
    def __init__(self, params: PhysicsRandomizationParams):
        self.params = params
        self.current_wind = np.zeros(3)  # [vx, vy, vz]
        self.current_mass_mult = 1.0
        self.current_inertia_mult = 1.0
        self.current_thrust_mult = 1.0
    
    def randomize(self, action: np.ndarray, state: Dict[str, Any], 
                 params: Optional[PhysicsRandomizationParams] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply physics randomization to drone action and state
        
        Args:
            action: Control action [vx, vy, vz, yaw_rate]
            state: Current drone state
            params: Optional override parameters
            
        Returns:
            Tuple of (modified_action, disturbance_info)
        """
        if params is None:
            params = self.params
        
        # Wind disturbance
        wind_velocity = np.random.uniform(*params.wind_velocity_range)
        wind_direction = np.random.uniform(*params.wind_direction_range)
        wind_direction_rad = np.radians(wind_direction)
        
        wind_x = wind_velocity * np.cos(wind_direction_rad)
        wind_y = wind_velocity * np.sin(wind_direction_rad)
        wind_z = np.random.uniform(-wind_velocity * 0.3, wind_velocity * 0.3)  # Vertical turbulence
        
        self.current_wind = np.array([wind_x, wind_y, wind_z])
        
        # Add wind turbulence
        turbulence_scale = np.random.uniform(*params.wind_turbulence_scale)
        turbulence = np.random.normal(0, turbulence_scale, 3)
        wind_disturbance = self.current_wind + turbulence
        
        # Apply wind to action (simulate external forces)
        modified_action = action.copy()
        modified_action[:3] += wind_disturbance * 0.1  # Scale wind effect
        
        # Drone dynamics variations
        self.current_mass_mult = np.random.uniform(*params.mass_variation_range)
        self.current_inertia_mult = np.random.uniform(*params.inertia_variation_range)
        self.current_thrust_mult = np.random.uniform(*params.thrust_efficiency_range)
        
        # Apply dynamics variations to action
        modified_action[:3] *= self.current_thrust_mult
        modified_action[3] *= self.current_inertia_mult  # Yaw rate affected by inertia
        
        disturbance_info = {
            'wind_velocity': wind_disturbance,
            'mass_multiplier': self.current_mass_mult,
            'inertia_multiplier': self.current_inertia_mult,
            'thrust_multiplier': self.current_thrust_mult
        }
        
        return modified_action, disturbance_info
    
    def reset(self):
        """Reset physics randomizer state"""
        self.current_wind = np.zeros(3)
        self.current_mass_mult = 1.0
        self.current_inertia_mult = 1.0
        self.current_thrust_mult = 1.0


class UnifiedDomainRandomizer:
    """
    Unified domain randomization manager for DeepFlyer
    Coordinates visual, physics, and environment randomization
    """
    
    def __init__(self, level: RandomizationLevel = RandomizationLevel.MODERATE):
        self.level = level
        self.visual_params, self.physics_params, self.env_params = self._get_params_for_level(level)
        
        self.visual_randomizer = VisualRandomizer(self.visual_params)
        self.physics_randomizer = PhysicsRandomizer(self.physics_params)
        
        # Statistics
        self.randomizations_applied = 0
        self.episode_randomizations = []
        
        logger.info(f"Domain randomizer initialized with level: {level.value}")
    
    def _get_params_for_level(self, level: RandomizationLevel) -> Tuple[
        VisualRandomizationParams, PhysicsRandomizationParams, EnvironmentRandomizationParams
    ]:
        """Get randomization parameters based on intensity level"""
        if level == RandomizationLevel.MINIMAL:
            visual = VisualRandomizationParams(
                brightness_range=(0.9, 1.1),
                contrast_range=(0.95, 1.05),
                gaussian_noise_std=(0.0, 0.01),
                motion_blur_kernel_range=(0, 3),
                fog_density_range=(0.0, 0.1)
            )
            physics = PhysicsRandomizationParams(
                wind_velocity_range=(0.0, 0.5),
                mass_variation_range=(0.95, 1.05),
                inertia_variation_range=(0.98, 1.02)
            )
        
        elif level == RandomizationLevel.MODERATE:
            visual = VisualRandomizationParams()  # Use defaults
            physics = PhysicsRandomizationParams()  # Use defaults
            
        elif level == RandomizationLevel.AGGRESSIVE:
            visual = VisualRandomizationParams(
                brightness_range=(0.5, 1.5),
                contrast_range=(0.6, 1.4),
                saturation_range=(0.5, 1.5),
                hue_shift_range=(-0.2, 0.2),
                gaussian_noise_std=(0.0, 0.05),
                motion_blur_kernel_range=(0, 9),
                fog_density_range=(0.0, 0.5),
                rain_intensity_range=(0.0, 0.4)
            )
            physics = PhysicsRandomizationParams(
                wind_velocity_range=(0.0, 3.0),
                wind_turbulence_scale=(0.0, 0.8),
                mass_variation_range=(0.7, 1.3),
                inertia_variation_range=(0.8, 1.2),
                thrust_efficiency_range=(0.8, 1.2)
            )
        
        else:  # STUDENT - configurable, start with moderate
            visual = VisualRandomizationParams()
            physics = PhysicsRandomizationParams()
        
        env = EnvironmentRandomizationParams()  # Use defaults for all levels
        return visual, physics, env
    
    def randomize_observation(self, observation: np.ndarray) -> np.ndarray:
        """Apply visual domain randomization to observation"""
        if len(observation.shape) == 3:  # RGB image
            return self.visual_randomizer.randomize(observation)
        return observation  # Non-visual observation, return as-is
    
    def randomize_action(self, action: np.ndarray, state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply physics domain randomization to action"""
        return self.physics_randomizer.randomize(action, state)
    
    def reset_episode(self):
        """Reset randomizers for new episode"""
        self.visual_randomizer.reset()
        self.physics_randomizer.reset()
        
        episode_info = {
            'level': self.level.value,
            'visual_params': self.visual_params.__dict__,
            'physics_params': self.physics_params.__dict__
        }
        self.episode_randomizations.append(episode_info)
        
        logger.debug(f"Domain randomization reset for episode {len(self.episode_randomizations)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get domain randomization statistics"""
        return {
            'level': self.level.value,
            'total_randomizations': self.randomizations_applied,
            'episodes_with_randomization': len(self.episode_randomizations),
            'current_params': {
                'visual': self.visual_params.__dict__,
                'physics': self.physics_params.__dict__,
                'environment': self.env_params.__dict__
            }
        }
    
    def update_level(self, new_level: RandomizationLevel):
        """Update randomization level during training"""
        logger.info(f"Updating domain randomization from {self.level.value} to {new_level.value}")
        self.level = new_level
        self.visual_params, self.physics_params, self.env_params = self._get_params_for_level(new_level)
        
        # Update randomizers with new parameters
        self.visual_randomizer.params = self.visual_params
        self.physics_randomizer.params = self.physics_params


def create_domain_randomizer(config: Dict[str, Any]) -> UnifiedDomainRandomizer:
    """
    Factory function to create domain randomizer from configuration
    
    Args:
        config: Configuration dictionary with randomization settings
        
    Returns:
        Configured domain randomizer
    """
    level_name = config.get('level', 'moderate').lower()
    
    # Map level names to enum
    level_map = {
        'minimal': RandomizationLevel.MINIMAL,
        'moderate': RandomizationLevel.MODERATE,
        'aggressive': RandomizationLevel.AGGRESSIVE,
        'student': RandomizationLevel.STUDENT
    }
    
    level = level_map.get(level_name, RandomizationLevel.MODERATE)
    randomizer = UnifiedDomainRandomizer(level)
    
    # Apply custom overrides if provided
    if 'visual_overrides' in config:
        for key, value in config['visual_overrides'].items():
            if hasattr(randomizer.visual_params, key):
                setattr(randomizer.visual_params, key, value)
    
    if 'physics_overrides' in config:
        for key, value in config['physics_overrides'].items():
            if hasattr(randomizer.physics_params, key):
                setattr(randomizer.physics_params, key, value)
    
    return randomizer
