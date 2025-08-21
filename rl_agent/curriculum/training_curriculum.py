"""
Training Curriculum System for DeepFlyer

Progressive difficulty curriculum for educational learning, following AWS DeepRacer principles
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    """Single stage in the training curriculum"""
    name: str
    duration_episodes: int
    hoop_count: int
    max_velocity: float
    reward_weights: Dict[str, float]
    difficulty_modifiers: Dict[str, Any]
    success_threshold: float = 0.7
    description: str = ""


class TrainingCurriculum:
    """Progressive difficulty curriculum for educational learning"""
    
    def __init__(self, curriculum_type: str = "standard"):
        """
        Initialize training curriculum
        
        Args:
            curriculum_type: Type of curriculum ("standard", "accelerated", "beginner")
        """
        self.curriculum_type = curriculum_type
        self.current_stage_index = 0
        self.current_stage_episodes = 0
        self.stage_success_history = []
        
        # Define curriculum stages
        if curriculum_type == "beginner":
            self.stages = self._create_beginner_curriculum()
        elif curriculum_type == "accelerated":
            self.stages = self._create_accelerated_curriculum()
        else:
            self.stages = self._create_standard_curriculum()
            
        logger.info(f"Initialized {curriculum_type} curriculum with {len(self.stages)} stages")
    
    def _create_standard_curriculum(self) -> List[CurriculumStage]:
        """Create standard curriculum for most students"""
        return [
            CurriculumStage(
                name='basic_hovering',
                duration_episodes=50,
                hoop_count=0,
                max_velocity=0.5,
                reward_weights={
                    'stability': 1.0,
                    'altitude_control': 0.8,
                    'exploration': 0.1,
                    'collision_penalty': -2.0
                },
                difficulty_modifiers={
                    'wind_disturbance': 0.0,
                    'sensor_noise': 0.1,
                    'control_delay': 0.0
                },
                success_threshold=0.6,
                description="Learn basic hovering and altitude control"
            ),
            CurriculumStage(
                name='single_hoop_approach',
                duration_episodes=100,
                hoop_count=1,
                max_velocity=1.0,
                reward_weights={
                    'hoop_approach': 0.8,
                    'visual_tracking': 0.6,
                    'stability': 0.4,
                    'collision_penalty': -1.5
                },
                difficulty_modifiers={
                    'hoop_size_variation': 0.1,
                    'lighting_variation': 0.2,
                    'sensor_noise': 0.15
                },
                success_threshold=0.65,
                description="Learn to approach and track a single hoop"
            ),
            CurriculumStage(
                name='single_hoop_navigation',
                duration_episodes=150,
                hoop_count=1,
                max_velocity=1.5,
                reward_weights={
                    'hoop_passage': 1.0,
                    'alignment': 0.7,
                    'approach_efficiency': 0.5,
                    'collision_penalty': -1.0
                },
                difficulty_modifiers={
                    'hoop_height_variation': 0.3,
                    'approach_angle_variation': 0.2,
                    'wind_disturbance': 0.1
                },
                success_threshold=0.7,
                description="Master flying through a single hoop with precision"
            ),
            CurriculumStage(
                name='dual_hoop_circuit',
                duration_episodes=200,
                hoop_count=2,
                max_velocity=1.8,
                reward_weights={
                    'course_completion': 0.9,
                    'navigation_efficiency': 0.6,
                    'speed_bonus': 0.3,
                    'collision_penalty': -0.8
                },
                difficulty_modifiers={
                    'hoop_spacing_variation': 0.2,
                    'wind_disturbance': 0.15,
                    'sensor_noise': 0.2
                },
                success_threshold=0.75,
                description="Navigate through two hoops in sequence"
            ),
            CurriculumStage(
                name='multi_hoop_circuit',
                duration_episodes=300,
                hoop_count=5,
                max_velocity=2.0,
                reward_weights={
                    'course_completion': 1.0,
                    'speed_efficiency': 0.4,
                    'precision_bonus': 0.3,
                    'collision_penalty': -0.5
                },
                difficulty_modifiers={
                    'dynamic_hoops': 0.1,
                    'wind_disturbance': 0.2,
                    'lighting_changes': 0.3,
                    'sensor_noise': 0.25
                },
                success_threshold=0.8,
                description="Complete full 5-hoop racing circuit"
            )
        ]
    
    def _create_beginner_curriculum(self) -> List[CurriculumStage]:
        """Create easier curriculum for beginners"""
        standard_stages = self._create_standard_curriculum()
        
        # Modify for beginners: longer duration, easier thresholds
        for stage in standard_stages:
            stage.duration_episodes = int(stage.duration_episodes * 1.5)
            stage.success_threshold *= 0.9
            stage.max_velocity *= 0.8
            
            # Reduce difficulty modifiers
            for key in stage.difficulty_modifiers:
                stage.difficulty_modifiers[key] *= 0.5
                
        return standard_stages
    
    def _create_accelerated_curriculum(self) -> List[CurriculumStage]:
        """Create faster curriculum for advanced students"""
        standard_stages = self._create_standard_curriculum()
        
        # Modify for acceleration: shorter duration, higher thresholds
        for stage in standard_stages:
            stage.duration_episodes = int(stage.duration_episodes * 0.7)
            stage.success_threshold = min(0.95, stage.success_threshold * 1.1)
            
            # Increase difficulty modifiers
            for key in stage.difficulty_modifiers:
                stage.difficulty_modifiers[key] = min(1.0, stage.difficulty_modifiers[key] * 1.5)
                
        return standard_stages
    
    def get_current_stage(self, episode: int = None) -> CurriculumStage:
        """Get current curriculum stage"""
        if episode is not None:
            self._update_stage_from_episode(episode)
        return self.stages[self.current_stage_index]
    
    def update_episode_result(self, episode_metrics: Dict[str, float]) -> bool:
        """
        Update curriculum based on episode results
        
        Returns:
            bool: True if advanced to next stage
        """
        self.current_stage_episodes += 1
        current_stage = self.stages[self.current_stage_index]
        
        # Calculate success for this episode
        success = self._calculate_episode_success(episode_metrics, current_stage)
        self.stage_success_history.append(success)
        
        # Check if we should advance to next stage
        should_advance = self._should_advance_stage(current_stage)
        
        if should_advance and self.current_stage_index < len(self.stages) - 1:
            self._advance_to_next_stage()
            return True
            
        return False
    
    def _calculate_episode_success(self, metrics: Dict[str, float], stage: CurriculumStage) -> bool:
        """Calculate if episode was successful for current stage"""
        if stage.name == 'basic_hovering':
            # Success = good stability and no crashes
            return (metrics.get('stability_score', 0) > 0.6 and 
                   not metrics.get('collision', False))
                   
        elif 'hoop' in stage.name:
            # Success = hoop completion above threshold
            success_rate = metrics.get('hoop_success_rate', 0)
            return success_rate >= stage.success_threshold
            
        else:
            # Default: overall reward above threshold
            reward_threshold = 50.0 * stage.success_threshold
            return metrics.get('total_reward', 0) >= reward_threshold
    
    def _should_advance_stage(self, current_stage: CurriculumStage) -> bool:
        """Determine if should advance to next stage"""
        # Need minimum episodes in stage
        if self.current_stage_episodes < current_stage.duration_episodes * 0.5:
            return False
            
        # Check recent success rate
        recent_episodes = min(20, len(self.stage_success_history))
        if recent_episodes < 10:
            return False
            
        recent_success_rate = sum(self.stage_success_history[-recent_episodes:]) / recent_episodes
        
        # Advance if success rate exceeded and minimum episodes completed
        if (recent_success_rate >= current_stage.success_threshold and 
            self.current_stage_episodes >= current_stage.duration_episodes * 0.5):
            return True
            
        # Force advance if spent too long in stage
        if self.current_stage_episodes >= current_stage.duration_episodes * 2:
            logger.warning(f"Force advancing from stage {current_stage.name} after {self.current_stage_episodes} episodes")
            return True
            
        return False
    
    def _advance_to_next_stage(self):
        """Advance to next curriculum stage"""
        old_stage = self.stages[self.current_stage_index]
        self.current_stage_index += 1
        new_stage = self.stages[self.current_stage_index]
        
        # Reset stage tracking
        recent_success = sum(self.stage_success_history[-20:]) / min(20, len(self.stage_success_history))
        logger.info(f"Advanced from '{old_stage.name}' to '{new_stage.name}' "
                   f"after {self.current_stage_episodes} episodes "
                   f"(success rate: {recent_success:.2f})")
        
        self.current_stage_episodes = 0
        self.stage_success_history = []
    
    def _update_stage_from_episode(self, episode: int):
        """Update stage based on absolute episode number"""
        cumulative_episodes = 0
        for i, stage in enumerate(self.stages):
            cumulative_episodes += stage.duration_episodes
            if episode <= cumulative_episodes:
                if i != self.current_stage_index:
                    self.current_stage_index = i
                    self.current_stage_episodes = episode - (cumulative_episodes - stage.duration_episodes)
                break
    
    def get_curriculum_progress(self) -> Dict[str, Any]:
        """Get detailed curriculum progress information"""
        current_stage = self.stages[self.current_stage_index]
        
        # Calculate overall progress
        total_episodes = sum(stage.duration_episodes for stage in self.stages)
        completed_episodes = sum(self.stages[i].duration_episodes for i in range(self.current_stage_index))
        completed_episodes += self.current_stage_episodes
        overall_progress = completed_episodes / total_episodes
        
        # Calculate stage progress
        stage_progress = self.current_stage_episodes / current_stage.duration_episodes
        
        # Recent success rate
        recent_episodes = min(10, len(self.stage_success_history))
        recent_success_rate = (sum(self.stage_success_history[-recent_episodes:]) / recent_episodes 
                              if recent_episodes > 0 else 0.0)
        
        return {
            'current_stage': current_stage.name,
            'stage_index': self.current_stage_index,
            'total_stages': len(self.stages),
            'stage_progress': min(1.0, stage_progress),
            'overall_progress': min(1.0, overall_progress),
            'stage_episodes_completed': self.current_stage_episodes,
            'stage_episodes_target': current_stage.duration_episodes,
            'recent_success_rate': recent_success_rate,
            'success_threshold': current_stage.success_threshold,
            'stage_description': current_stage.description,
            'ready_for_next_stage': self._should_advance_stage(current_stage)
        }
    
    def get_stage_configuration(self) -> Dict[str, Any]:
        """Get current stage configuration for environment setup"""
        current_stage = self.stages[self.current_stage_index]
        
        return {
            'stage_name': current_stage.name,
            'hoop_count': current_stage.hoop_count,
            'max_velocity': current_stage.max_velocity,
            'reward_weights': current_stage.reward_weights.copy(),
            'difficulty_modifiers': current_stage.difficulty_modifiers.copy(),
            'success_threshold': current_stage.success_threshold
        }
    
    def is_curriculum_complete(self) -> bool:
        """Check if curriculum is completed"""
        return self.current_stage_index >= len(self.stages) - 1
