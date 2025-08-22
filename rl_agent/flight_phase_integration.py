#!/usr/bin/env python3
"""
Flight Phase Integration for Training
Integrates flight phase management with P3O training
"""

import numpy as np
from typing import Dict, Any, List, Optional
from enum import Enum

from rl_agent.trajectory import FlightPhase, PhaseController, TrajectoryConfig
from rl_agent.env.safety_layer import SafetyLayer
from rl_agent.rewards.rewards import HoopNavigationReward


class TrainingPhaseManager:
    """Manages flight phases during RL training"""
    
    def __init__(self, config: TrajectoryConfig, safety_layer: SafetyLayer):
        self.phase_controller = PhaseController(config)
        self.safety_layer = safety_layer
        self.config = config
        
        # Episode tracking
        self.current_episode = 0
        self.phase_history = []
        self.phase_transition_rewards = {
            FlightPhase.TAKEOFF: 10.0,
            FlightPhase.SCAN_360: 15.0,
            FlightPhase.NAVIGATE_TO_HOOP: 20.0,
            FlightPhase.THROUGH_HOOP_FIRST: 50.0,
            FlightPhase.RETURN_TO_HOOP: 20.0,
            FlightPhase.THROUGH_HOOP_SECOND: 50.0,
            FlightPhase.RETURN_TO_ORIGIN: 25.0,
            FlightPhase.LANDING: 30.0,
            FlightPhase.COMPLETED: 100.0
        }
        
    def reset_episode(self):
        """Reset for new episode"""
        self.phase_controller = PhaseController(self.config)
        self.current_episode += 1
        self.phase_history = []
        
    def update_phase(self, observation: np.ndarray, drone_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update flight phase and return phase-specific information
        
        Args:
            observation: 8D RL observation
            drone_state: Complete drone state
            
        Returns:
            Dictionary with phase information and rewards
        """
        previous_phase = self.phase_controller.phase
        
        # Update phase based on observation and state
        current_phase = self.phase_controller.update_phase(observation, drone_state)
        
        # Check for phase transition
        phase_transition_reward = 0.0
        if current_phase != previous_phase:
            self.phase_history.append(current_phase)
            phase_transition_reward = self.phase_transition_rewards.get(current_phase, 0.0)
            print(f"Phase transition: {previous_phase} -> {current_phase} (+{phase_transition_reward:.1f})")
        
        # Get phase-specific action constraints
        action_constraints = self._get_phase_action_constraints(current_phase)
        
        # Check if episode should terminate
        episode_done = current_phase == FlightPhase.COMPLETED
        
        return {
            'current_phase': current_phase,
            'previous_phase': previous_phase,
            'phase_transition_reward': phase_transition_reward,
            'action_constraints': action_constraints,
            'episode_done': episode_done,
            'phase_progress': self.phase_controller.phase_progress if hasattr(self.phase_controller, 'phase_progress') else 0.0,
            'phase_history': self.phase_history.copy()
        }
    
    def _get_phase_action_constraints(self, phase: FlightPhase) -> Dict[str, Any]:
        """Get action constraints for current phase"""
        constraints = {
            'max_velocity': 1.0,
            'max_yaw_rate': 0.5,
            'min_altitude': 0.5,
            'max_altitude': 3.0
        }
        
        if phase == FlightPhase.TAKEOFF:
            # Conservative during takeoff
            constraints['max_velocity'] = 0.3
            constraints['max_yaw_rate'] = 0.2
            
        elif phase == FlightPhase.LANDING:
            # Very conservative during landing
            constraints['max_velocity'] = 0.2
            constraints['max_yaw_rate'] = 0.1
            
        elif phase in [FlightPhase.THROUGH_HOOP_FIRST, FlightPhase.THROUGH_HOOP_SECOND]:
            # Moderate speed through hoops
            constraints['max_velocity'] = 0.6
            constraints['max_yaw_rate'] = 0.3
            
        elif phase == FlightPhase.SCAN_360:
            # Slow movement, fast yaw for scanning
            constraints['max_velocity'] = 0.2
            constraints['max_yaw_rate'] = 0.8
            
        return constraints
    
    def get_phase_specific_reward_bonus(self, phase: FlightPhase, observation: np.ndarray, 
                                      action: np.ndarray) -> float:
        """Calculate phase-specific reward bonuses"""
        bonus = 0.0
        
        if phase == FlightPhase.SCAN_360:
            # Reward for scanning behavior (yaw movement)
            yaw_rate = abs(action[-1]) if len(action) > 3 else 0.0
            bonus += yaw_rate * 2.0  # Encourage scanning
            
        elif phase == FlightPhase.NAVIGATE_TO_HOOP:
            # Reward for approaching hoop
            if observation[2] > 0.5:  # Hoop visible
                distance = observation[3]
                bonus += (1.0 - distance) * 5.0  # Closer is better
                
                # Alignment bonus
                alignment_error = abs(observation[0])  # How far from center
                bonus += (1.0 - alignment_error) * 3.0
                
        elif phase in [FlightPhase.THROUGH_HOOP_FIRST, FlightPhase.THROUGH_HOOP_SECOND]:
            # Reward for good hoop passage
            if observation[2] > 0.5:  # Hoop visible
                alignment_error = abs(observation[0])
                bonus += (1.0 - alignment_error) * 10.0  # High alignment reward
                
                # Forward velocity bonus
                forward_velocity = action[0] if len(action) > 0 else 0.0
                if forward_velocity > 0:
                    bonus += forward_velocity * 5.0
        
        elif phase == FlightPhase.RETURN_TO_ORIGIN:
            # Reward for returning efficiently
            bonus += 2.0  # Base return bonus
            
        return bonus
    
    def should_force_episode_end(self, observation: np.ndarray, step: int) -> bool:
        """Check if episode should be force-ended due to phase constraints"""
        current_phase = self.phase_controller.phase
        
        # Force end if taking too long in any phase
        phase_duration = self.phase_controller.phase_duration if hasattr(self.phase_controller, 'phase_duration') else 0
        
        max_phase_durations = {
            FlightPhase.TAKEOFF: 30.0,      # 30 seconds
            FlightPhase.SCAN_360: 60.0,     # 1 minute
            FlightPhase.NAVIGATE_TO_HOOP: 90.0,  # 1.5 minutes
            FlightPhase.THROUGH_HOOP_FIRST: 30.0,
            FlightPhase.RETURN_TO_HOOP: 60.0,
            FlightPhase.THROUGH_HOOP_SECOND: 30.0,
            FlightPhase.RETURN_TO_ORIGIN: 90.0,
            FlightPhase.LANDING: 45.0,
        }
        
        max_duration = max_phase_durations.get(current_phase, 120.0)
        if phase_duration > max_duration:
            print(f"Force ending episode: {current_phase} exceeded {max_duration}s")
            return True
            
        return False


def integrate_phase_management(trainer_instance):
    """
    Integrate phase management into existing trainer
    This function can be called to add phase management to P3OTrainer
    """
    from rl_agent.trajectory import TrajectoryConfig
    
    # Add phase manager to trainer
    if not hasattr(trainer_instance, 'phase_manager'):
        config = TrajectoryConfig()
        trainer_instance.phase_manager = TrainingPhaseManager(
            config, 
            trainer_instance.safety_layer
        )
    
    # Modify trainer's simulate_episode method to use phases
    original_simulate_episode = trainer_instance.simulate_episode
    
    def enhanced_simulate_episode(max_steps: int = 500) -> float:
        """Enhanced episode simulation with phase management"""
        trainer_instance.phase_manager.reset_episode()
        
        total_reward = 0.0
        trainer_instance.agent.reset_episode()
        trainer_instance.reward_fn.reset()
        
        obs = trainer_instance.generate_observation(step=0)
        
        for step in range(max_steps):
            # Mock drone state for phase management
            drone_state = {
                'position': np.array([obs[0], obs[1], 1.5]),
                'velocity': obs[4:7],
                'yaw': 0.0
            }
            
            # Update phase
            phase_info = trainer_instance.phase_manager.update_phase(obs, drone_state)
            
            # Get action from agent
            action = trainer_instance.agent.get_action(obs, training=True)
            
            # Apply phase constraints to action
            constraints = phase_info['action_constraints']
            action_magnitude = np.linalg.norm(action[:3])
            if action_magnitude > constraints['max_velocity']:
                action[:3] = action[:3] * constraints['max_velocity'] / action_magnitude
            
            if len(action) > 3:
                action[3] = np.clip(action[3], -constraints['max_yaw_rate'], constraints['max_yaw_rate'])
            
            # Simulate environment step
            next_obs, reward, done, info = trainer_instance.simulate_step(obs, action, step)
            
            # Add phase-specific rewards
            phase_bonus = trainer_instance.phase_manager.get_phase_specific_reward_bonus(
                phase_info['current_phase'], obs, action
            )
            reward += phase_bonus + phase_info['phase_transition_reward']
            
            # Store experience
            trainer_instance.agent.store_experience(obs, action, next_obs, reward, done)
            
            total_reward += reward
            obs = next_obs
            
            # Check phase-based termination
            if phase_info['episode_done'] or trainer_instance.phase_manager.should_force_episode_end(obs, step):
                done = True
                
            if done:
                break
        
        return total_reward
    
    # Replace the method
    trainer_instance.simulate_episode = enhanced_simulate_episode
    
    print("Phase management integration complete")
    return trainer_instance
