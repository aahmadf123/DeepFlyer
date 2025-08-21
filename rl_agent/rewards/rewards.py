"""
DeepFlyer Reward Functions for Hoop Navigation
AWS DeepRacer-style tunable reward system
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Student-tunable reward configuration"""
    # Hoop detection and alignment
    hoop_visible_reward: float = 2.0
    horizontal_alignment_scale: float = 10.0
    vertical_alignment_scale: float = 10.0
    perfect_alignment_bonus: float = 20.0
    
    # Distance and approach
    approach_reward_scale: float = 15.0
    proximity_bonus_threshold: float = 0.3  # Normalized distance
    proximity_bonus: float = 30.0
    
    # Hoop passage
    hoop_passage_reward: float = 100.0
    clean_passage_bonus: float = 50.0  # Bonus for centered passage
    
    # Movement and control
    forward_progress_scale: float = 5.0
    smooth_control_scale: float = 2.0
    hover_penalty: float = -1.0
    
    # Penalties
    collision_penalty: float = -50.0
    out_of_bounds_penalty: float = -30.0
    excessive_yaw_penalty: float = -5.0
    lost_visual_penalty: float = -3.0
    
    # Time efficiency
    time_penalty_per_step: float = -0.1
    efficiency_bonus_threshold: float = 0.8  # Reward/time ratio
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'hoop_visible_reward': self.hoop_visible_reward,
            'horizontal_alignment_scale': self.horizontal_alignment_scale,
            'vertical_alignment_scale': self.vertical_alignment_scale,
            'perfect_alignment_bonus': self.perfect_alignment_bonus,
            'approach_reward_scale': self.approach_reward_scale,
            'proximity_bonus_threshold': self.proximity_bonus_threshold,
            'proximity_bonus': self.proximity_bonus,
            'hoop_passage_reward': self.hoop_passage_reward,
            'clean_passage_bonus': self.clean_passage_bonus,
            'forward_progress_scale': self.forward_progress_scale,
            'smooth_control_scale': self.smooth_control_scale,
            'hover_penalty': self.hover_penalty,
            'collision_penalty': self.collision_penalty,
            'out_of_bounds_penalty': self.out_of_bounds_penalty,
            'excessive_yaw_penalty': self.excessive_yaw_penalty,
            'lost_visual_penalty': self.lost_visual_penalty,
            'time_penalty_per_step': self.time_penalty_per_step
        }


class HoopNavigationReward:
    """Main reward function for hoop navigation task"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.prev_distance = None
        self.prev_action = None
        self.steps_without_visual = 0
        self.total_steps = 0
        
        # Track hoop passages
        self.hoops_passed = 0
        self.was_in_front = True  # Track which side of hoop
        
        logger.info("HoopNavigationReward initialized")
    
    def calculate_reward(self, obs: np.ndarray, action: np.ndarray, 
                        info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward and component breakdown
        
        Args:
            obs: 8D observation [hoop_x, hoop_y, visible, distance, vx, vy, vz, yaw_rate]
            action: 4D action [vx, vy, vz, yaw_rate]
            info: Additional information (collision, out_of_bounds, hoop_passed, etc.)
        
        Returns:
            total_reward: Total reward value
            components: Dictionary of reward components for visualization
        """
        components = {}
        total_reward = 0.0
        
        # Extract observation components
        hoop_x, hoop_y, hoop_visible, hoop_distance = obs[:4]
        vx, vy, vz, yaw_rate = obs[4:8]
        
        # 1. Visual tracking reward
        if hoop_visible > 0.5:
            components['visual_tracking'] = self.config.hoop_visible_reward
            self.steps_without_visual = 0
            
            # 2. Alignment rewards
            # Horizontal alignment (left/right)
            horizontal_error = abs(hoop_x)
            horizontal_reward = self.config.horizontal_alignment_scale * (1.0 - horizontal_error)
            components['horizontal_alignment'] = horizontal_reward
            
            # Vertical alignment (up/down)
            vertical_error = abs(hoop_y)
            vertical_reward = self.config.vertical_alignment_scale * (1.0 - vertical_error)
            components['vertical_alignment'] = vertical_reward
            
            # Perfect alignment bonus
            if horizontal_error < 0.1 and vertical_error < 0.1:
                components['perfect_alignment'] = self.config.perfect_alignment_bonus
            else:
                components['perfect_alignment'] = 0.0
            
            # 3. Distance and approach reward
            if self.prev_distance is not None:
                distance_change = self.prev_distance - hoop_distance
                if distance_change > 0:  # Getting closer
                    approach_reward = self.config.approach_reward_scale * distance_change
                    components['approach'] = approach_reward
                else:
                    components['approach'] = 0.0
            else:
                components['approach'] = 0.0
            
            self.prev_distance = hoop_distance
            
            # Proximity bonus
            if hoop_distance < self.config.proximity_bonus_threshold:
                components['proximity_bonus'] = self.config.proximity_bonus
            else:
                components['proximity_bonus'] = 0.0
        else:
            # No visual contact
            self.steps_without_visual += 1
            components['visual_tracking'] = 0.0
            components['horizontal_alignment'] = 0.0
            components['vertical_alignment'] = 0.0
            components['perfect_alignment'] = 0.0
            components['approach'] = 0.0
            components['proximity_bonus'] = 0.0
            
            # Penalty for losing visual
            if self.steps_without_visual > 5:
                components['lost_visual_penalty'] = self.config.lost_visual_penalty
            else:
                components['lost_visual_penalty'] = 0.0
        
        # 4. Hoop passage reward
        if info.get('hoop_passed', False):
            components['hoop_passage'] = self.config.hoop_passage_reward
            
            # Clean passage bonus (if well-centered)
            if horizontal_error < 0.2 and vertical_error < 0.2:
                components['clean_passage'] = self.config.clean_passage_bonus
            else:
                components['clean_passage'] = 0.0
            
            self.hoops_passed += 1
        else:
            components['hoop_passage'] = 0.0
            components['clean_passage'] = 0.0
        
        # 5. Movement and control rewards
        # Forward progress
        forward_speed = vx  # Normalized velocity
        if forward_speed > 0:
            components['forward_progress'] = self.config.forward_progress_scale * forward_speed
        else:
            components['forward_progress'] = 0.0
        
        # Smooth control (penalize jerky movements)
        if self.prev_action is not None:
            action_change = np.linalg.norm(action - self.prev_action)
            smooth_reward = self.config.smooth_control_scale * (1.0 - min(action_change, 1.0))
            components['smooth_control'] = smooth_reward
        else:
            components['smooth_control'] = 0.0
        
        self.prev_action = action.copy()
        
        # Hover penalty
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        if speed < 0.1:  # Nearly stationary
            components['hover_penalty'] = self.config.hover_penalty
        else:
            components['hover_penalty'] = 0.0
        
        # 6. Penalties
        # Collision
        if info.get('collision', False):
            components['collision_penalty'] = self.config.collision_penalty
        else:
            components['collision_penalty'] = 0.0
        
        # Out of bounds
        if info.get('out_of_bounds', False):
            components['out_of_bounds_penalty'] = self.config.out_of_bounds_penalty
        else:
            components['out_of_bounds_penalty'] = 0.0
        
        # Excessive yaw
        if abs(yaw_rate) > 0.8:  # Normalized threshold
            components['excessive_yaw_penalty'] = self.config.excessive_yaw_penalty
        else:
            components['excessive_yaw_penalty'] = 0.0
        
        # 7. Time efficiency
        components['time_penalty'] = self.config.time_penalty_per_step
        self.total_steps += 1
        
        # Calculate total reward
        total_reward = sum(components.values())
        
        # Add efficiency tracking
        if self.total_steps > 0:
            efficiency = (total_reward + abs(components.get('time_penalty', 0))) / self.total_steps
            if efficiency > self.config.efficiency_bonus_threshold:
                efficiency_bonus = 10.0
                components['efficiency_bonus'] = efficiency_bonus
                total_reward += efficiency_bonus
        
        return total_reward, components
    
    def reset(self):
        """Reset reward function for new episode"""
        self.prev_distance = None
        self.prev_action = None
        self.steps_without_visual = 0
        self.total_steps = 0
        self.hoops_passed = 0
        self.was_in_front = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reward statistics"""
        return {
            'hoops_passed': self.hoops_passed,
            'total_steps': self.total_steps,
            'steps_without_visual': self.steps_without_visual
        }


# Registry for different reward presets (AWS DeepRacer style)
REWARD_PRESETS = {
    'beginner': RewardConfig(
        hoop_visible_reward=5.0,
        horizontal_alignment_scale=15.0,
        vertical_alignment_scale=15.0,
        approach_reward_scale=20.0,
        hoop_passage_reward=150.0,
        collision_penalty=-100.0,
        time_penalty_per_step=-0.05
    ),
    'intermediate': RewardConfig(
        hoop_visible_reward=2.0,
        horizontal_alignment_scale=10.0,
        vertical_alignment_scale=10.0,
        approach_reward_scale=15.0,
        hoop_passage_reward=100.0,
        collision_penalty=-50.0,
        time_penalty_per_step=-0.1
    ),
    'advanced': RewardConfig(
        hoop_visible_reward=1.0,
        horizontal_alignment_scale=5.0,
        vertical_alignment_scale=5.0,
        approach_reward_scale=10.0,
        hoop_passage_reward=75.0,
        collision_penalty=-25.0,
        time_penalty_per_step=-0.2
    ),
    'speed_focused': RewardConfig(
        forward_progress_scale=15.0,
        time_penalty_per_step=-0.3,
        hoop_passage_reward=50.0,
        smooth_control_scale=5.0
    ),
    'precision_focused': RewardConfig(
        horizontal_alignment_scale=20.0,
        vertical_alignment_scale=20.0,
        perfect_alignment_bonus=50.0,
        clean_passage_bonus=100.0,
        hover_penalty=-0.5
    )
}


def get_reward_preset(name: str) -> RewardConfig:
    """Get a preset reward configuration"""
    if name not in REWARD_PRESETS:
        logger.warning(f"Unknown preset {name}, using default")
        return RewardConfig()
    return REWARD_PRESETS[name]


def reward_function(params: Dict[str, Any]) -> float:
    """
    DeepRacer-style reward entrypoint with flight phase awareness.

    Expected params (best-effort; missing keys are handled safely):
    - hoop_detected: bool
    - hoop_center_x: float in [-1, 1]
    - hoop_center_y: float in [-1, 1]
    - hoop_distance: float in [0, 1] (0=close)
    - vx_norm, vy_norm, vz_norm, yaw_rate_norm: floats in [-1, 1]
    - hoop_passed: bool
    - collision: bool
    - out_of_bounds: bool
    
    # Flight phase specific params:
    - flight_phase: str (takeoff, scan, plan, navigate, return, land, failed)
    - phase_progress: float in [0, 1]
    - hoops_detected: int (number of hoops found during scan)
    - scan_progress: float in [0, 1] (360° scan completion)
    """
    cfg = RewardConfig()
    total = 0.0

    hoop_detected = bool(params.get('hoop_detected', False))
    cx = float(params.get('hoop_center_x', 0.0))
    cy = float(params.get('hoop_center_y', 0.0))
    dist = float(params.get('hoop_distance', 1.0))
    vx = float(params.get('vx_norm', 0.0))
    vy = float(params.get('vy_norm', 0.0))
    vz = float(params.get('vz_norm', 0.0))
    yaw_rate = float(params.get('yaw_rate_norm', 0.0))

    # 1) Visual tracking and alignment
    if hoop_detected:
        total += cfg.hoop_visible_reward
        horizontal_reward = cfg.horizontal_alignment_scale * (1.0 - abs(cx))
        vertical_reward = cfg.vertical_alignment_scale * (1.0 - abs(cy))
        total += max(horizontal_reward, 0.0)
        total += max(vertical_reward, 0.0)
        if abs(cx) < 0.1 and abs(cy) < 0.1:
            total += cfg.perfect_alignment_bonus
        # Approach and proximity
        total += cfg.approach_reward_scale * max(1.0 - dist, 0.0)
        if dist < cfg.proximity_bonus_threshold:
            total += cfg.proximity_bonus
    else:
        total += cfg.lost_visual_penalty

    # 2) Movement and smoothness
    forward_speed = max(vx, 0.0)
    total += cfg.forward_progress_scale * forward_speed
    speed = (vx**2 + vy**2 + vz**2) ** 0.5
    if speed < 0.1:
        total += cfg.hover_penalty

    # 3) Events and penalties
    if bool(params.get('hoop_passed', False)):
        total += cfg.hoop_passage_reward
        if abs(cx) < 0.2 and abs(cy) < 0.2:
            total += cfg.clean_passage_bonus
    if bool(params.get('collision', False)):
        total += cfg.collision_penalty
    if bool(params.get('out_of_bounds', False)):
        total += cfg.out_of_bounds_penalty
    if abs(yaw_rate) > 0.8:
        total += cfg.excessive_yaw_penalty

    # 4) Flight phase bonuses (AWS DeepRacer-style progress rewards)
    flight_phase = params.get('flight_phase', 'takeoff')
    phase_progress = float(params.get('phase_progress', 0.0))
    
    if flight_phase == 'takeoff':
        total += 5.0 * phase_progress  # Reward taking off
    elif flight_phase == 'scan':
        scan_progress = float(params.get('scan_progress', 0.0))
        hoops_detected = int(params.get('hoops_detected', 0))
        total += 10.0 * scan_progress  # Reward 360° scanning
        total += 15.0 * hoops_detected  # Big bonus for finding hoops
    elif flight_phase == 'navigate':
        total += 20.0 * phase_progress  # Major reward for navigation progress
    elif flight_phase == 'return':
        total += 10.0 * phase_progress  # Reward returning home
    elif flight_phase == 'land':
        total += 25.0 * phase_progress  # Reward safe landing
        if phase_progress >= 1.0:
            total += 100.0  # Huge bonus for successful episode completion
    elif flight_phase == 'failed':
        total -= 100.0  # Heavy penalty for failure (like DeepRacer off-track)

    # 5) Time penalty per step
    total += cfg.time_penalty_per_step

    return float(total)