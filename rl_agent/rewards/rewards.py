"""
DeepFlyer Hoop Navigation Reward Function

This function receives sensor data and returns a reward value.
Parameters can be modified below or the function can be completely rewritten.

Args:
    params (dict): Dictionary containing sensor data and environment state
    
Available parameters:
    - hoop_detected (bool): Is a hoop visible in camera?
    - hoop_distance (float): Distance to visible hoop in meters
    - hoop_alignment (float): How centered hoop is (-1.0 to 1.0, 0=center)
    - approaching_hoop (bool): Is drone getting closer to target hoop?
    - hoop_passed (bool): Did drone just pass through a hoop?
    - center_passage (bool): Did drone pass through center of hoop?
    - making_progress (bool): Is drone making forward progress?
    - lap_completed (bool): Did drone just complete a lap (5 hoops)?
    - course_completed (bool): Did drone complete all 3 laps?
    - missed_hoop (bool): Did drone miss/go around a hoop?
    - collision (bool): Did drone hit something?
    - slow_progress (bool): Is drone taking too long?
    - out_of_bounds (bool): Is drone outside safe flight area?
    
Returns:
    float: Reward value (positive for good behavior, negative for bad)
"""

def reward_function(params):
    
    # Parameters
    
    # Hoop Navigation Rewards
    HOOP_APPROACH_REWARD = 10.0      # Getting closer to target hoop
    HOOP_PASSAGE_REWARD = 50.0       # Successfully passing through hoop
    HOOP_CENTER_BONUS = 20.0         # Passing through center of hoop
    VISUAL_ALIGNMENT_REWARD = 5.0    # Keeping hoop centered in camera view
    
    # Speed and Efficiency
    FORWARD_PROGRESS_REWARD = 3.0    # Making progress toward goal
    SPEED_EFFICIENCY_BONUS = 2.0     # Maintaining good speed
    
    # Lap Completion Bonuses
    LAP_COMPLETION_BONUS = 100.0     # Completing a full lap (5 hoops)
    COURSE_COMPLETION_BONUS = 500.0  # Completing all 3 laps
    
    # Safety Penalties
    HOOP_MISS_PENALTY = -25.0        # Flying around instead of through hoop
    COLLISION_PENALTY = -100.0       # Hitting hoop or obstacle
    SLOW_PROGRESS_PENALTY = -1.0     # Taking too long
    OUT_OF_BOUNDS_PENALTY = -200.0   # Flying outside safe area
    
    # Reward Calculation
    
    total_reward = 0.0
    
    # Reward for approaching hoops
    if params.get('approaching_hoop', False):
        total_reward += HOOP_APPROACH_REWARD
    
    # Hoop passage rewards
    if params.get('hoop_passed', False):
        total_reward += HOOP_PASSAGE_REWARD
        # Extra bonus for center passage
        if params.get('center_passage', False):
            total_reward += HOOP_CENTER_BONUS
    
    # Visual alignment reward (hoop centered in camera)
    hoop_alignment = params.get('hoop_alignment', 0.0)
    if abs(hoop_alignment) < 0.2:  # Well centered
        total_reward += VISUAL_ALIGNMENT_REWARD
    
    # Progress rewards
    if params.get('making_progress', False):
        total_reward += FORWARD_PROGRESS_REWARD
    
    # Speed efficiency (example of using multiple parameters)
    if params.get('making_progress', False) and params.get('hoop_detected', False):
        total_reward += SPEED_EFFICIENCY_BONUS
    
    # Major completion bonuses
    if params.get('lap_completed', False):
        total_reward += LAP_COMPLETION_BONUS
    
    if params.get('course_completed', False):
        total_reward += COURSE_COMPLETION_BONUS
    
    # Penalties for undesired behavior
    if params.get('missed_hoop', False):
        total_reward += HOOP_MISS_PENALTY
    
    if params.get('collision', False):
        total_reward += COLLISION_PENALTY
    
    if params.get('slow_progress', False):
        total_reward += SLOW_PROGRESS_PENALTY
    
    if params.get('out_of_bounds', False):
        total_reward += OUT_OF_BOUNDS_PENALTY
    
    return float(total_reward)

