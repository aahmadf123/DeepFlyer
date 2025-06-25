"""
DeepFlyer Reward Function
Students can modify the parameters below or completely rewrite this function
"""

def reward_function(params):
    """
    DeepFlyer Hoop Navigation Reward Function
    
    This function receives sensor data and returns a reward value.
    Students can modify the parameters below or completely rewrite this function.
    
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
    
    # ============================================================================
    # STUDENT-MODIFIABLE PARAMETERS
    # ============================================================================
    
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
    
    # ============================================================================
    # REWARD CALCULATION (Students can modify this logic too)
    # ============================================================================
    
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


# ============================================================================
# EXAMPLES OF DIFFERENT REWARD STRATEGIES
# ============================================================================

def speed_focused_reward(params):
    """
    Example: Reward function focused on speed over precision
    Students can use this as inspiration or copy this approach
    """
    reward = 0.0
    
    # High rewards for speed and progress
    if params.get('making_progress', False):
        reward += 15.0  # Higher than default
    
    if params.get('hoop_passed', False):
        reward += 30.0  # Lower than default (less precision focus)
    
    # Lower penalties for missing hoops (accepting some misses for speed)
    if params.get('missed_hoop', False):
        reward -= 10.0  # Less harsh than default
    
    return reward


def precision_focused_reward(params):
    """
    Example: Reward function focused on precision over speed
    Students can use this as inspiration or copy this approach
    """
    reward = 0.0
    
    # High rewards for precision
    if params.get('hoop_passed', False):
        reward += 80.0  # Much higher than default
        if params.get('center_passage', False):
            reward += 50.0  # Big bonus for center passages
    
    # High penalties for missing
    if params.get('missed_hoop', False):
        reward -= 60.0  # Much harsher than default
    
    # Reward good alignment
    hoop_alignment = params.get('hoop_alignment', 0.0)
    if abs(hoop_alignment) < 0.1:  # Very well centered
        reward += 10.0
    
    return reward


def balanced_reward(params):
    """
    Example: Balanced approach between speed and precision
    This is similar to the main reward_function above
    """
    # This would be similar to the main function
    # Students can study this as an example of balanced design
    pass


# ============================================================================
# STUDENT NOTES AND TIPS
# ============================================================================

"""
TIPS FOR STUDENTS:

1. START SIMPLE: Modify just one or two parameters at first to see the effect

2. OBSERVE BEHAVIOR: Run the simulation and watch how the drone behaves
   - Does it approach hoops but miss them? Increase HOOP_PASSAGE_REWARD
   - Does it fly too slowly? Increase FORWARD_PROGRESS_REWARD
   - Does it ignore hoops? Increase HOOP_APPROACH_REWARD

3. BALANCE REWARDS: Make sure rewards are proportional
   - HOOP_PASSAGE_REWARD should be larger than HOOP_APPROACH_REWARD
   - Completion bonuses should be much larger than individual action rewards

4. TEST DIFFERENT STRATEGIES:
   - Speed-focused: High progress rewards, lower precision requirements
   - Precision-focused: High accuracy rewards, harsh miss penalties
   - Balanced: Moderate rewards for both speed and accuracy

5. EXPERIMENT: Try completely different approaches!
   - What if you only reward hoop passages?
   - What if you heavily penalize slow progress?
   - What if you give bigger bonuses for staying centered?

6. DEBUG: If drone behavior seems strange, check:
   - Are any rewards much larger/smaller than others?
   - Are you accidentally rewarding the wrong behavior?
   - Are penalties too harsh or too lenient?

REMEMBER: There's no "perfect" reward function - it depends on what behavior 
you want the drone to learn. Experiment and have fun!
""" 