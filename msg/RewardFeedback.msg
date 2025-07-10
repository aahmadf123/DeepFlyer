# RewardFeedback.msg
# Detailed reward feedback for RL training and monitoring
# Compatible with both DeepRacer-style and component-based reward functions

# Header for timestamp and frame coordination
std_msgs/Header header

# Total Reward Information
float32 total_reward            # Current step reward
float32 cumulative_reward       # Total reward accumulated this episode
float32 total_episode_reward    # Same as cumulative_reward (for compatibility)
int32 episode_step              # Current step in episode
float32 episode_time            # Time elapsed in current episode (seconds)

# Individual Reward Components (DeepRacer-style naming)
float32 hoop_detected_reward        # Reward for detecting hoop
float32 horizontal_align_reward     # Reward for horizontal alignment
float32 vertical_align_reward       # Reward for vertical alignment  
float32 depth_closer_reward         # Reward for getting closer to hoop
float32 hoop_passage_reward         # Reward for passing through hoop
float32 roundtrip_finish_reward     # Reward for completing full trajectory

# Extended Component Rewards (detailed breakdown)
float32 hoop_approach_reward        # Reward for approaching hoop
float32 hoop_center_bonus           # Bonus for good center alignment
float32 visual_alignment_reward     # Reward for visual tracking quality
float32 forward_progress_reward     # Reward for forward motion
float32 speed_efficiency_bonus      # Bonus for optimal speed
float32 lap_completion_bonus        # Bonus for completing lap
float32 course_completion_bonus     # Bonus for finishing course
float32 smooth_flight_bonus         # Bonus for smooth control
float32 precision_bonus             # Bonus for precise navigation

# Penalties (negative rewards)
float32 collision_penalty           # Penalty for collisions
float32 missed_hoop_penalty         # Penalty for missing hoop
float32 drift_lost_penalty          # Penalty for losing track
float32 time_penalty               # Penalty for time usage
float32 wrong_direction_penalty     # Penalty for going wrong way
float32 hoop_miss_penalty          # Penalty for hoop miss
float32 slow_progress_penalty      # Penalty for slow progress
float32 erratic_flight_penalty     # Penalty for erratic control
float32 boundary_violation_penalty  # Penalty for boundary violations
float32 emergency_landing_penalty   # Penalty for emergency situations

# Current Context Information
float32 distance_to_target         # Current distance to target (meters)
int32 current_hoop_id             # ID of current target hoop
int32 current_lap                 # Current lap number
float32 hoop_alignment_error      # Current alignment error
bool hoop_visible                 # Whether hoop is currently visible
int32 hoop_passages_completed     # Number of hoops passed through
string current_flight_phase       # Current phase of flight

# Performance Metrics
float32 best_episode_reward        # Best reward achieved so far
float32 average_reward_last_10     # Average reward over last 10 episodes
int32 successful_trajectories      # Number of successful completions
int32 total_episodes              # Total episodes attempted
float32 episode_efficiency        # Reward per time ratio
float32 learning_progress         # Overall learning progress [0.0, 1.0]
bool goal_achieved                # Whether main goal was achieved

# Configurable Reward Settings (for tuning)
float32 hoop_detected_setting      # Current setting for hoop detection reward
float32 horizontal_align_setting   # Current setting for horizontal alignment
float32 vertical_align_setting     # Current setting for vertical alignment
float32 depth_closer_setting       # Current setting for depth approach
float32 hoop_passage_setting       # Current setting for hoop passage
float32 roundtrip_finish_setting   # Current setting for trajectory completion
float32 collision_penalty_setting  # Current setting for collision penalty
float32 missed_hoop_penalty_setting # Current setting for miss penalty
float32 drift_lost_penalty_setting  # Current setting for drift penalty
float32 time_penalty_setting       # Current setting for time penalty 