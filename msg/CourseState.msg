# CourseState.msg
# MVP trajectory state and progress tracking
# Single hoop navigation: takeoff → scan → through hoop → return → land

# Header for timestamp and frame coordination
std_msgs/Header header

# MVP Flight Phase Information
string current_phase               # Current MVP phase: TAKEOFF, SCAN_360, NAVIGATE_TO_HOOP, etc.
float32 phase_progress            # Progress through current phase [0.0 - 1.0]
float32 phase_duration            # Time spent in current phase (seconds)
float32 total_flight_time         # Total time since takeoff (seconds)

# Hoop Information
bool hoop_detected                # Whether target hoop has been detected
geometry_msgs/Point hoop_position # 3D position of detected hoop
float32 hoop_diameter             # Estimated hoop diameter (meters)
int32 hoop_passages_completed     # Number of times passed through hoop (0, 1, or 2)

# Navigation State
geometry_msgs/Point drone_position    # Current drone position
geometry_msgs/Point takeoff_position  # Point A - takeoff/landing position
float32 distance_to_hoop          # Distance to target hoop (meters)
float32 distance_to_origin        # Distance back to Point A (meters)
float32 altitude                  # Current altitude (meters)

# MVP Trajectory Progress
bool takeoff_completed            # Reached target altitude
bool scan_completed               # Completed 360° scan
bool hoop_found                   # Hoop detected during scan
bool first_passage_completed      # First passage through hoop
bool return_passage_completed     # Return passage through hoop  
bool returned_to_origin           # Back to Point A
bool landing_completed            # Successfully landed
bool trajectory_completed         # Full MVP trajectory complete

# Episode Management
int32 episode_id                  # Current episode number
int32 episode_step                # Current step in episode
float32 episode_time              # Time elapsed in current episode (seconds)
bool episode_active               # Whether episode is currently running

# Performance Metrics
int32 successful_episodes         # Number of completed trajectories
float32 best_completion_time      # Fastest successful trajectory (seconds)
float32 average_completion_time   # Average time for successful trajectories
int32 collision_count             # Collisions in current episode
bool emergency_landing            # Whether emergency landing was triggered

# Scan Progress (during SCAN_360 phase)
float32 scan_start_yaw           # Yaw angle when scan started (radians)
float32 current_yaw              # Current yaw angle (radians)
float32 scan_progress            # Scan completion [0.0 - 1.0]
int32 hoops_detected_during_scan # Number of hoops found during scan

# Safety and Status
bool safety_override_active      # Safety constraints currently applied
bool in_safe_zone               # Whether drone is in designated safe area
string status_message           # Human-readable status information
int32 error_code                # 0=OK, >0=error condition 