# CourseState.msg - Course navigation state and management

# Header for timestamp and frame coordination
std_msgs/Header header

# Current course progress
uint32 current_target_hoop         # ID of current target hoop (0-based indexing)
uint32 current_lap                 # Current lap number (1-based)
uint32 total_laps                  # Total laps to complete
uint32 hoops_completed_this_lap    # Number of hoops passed in current lap
uint32 total_hoops_completed       # Total hoops passed in entire course

# Course layout information
uint32 num_hoops_in_course         # Total number of hoops in course layout
geometry_msgs/Point[] hoop_positions   # 3D positions of all hoops in course
float32[] hoop_diameters           # Diameter of each hoop (meters)

# Current target hoop details
geometry_msgs/Point target_position    # 3D position of current target hoop
float32 target_diameter            # Diameter of current target hoop
float32 distance_to_target         # Distance from drone to target (meters)
float32 bearing_to_target          # Bearing to target from current position (radians)

# Progress tracking
float32 lap_progress               # Progress through current lap (0.0 - 1.0)
float32 overall_progress           # Overall course progress (0.0 - 1.0)
float32 estimated_time_remaining   # Estimated time to course completion (seconds)

# Course events
bool hoop_passed                   # True if a hoop was just passed
bool lap_completed                 # True if a lap was just completed  
bool course_completed              # True if entire course is completed
bool new_target_hoop              # True if target hoop just changed

# Episode management
uint64 episode_id                  # Unique identifier for current episode
float32 episode_time_elapsed       # Time elapsed in current episode (seconds)
bool episode_reset_requested       # True if episode reset has been requested
uint32 episode_step_count          # Number of steps taken in current episode

# Performance metrics
float32 average_hoop_time          # Average time per hoop completion
float32 navigation_efficiency      # Efficiency score for navigation
uint32 collision_count             # Number of collisions in current episode
uint32 boundary_violations         # Number of boundary violations 