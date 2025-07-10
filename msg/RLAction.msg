# RLAction.msg
# Reinforcement learning action commands for MVP hoop navigation
# 4D action space: vx, vy, vz, yaw_rate

# Header for timestamp and frame coordination
std_msgs/Header header

# Primary 4D Action Commands (normalized values)
float32 vx_cmd              # Forward/backward velocity command [-1.0, 1.0]
float32 vy_cmd              # Left/right velocity command [-1.0, 1.0]  
float32 vz_cmd              # Up/down velocity command [-1.0, 1.0]
float32 yaw_rate_cmd        # Yaw rate command [-1.0, 1.0]

# Raw Action Values (before normalization/scaling)
float32 raw_vx_cmd          # Raw vx value from RL agent
float32 raw_vy_cmd          # Raw vy value from RL agent
float32 raw_vz_cmd          # Raw vz value from RL agent
float32 raw_yaw_rate_cmd    # Raw yaw rate value from RL agent

# Action Metadata
float32 action_confidence   # Confidence score for this action [0.0, 1.0]
int32 episode_step          # Current step in episode
float32 episode_time        # Time elapsed in current episode (seconds)

# Safety and Control Status
bool safety_override        # True if safety constraints are applied
bool emergency_stop         # True if emergency stop is triggered
string flight_phase         # Current flight phase (e.g., "ACTIVE", "HOVERING")

# Action Source Information
string action_source        # Source of action (e.g., "P3O_AGENT", "MANUAL")
float32 action_latency      # Time between observation and action (milliseconds) 