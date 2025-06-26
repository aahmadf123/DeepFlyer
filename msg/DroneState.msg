# DroneState.msg - Comprehensive drone state information for RL

# Header for timestamp and frame coordination
std_msgs/Header header

# Position and orientation (ENU coordinate frame)
geometry_msgs/Point position          # Current position (x, y, z) in meters
geometry_msgs/Quaternion orientation  # Current orientation as quaternion
geometry_msgs/Vector3 euler_angles    # Euler angles (roll, pitch, yaw) in radians

# Velocity and acceleration
geometry_msgs/Vector3 linear_velocity     # Linear velocity (vx, vy, vz) m/s
geometry_msgs/Vector3 angular_velocity    # Angular velocity (wx, wy, wz) rad/s
geometry_msgs/Vector3 linear_acceleration # Linear acceleration (ax, ay, az) m/s²

# Flight controller status
bool armed                         # True if drone is armed
bool offboard_mode                # True if in offboard control mode
string flight_mode                # Current flight mode (e.g., "OFFBOARD", "MANUAL")
uint32 system_status              # Overall system status code
bool position_valid               # True if position estimate is valid
bool velocity_valid               # True if velocity estimate is valid

# Battery and power
float32 battery_voltage           # Battery voltage (V)
float32 battery_current           # Battery current (A) 
float32 battery_remaining         # Battery remaining (0.0 - 1.0)
uint32 battery_warning_level      # Battery warning level (0=OK, 1=LOW, 2=CRITICAL)

# Safety status
bool safety_switch_enabled        # True if safety switch is enabled
bool geofence_violated            # True if geofence has been violated
bool collision_imminent           # True if collision is detected as imminent
uint32 failure_flags              # Bitmask of failure conditions

# Control inputs and outputs
geometry_msgs/Vector3 control_input    # Current control input (normalized)
float32[] motor_outputs               # Individual motor outputs (0.0 - 1.0)
float32 throttle_position           # Current throttle position (0.0 - 1.0)

# Environmental sensing
float32 ground_distance           # Distance to ground (meters)
float32 obstacle_distance         # Distance to nearest obstacle (meters)
geometry_msgs/Vector3 wind_velocity   # Estimated wind velocity (m/s)
float32 air_pressure              # Barometric pressure (hPa)
float32 temperature               # Environmental temperature (°C)

# Performance metrics
float32 position_accuracy         # Estimated position accuracy (meters)
float32 control_latency          # Control loop latency (milliseconds)
uint32 dropped_messages          # Count of dropped communication messages
float32 cpu_usage                # CPU usage percentage (0.0 - 100.0) 