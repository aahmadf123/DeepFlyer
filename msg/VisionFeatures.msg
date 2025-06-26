# VisionFeatures.msg - YOLO11 vision processing results for hoop detection

# Header for timestamp and frame_id  
std_msgs/Header header

# Primary hoop detection
bool hoop_detected                    # Whether any hoop is visible
uint32 num_hoops_detected            # Number of hoops detected in frame

# Primary target hoop (closest or most centered)
float32 hoop_center_x                # Normalized x position (-1.0 = left, 1.0 = right)
float32 hoop_center_y                # Normalized y position (-1.0 = bottom, 1.0 = top)
float32 hoop_diameter_pixels         # Hoop diameter in pixels
float32 hoop_distance                # Distance to hoop in meters (from depth)
float32 confidence                   # Detection confidence (0.0 - 1.0)

# Alignment metrics for RL
float32 alignment_error              # Horizontal alignment error (-1.0 to 1.0)
float32 approach_angle               # Angle of approach to hoop (radians)
bool centered_in_frame              # True if hoop is well-centered

# Multi-hoop context
uint32[] other_hoop_ids             # IDs of other detected hoops
float32[] other_hoop_distances      # Distances to other hoops

# Processing performance
uint32 processing_time_ms           # Time taken for vision processing
float32 fps                         # Current processing frame rate
bool tracking_stable                # Whether tracking is stable across frames

# Error and status flags  
uint32 error_code                   # 0=OK, >0=error conditions
string error_message               # Human readable error description 