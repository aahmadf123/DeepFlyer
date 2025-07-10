# VisionFeatures.msg
# Vision processing output for MVP hoop navigation
# Compatible with 8D observation space

# Header for timestamp and frame info
std_msgs/Header header

# Hoop detection status
bool hoop_detected

# Hoop center position in image coordinates (pixels)
int32 hoop_center_u
int32 hoop_center_v

# Hoop center position normalized to [-1, 1] range
# Used directly in 8D observation space
float32 hoop_center_x_norm
float32 hoop_center_y_norm

# Distance measurements
float32 hoop_distance_meters    # Raw distance in meters
float32 hoop_distance_norm      # Normalized distance [0, 1] for 8D observation

# Hoop geometry
float32 hoop_diameter_pixels    # Hoop diameter in pixels
float32 hoop_area_ratio         # Hoop area / total image area

# Detection confidence and quality
float32 detection_confidence    # YOLO11 confidence score [0, 1]
float32 depth_confidence       # Depth measurement confidence [0, 1]

# Additional hoop information
int32 hoop_id                  # Unique identifier for this hoop
bool next_hoop_visible         # Whether next target hoop is visible

# Image dimensions for reference
int32 image_width
int32 image_height 