#!/usr/bin/env python3
"""
MVP System Launch File

Launches the complete MVP hoop navigation system with:
- Vision processing (YOLO11 + ZED)
- PX4 flight control interface  
- RL agent with P3O algorithm
- Course management and reward feedback

For educational drone RL with real-time training capability.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for MVP system"""
    
    # Get package directory
    pkg_dir = get_package_share_directory('deepflyer')
    
    # Declare launch arguments
    training_mode_arg = DeclareLaunchArgument(
        'training_mode',
        default_value='true',
        description='Enable RL training mode'
    )
    
    debug_vision_arg = DeclareLaunchArgument(
        'debug_vision',
        default_value='false',
        description='Enable vision debug visualization'
    )
    
    enable_safety_arg = DeclareLaunchArgument(
        'enable_safety',
        default_value='true',
        description='Enable safety constraints'
    )
    
    auto_arm_arg = DeclareLaunchArgument(
        'auto_arm',
        default_value='false',
        description='Automatically arm vehicle when episode starts'
    )
    
    yolo_model_arg = DeclareLaunchArgument(
        'yolo_model',
        default_value='weights/best.pt',
        description='Path to YOLO11 model file'
    )
    
    takeoff_altitude_arg = DeclareLaunchArgument(
        'takeoff_altitude',
        default_value='1.5',
        description='Takeoff altitude in meters'
    )
    
    control_frequency_arg = DeclareLaunchArgument(
        'control_frequency',
        default_value='20.0',
        description='Control loop frequency (Hz)'
    )
    
    # Get launch configurations
    training_mode = LaunchConfiguration('training_mode')
    debug_vision = LaunchConfiguration('debug_vision')
    enable_safety = LaunchConfiguration('enable_safety')
    auto_arm = LaunchConfiguration('auto_arm')
    yolo_model = LaunchConfiguration('yolo_model')
    takeoff_altitude = LaunchConfiguration('takeoff_altitude')
    control_frequency = LaunchConfiguration('control_frequency')
    
    # Vision Processor Node
    vision_processor_node = Node(
        package='deepflyer',
        executable='vision_processor_node.py',
        name='vision_processor_node',
        output='screen',
        parameters=[{
            'yolo_model_path': yolo_model,
            'confidence_threshold': 0.5,
            'processing_fps': control_frequency,
            'debug_visualization': debug_vision
        }],
        remappings=[
            # ZED camera topics
            ('/zed_mini/zed_node/rgb/image_rect_color', '/zed_mini/zed_node/rgb/image_rect_color'),
            ('/zed_mini/zed_node/depth/depth_registered', '/zed_mini/zed_node/depth/depth_registered'),
            # Output topics
            ('/deepflyer/vision_features', '/deepflyer/vision_features')
        ]
    )
    
    # PX4 Interface Node
    px4_interface_node = Node(
        package='deepflyer',
        executable='px4_interface_node.py',
        name='px4_interface_node',
        output='screen',
        parameters=[{
            'enable_safety_limits': enable_safety,
            'takeoff_altitude': takeoff_altitude,
            'control_frequency': control_frequency,
            'auto_arm': auto_arm
        }],
        remappings=[
            # PX4 topics
            ('/fmu/in/vehicle_command', '/fmu/in/vehicle_command'),
            ('/fmu/in/offboard_control_mode', '/fmu/in/offboard_control_mode'),
            ('/fmu/in/trajectory_setpoint', '/fmu/in/trajectory_setpoint'),
            ('/fmu/out/vehicle_local_position', '/fmu/out/vehicle_local_position'),
            ('/fmu/out/vehicle_status', '/fmu/out/vehicle_status'),
            # DeepFlyer topics
            ('/deepflyer/rl_action', '/deepflyer/rl_action'),
            ('/deepflyer/course_state', '/deepflyer/course_state')
        ]
    )
    
    # RL Agent Node
    rl_agent_node = Node(
        package='deepflyer',
        executable='rl_agent_node.py',
        name='rl_agent_node',
        output='screen',
        parameters=[{
            'training_mode': training_mode,
            'model_save_path': 'models/mvp_p3o_model.pt',
            'training_config_file': 'config/mvp_training.json',
            'action_frequency': control_frequency
        }],
        remappings=[
            # Input topics
            ('/deepflyer/vision_features', '/deepflyer/vision_features'),
            ('/deepflyer/course_state', '/deepflyer/course_state'),
            ('/fmu/out/vehicle_local_position', '/fmu/out/vehicle_local_position'),
            # Output topics
            ('/deepflyer/rl_action', '/deepflyer/rl_action'),
            ('/deepflyer/reward_feedback', '/deepflyer/reward_feedback')
        ]
    )
    
    # Course Manager Node (optional - for episode management)
    course_manager_node = Node(
        package='deepflyer',
        executable='course_manager_node.py',
        name='course_manager_node',
        output='screen',
        parameters=[{
            'episode_timeout': 300.0,  # 5 minutes max episode time
            'auto_reset': True,
            'hoop_position_x': 5.0,    # Hoop 5m forward from takeoff
            'hoop_position_y': 0.0,
            'hoop_position_z': 1.5
        }],
        condition=IfCondition(training_mode)
    )
    
    # Reward Calculator Node (optional - for advanced reward processing)
    reward_calculator_node = Node(
        package='deepflyer',
        executable='reward_calculator_node.py',
        name='reward_calculator_node',
        output='screen',
        parameters=[{
            'enable_advanced_rewards': True,
            'trajectory_smoothing': True,
            'performance_scaling': True
        }],
        condition=IfCondition(training_mode)
    )
    
    # ZED Camera Node (if not already running)
    zed_camera_node = Node(
        package='zed_ros2',
        executable='zed_node',
        name='zed_node',
        namespace='zed_mini',
        output='screen',
        parameters=[{
            'camera_model': 'zedmini',
            'resolution': 2,  # 720p
            'frame_rate': 60,
            'depth_mode': 1,  # Performance mode
            'depth_stabilization': True,
            'point_cloud_freq': 10.0,
            'mapping_enabled': False,
            'odometry_enabled': False
        }],
        condition=UnlessCondition(LaunchConfiguration('use_bag_data', default='false'))
    )
    
    # Static transform publisher for camera frame
    camera_transform_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='zed_camera_transform',
        arguments=[
            '0', '0', '0',          # translation
            '0', '0', '0', '1',     # rotation (quaternion)
            'base_link',            # parent frame
            'zed_left_camera_frame' # child frame
        ]
    )
    
    # ROS2 bag recording (optional)
    bag_record_node = Node(
        package='ros2bag',
        executable='record',
        name='bag_recorder',
        arguments=[
            '-o', 'mvp_flight_data',
            '/deepflyer/vision_features',
            '/deepflyer/course_state', 
            '/deepflyer/rl_action',
            '/deepflyer/reward_feedback',
            '/fmu/out/vehicle_local_position',
            '/fmu/out/vehicle_status',
            '/zed_mini/zed_node/rgb/image_rect_color',
            '/zed_mini/zed_node/depth/depth_registered'
        ],
        condition=IfCondition(LaunchConfiguration('record_bag', default='false'))
    )
    
    # Create launch description
    return LaunchDescription([
        # Launch arguments
        training_mode_arg,
        debug_vision_arg,
        enable_safety_arg,
        auto_arm_arg,
        yolo_model_arg,
        takeoff_altitude_arg,
        control_frequency_arg,
        
        # System startup message
        LogInfo(msg="Starting DeepFlyer MVP System..."),
        LogInfo(msg=["Training mode: ", training_mode]),
        LogInfo(msg=["Safety enabled: ", enable_safety]),
        LogInfo(msg=["Control frequency: ", control_frequency, " Hz"]),
        
        # Core system nodes
        GroupAction([
            camera_transform_node,
            zed_camera_node,
            vision_processor_node,
            px4_interface_node,
            rl_agent_node
        ], scoped=True),
        
        # Optional training nodes
        GroupAction([
            course_manager_node,
            reward_calculator_node
        ], condition=IfCondition(training_mode), scoped=True),
        
        # Optional data recording
        GroupAction([
            bag_record_node
        ], condition=IfCondition(LaunchConfiguration('record_bag', default='false')), scoped=True),
        
        # System ready message
        LogInfo(msg="MVP System launched successfully!")
    ])


if __name__ == '__main__':
    generate_launch_description() 