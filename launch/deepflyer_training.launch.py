#!/usr/bin/env python3
"""
DeepFlyer Training System Launch File
Launches all components for P3O training with drone
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Declare launch arguments
    training_mode_arg = DeclareLaunchArgument(
        'training_mode',
        default_value='true',
        description='Enable training mode'
    )
    
    use_gazebo_arg = DeclareLaunchArgument(
        'use_gazebo',
        default_value='true',
        description='Use Gazebo simulation'
    )
    
    use_zed_arg = DeclareLaunchArgument(
        'use_zed',
        default_value='true',
        description='Use ZED Mini camera'
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='trained_models/p3o/p3o_training.pt',
        description='Path to P3O model'
    )
    
    reward_preset_arg = DeclareLaunchArgument(
        'reward_preset',
        default_value='intermediate',
        description='Reward function preset'
    )
    
    # Get launch configurations
    training_mode = LaunchConfiguration('training_mode')
    use_gazebo = LaunchConfiguration('use_gazebo')
    use_zed = LaunchConfiguration('use_zed')
    model_path = LaunchConfiguration('model_path')
    reward_preset = LaunchConfiguration('reward_preset')
    
    # P3O RL Agent Node
    rl_agent_node = Node(
        package='deepflyer',
        executable='rl_agent_node.py',
        name='rl_agent_node',
        output='screen',
        parameters=[{
            'training_mode': training_mode,
            'model_save_path': model_path,
            'action_frequency': 20.0,
            'enable_clearml': True,
            'clearml_project': 'DeepFlyer Training',
            'clearml_task': 'Hoop Navigation Training'
        }],
        remappings=[
            # PX4 topics
            ('/fmu/out/vehicle_local_position', '/fmu/out/vehicle_local_position'),
            ('/fmu/out/vehicle_status', '/fmu/out/vehicle_status'),
            ('/fmu/in/trajectory_setpoint', '/fmu/in/trajectory_setpoint'),
            ('/fmu/in/offboard_control_mode', '/fmu/in/offboard_control_mode'),
            # Vision topics
            ('/deepflyer/vision_features', '/deepflyer/vision_features'),
            ('/deepflyer/course_state', '/deepflyer/course_state'),
            # Output topics
            ('/deepflyer/rl_action', '/deepflyer/rl_action'),
            ('/deepflyer/reward_feedback', '/deepflyer/reward_feedback')
        ]
    )
    
    # Vision Processing Node (YOLO11 + ZED)
    vision_processor_node = Node(
        package='deepflyer',
        executable='vision_processor_node.py',
        name='vision_processor_node',
        output='screen',
        parameters=[{
            'use_zed': use_zed,
            'yolo_model': 'weights/best.pt',
            'detection_threshold': 0.5,
            'camera_resolution': 'HD720',
            'camera_fps': 30
        }],
        condition=IfCondition(use_zed),
        remappings=[
            # ZED topics
            ('/zed_mini/zed_node/rgb/image_rect_color', '/zed_mini/zed_node/rgb/image_rect_color'),
            ('/zed_mini/zed_node/depth/depth_registered', '/zed_mini/zed_node/depth/depth_registered'),
            # Output
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
            'use_px4_com': True,  # Use PX4-ROS-COM instead of MAVROS
            'safety_enabled': True,
            'geofence_radius': 5.0,
            'min_altitude': 0.3,
            'max_altitude': 3.0
        }],
        remappings=[
            # Input from RL agent
            ('/deepflyer/rl_action', '/deepflyer/rl_action'),
            # PX4 communication
            ('/fmu/in/trajectory_setpoint', '/fmu/in/trajectory_setpoint'),
            ('/fmu/in/offboard_control_mode', '/fmu/in/offboard_control_mode'),
            ('/fmu/in/vehicle_command', '/fmu/in/vehicle_command'),
            ('/fmu/out/vehicle_local_position', '/fmu/out/vehicle_local_position'),
            ('/fmu/out/vehicle_status', '/fmu/out/vehicle_status')
        ]
    )
    
    # Reward Calculator Node
    reward_calculator_node = Node(
        package='deepflyer',
        executable='reward_calculator_node.py',
        name='reward_calculator_node',
        output='screen',
        parameters=[{
            'reward_preset': reward_preset,
            'enable_student_mode': True
        }],
        remappings=[
            # Inputs
            ('/deepflyer/vision_features', '/deepflyer/vision_features'),
            ('/deepflyer/drone_state', '/deepflyer/drone_state'),
            ('/deepflyer/rl_action', '/deepflyer/rl_action'),
            # Output
            ('/deepflyer/reward_feedback', '/deepflyer/reward_feedback')
        ]
    )
    
    # Training Monitor Node (for dashboard)
    training_monitor_node = Node(
        package='deepflyer',
        executable='training_monitor_node.py',
        name='training_monitor_node',
        output='screen',
        parameters=[{
            'update_rate': 2.0,  # Hz
            'enable_clearml': True,
            'clearml_project': 'DeepFlyer',
            'clearml_task': 'P3O_Training'
        }],
        condition=IfCondition(training_mode),
        remappings=[
            ('/deepflyer/reward_feedback', '/deepflyer/reward_feedback'),
            ('/deepflyer/training_metrics', '/deepflyer/training_metrics')
        ]
    )
    
    # Include Gazebo simulation if enabled
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('deepflyer'),
                'launch',
                'gazebo_world.launch.py'
            ])
        ]),
        condition=IfCondition(use_gazebo),
        launch_arguments={
            'world_name': 'hoop_course',
            'drone_model': 'x500_v2'
        }.items()
    )
    
    # Include ZED camera launch if enabled
    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('zed_wrapper'),
                'launch',
                'zed_mini.launch.py'
            ])
        ]),
        condition=IfCondition(use_zed),
        launch_arguments={
            'camera_model': 'zed_mini',
            'publish_tf': 'true',
            'publish_map_tf': 'false'
        }.items()
    )
    
    return LaunchDescription([
        # Launch arguments
        training_mode_arg,
        use_gazebo_arg,
        use_zed_arg,
        model_path_arg,
        reward_preset_arg,
        
        # Nodes
        rl_agent_node,
        vision_processor_node,
        px4_interface_node,
        reward_calculator_node,
        training_monitor_node,
        
        # Includes
        gazebo_launch,
        zed_launch
    ])