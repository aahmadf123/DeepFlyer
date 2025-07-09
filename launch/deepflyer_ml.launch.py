#!/usr/bin/env python3
"""
DeepFlyer ML Components Launch File
Starts all ML/RL components for integration testing
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for DeepFlyer ML components"""
    
    # Launch arguments
    custom_model_path_arg = DeclareLaunchArgument(
        'custom_model_path',
        default_value='weights/best.pt',
        description='Path to custom trained YOLO model'
    )
    
    yolo_model_size_arg = DeclareLaunchArgument(
        'yolo_model_size', 
        default_value='n',
        description='YOLO model size (n, s, m, l, x)'
    )
    
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.3',
        description='YOLO confidence threshold'
    )
    
    enable_clearml_arg = DeclareLaunchArgument(
        'enable_clearml',
        default_value='true',
        description='Enable ClearML experiment tracking'
    )
    
    # Nodes
    vision_processor_node = Node(
        package='deepflyer_msgs',
        executable='vision_processor_node.py',
        name='vision_processor',
        output='screen',
        parameters=[{
            'custom_model_path': LaunchConfiguration('custom_model_path'),
            'yolo_model_size': LaunchConfiguration('yolo_model_size'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'use_zed': True,
            'processing_frequency': 30.0,
            'publish_debug_images': True
        }]
    )
    
    p3o_agent_node = Node(
        package='deepflyer_msgs',
        executable='p3o_agent_node.py',
        name='p3o_agent',
        output='screen',
        parameters=[{
            'enable_clearml': LaunchConfiguration('enable_clearml'),
        }]
    )
    
    reward_calculator_node = Node(
        package='deepflyer_msgs', 
        executable='reward_calculator_node.py',
        name='reward_calculator',
        output='screen',
        parameters=[{
            'publish_frequency': 20.0,
            'enable_detailed_breakdown': True
        }]
    )
    
    course_manager_node = Node(
        package='deepflyer_msgs',
        executable='course_manager_node.py', 
        name='course_manager',
        output='screen',
        parameters=[{
            'spawn_x': 0.0,
            'spawn_y': 0.0, 
            'spawn_z': 0.8,
            'hoop_completion_radius': 0.4,
            'publish_frequency': 10.0
        }]
    )
    
    # MVP trajectory controller (optional)
    mvp_trajectory_node = Node(
        package='deepflyer_msgs',
        executable='mvp_trajectory.py',
        name='mvp_trajectory_controller',
        output='screen',
        condition=LaunchConfiguration('enable_mvp', default='false')
    )
    
    return LaunchDescription([
        # Arguments
        custom_model_path_arg,
        yolo_model_size_arg,
        confidence_threshold_arg,
        enable_clearml_arg,
        
        # Info
        LogInfo(msg="Starting DeepFlyer ML Components..."),
        LogInfo(msg="- Vision Processor (YOLO11)"),
        LogInfo(msg="- P3O RL Agent"),
        LogInfo(msg="- Reward Calculator"),
        LogInfo(msg="- Course Manager"),
        
        # Nodes
        vision_processor_node,
        p3o_agent_node,
        reward_calculator_node,
        course_manager_node,
        mvp_trajectory_node
    ]) 