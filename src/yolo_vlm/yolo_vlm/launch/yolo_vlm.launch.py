#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'api',
            default_value='mlc',
            description='The model backend to use'
        ),
        DeclareLaunchArgument(
            'quantization',
            default_value='q4f16_ft',
            description='The quantization method to use'
        )
    ]
    
    yolo_node = Node(
        package='yolo_vlm',
        executable='yolo',
        name='yolo',
        output='screen'
    )
    
    vlm_node = Node(
        package='yolo_vlm',
        executable='nano_llm',
        name='nano_llm',
        output='screen',
        parameters=[{
            'model': 'Efficient-Large-Model/VILA-2.7b',
            'api': LaunchConfiguration('api'),
            'quantization': LaunchConfiguration('quantization'),
        }]
    )
    
    return LaunchDescription(launch_args + [yolo_node, vlm_node])
