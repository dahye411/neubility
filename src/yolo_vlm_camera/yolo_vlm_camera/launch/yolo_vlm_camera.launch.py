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
        package='yolo_vlm_camera',   # Package name (modify to match the actual package name)
        executable='yolo',    # Executable name of the yolo.py file (register the entry point if required)
        name='yolo',
        output='screen',
        parameters=[{'video_path': '/dev/video0'}]  # Use camera input
    )
    
    vlm_node = Node(
        package='yolo_vlm_camera',
        executable='nano_llm',  # Executable name of the nano_llm.py file
        name='nano_llm',
        output='screen',
        parameters=[{
            'model': 'Efficient-Large-Model/VILA-2.7b',
            'api': LaunchConfiguration('api'),
            'quantization': LaunchConfiguration('quantization'),
        }]
    )
    
    return LaunchDescription(launch_args + [yolo_node, vlm_node])
