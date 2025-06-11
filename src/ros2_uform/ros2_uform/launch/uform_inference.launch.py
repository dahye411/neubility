from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    cam2image_node = Node(
        package='image_tools',
        executable='cam2image',
        name='cam2image_node',
        remappings=[('image', 'input_image')],
        parameters=[{
            'device': '/dev/video0',
            'width': 640,
            'height': 480,
            'fps': 10
        }]
    )

    uform_node = Node(
        package='ros2_uform',
        executable='uform_node',
        name='uform_node',
        output='screen',
    )

    return LaunchDescription([
        cam2image_node,
        uform_node
    ])
