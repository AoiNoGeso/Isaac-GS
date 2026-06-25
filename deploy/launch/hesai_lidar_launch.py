from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    hesai_package_dir = get_package_share_directory('hesai_ros_driver')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(hesai_package_dir, 'launch',
                             'start.py'))),
        Node(
            package='pointcloud_to_laserscan', executable='pointcloud_to_laserscan_node',
            remappings=[('cloud_in', '/lidar_points'),
                        ('scan', '/scan')],
            parameters=[{
                'target_frame': '',
                'transform_tolerance': 0.01,
                'min_height': -0.2,
                'max_height': 0.2,
                'angle_min': -3.1415,  # -M_PI
                'angle_max': 3.1415,  # M_PI
                # XT32仕様: 10Hz時の水平分解能は0.18度 -> 0.00314 rad
                'angle_increment': 0.00628,
                'scan_time': 0.05,
                'range_min': 0.05,
                'range_max': 120.0,
                'use_inf': True,
                'inf_epsilon': 1.0
            }],
            name='pointcloud_to_laserscan'
        )
    ])