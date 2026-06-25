from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        parameters=[{'use_mag': False},{'world_frame':"enu"},{"publish_tf":False}],
    )
    sony_imu = Node(
        package='cxd5602pwbimu_localizer_node',
        executable='localizer_node',
    )
    return LaunchDescription([
        sony_imu,
        # imu_filter,
    ])