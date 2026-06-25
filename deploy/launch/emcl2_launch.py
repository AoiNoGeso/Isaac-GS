import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

_DEPLOY_DIR = Path(__file__).resolve().parents[1]

# SLAM で生成した地図の yaml ファイルパスをここに指定する
MAP_YAML = str(_DEPLOY_DIR / 'map' / 'map.yaml')


def generate_launch_description():
    emcl_params = str(_DEPLOY_DIR / 'config' / 'emcl_config.yaml')
    emcl_package_dir = get_package_share_directory('emcl2')

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(_DEPLOY_DIR / 'launch' / 'sony_imu_launch.py'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(emcl_package_dir, 'launch', 'emcl2.launch.py')),
            launch_arguments=[
                ('params_file', emcl_params),
                ('map', MAP_YAML),
                ('use_sim_time', 'false'),
            ]),
        Node(
            package='serial_robot_driver',
            executable='mecanumrover_v2_serial_driver',
            name='mecanumrover_v2_serial_driver',
            parameters=[{'serial_port': "/dev/ttyACM0"},
                        {'print_tf': True},
                        {'coordinate': "world"},
                        {'imu_topic_name': "imu/data_raw"},
                        {'use_omega': False},
                        {'odom_angle_coefficient': 1.0},
                        {'control_posture': True},
                        {'imu_name': "livox"}]),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_laser',
            arguments=['0', '0', '0.3', '0', '0', '0', '1', 'base_footprint', 'hesai_lidar']),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_pub_map',
            arguments=['0', '0', '0', '0', '0', '0', '1', 'map', 'odom']),
    ])
