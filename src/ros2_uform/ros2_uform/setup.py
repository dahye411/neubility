import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ros2_uform'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
	('share/' + package_name, glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dahye Kim',
    maintainer_email='sere411@gachon.ac.kr',
    description='ROS2 package for running uform',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
		'uform_node = ros2_uform.uform_node:main'
        ],
    },
)