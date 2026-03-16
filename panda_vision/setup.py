from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'panda_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models',
            glob('panda_vision/models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='utk',
    maintainer_email='kutkarsh706@gmail.com',
    description='YOLOv12 fastener detection package for Panda robot vision',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'color_detector = panda_vision.color_detector:main',
            'fastener_detector = panda_vision.fastener_detector:main',
        ],
    },
)
