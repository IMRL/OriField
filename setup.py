import os
import sys
from setuptools import Extension
from setuptools import setup, find_packages

__version__ = '0.0.0'

py_dir = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages') + '/'
print("py_dir", py_dir)

# lidar BEV cpp extension
lidar_bev_ext_module = Extension(
    name = 'lidardet.ops.lidar_bev.bev',
    sources = ['lidardet/ops/lidar_bev/bev.cpp'], 
    include_dirs = ['/usr/include/eigen3', py_dir + 'pybind11/include'],
    language='c++'
)

# planning cpp extension
planning_ext_module = Extension(
    name = 'lidardet.ops.planning.planning',
    sources = ['lidardet/ops/planning/planning.cpp', 'lidardet/ops/planning/dubins.cpp'], 
    include_dirs = ['/usr/include/eigen3', py_dir + 'pybind11/include', 'lidardet/ops/planning/include'],
    language='c++'
)

if __name__ == '__main__':

    setup(
        name='lidardet',
        version="0.0.0",
        description='LidarDet is a general codebase for learning based perception task',
        install_requires=[
            'numpy',
            'torch',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml'
        ],
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output', 'cache', 'ros', 'docker', 'config']),
        ext_modules=[lidar_bev_ext_module, planning_ext_module]
        )
