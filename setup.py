import os
from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    raise ImportError('scikit-build is required for installing')

import torch
torch_root = os.path.dirname(torch.__file__)

setup(
    name="d3d",
    version="0.0.4",
    description="Customized tools for 3D object detection",
    long_description='(see project homepage)',
    author='Jacob Zhong',
    author_email='cmpute@gmail.com',
    url='https://github.com/cmpute/d3d',
    download_url='https://github.com/cmpute/d3d/archive/master.zip',
    license='BSD-3-Clause',
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'py3nvml', 'scipy>=1.4'],
    setup_requires=['pybind11', 'torch', 'scikit-build'],
    extras_require={'test': ['pytest']},
    classifiers=[
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Development Status :: 2 - Pre-Alpha',
        'Topic :: Scientific/Engineering'
    ],
    keywords=['detection', '3d'],
    entry_points={
        'console_scripts': [
            'd3d_waymo_convert = d3d.dataset.waymo.converter:main',
            'd3d_nuscenes_convert = d3d.dataset.nuscenes.converter:main',
            'd3d_kitti_parse_result = d3d.dataset.kitti.object:parse_detection_output',
        ],
    },
    cmake_args=[f'-DCMAKE_PREFIX_PATH={torch_root}']
)