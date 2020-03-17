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
    install_requires=['numpy', 'torch', 'py3nvml'],
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
    cmake_args=[f'-DCMAKE_PREFIX_PATH={torch_root}']
)