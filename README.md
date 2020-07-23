# D3D
Devkit for 3D: Some utils for 3D object detection and tracking based on Numpy and Pytorch

## Features
- Unified interface
- Support loading KITTI, Waymo, Nuscenes dataset
- 2D IoU, NMS
- Voxelization
- Visualization
- Benchmarking

## Package structure

- `d3d.abstraction`: Common interface definitions
- `d3d.benchmark`: Implementation of benchmarks
- `d3d.box`: Modules for bounding box related calculations
- `d3d.dataset`: Modules for dataset loading
- `d3d.math`: Implementation of some special math functions
- `d3d.point`: Modules for point array related components
- `d3d.vis`: Modules for visualizations
- `d3d.voxel`: Moduels for voxel related components

# Requirements

Installation requirements:
- `python >= 3.6`
- `pytorch == 1.4`
- `scipy >= 1.4`
- `addict`
- `pillow`

Build requirements:
- `cython >= 0.29.16`
- `scikit-build`
- `setuptools-scm`

Optional requirements:
- `utm`: support converting GPS coordinate to local frame
- `pcl.py`: support visualization in PCL
- `matplotlib`: support visualization in 2D figures
- `waymo_open_dataset`: support converting Waymo Dataset
- `msgpack`: support serialization/deserialization

# Build

- create build environment in conda: `conda create -f conda/env-dev.yaml`
- build and install: `python setup.py install`
- build wheel: `python setup.py bdist_wheel`
- build in-place: `python setup.py develop`
- build debug: `python setup.py develop --build-type Debug`

## Build on cluster

Some tips about building the library in a cluster: The default behavior of building is using all the CPU cores, so if you find the compiler crashed during compilation, that's usually due to insufficient memory. You can choose the number of parallel building by using `-jN` switch along with those building commands

## Wheels

Prebuilt wheels will be distributed in the future, through either release page or conda channel. Only source distribution will be uploaded to PyPI.

# Versioning
- Major version will be increased when big feature is added
- Minor version will be increased when API compatibility is broken
- Patch version will be increased when new feature is completed.
