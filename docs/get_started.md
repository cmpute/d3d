# Requirements

Installation requirements:
- `python >= 3.6`
- `numpy >= 1.17.0`
- `scipy >= 1.4`
- `addict`
- `pillow`
- `tqdm`
- `msgpack`
- `filterpy`
- `PyYAML`
- `sortedcontainers`

Build requirements:
- `cython >= 0.29.16`
- `scikit-build`
- `setuptools-scm`

Optional requirements:
- `utm`: support converting GPS coordinate to local frame
- `pytorch >= 1.4`: support custom pytorch operators
- `pcl.py`: support visualization in PCL
- `matplotlib`: support visualization in 2D figures
- `waymo_open_dataset`: support converting Waymo Dataset
- `intervaltree`: support indexing in some datasets
- `pyyaml`: support calibration loading in some datasets
- `filelock`: support indexing in some datasets
- `scikit-learn`: support indexing in some datasets

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

## Versioning
- Major version will be increased when big feature is added
- Minor version will be increased when API compatibility is broken
- Patch version will be increased when new feature is completed.

# Features
- Unified data representation
- Support loading KITTI, Waymo, Nuscenes dataset
- Rotated 2D IoU, NMS with clear CUDA implementations
- Point Cloud Voxelization
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
