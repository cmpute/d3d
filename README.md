# D3D
Devkit for 3D: Some utils for 3D object detection and tracking based on Numpy and Pytorch

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
