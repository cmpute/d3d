[pypi-image]: https://badge.fury.io/py/d3d.svg
[pypi-url]: https://pypi.org/project/d3d/
[docs-image]: https://readthedocs.org/projects/d3d/badge/?version=latest
[docs-url]: https:/d3d.readthedocs.io/en/latest/?badge=latest

# D3D
**Devkit for 3D: Some utils for 3D object detection and tracking based on Numpy and Pytorch**

[![PyPI Version][pypi-image]][pypi-url]
[![Docs Status][docs-image]][docs-url]

<hr/>
Please consider siting my work if you find this library useful in your research :)

```bibtex
@article{zhong2020uncertainty,
  title={Uncertainty-Aware Voxel based 3D Object Detection and Tracking with von-Mises Loss},
  author={Zhong, Yuanxin and Zhu, Minghan and Peng, Huei},
  journal={arXiv preprint arXiv:2011.02553},
  year={2020}
}
```

## Features
- Unified data representation
- Support loading KITTI, Waymo, Nuscenes dataset
- Rotated 2D IoU, NMS with clear CUDA implementations
- Point Cloud Voxelization
- Visualization
- Benchmarking

# Tips
- Current polygon intersecting algorithm is not very stable, so try to convert the input to double precision if you met error with the iou functions
