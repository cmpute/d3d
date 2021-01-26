.. d3d documentation master file, created by
   sphinx-quickstart on Sat Jan 16 18:37:47 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

D3D: Devkit for 3D Machine Learning
===================================

D3D is a collections of tools built for 3D Machine Learning, currently it's mainly designed for 3D object detection and tracking tasks.

Please consider siting my work if you find this library useful in your research :)

.. code-block:: bibtex

   @article{zhong2020uncertainty,
     title={Uncertainty-Aware Voxel based 3D Object Detection and Tracking with von-Mises Loss},
     author={Zhong, Yuanxin and Zhu, Minghan and Peng, Huei},
     journal={arXiv preprint arXiv:2011.02553},
     year={2020}
   }

.. toctree::
   :caption: Get Started
   :maxdepth: 2

   get_started.md

.. toctree::
   :caption: Functionalities
   :maxdepth: 2

   abstraction.md
   datasets.md
   operators.md
   tracking.md
   utils.md

.. toctree::
   :caption: API Reference
   :maxdepth: 2

   apis/abstraction.rst
   apis/benchmarks.rst
   apis/box.rst
   apis/dataset.rst
   apis/dataset-kitti.rst
   apis/dataset-kitti360.rst
   apis/dataset-nuscenes.rst
   apis/dataset-waymo.rst
   apis/math.rst
   apis/point.rst
   apis/tracking.rst
   apis/vis.rst
   apis/voxel.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
