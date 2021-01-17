# Requirements

Installation requirements:
- `python >= 3.6`
- `numpy >= 1.17.0`
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
- `filterpy`: support KF tracking
- `intervaltree`: support indexing in some datasets
- `pyyaml`: support calibration loading in some datasets
- `filelock`: support indexing in some datasets

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
