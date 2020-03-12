from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    raise ImportError('scikit-build is required for installing')

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
    setup_requires=['pybind11', 'torch', 'skbuild'],
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

    ext_modules=[
        CUDAExtension('d3d.box._impl', [
            'd3d/box/impl.cpp',
            'd3d/box/iou.cpp', 'd3d/box/iou_cuda.cu',
            'd3d/box/nms.cpp', 'd3d/box/nms_cuda.cu'
            ], include_dirs=["."]),
        CppExtension('d3d.voxel._impl', ['d3d/voxel/impl.cpp'], include_dirs=["./robin-map/include"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)