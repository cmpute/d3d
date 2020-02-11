from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="d3d",
    version="0.0.1",
    description="Customized tools for 3D object detection",
    long_description='(see project homepage)',
    author='Jacob Zhong',
    author_email='cmpute@gmail.com',
    url='https://github.com/cmpute/d3d',
    download_url='https://github.com/cmpute/d3d/archive/master.zip',
    license='BSD-3-Clause',
    packages=['d3d'],
    install_requires=['numpy', 'torch'],
    setup_requires=['pybind11', 'torch'],
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
        CUDAExtension('d3d.nms._impl', ['d3d/nms/nms.cpp', 'd3d/nms/nms_cuda.cu']),
        CUDAExtension('d3d.utils._impl', ['d3d/utils/utils.cpp'], include_dirs=["robin-map/include"])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)