from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='d3d',
    ext_modules=[
        CUDAExtension('d3d.nms._impl', ['d3d/nms/nms.cpp', 'd3d/nms/nms_cuda.cu'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)