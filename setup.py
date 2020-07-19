import os
from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    raise ImportError('scikit-build is required for installing')

import torch
torch_root = os.path.dirname(torch.__file__)

def cuda_version(): # get 
    from subprocess import check_output
    from setuptools_scm.version import get_local_node_and_date

    cuda_ver = check_output(["nvcc", "-V"]).decode()
    ver_aidx = cuda_ver.find("release")
    ver_bidx = cuda_ver.find(',', ver_aidx)
    cuda_ver = 'cu' + cuda_ver[ver_aidx+8:ver_bidx].replace('.', '')
    def cuda_scheme(version):
        ver_str = get_local_node_and_date(version)
        if ver_str.find("+") < 0:
            ver_str += "+" + cuda_ver
        else:
            ver_str += "." + cuda_ver
        return ver_str

    return {'local_scheme': cuda_scheme}

setup(
    name="d3d",
    use_scm_version=cuda_version,
    description="Customized tools for 3D object detection",
    long_description='(see project homepage)',
    author='Jacob Zhong',
    author_email='cmpute@gmail.com',
    url='https://github.com/cmpute/d3d',
    download_url='https://github.com/cmpute/d3d/archive/master.zip',
    license='BSD-3-Clause',
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'scipy>=1.4', 'addict', 'pillow'],
    setup_requires=['torch', 'scikit-build', 'setuptools_scm'],
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
    entry_points={
        'console_scripts': [
            'd3d_waymo_convert = d3d.dataset.waymo.converter:main',
            'd3d_nuscenes_convert = d3d.dataset.nuscenes.converter:main',
            'd3d_kitti_parse_result = d3d.dataset.kitti.object:parse_detection_output',
        ],
    },
    cmake_args=[f'-DCMAKE_PREFIX_PATH={torch_root}']
)