cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(D3D LANGUAGES C CXX CUDA VERSION 1.0)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CUDA_FLAGS_DEBUG "-G -g") # disable CUDA optimization
endif()

find_package(Torch REQUIRED)
find_package(PythonExtensions REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

if (TARGET torch_cuda)
    # tweak torch compiling flags, see https://github.com/pytorch/pytorch/pull/36021
    set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")  
    set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")

    # https://github.com/pytorch/pytorch/issues/38122
    find_library(TORCH_PYTHON torch_python)
    list(APPEND TORCH_LIBRARIES ${TORCH_PYTHON})

    # tweak torch dependencies, see https://github.com/pytorch/pytorch/issues/33928
    get_target_property(TORCH_INTERFACE_LIB torch_cuda INTERFACE_LINK_LIBRARIES)
    string(REPLACE "/usr/local/cuda" ${CUDA_TOOLKIT_ROOT_DIR} TORCH_INTERFACE_LIB "${TORCH_INTERFACE_LIB}")
    set_target_properties(torch_cuda PROPERTIES INTERFACE_LINK_LIBRARIES ${TORCH_INTERFACE_LIB})
else ()
    # this case is for pytorch 1.4
    get_target_property(TORCH_INTERFACE_LIB torch INTERFACE_LINK_LIBRARIES)
    string(REPLACE "/usr/local/cuda" ${CUDA_TOOLKIT_ROOT_DIR} TORCH_INTERFACE_LIB "${TORCH_INTERFACE_LIB}")
    set_target_properties(torch PROPERTIES INTERFACE_LINK_LIBRARIES ${TORCH_INTERFACE_LIB})
endif ()

# Cython related dependencies
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIRS})

add_subdirectory(d3d)
