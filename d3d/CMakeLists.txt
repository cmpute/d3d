if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CYTHON_DEFINITIONS CYTHON_WITHOUT_ASSERTIONS)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_subdirectory(box)
add_subdirectory(math)
add_subdirectory(point)
add_subdirectory(tracking)
add_subdirectory(voxel)

# compile abstraction cython module
add_cython_target(abstraction CXX)
add_library(abstraction MODULE ${abstraction})
target_compile_definitions(abstraction PRIVATE ${CYTHON_DEFINITIONS})
python_extension_module(abstraction)
install(TARGETS abstraction LIBRARY DESTINATION d3d)

# compile benchmarks cython module
add_cython_target(benchmarks CXX)
add_library(benchmarks MODULE ${benchmarks})
target_compile_definitions(benchmarks PRIVATE ${CYTHON_DEFINITIONS})
python_extension_module(benchmarks)
install(TARGETS benchmarks LIBRARY DESTINATION d3d)

# compile geometry pybind11 module
add_library(geometry MODULE geometry_binding.cpp)
python_extension_module(geometry)
target_include_directories(geometry PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET geometry PROPERTY CXX_STANDARD 14)
install(TARGETS geometry DESTINATION d3d)
