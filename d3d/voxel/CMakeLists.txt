add_library(voxel_impl MODULE
    impl.cpp
    voxelize.cpp
)
python_extension_module(voxel_impl)
target_link_libraries(voxel_impl ${TORCH_LIBRARIES})
target_include_directories(voxel_impl PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(voxel_impl PRIVATE TORCH_EXTENSION_NAME=voxel_impl)
set_property(TARGET voxel_impl PROPERTY CXX_STANDARD 14)

install(TARGETS voxel_impl DESTINATION d3d/voxel)
