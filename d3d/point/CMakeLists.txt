add_library(point_impl MODULE
    impl.cpp
    scatter.cpp scatter_cuda.cu
)

python_extension_module(point_impl)
target_link_libraries(point_impl ${TORCH_LIBRARIES})
target_include_directories(point_impl PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(point_impl PRIVATE TORCH_EXTENSION_NAME=point_impl)
set_property(TARGET point_impl PROPERTY CXX_STANDARD 14)

install(TARGETS point_impl DESTINATION d3d/point)
