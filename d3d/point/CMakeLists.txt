set(POINT_SRC impl.cpp scatter.cpp)
if (BUILD_WITH_CUDA)
    set(POINT_SRC ${POINT_SRC} scatter_cuda.cu)
endif ()

add_library(point_impl MODULE ${POINT_SRC})
python_extension_module(point_impl)
target_link_libraries(point_impl ${TORCH_LIBRARIES})
target_include_directories(point_impl PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(point_impl PRIVATE TORCH_EXTENSION_NAME=point_impl)
set_property(TARGET point_impl PROPERTY CXX_STANDARD 14)

install(TARGETS point_impl DESTINATION d3d/point)
