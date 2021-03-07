set(BOX_SRC impl.cpp utils.cpp iou.cpp nms.cpp dist.cpp)
if (BUILD_WITH_CUDA)
    set(BOX_SRC ${BOX_SRC} iou_cuda.cu nms_cuda.cu dist_cuda.cu)
endif()

add_library(box_impl MODULE ${BOX_SRC})
python_extension_module(box_impl)
target_link_libraries(box_impl ${TORCH_LIBRARIES})
target_include_directories(box_impl PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(box_impl PRIVATE TORCH_EXTENSION_NAME=box_impl)
set_property(TARGET box_impl PROPERTY CXX_STANDARD 14)

install(TARGETS box_impl DESTINATION d3d/box)
