add_library(box_impl MODULE
    impl.cpp utils.cpp
    iou.cpp iou_cuda.cu
    nms.cpp nms_cuda.cu
)

python_extension_module(box_impl)
target_link_libraries(box_impl ${TORCH_LIBRARIES})
target_include_directories(box_impl PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(box_impl PRIVATE TORCH_EXTENSION_NAME=box_impl)
set_property(TARGET box_impl PROPERTY CXX_STANDARD 14)

install(TARGETS box_impl DESTINATION d3d/box)
