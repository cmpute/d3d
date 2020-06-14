add_library(math_impl MODULE
    impl.cpp math.cpp math_cuda.cu
)

python_extension_module(math_impl)
target_link_libraries(math_impl ${TORCH_LIBRARIES})
target_include_directories(math_impl PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(math_impl PRIVATE TORCH_EXTENSION_NAME=math_impl)
set_property(TARGET math_impl PROPERTY CXX_STANDARD 14)

install(TARGETS math_impl DESTINATION d3d/math)
