add_cython_target(matcher CXX)
add_library(matcher MODULE ${matcher})
target_compile_definitions(matcher PRIVATE ${CYTHON_DEFINITIONS})
python_extension_module(matcher)
install(TARGETS matcher LIBRARY DESTINATION d3d/tracking)
