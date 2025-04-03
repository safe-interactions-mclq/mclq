#cmake_minimum_required(VERSION 3.12)
#project(mclq C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

find_package(Python REQUIRED COMPONENTS Interpreter Development)

add_library(mclq MODULE
    src/utils/random.c
    src/utils/alloc.c
    src/module.c
)

set_target_properties(mclq PROPERTIES PREFIX "lib" OUTPUT_NAME "mclq" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/mclq")
if(WIN32)
    set_target_properties(mclq PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(mclq PROPERTIES SUFFIX ".so")
endif()

target_include_directories(mclq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(mclq PRIVATE -O3 -march=native -ftree-vectorize)
target_link_libraries(mclq PRIVATE
    m
    Python::Python
)
