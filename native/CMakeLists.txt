cmake_minimum_required(VERSION 3.21)
project(zoedepth_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_library(zoedepth_model STATIC src/model.cpp)
target_include_directories(zoedepth_model PUBLIC src)

add_executable(
    zoedepth_model_probe tools/model_probe.cpp)
target_link_libraries(
    zoedepth_model_probe PRIVATE zoedepth_model)

if(MSVC)
    target_compile_options(
        zoedepth_model PRIVATE /W4 /permissive-)
    target_compile_options(
        zoedepth_model_probe PRIVATE /W4 /permissive-)
else()
    target_compile_options(
        zoedepth_model PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        zoedepth_model_probe PRIVATE -Wall -Wextra -Wpedantic)
endif()
