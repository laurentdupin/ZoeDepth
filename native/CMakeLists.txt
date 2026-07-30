cmake_minimum_required(VERSION 3.21)
project(zoedepth_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_library(zoedepth_core STATIC
    src/model.cpp
    src/encoder_cpu.cpp
    src/graph_cpu.cpp)
target_include_directories(zoedepth_core PUBLIC src)

add_library(zoedepth_model SHARED src/zoedepth.cpp)
target_include_directories(
    zoedepth_model PUBLIC include PRIVATE src)
target_link_libraries(zoedepth_model PRIVATE zoedepth_core)
target_compile_definitions(
    zoedepth_model PRIVATE ZOEDEPTH_BUILD_DLL)
set_target_properties(
    zoedepth_model PROPERTIES OUTPUT_NAME zoedepth_native)

add_executable(
    zoedepth_model_probe tools/model_probe.cpp)
target_link_libraries(
    zoedepth_model_probe PRIVATE zoedepth_core)
target_include_directories(
    zoedepth_model_probe PRIVATE src)

add_executable(
    zoedepth_encoder_probe tools/encoder_probe.cpp)
target_link_libraries(
    zoedepth_encoder_probe PRIVATE zoedepth_core)
target_include_directories(
    zoedepth_encoder_probe PRIVATE src)

add_executable(
    zoedepth_decoder_probe tools/decoder_probe.cpp)
target_link_libraries(
    zoedepth_decoder_probe PRIVATE zoedepth_core)
target_include_directories(
    zoedepth_decoder_probe PRIVATE src)

add_executable(
    zoedepth_full_graph_probe tools/full_graph_probe.cpp)
target_link_libraries(
    zoedepth_full_graph_probe PRIVATE zoedepth_model)

include(CTest)
if(BUILD_TESTING)
    add_executable(
        zoedepth_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(
        zoedepth_c_abi_smoke PRIVATE zoedepth_model)
    add_test(
        NAME zoedepth_c_abi_smoke
        COMMAND zoedepth_c_abi_smoke)
endif()

if(MSVC)
    target_compile_options(
        zoedepth_core PRIVATE /W4 /permissive-)
    target_compile_options(
        zoedepth_model PRIVATE /W4 /permissive-)
    target_compile_options(
        zoedepth_model_probe PRIVATE /W4 /permissive-)
    target_compile_options(
        zoedepth_encoder_probe PRIVATE /W4 /permissive-)
    target_compile_options(
        zoedepth_decoder_probe PRIVATE /W4 /permissive-)
    target_compile_options(
        zoedepth_full_graph_probe PRIVATE /W4 /permissive-)
else()
    target_compile_options(
        zoedepth_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        zoedepth_model PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        zoedepth_model_probe PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        zoedepth_encoder_probe PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        zoedepth_decoder_probe PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        zoedepth_full_graph_probe PRIVATE
        -Wall -Wextra -Wpedantic)
endif()
