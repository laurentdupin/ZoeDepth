cmake_minimum_required(VERSION 3.21)
project(zoedepth_native LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../native_support/InferBridgeNativeHarnessSupport.cmake")
link_libraries(InferBridge::NativeHarnessSupport)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(ZOEDEPTH_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)
option(ZOEDEPTH_WITH_METAL
    "Build the native Metal executor" ${APPLE})

if(ZOEDEPTH_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "ZOEDEPTH_WITH_METAL requires Apple platforms")
    endif()
    enable_language(OBJCXX)
endif()

if(ZOEDEPTH_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(ZOEDEPTH_GENERATED_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${ZOEDEPTH_GENERATED_DIR}")
    function(zoedepth_compile_shader NAME)
        set(SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${ZOEDEPTH_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${ZOEDEPTH_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -O -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM)
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SPIRV}" "-DOUTPUT=${HEADER}"
                "-DVARIABLE=zoe_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(ZOEDEPTH_SHADER_HEADERS
            ${ZOEDEPTH_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()
    foreach(SHADER
            add_scaled add add_position bilinear_align_true
            bilinear_align_true_image bmm bmm_score_half bmm_value_half
            conv2d conv2d8 conv2d_half conv2d8_half conv2d_tiled
            conv2d_tiled4
            conv_transpose_nonoverlap conv_transpose_nonoverlap4
            conv_transpose_nonoverlap_half conv_transpose_nonoverlap4_half
            elementwise gelu layer_norm linear linear16 linear_half
            linear16_half linear_vec8 linear_vec8_rows24
            linear_vec8_rows24_half linear_int8_tiled16
            prepare_tokens position_bicubic project_tokens
            project_tokens_half relu softmax_lastdim softmax_lastdim_half
            group_norm silu nearest concatenate add_channel nchw_tokens
            geglu attention_scores attention_values preprocess_rgb
            posterior_sample scale_values depth_output
            prepare_beit qv_bias relative_bias readout_concat
            tokens_to_nchw_plain softplus router_tokens qkv_split
            seed_centers attractor_activate attractor_update
            distribution_depth select_depth)
        zoedepth_compile_shader(${SHADER})
    endforeach()
    zoedepth_compile_shader(preprocess_texture)
    zoedepth_compile_shader(combine_depth_image)
    zoedepth_compile_shader(combine_depth_buffer)
    zoedepth_compile_shader(reduce_minmax)
    zoedepth_compile_shader(normalize_inverse)
    zoedepth_compile_shader(depth_buffer_to_image)
    foreach(SHARED_SHADER quantize_rows_int8 linear_int8_tiled
            layer_norm_parallel)
        inferbridge_compile_native_support_shader(
            NAME "${SHARED_SHADER}" PREFIX zoe
            GENERATED_DIR "${ZOEDEPTH_GENERATED_DIR}"
            EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            OUT_VAR ZOEDEPTH_SHADER_HEADERS)
    endforeach()
endif()

add_library(zoedepth_core STATIC
    src/model.cpp
    src/encoder_cpu.cpp
    src/graph_cpu.cpp)
set_target_properties(zoedepth_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(zoedepth_core PUBLIC src)

add_library(zoedepth_model SHARED
    src/zoedepth.cpp
    src/inferbridge_harness.cpp)
if(ZOEDEPTH_WITH_METAL)
    target_sources(zoedepth_model PRIVATE src/metal_executor.mm)
    target_link_libraries(zoedepth_model PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShadersGraph")
    target_compile_definitions(
        zoedepth_model PRIVATE ZOEDEPTH_WITH_METAL=1)
    set_source_files_properties(src/metal_executor.mm PROPERTIES
        COMPILE_OPTIONS "-fobjc-arc")
endif()
if(ZOEDEPTH_WITH_VULKAN)
    target_sources(zoedepth_model PRIVATE
        src/vulkan.cpp src/operators.cpp src/gpu_model.cpp
        src/encoder_gpu.cpp src/graph_gpu.cpp src/gpu_io.cpp
        src/external_gpu.cpp
        ${ZOEDEPTH_SHADER_HEADERS})
    target_include_directories(
        zoedepth_model PRIVATE "${ZOEDEPTH_GENERATED_DIR}")
    target_link_libraries(zoedepth_model PRIVATE Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(zoedepth_model PRIVATE d3d12 dxgi)
    endif()
    target_compile_definitions(
        zoedepth_model PRIVATE ZOEDEPTH_WITH_VULKAN=1)
endif()
target_include_directories(
    zoedepth_model PUBLIC include PRIVATE src)
target_link_libraries(zoedepth_model PRIVATE zoedepth_core)
target_compile_definitions(
    zoedepth_model PRIVATE
        ZOEDEPTH_BUILD_DLL
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
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

if(ZOEDEPTH_WITH_METAL)
    add_executable(
        zoedepth_metal_parity_probe tools/metal_parity_probe.cpp)
    target_link_libraries(
        zoedepth_metal_parity_probe PRIVATE zoedepth_model)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(
        zoedepth_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(
        zoedepth_c_abi_smoke PRIVATE zoedepth_model)
    add_test(
        NAME zoedepth_c_abi_smoke
        COMMAND zoedepth_c_abi_smoke)
    add_executable(
        zoedepth_harness_abi_smoke tests/harness_abi_smoke.c)
    target_link_libraries(
        zoedepth_harness_abi_smoke PRIVATE zoedepth_model)
    target_include_directories(
        zoedepth_harness_abi_smoke PRIVATE include)
    add_test(
        NAME zoedepth_harness_abi_smoke
        COMMAND zoedepth_harness_abi_smoke)
    add_executable(
        zoedepth_harness_full_graph tests/harness_full_graph.cpp)
    target_link_libraries(
        zoedepth_harness_full_graph PRIVATE zoedepth_model)
    target_include_directories(
        zoedepth_harness_full_graph PRIVATE include)
    add_test(
        NAME zoedepth_harness_full_graph
        COMMAND zoedepth_harness_full_graph)
    set_tests_properties(
        zoedepth_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)
    if(WIN32 AND ZOEDEPTH_WITH_VULKAN)
        add_executable(
            zoedepth_d3d12_full_graph tests/d3d12_full_graph.cpp)
        target_link_libraries(
            zoedepth_d3d12_full_graph PRIVATE zoedepth_model d3d12 dxgi)
        target_include_directories(
            zoedepth_d3d12_full_graph PRIVATE include)
        target_compile_options(
            zoedepth_d3d12_full_graph PRIVATE /W4 /permissive-)
        add_test(NAME zoedepth_d3d12_full_graph
                 COMMAND zoedepth_d3d12_full_graph)
        set_tests_properties(
            zoedepth_d3d12_full_graph PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if(ZOEDEPTH_WITH_VULKAN)
        add_executable(
            zoedepth_gpu_encoder_probe
            tools/gpu_encoder_probe.cpp
            src/model.cpp src/vulkan.cpp src/operators.cpp
            src/gpu_model.cpp src/encoder_gpu.cpp
            ${ZOEDEPTH_SHADER_HEADERS})
        target_include_directories(
            zoedepth_gpu_encoder_probe PRIVATE
            include src "${ZOEDEPTH_GENERATED_DIR}")
        target_link_libraries(
            zoedepth_gpu_encoder_probe PRIVATE Vulkan::Vulkan)
        add_executable(
            zoedepth_gpu_full_graph_probe
            tools/gpu_full_graph_probe.cpp)
        target_link_libraries(
            zoedepth_gpu_full_graph_probe PRIVATE zoedepth_model)
        target_include_directories(
            zoedepth_gpu_full_graph_probe PRIVATE include)
    endif()
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
