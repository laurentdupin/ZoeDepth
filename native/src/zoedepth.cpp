#include "zoedepth_native.h"

#include "encoder_cpu.h"
#include "graph_cpu.h"
#include "model.h"
#if defined(ZOEDEPTH_WITH_VULKAN)
#include "encoder_gpu.h"
#include "gpu_model.h"
#include "graph_gpu.h"
#include "operators.h"
#include "vulkan.h"
#endif

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct zoedepth_context {
    std::unique_ptr<zoe_native::ModelFile> model;
#if defined(ZOEDEPTH_WITH_VULKAN)
    std::unique_ptr<zoe_native::VulkanContext> vulkan;
    std::unique_ptr<zoe_native::GpuModel> gpu_model;
    std::unique_ptr<zoe_native::VulkanOperators> operators;
#endif
};

namespace {
thread_local std::string last_error;

zoedepth_status fail(
    zoedepth_status status,
    const char* message) {
    last_error = message ? message : "";
    return status;
}

template <typename Function>
zoedepth_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return ZOEDEPTH_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(
            ZOEDEPTH_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& error) {
        return fail(
            ZOEDEPTH_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(
            ZOEDEPTH_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(
            ZOEDEPTH_STATUS_INTERNAL_ERROR,
            "unknown internal error");
    }
}
}

extern "C" {

uint32_t ZOEDEPTH_CALL zoedepth_abi_version(void) {
    return ZOEDEPTH_ABI_VERSION;
}

const char* ZOEDEPTH_CALL zoedepth_version_string(void) {
    return "0.4.0-zoed-n-k-nk-cpu-vulkan";
}

zoedepth_status ZOEDEPTH_CALL zoedepth_create_vulkan(
    const char* path,
    zoedepth_variant variant,
    uint32_t device_index,
    zoedepth_context** context) {
    if (!context || !path || path[0] == '\0' ||
        (variant != ZOEDEPTH_VARIANT_N &&
         variant != ZOEDEPTH_VARIANT_K &&
         variant != ZOEDEPTH_VARIANT_NK)) {
        return fail(
            ZOEDEPTH_STATUS_INVALID_ARGUMENT,
            "invalid ZoeDepth Vulkan creation input");
    }
    *context = nullptr;
#if !defined(ZOEDEPTH_WITH_VULKAN)
    (void)device_index;
    return fail(
        ZOEDEPTH_STATUS_UNSUPPORTED,
        "this DLL was built without Vulkan");
#else
    return protect([&] {
        auto result = std::make_unique<zoedepth_context>();
        const zoe_native::Variant native_variant =
            variant == ZOEDEPTH_VARIANT_K ? zoe_native::Variant::k :
            variant == ZOEDEPTH_VARIANT_NK ? zoe_native::Variant::nk :
            zoe_native::Variant::n;
        result->model =
            std::make_unique<zoe_native::ModelFile>(path, native_variant);
        result->vulkan =
            std::make_unique<zoe_native::VulkanContext>(device_index);
        result->gpu_model =
            std::make_unique<zoe_native::GpuModel>(
                *result->model, *result->vulkan);
        result->operators =
            std::make_unique<zoe_native::VulkanOperators>(*result->vulkan);
        *context = result.release();
    });
#endif
}

const char* ZOEDEPTH_CALL zoedepth_last_error(void) {
    return last_error.c_str();
}

zoedepth_status ZOEDEPTH_CALL zoedepth_create(
    const char* path,
    zoedepth_variant variant,
    zoedepth_context** context) {
    if (!context) {
        return fail(
            ZOEDEPTH_STATUS_INVALID_ARGUMENT,
            "context output is null");
    }
    *context = nullptr;
    if (!path || path[0] == '\0') {
        return fail(
            ZOEDEPTH_STATUS_INVALID_ARGUMENT,
            "model path is empty");
    }
    if (variant != ZOEDEPTH_VARIANT_N &&
        variant != ZOEDEPTH_VARIANT_K &&
        variant != ZOEDEPTH_VARIANT_NK) {
        return fail(
            ZOEDEPTH_STATUS_INVALID_ARGUMENT,
            "invalid ZoeDepth variant");
    }
    return protect([&] {
        auto result = std::make_unique<zoedepth_context>();
        result->model = std::make_unique<zoe_native::ModelFile>(
            path,
            variant == ZOEDEPTH_VARIANT_K
                ? zoe_native::Variant::k
                : variant == ZOEDEPTH_VARIANT_NK
                    ? zoe_native::Variant::nk
                    : zoe_native::Variant::n);
        *context = result.release();
    });
}

void ZOEDEPTH_CALL zoedepth_destroy(
    zoedepth_context* context) {
    delete context;
}

zoedepth_status ZOEDEPTH_CALL zoedepth_infer_rgb_f32(
    zoedepth_context* context,
    const float* rgb,
    int32_t width,
    int32_t height,
    float* depth,
    uint64_t depth_elements) {
    if (!context || !context->model || !rgb || !depth ||
        width <= 0 || height <= 0 ||
        width % 32 != 0 || height % 32 != 0 ||
        depth_elements <
            static_cast<std::uint64_t>(width) *
                static_cast<std::uint64_t>(height)) {
        return fail(
            ZOEDEPTH_STATUS_INVALID_ARGUMENT,
            "invalid ZoeDepth tensor inference input");
    }
    return protect([&] {
#if defined(ZOEDEPTH_WITH_VULKAN)
        if (context->vulkan) {
            const std::uint64_t input_elements =
                std::uint64_t(3) * width * height;
            std::vector<float> prepared(
                static_cast<std::size_t>(input_elements));
            for (std::uint64_t index = 0;
                 index < input_elements; ++index) {
                prepared[static_cast<std::size_t>(index)] =
                    (rgb[index] - 0.5f) / 0.5f;
            }
            zoe_native::VulkanBuffer image =
                context->vulkan->create_device_buffer(
                    input_elements * sizeof(float));
            context->vulkan->upload(
                image, prepared.data(),
                prepared.size() * sizeof(float));
            zoe_native::GpuFeature gpu_result =
                zoe_native::full_graph_gpu(
                    *context->vulkan, *context->gpu_model,
                    *context->operators,
                    zoe_native::encoder_gpu(
                        *context->vulkan, *context->gpu_model,
                        *context->operators, image,
                        static_cast<std::uint32_t>(width),
                        static_cast<std::uint32_t>(height)));
            context->vulkan->download(
                gpu_result.buffer, depth,
                std::uint64_t(width) * height * sizeof(float));
            return;
        }
#endif
        const std::uint64_t plane =
            static_cast<std::uint64_t>(width) *
            static_cast<std::uint64_t>(height);
        std::vector<float> prepared(
            static_cast<std::size_t>(3 * plane));
        for (std::uint64_t index = 0;
             index < 3 * plane; ++index) {
            prepared[static_cast<std::size_t>(index)] =
                (rgb[index] - 0.5f) / 0.5f;
        }
        zoe_native::Image result =
            zoe_native::metric_depth_cpu(
                *context->model,
                zoe_native::midas_decoder_cpu(
                    *context->model,
                    zoe_native::encoder_cpu(
                        *context->model, prepared.data(),
                        static_cast<std::uint32_t>(width),
                        static_cast<std::uint32_t>(height))));
        std::copy(
            result.values.begin(), result.values.end(), depth);
    });
}

}
