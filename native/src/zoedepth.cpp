#include "zoedepth_native.h"

#include "encoder_cpu.h"
#include "graph_cpu.h"
#include "model.h"

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct zoedepth_context {
    std::unique_ptr<zoe_native::ModelFile> model;
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
    return "0.3.0-zoed-n-k-nk-cpu";
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
