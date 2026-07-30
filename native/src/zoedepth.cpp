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
#include <cmath>
#include <cstring>
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

std::uint32_t reflect_index(
    std::int64_t index, std::uint32_t size) {
    if (size < 2) {
        throw std::invalid_argument(
            "reflection padding requires dimensions greater than one");
    }
    const std::int64_t period =
        2 * (static_cast<std::int64_t>(size) - 1);
    index %= period;
    if (index < 0) {
        index += period;
    }
    if (index >= static_cast<std::int64_t>(size)) {
        index = period - index;
    }
    return static_cast<std::uint32_t>(index);
}

std::uint32_t nearest_multiple_32(double value) {
    // nearbyint matches numpy.round's ties-to-even behavior under the
    // default IEEE rounding mode.
    const double rounded = std::nearbyint(value / 32.0) * 32.0;
    return static_cast<std::uint32_t>(
        std::max(32.0, rounded));
}

std::pair<std::uint32_t, std::uint32_t> midas_size(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t model_size) {
    double scale_height =
        static_cast<double>(model_size) / height;
    double scale_width =
        static_cast<double>(model_size) / width;
    if (std::abs(1.0 - scale_width) <
        std::abs(1.0 - scale_height)) {
        scale_height = scale_width;
    } else {
        scale_width = scale_height;
    }
    return {
        nearest_multiple_32(scale_width * width),
        nearest_multiple_32(scale_height * height)};
}

std::vector<float> resize_bilinear_align_corners(
    const std::vector<float>& input,
    std::uint32_t channels,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    std::vector<float> output(
        static_cast<std::size_t>(
            std::uint64_t(channels) * output_width * output_height));
    const std::uint64_t input_plane =
        std::uint64_t(input_width) * input_height;
    const std::uint64_t output_plane =
        std::uint64_t(output_width) * output_height;
    for (std::uint32_t y = 0; y < output_height; ++y) {
        const float source_y = output_height > 1
            ? static_cast<float>(y) *
                static_cast<float>(input_height - 1) /
                static_cast<float>(output_height - 1)
            : 0.0f;
        const auto y0 = static_cast<std::uint32_t>(source_y);
        const auto y1 = std::min(y0 + 1, input_height - 1);
        const float fy = source_y - static_cast<float>(y0);
        for (std::uint32_t x = 0; x < output_width; ++x) {
            const float source_x = output_width > 1
                ? static_cast<float>(x) *
                    static_cast<float>(input_width - 1) /
                    static_cast<float>(output_width - 1)
                : 0.0f;
            const auto x0 = static_cast<std::uint32_t>(source_x);
            const auto x1 = std::min(x0 + 1, input_width - 1);
            const float fx = source_x - static_cast<float>(x0);
            for (std::uint32_t channel = 0;
                 channel < channels; ++channel) {
                const std::uint64_t base =
                    std::uint64_t(channel) * input_plane;
                const float top =
                    input[static_cast<std::size_t>(
                        base + std::uint64_t(y0) * input_width + x0)] *
                        (1.0f - fx) +
                    input[static_cast<std::size_t>(
                        base + std::uint64_t(y0) * input_width + x1)] * fx;
                const float bottom =
                    input[static_cast<std::size_t>(
                        base + std::uint64_t(y1) * input_width + x0)] *
                        (1.0f - fx) +
                    input[static_cast<std::size_t>(
                        base + std::uint64_t(y1) * input_width + x1)] * fx;
                output[static_cast<std::size_t>(
                    std::uint64_t(channel) * output_plane +
                    std::uint64_t(y) * output_width + x)] =
                    top * (1.0f - fy) + bottom * fy;
            }
        }
    }
    return output;
}

float cubic_weight(float distance) {
    constexpr float a = -0.75f;
    distance = std::abs(distance);
    if (distance < 1.0f) {
        return ((a + 2.0f) * distance -
            (a + 3.0f)) * distance * distance + 1.0f;
    }
    if (distance < 2.0f) {
        return ((a * distance - 5.0f * a) * distance +
            8.0f * a) * distance - 4.0f * a;
    }
    return 0.0f;
}

std::vector<float> resize_bicubic_align_corners_false(
    const std::vector<float>& input,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    std::vector<float> output(
        static_cast<std::size_t>(
            std::uint64_t(output_width) * output_height));
    const float scale_x =
        static_cast<float>(input_width) / output_width;
    const float scale_y =
        static_cast<float>(input_height) / output_height;
    for (std::uint32_t y = 0; y < output_height; ++y) {
        const float source_y =
            (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y_base =
            static_cast<int>(std::floor(source_y));
        for (std::uint32_t x = 0; x < output_width; ++x) {
            const float source_x =
                (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x_base =
                static_cast<int>(std::floor(source_x));
            float value = 0.0f;
            for (int ky = -1; ky <= 2; ++ky) {
                const int iy = std::clamp(
                    y_base + ky, 0,
                    static_cast<int>(input_height) - 1);
                const float wy = cubic_weight(
                    source_y - static_cast<float>(y_base + ky));
                for (int kx = -1; kx <= 2; ++kx) {
                    const int ix = std::clamp(
                        x_base + kx, 0,
                        static_cast<int>(input_width) - 1);
                    value += input[static_cast<std::size_t>(
                        std::uint64_t(iy) * input_width + ix)] *
                        wy * cubic_weight(
                            source_x -
                            static_cast<float>(x_base + kx));
                }
            }
            output[static_cast<std::size_t>(
                std::uint64_t(y) * output_width + x)] = value;
        }
    }
    return output;
}

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

std::vector<float> execute_prepared(
    zoedepth_context& context,
    const std::vector<float>& prepared,
    std::uint32_t width,
    std::uint32_t height) {
#if defined(ZOEDEPTH_WITH_VULKAN)
    if (context.vulkan) {
        zoe_native::VulkanBuffer image =
            context.vulkan->create_device_buffer(
                prepared.size() * sizeof(float));
        context.vulkan->upload(
            image, prepared.data(),
            prepared.size() * sizeof(float));
        zoe_native::GpuFeature gpu_result =
            zoe_native::full_graph_gpu(
                *context.vulkan, *context.gpu_model,
                *context.operators,
                zoe_native::encoder_gpu(
                    *context.vulkan, *context.gpu_model,
                    *context.operators, image, width, height));
        std::vector<float> result(
            static_cast<std::size_t>(
                std::uint64_t(width) * height));
        context.vulkan->download(
            gpu_result.buffer, result.data(),
            result.size() * sizeof(float));
        return result;
    }
#endif
    zoe_native::Image result =
        zoe_native::metric_depth_cpu(
            *context.model,
            zoe_native::midas_decoder_cpu(
                *context.model,
                zoe_native::encoder_cpu(
                    *context.model, prepared.data(), width, height)));
    return std::move(result.values);
}
}

extern "C" {

uint32_t ZOEDEPTH_CALL zoedepth_abi_version(void) {
    return ZOEDEPTH_ABI_VERSION;
}

const char* ZOEDEPTH_CALL zoedepth_version_string(void) {
    return "0.5.0-zoed-n-k-nk-image-cpu-vulkan";
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
        std::vector<float> prepared(
            static_cast<std::size_t>(
                std::uint64_t(3) * width * height));
        for (std::uint64_t index = 0;
             index < prepared.size(); ++index) {
            prepared[static_cast<std::size_t>(index)] =
                (rgb[index] - 0.5f) / 0.5f;
        }
        std::vector<float> result = execute_prepared(
            *context, prepared,
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height));
        std::copy(
            result.begin(), result.end(), depth);
    });
}

zoedepth_status ZOEDEPTH_CALL zoedepth_infer_bgra8_f32(
    zoedepth_context* context,
    const uint8_t* bgra,
    uint64_t bgra_stride_bytes,
    int32_t width,
    int32_t height,
    int32_t model_size,
    float* depth,
    uint64_t depth_elements) {
    if (!context || !context->model || !bgra || !depth ||
        width <= 1 || height <= 1 ||
        model_size <= 0 || model_size % 32 != 0 ||
        bgra_stride_bytes <
            static_cast<std::uint64_t>(width) * 4 ||
        depth_elements <
            static_cast<std::uint64_t>(width) *
                static_cast<std::uint64_t>(height)) {
        return fail(
            ZOEDEPTH_STATUS_INVALID_ARGUMENT,
            "invalid ZoeDepth BGRA image inference input");
    }
    return protect([&] {
        const std::uint32_t source_width =
            static_cast<std::uint32_t>(width);
        const std::uint32_t source_height =
            static_cast<std::uint32_t>(height);
        const std::uint32_t pad_h = static_cast<std::uint32_t>(
            std::sqrt(static_cast<double>(height) / 2.0) * 3.0);
        const std::uint32_t pad_w = static_cast<std::uint32_t>(
            std::sqrt(static_cast<double>(width) / 2.0) * 3.0);
        if (pad_h >= source_height || pad_w >= source_width) {
            throw std::invalid_argument(
                "image is too small for ZoeDepth reflection padding");
        }
        const std::uint32_t padded_width =
            source_width + 2 * pad_w;
        const std::uint32_t padded_height =
            source_height + 2 * pad_h;
        const auto [network_width, network_height] =
            midas_size(
                padded_width, padded_height,
                static_cast<std::uint32_t>(model_size));
        const std::uint64_t padded_plane =
            std::uint64_t(padded_width) * padded_height;

        auto run_pass = [&](bool flip) {
            std::vector<float> padded(
                static_cast<std::size_t>(3 * padded_plane));
            for (std::uint32_t y = 0; y < padded_height; ++y) {
                const std::uint32_t sy = reflect_index(
                    static_cast<std::int64_t>(y) - pad_h,
                    source_height);
                const auto* row =
                    bgra + std::uint64_t(sy) * bgra_stride_bytes;
                for (std::uint32_t x = 0; x < padded_width; ++x) {
                    std::uint32_t sx = reflect_index(
                        static_cast<std::int64_t>(x) - pad_w,
                        source_width);
                    if (flip) {
                        sx = source_width - 1 - sx;
                    }
                    const std::uint64_t destination =
                        std::uint64_t(y) * padded_width + x;
                    // InferBridge's BGRA -> BGR slice -> RGB conversion.
                    padded[static_cast<std::size_t>(destination)] =
                        row[std::uint64_t(sx) * 4 + 2] / 255.0f;
                    padded[static_cast<std::size_t>(
                        padded_plane + destination)] =
                        row[std::uint64_t(sx) * 4 + 1] / 255.0f;
                    padded[static_cast<std::size_t>(
                        2 * padded_plane + destination)] =
                        row[std::uint64_t(sx) * 4] / 255.0f;
                }
            }
            std::vector<float> prepared =
                resize_bilinear_align_corners(
                    padded, 3, padded_width, padded_height,
                    network_width, network_height);
            for (float& value : prepared) {
                value = (value - 0.5f) / 0.5f;
            }
            std::vector<float> prediction = execute_prepared(
                *context, prepared, network_width, network_height);
            return resize_bicubic_align_corners_false(
                prediction, network_width, network_height,
                padded_width, padded_height);
        };

        std::vector<float> direct = run_pass(false);
        std::vector<float> flipped = run_pass(true);
        for (std::uint32_t y = 0; y < source_height; ++y) {
            for (std::uint32_t x = 0; x < source_width; ++x) {
                const std::uint64_t direct_index =
                    std::uint64_t(y + pad_h) * padded_width +
                    (x + pad_w);
                const std::uint64_t flipped_index =
                    std::uint64_t(y + pad_h) * padded_width +
                    (source_width - 1 - x + pad_w);
                depth[std::uint64_t(y) * source_width + x] =
                    0.5f * (
                        direct[static_cast<std::size_t>(direct_index)] +
                        flipped[static_cast<std::size_t>(flipped_index)]);
            }
        }
    });
}

}
