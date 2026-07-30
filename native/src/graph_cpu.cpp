#include "graph_cpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zoe_native {
namespace {

std::uint64_t element_count(
    std::uint32_t channels,
    std::uint32_t height,
    std::uint32_t width) {
    return std::uint64_t(channels) * height * width;
}

const TensorView& checked_tensor(
    const ModelFile& model,
    const std::string& name,
    std::uint32_t rank) {
    const TensorView& tensor = model.tensor(name);
    if (tensor.rank != rank) {
        throw std::runtime_error(
            "unexpected ZoeDepth graph tensor rank: " + name);
    }
    return tensor;
}

float gelu(float value) {
    return 0.5f * value *
        (1.0f + std::erf(value * 0.7071067811865475244f));
}

Image conv2d(
    const ModelFile& model,
    const Image& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t stride,
    std::uint32_t padding) {
    const TensorView& weight =
        checked_tensor(model, weight_name, 4);
    const TensorView* bias = bias_name.empty()
        ? nullptr
        : &checked_tensor(model, bias_name, 1);
    if (weight.dimensions[1] != input.channels ||
        weight.dimensions[2] != weight.dimensions[3] ||
        (bias && bias->dimensions[0] != weight.dimensions[0])) {
        throw std::runtime_error(
            "ZoeDepth convolution shape mismatch: " + weight_name);
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight.dimensions[0]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(weight.dimensions[2]);
    if (input.height + 2 * padding < kernel ||
        input.width + 2 * padding < kernel) {
        throw std::runtime_error(
            "ZoeDepth convolution input is too small");
    }
    Image output{
        output_channels,
        (input.height + 2 * padding - kernel) / stride + 1,
        (input.width + 2 * padding - kernel) / stride + 1,
        {}};
    output.values.resize(element_count(
        output.channels, output.height, output.width));
    for (std::uint32_t out = 0; out < output.channels; ++out) {
        for (std::uint32_t oy = 0; oy < output.height; ++oy) {
            for (std::uint32_t ox = 0; ox < output.width; ++ox) {
                float value = bias ? bias->data[out] : 0.0f;
                for (std::uint32_t in = 0;
                     in < input.channels; ++in) {
                    for (std::uint32_t ky = 0; ky < kernel; ++ky) {
                        const int iy =
                            static_cast<int>(oy * stride + ky) -
                            static_cast<int>(padding);
                        if (iy < 0 ||
                            iy >= static_cast<int>(input.height)) {
                            continue;
                        }
                        for (std::uint32_t kx = 0;
                             kx < kernel; ++kx) {
                            const int ix =
                                static_cast<int>(ox * stride + kx) -
                                static_cast<int>(padding);
                            if (ix < 0 ||
                                ix >= static_cast<int>(input.width)) {
                                continue;
                            }
                            value += input.values[
                                (std::uint64_t(in) * input.height +
                                 static_cast<std::uint32_t>(iy)) *
                                    input.width +
                                static_cast<std::uint32_t>(ix)] *
                                weight.data[
                                    ((std::uint64_t(out) *
                                          input.channels +
                                      in) *
                                         kernel +
                                     ky) *
                                        kernel +
                                    kx];
                        }
                    }
                }
                output.values[
                    (std::uint64_t(out) * output.height + oy) *
                        output.width +
                    ox] = value;
            }
        }
    }
    return output;
}

Image transpose_conv_nonoverlap(
    const ModelFile& model,
    const Image& input,
    const std::string& weight_name,
    const std::string& bias_name) {
    const TensorView& weight =
        checked_tensor(model, weight_name, 4);
    const TensorView& bias =
        checked_tensor(model, bias_name, 1);
    if (weight.dimensions[0] != input.channels ||
        weight.dimensions[2] != weight.dimensions[3] ||
        bias.dimensions[0] != weight.dimensions[1]) {
        throw std::runtime_error(
            "ZoeDepth transpose convolution shape mismatch");
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight.dimensions[1]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(weight.dimensions[2]);
    Image output{
        output_channels,
        input.height * kernel,
        input.width * kernel,
        {}};
    output.values.resize(element_count(
        output.channels, output.height, output.width));
    for (std::uint32_t out = 0; out < output.channels; ++out) {
        std::fill(
            output.values.begin() +
                std::uint64_t(out) * output.height * output.width,
            output.values.begin() +
                std::uint64_t(out + 1) *
                    output.height * output.width,
            bias.data[out]);
    }
    for (std::uint32_t in = 0; in < input.channels; ++in) {
        for (std::uint32_t iy = 0; iy < input.height; ++iy) {
            for (std::uint32_t ix = 0; ix < input.width; ++ix) {
                const float source = input.values[
                    (std::uint64_t(in) * input.height + iy) *
                        input.width +
                    ix];
                for (std::uint32_t out = 0;
                     out < output.channels; ++out) {
                    for (std::uint32_t ky = 0; ky < kernel; ++ky) {
                        for (std::uint32_t kx = 0;
                             kx < kernel; ++kx) {
                            output.values[
                                (std::uint64_t(out) * output.height +
                                 iy * kernel + ky) *
                                    output.width +
                                ix * kernel + kx] +=
                                source * weight.data[
                                    ((std::uint64_t(in) *
                                          output.channels +
                                      out) *
                                         kernel +
                                     ky) *
                                        kernel +
                                    kx];
                        }
                    }
                }
            }
        }
    }
    return output;
}

Image resize_align_corners(
    const Image& input,
    std::uint32_t output_height,
    std::uint32_t output_width) {
    if (output_height == 0 || output_width == 0) {
        throw std::runtime_error("invalid ZoeDepth resize output");
    }
    Image output{
        input.channels, output_height, output_width, {}};
    output.values.resize(element_count(
        output.channels, output.height, output.width));
    for (std::uint32_t channel = 0;
         channel < input.channels; ++channel) {
        for (std::uint32_t oy = 0; oy < output.height; ++oy) {
            const float source_y = output.height > 1
                ? static_cast<float>(oy) *
                    static_cast<float>(input.height - 1) /
                    static_cast<float>(output.height - 1)
                : 0.0f;
            const std::uint32_t y0 =
                static_cast<std::uint32_t>(source_y);
            const std::uint32_t y1 =
                std::min(y0 + 1, input.height - 1);
            const float wy = source_y - static_cast<float>(y0);
            for (std::uint32_t ox = 0; ox < output.width; ++ox) {
                const float source_x = output.width > 1
                    ? static_cast<float>(ox) *
                        static_cast<float>(input.width - 1) /
                        static_cast<float>(output.width - 1)
                    : 0.0f;
                const std::uint32_t x0 =
                    static_cast<std::uint32_t>(source_x);
                const std::uint32_t x1 =
                    std::min(x0 + 1, input.width - 1);
                const float wx =
                    source_x - static_cast<float>(x0);
                const auto at = [&](std::uint32_t y, std::uint32_t x) {
                    return input.values[
                        (std::uint64_t(channel) * input.height + y) *
                            input.width +
                        x];
                };
                output.values[
                    (std::uint64_t(channel) * output.height + oy) *
                        output.width +
                    ox] =
                    (at(y0, x0) * (1.0f - wx) +
                     at(y0, x1) * wx) *
                        (1.0f - wy) +
                    (at(y1, x0) * (1.0f - wx) +
                     at(y1, x1) * wx) *
                        wy;
            }
        }
    }
    return output;
}

void relu(Image& image) {
    for (float& value : image.values) {
        value = std::max(value, 0.0f);
    }
}

Image add(Image left, const Image& right) {
    if (left.channels != right.channels ||
        left.height != right.height ||
        left.width != right.width) {
        throw std::runtime_error(
            "ZoeDepth feature addition shape mismatch");
    }
    for (std::size_t index = 0;
         index < left.values.size(); ++index) {
        left.values[index] += right.values[index];
    }
    return left;
}

Image residual_unit(
    const ModelFile& model,
    const Image& input,
    const std::string& base) {
    Image branch = input;
    relu(branch);
    branch = conv2d(
        model, branch, base + ".conv1.weight",
        base + ".conv1.bias", 1, 1);
    relu(branch);
    branch = conv2d(
        model, branch, base + ".conv2.weight",
        base + ".conv2.bias", 1, 1);
    return add(std::move(branch), input);
}

Image fusion(
    const ModelFile& model,
    Image path,
    const Image* skip,
    const std::string& base,
    std::uint32_t output_height,
    std::uint32_t output_width) {
    if (skip) {
        path = add(
            std::move(path),
            residual_unit(
                model, *skip, base + ".resConfUnit1"));
    }
    path = residual_unit(
        model, path, base + ".resConfUnit2");
    path = resize_align_corners(
        path, output_height, output_width);
    return conv2d(
        model, path, base + ".out_conv.weight",
        base + ".out_conv.bias", 1, 0);
}

Image postprocess_capture(
    const ModelFile& model,
    const std::vector<float>& capture,
    std::uint32_t patch_height,
    std::uint32_t patch_width,
    std::uint32_t level) {
    const std::uint32_t patches =
        patch_height * patch_width;
    if (capture.size() !=
        std::uint64_t(1 + patches) * 1024) {
        throw std::runtime_error(
            "invalid ZoeDepth encoder capture");
    }
    const std::string base =
        "core.core.pretrained.act_postprocess" +
        std::to_string(level + 1);
    const TensorView& readout_weight = checked_tensor(
        model, base + ".0.project.0.weight", 2);
    const TensorView& readout_bias = checked_tensor(
        model, base + ".0.project.0.bias", 1);
    if (readout_weight.dimensions[0] != 1024 ||
        readout_weight.dimensions[1] != 2048 ||
        readout_bias.dimensions[0] != 1024) {
        throw std::runtime_error(
            "unexpected ZoeDepth readout projection");
    }
    Image projected_tokens{
        1024, patch_height, patch_width, {}};
    projected_tokens.values.resize(
        element_count(1024, patch_height, patch_width));
    const float* cls = capture.data();
    for (std::uint32_t patch = 0; patch < patches; ++patch) {
        const float* token =
            capture.data() + std::uint64_t(1 + patch) * 1024;
        for (std::uint32_t out = 0; out < 1024; ++out) {
            const float* kernel =
                readout_weight.data + std::uint64_t(out) * 2048;
            float value = readout_bias.data[out];
            for (std::uint32_t in = 0; in < 1024; ++in) {
                value += token[in] * kernel[in];
            }
            for (std::uint32_t in = 0; in < 1024; ++in) {
                value += cls[in] * kernel[1024 + in];
            }
            projected_tokens.values[
                (std::uint64_t(out) * patch_height +
                 patch / patch_width) *
                    patch_width +
                patch % patch_width] = gelu(value);
        }
    }
    Image result = conv2d(
        model, projected_tokens, base + ".3.weight",
        base + ".3.bias", 1, 0);
    if (level < 2) {
        result = transpose_conv_nonoverlap(
            model, result, base + ".4.weight",
            base + ".4.bias");
    } else if (level == 3) {
        result = conv2d(
            model, result, base + ".4.weight",
            base + ".4.bias", 2, 1);
    }
    return result;
}

}  // namespace

DecoderOutput midas_decoder_cpu(
    const ModelFile& model,
    EncoderOutput&& encoded) {
    if (encoded.captures.size() != 4 ||
        encoded.patch_width == 0 ||
        encoded.patch_height == 0) {
        throw std::invalid_argument(
            "invalid ZoeDepth encoder output");
    }
    Image layers[4];
    for (std::uint32_t level = 0; level < 4; ++level) {
        layers[level] = postprocess_capture(
            model, encoded.captures[level],
            encoded.patch_height, encoded.patch_width, level);
    }
    Image refined[4];
    for (std::uint32_t level = 0; level < 4; ++level) {
        refined[level] = conv2d(
            model, layers[level],
            "core.core.scratch.layer" +
                std::to_string(level + 1) + "_rn.weight",
            "", 1, 1);
    }
    DecoderOutput output;
    output.bottleneck = refined[3];
    Image path = fusion(
        model, refined[3], nullptr,
        "core.core.scratch.refinenet4",
        refined[2].height, refined[2].width);
    output.refinement_blocks.push_back(path);
    path = fusion(
        model, std::move(path), &refined[2],
        "core.core.scratch.refinenet3",
        refined[1].height, refined[1].width);
    output.refinement_blocks.push_back(path);
    path = fusion(
        model, std::move(path), &refined[1],
        "core.core.scratch.refinenet2",
        refined[0].height, refined[0].width);
    output.refinement_blocks.push_back(path);
    path = fusion(
        model, std::move(path), &refined[0],
        "core.core.scratch.refinenet1",
        refined[0].height * 2, refined[0].width * 2);
    output.refinement_blocks.push_back(path);

    path = conv2d(
        model, path,
        "core.core.scratch.output_conv.0.weight",
        "core.core.scratch.output_conv.0.bias", 1, 1);
    path = resize_align_corners(
        path, path.height * 2, path.width * 2);
    path = conv2d(
        model, path,
        "core.core.scratch.output_conv.2.weight",
        "core.core.scratch.output_conv.2.bias", 1, 1);
    relu(path);
    output.out_conv = path;
    path = conv2d(
        model, path,
        "core.core.scratch.output_conv.4.weight",
        "core.core.scratch.output_conv.4.bias", 1, 0);
    relu(path);
    output.relative_depth = std::move(path);
    return output;
}

}  // namespace zoe_native

