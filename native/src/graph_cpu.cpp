#include "graph_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

void softplus(Image& image) {
    for (float& value : image.values) {
        value = std::log1p(std::exp(-std::abs(value))) +
            std::max(value, 0.0f);
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

Image concatenate(
    const Image& first,
    const Image& second) {
    if (first.height != second.height ||
        first.width != second.width) {
        throw std::runtime_error(
            "ZoeDepth concatenation shape mismatch");
    }
    Image output{
        first.channels + second.channels,
        first.height, first.width, {}};
    output.values.reserve(
        first.values.size() + second.values.size());
    output.values.insert(
        output.values.end(),
        first.values.begin(), first.values.end());
    output.values.insert(
        output.values.end(),
        second.values.begin(), second.values.end());
    return output;
}

Image projector(
    const ModelFile& model,
    const Image& input,
    const std::string& base) {
    Image output = conv2d(
        model, input, base + ".0.weight",
        base + ".0.bias", 1, 0);
    relu(output);
    return conv2d(
        model, output, base + ".2.weight",
        base + ".2.bias", 1, 0);
}

std::vector<float> linear_vector(
    const ModelFile& model,
    const std::vector<float>& input,
    const std::string& weight_name,
    const std::string& bias_name) {
    const TensorView& weight =
        checked_tensor(model, weight_name, 2);
    const TensorView& bias =
        checked_tensor(model, bias_name, 1);
    if (weight.dimensions[1] != input.size() ||
        bias.dimensions[0] != weight.dimensions[0]) {
        throw std::runtime_error(
            "ZoeDepth linear shape mismatch: " + weight_name);
    }
    std::vector<float> output(
        static_cast<std::size_t>(weight.dimensions[0]));
    for (std::uint64_t out = 0;
         out < weight.dimensions[0]; ++out) {
        float value = bias.data[out];
        for (std::uint64_t in = 0;
             in < weight.dimensions[1]; ++in) {
            value += input[static_cast<std::size_t>(in)] *
                weight.data[out * weight.dimensions[1] + in];
        }
        output[static_cast<std::size_t>(out)] = value;
    }
    return output;
}

void layer_norm_tokens(
    const ModelFile& model,
    std::vector<float>& tokens,
    std::uint32_t token_count,
    const std::string& weight_name,
    const std::string& bias_name) {
    constexpr std::uint32_t features = 128;
    constexpr float epsilon = 1.0e-5f;
    const TensorView& weight =
        checked_tensor(model, weight_name, 1);
    const TensorView& bias =
        checked_tensor(model, bias_name, 1);
    if (weight.dimensions[0] != features ||
        bias.dimensions[0] != features ||
        tokens.size() !=
            std::uint64_t(token_count) * features) {
        throw std::runtime_error(
            "ZoeDepth router layer norm shape mismatch");
    }
    for (std::uint32_t token = 0;
         token < token_count; ++token) {
        float mean = 0.0f;
        for (std::uint32_t feature = 0;
             feature < features; ++feature) {
            mean += tokens[
                std::uint64_t(token) * features + feature];
        }
        mean /= static_cast<float>(features);
        float variance = 0.0f;
        for (std::uint32_t feature = 0;
             feature < features; ++feature) {
            const float centered = tokens[
                std::uint64_t(token) * features + feature] - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(features);
        const float inverse =
            1.0f / std::sqrt(variance + epsilon);
        for (std::uint32_t feature = 0;
             feature < features; ++feature) {
            const std::uint64_t index =
                std::uint64_t(token) * features + feature;
            tokens[index] =
                (tokens[index] - mean) * inverse *
                    weight.data[feature] +
                bias.data[feature];
        }
    }
}

std::string route_nk(
    const ModelFile& model,
    const Image& bottleneck) {
    constexpr std::uint32_t features = 128;
    constexpr std::uint32_t heads = 4;
    constexpr std::uint32_t head_features = features / heads;
    Image embedded = conv2d(
        model, bottleneck,
        "patch_transformer.embedding_convPxP.weight",
        "patch_transformer.embedding_convPxP.bias", 1, 0);
    if (embedded.channels != features) {
        throw std::runtime_error(
            "unexpected ZoeDepth NK router embedding");
    }
    const std::uint32_t token_count =
        embedded.height * embedded.width + 1;
    std::vector<float> tokens(
        std::uint64_t(token_count) * features, 0.0f);
    constexpr float log_10000 = 9.210340371976184f;
    for (std::uint32_t token = 0;
         token < token_count; ++token) {
        for (std::uint32_t pair = 0;
             pair < features / 2; ++pair) {
            const float angle = static_cast<float>(token) *
                std::exp(
                    static_cast<float>(pair * 2) *
                    (-log_10000 / static_cast<float>(features)));
            tokens[std::uint64_t(token) * features + pair] =
                std::sin(angle);
            tokens[std::uint64_t(token) * features +
                features / 2 + pair] = std::cos(angle);
        }
    }
    const std::uint64_t embedded_plane =
        std::uint64_t(embedded.height) * embedded.width;
    for (std::uint32_t token = 1;
         token < token_count; ++token) {
        const std::uint64_t pixel = token - 1;
        for (std::uint32_t feature = 0;
             feature < features; ++feature) {
            tokens[std::uint64_t(token) * features + feature] +=
                embedded.values[
                    std::uint64_t(feature) * embedded_plane +
                    pixel];
        }
    }

    for (std::uint32_t layer = 0; layer < 4; ++layer) {
        const std::string base =
            "patch_transformer.transformer_encoder.layers." +
            std::to_string(layer);
        const TensorView& in_weight = checked_tensor(
            model, base + ".self_attn.in_proj_weight", 2);
        const TensorView& in_bias = checked_tensor(
            model, base + ".self_attn.in_proj_bias", 1);
        if (in_weight.dimensions[0] != 3 * features ||
            in_weight.dimensions[1] != features ||
            in_bias.dimensions[0] != 3 * features) {
            throw std::runtime_error(
                "unexpected ZoeDepth NK attention projection");
        }
        std::vector<float> qkv(
            std::uint64_t(token_count) * 3 * features);
        for (std::uint32_t token = 0;
             token < token_count; ++token) {
            for (std::uint32_t out = 0;
                 out < 3 * features; ++out) {
                float value = in_bias.data[out];
                for (std::uint32_t in = 0;
                     in < features; ++in) {
                    value += tokens[
                        std::uint64_t(token) * features + in] *
                        in_weight.data[
                            std::uint64_t(out) * features + in];
                }
                qkv[
                    std::uint64_t(token) * 3 * features + out] =
                    value;
            }
        }
        std::vector<float> attended(
            std::uint64_t(token_count) * features);
        std::vector<float> scores(token_count);
        const float scale =
            1.0f / std::sqrt(static_cast<float>(head_features));
        for (std::uint32_t query = 0;
             query < token_count; ++query) {
            for (std::uint32_t head = 0;
                 head < heads; ++head) {
                float maximum =
                    -std::numeric_limits<float>::infinity();
                for (std::uint32_t key = 0;
                     key < token_count; ++key) {
                    float score = 0.0f;
                    for (std::uint32_t feature = 0;
                         feature < head_features; ++feature) {
                        const std::uint32_t offset =
                            head * head_features + feature;
                        score += qkv[
                            std::uint64_t(query) *
                                3 * features +
                            offset] *
                            qkv[
                                std::uint64_t(key) *
                                    3 * features +
                                features + offset];
                    }
                    scores[key] = score * scale;
                    maximum = std::max(maximum, scores[key]);
                }
                float denominator = 0.0f;
                for (float& score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (std::uint32_t feature = 0;
                     feature < head_features; ++feature) {
                    const std::uint32_t offset =
                        head * head_features + feature;
                    float value = 0.0f;
                    for (std::uint32_t key = 0;
                         key < token_count; ++key) {
                        value += scores[key] / denominator *
                            qkv[
                                std::uint64_t(key) *
                                    3 * features +
                                2 * features + offset];
                    }
                    attended[
                        std::uint64_t(query) * features + offset] =
                        value;
                }
            }
        }
        const TensorView& out_weight = checked_tensor(
            model, base + ".self_attn.out_proj.weight", 2);
        const TensorView& out_bias = checked_tensor(
            model, base + ".self_attn.out_proj.bias", 1);
        std::vector<float> attention_output(tokens.size());
        for (std::uint32_t token = 0;
             token < token_count; ++token) {
            for (std::uint32_t out = 0;
                 out < features; ++out) {
                float value = out_bias.data[out];
                for (std::uint32_t in = 0;
                     in < features; ++in) {
                    value += attended[
                        std::uint64_t(token) * features + in] *
                        out_weight.data[
                            std::uint64_t(out) * features + in];
                }
                attention_output[
                    std::uint64_t(token) * features + out] =
                    tokens[
                        std::uint64_t(token) * features + out] +
                    value;
            }
        }
        layer_norm_tokens(
            model, attention_output, token_count,
            base + ".norm1.weight", base + ".norm1.bias");

        const TensorView& linear1_weight = checked_tensor(
            model, base + ".linear1.weight", 2);
        const TensorView& linear1_bias = checked_tensor(
            model, base + ".linear1.bias", 1);
        const TensorView& linear2_weight = checked_tensor(
            model, base + ".linear2.weight", 2);
        const TensorView& linear2_bias = checked_tensor(
            model, base + ".linear2.bias", 1);
        if (linear1_weight.dimensions[0] != 1024 ||
            linear1_weight.dimensions[1] != features ||
            linear2_weight.dimensions[0] != features ||
            linear2_weight.dimensions[1] != 1024) {
            throw std::runtime_error(
                "unexpected ZoeDepth NK feed-forward shape");
        }
        std::vector<float> feed_forward(tokens.size());
        std::vector<float> hidden(1024);
        for (std::uint32_t token = 0;
             token < token_count; ++token) {
            for (std::uint32_t out = 0; out < 1024; ++out) {
                float value = linear1_bias.data[out];
                for (std::uint32_t in = 0;
                     in < features; ++in) {
                    value += attention_output[
                        std::uint64_t(token) * features + in] *
                        linear1_weight.data[
                            std::uint64_t(out) * features + in];
                }
                hidden[out] = std::max(value, 0.0f);
            }
            for (std::uint32_t out = 0;
                 out < features; ++out) {
                float value = linear2_bias.data[out];
                for (std::uint32_t in = 0; in < 1024; ++in) {
                    value += hidden[in] *
                        linear2_weight.data[
                            std::uint64_t(out) * 1024 + in];
                }
                feed_forward[
                    std::uint64_t(token) * features + out] =
                    attention_output[
                        std::uint64_t(token) * features + out] +
                    value;
            }
        }
        layer_norm_tokens(
            model, feed_forward, token_count,
            base + ".norm2.weight", base + ".norm2.bias");
        tokens = std::move(feed_forward);
    }

    std::vector<float> class_token(
        tokens.begin(), tokens.begin() + features);
    class_token = linear_vector(
        model, class_token,
        "mlp_classifier.0.weight", "mlp_classifier.0.bias");
    for (float& value : class_token) {
        value = std::max(value, 0.0f);
    }
    const std::vector<float> logits = linear_vector(
        model, class_token,
        "mlp_classifier.2.weight", "mlp_classifier.2.bias");
    if (logits.size() != 2) {
        throw std::runtime_error(
            "unexpected ZoeDepth NK classifier output");
    }
    return logits[0] >= logits[1] ? "nyu" : "kitti";
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

Image metric_depth_cpu(
    const ModelFile& model,
    DecoderOutput&& decoded) {
    if (decoded.bottleneck.channels != 256 ||
        decoded.out_conv.channels != 32 ||
        decoded.relative_depth.channels != 1 ||
        decoded.refinement_blocks.size() != 4) {
        throw std::invalid_argument(
            "invalid ZoeDepth decoder output");
    }
    const bool normed =
        model.derivation().variant == Variant::k;
    const bool dual_head =
        model.derivation().variant == Variant::nk;
    Image bottleneck = conv2d(
        model, decoded.bottleneck,
        "conv2.weight", "conv2.bias", 1, 0);
    const std::string domain =
        dual_head ? route_nk(model, bottleneck) : "";
    const std::string seed_base = dual_head
        ? "seed_bin_regressors." + domain
        : "seed_bin_regressor";
    const std::string attractor_base = dual_head
        ? "attractors." + domain
        : "attractors";
    const std::string distribution_base = dual_head
        ? "conditional_log_binomial." + domain
        : "conditional_log_binomial";

    Image b_previous = conv2d(
        model, bottleneck,
        seed_base + "._net.0.weight",
        seed_base + "._net.0.bias", 1, 0);
    relu(b_previous);
    b_previous = conv2d(
        model, b_previous,
        seed_base + "._net.2.weight",
        seed_base + "._net.2.bias", 1, 0);
    if (normed) {
        relu(b_previous);
        const std::uint64_t pixels =
            std::uint64_t(b_previous.height) * b_previous.width;
        Image normalized_centers = b_previous;
        for (std::uint64_t pixel = 0; pixel < pixels; ++pixel) {
            float sum = 0.0f;
            for (std::uint32_t bin = 0;
                 bin < b_previous.channels; ++bin) {
                b_previous.values[
                    std::uint64_t(bin) * pixels + pixel] += 1.0e-3f;
                sum += b_previous.values[
                    std::uint64_t(bin) * pixels + pixel];
            }
            float edge = 0.0f;
            for (std::uint32_t bin = 0;
                 bin < b_previous.channels; ++bin) {
                const float width = b_previous.values[
                    std::uint64_t(bin) * pixels + pixel] / sum;
                normalized_centers.values[
                    std::uint64_t(bin) * pixels + pixel] =
                    edge + 0.5f * width;
                edge += width;
            }
        }
        b_previous = std::move(normalized_centers);
    } else {
        softplus(b_previous);
    }
    Image previous_embedding = projector(
        model, bottleneck, "seed_projector._net");

    Image centers;
    Image embedding;
    const std::uint32_t attractor_counts[4] = {
        16, dual_head ? 16u : 8u,
        dual_head ? 16u : 4u,
        dual_head ? 16u : 1u};
    for (std::uint32_t level = 0; level < 4; ++level) {
        embedding = projector(
            model, decoded.refinement_blocks[level],
            "projectors." + std::to_string(level) + "._net");
        Image previous_embedding_resized = resize_align_corners(
            previous_embedding, embedding.height, embedding.width);
        Image attractor_input = add(
            embedding, previous_embedding_resized);
        Image attractors = conv2d(
            model, attractor_input,
            attractor_base + "." + std::to_string(level) +
                "._net.0.weight",
            attractor_base + "." + std::to_string(level) +
                "._net.0.bias",
            1, 0);
        relu(attractors);
        attractors = conv2d(
            model, attractors,
            attractor_base + "." + std::to_string(level) +
                "._net.2.weight",
            attractor_base + "." + std::to_string(level) +
                "._net.2.bias",
            1, 0);
        if (normed) {
            relu(attractors);
            for (float& value : attractors.values) {
                value += 1.0e-3f;
            }
        } else {
            softplus(attractors);
        }
        const std::uint32_t expected_attractor_channels =
            attractor_counts[level] * (normed ? 2u : 1u);
        if (attractors.channels != expected_attractor_channels) {
            throw std::runtime_error(
                "unexpected ZoeDepth attractor count");
        }
        b_previous = resize_align_corners(
            b_previous, embedding.height, embedding.width);
        centers = b_previous;
        const std::uint64_t pixels =
            std::uint64_t(embedding.height) * embedding.width;
        for (std::uint32_t bin = 0;
             bin < centers.channels; ++bin) {
            for (std::uint64_t pixel = 0;
                 pixel < pixels; ++pixel) {
                const float center =
                    b_previous.values[
                        std::uint64_t(bin) * pixels + pixel];
                float delta = 0.0f;
                for (std::uint32_t attractor = 0;
                     attractor < attractor_counts[level];
                     ++attractor) {
                    const std::uint32_t attractor_channel =
                        normed ? attractor * 2 : attractor;
                    const float difference =
                        attractors.values[
                            std::uint64_t(attractor_channel) * pixels +
                            pixel] -
                        center;
                    delta += difference /
                        (1.0f + 1000.0f *
                            difference * difference);
                }
                centers.values[
                    std::uint64_t(bin) * pixels + pixel] =
                    center + delta /
                        static_cast<float>(
                            attractor_counts[level]);
            }
        }
        if (normed) {
            b_previous = centers;
            constexpr float minimum_depth = 1.0e-3f;
            constexpr float maximum_depth = 10.0f;
            for (std::uint64_t pixel = 0; pixel < pixels; ++pixel) {
                std::vector<float> sorted(centers.channels);
                for (std::uint32_t bin = 0;
                     bin < centers.channels; ++bin) {
                    sorted[bin] = std::clamp(
                        (maximum_depth - minimum_depth) *
                                centers.values[
                                    std::uint64_t(bin) * pixels +
                                    pixel] +
                            minimum_depth,
                        minimum_depth, maximum_depth);
                }
                std::sort(sorted.begin(), sorted.end());
                for (std::uint32_t bin = 0;
                     bin < centers.channels; ++bin) {
                    centers.values[
                        std::uint64_t(bin) * pixels + pixel] =
                        sorted[bin];
                }
            }
        } else {
            b_previous = centers;
        }
        previous_embedding = embedding;
    }

    Image last = decoded.out_conv;
    if (!dual_head) {
        Image relative = resize_align_corners(
            decoded.relative_depth,
            decoded.out_conv.height,
            decoded.out_conv.width);
        last = concatenate(decoded.out_conv, relative);
    }
    embedding = resize_align_corners(
        embedding, last.height, last.width);
    Image probability_parameters = concatenate(last, embedding);
    probability_parameters = conv2d(
        model, probability_parameters,
        distribution_base + ".mlp.0.weight",
        distribution_base + ".mlp.0.bias", 1, 0);
    for (float& value : probability_parameters.values) {
        value = gelu(value);
    }
    probability_parameters = conv2d(
        model, probability_parameters,
        distribution_base + ".mlp.2.weight",
        distribution_base + ".mlp.2.bias", 1, 0);
    softplus(probability_parameters);
    if (probability_parameters.channels != 4 ||
        centers.channels != 64) {
        throw std::runtime_error(
            "unexpected ZoeDepth distribution shape");
    }
    centers = resize_align_corners(
        centers, last.height, last.width);
    const std::uint64_t pixels =
        std::uint64_t(last.height) * last.width;
    Image depth{1, last.height, last.width, {}};
    depth.values.resize(static_cast<std::size_t>(pixels));
    std::vector<float> logits(64);
    constexpr float epsilon = 1.0e-4f;
    constexpr float log_epsilon = 1.0e-7f;
    for (std::uint64_t pixel = 0; pixel < pixels; ++pixel) {
        const float p0 =
            probability_parameters.values[pixel] + epsilon;
        const float p1 =
            probability_parameters.values[pixels + pixel] + epsilon;
        const float probability = p0 / (p0 + p1);
        const float t0 =
            probability_parameters.values[2 * pixels + pixel] +
            epsilon;
        const float t1 =
            probability_parameters.values[3 * pixels + pixel] +
            epsilon;
        const float temperature =
            (50.0f - 0.0212f) * (t0 / (t0 + t1)) +
            0.0212f;
        const float n = 63.0f + log_epsilon;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::uint32_t bin = 0; bin < 64; ++bin) {
            const float k =
                static_cast<float>(bin) + log_epsilon;
            const float n_minus_k = n - k;
            const float log_combination =
                n * std::log(n) -
                k * std::log(k) -
                n_minus_k *
                    std::log(n_minus_k + log_epsilon);
            const float one_minus_probability =
                std::clamp(
                    1.0f - probability, epsilon, 1.0f);
            const float clamped_probability =
                std::clamp(probability, epsilon, 1.0f);
            logits[bin] = (
                log_combination +
                k * std::log(clamped_probability) +
                (64.0f - 1.0f - k) *
                    std::log(one_minus_probability)) /
                temperature;
            maximum = std::max(maximum, logits[bin]);
        }
        float denominator = 0.0f;
        for (float& logit : logits) {
            logit = std::exp(logit - maximum);
            denominator += logit;
        }
        float value = 0.0f;
        for (std::uint32_t bin = 0; bin < 64; ++bin) {
            value +=
                (logits[bin] / denominator) *
                centers.values[
                    std::uint64_t(bin) * pixels + pixel];
        }
        depth.values[pixel] = value;
    }
    return depth;
}

}  // namespace zoe_native
