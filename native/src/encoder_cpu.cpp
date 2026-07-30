#include "encoder_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace zoe_native {
namespace {

constexpr std::uint32_t embedding = 1024;
constexpr std::uint32_t heads = 16;
constexpr std::uint32_t head_channels = 64;
constexpr const char* prefix =
    "core.core.pretrained.model.";

const TensorView& tensor(
    const ModelFile& model,
    const std::string& name,
    std::uint32_t rank) {
    const TensorView& result = model.tensor(name);
    if (result.rank != rank) {
        throw std::runtime_error(
            "unexpected ZoeDepth encoder tensor rank: " + name);
    }
    return result;
}

void linear(
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t input_channels,
    const TensorView& weight,
    const TensorView* bias,
    std::vector<float>& output) {
    if (weight.rank != 2 ||
        weight.dimensions[1] != input_channels ||
        input.size() != std::uint64_t(rows) * input_channels ||
        (bias && (
            bias->rank != 1 ||
            bias->dimensions[0] != weight.dimensions[0]))) {
        throw std::runtime_error(
            "ZoeDepth encoder linear shape mismatch");
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight.dimensions[0]);
    output.resize(std::uint64_t(rows) * output_channels);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            input.data() + std::uint64_t(row) * input_channels;
        float* destination =
            output.data() + std::uint64_t(row) * output_channels;
        for (std::uint32_t out = 0;
             out < output_channels; ++out) {
            const float* kernel =
                weight.data + std::uint64_t(out) * input_channels;
            float value = bias ? bias->data[out] : 0.0f;
            for (std::uint32_t in = 0;
                 in < input_channels; ++in) {
                value += source[in] * kernel[in];
            }
            destination[out] = value;
        }
    }
}

void layer_norm(
    const std::vector<float>& input,
    std::uint32_t rows,
    const TensorView& scale,
    const TensorView& bias,
    std::vector<float>& output) {
    if (input.size() != std::uint64_t(rows) * embedding ||
        scale.rank != 1 || bias.rank != 1 ||
        scale.dimensions[0] != embedding ||
        bias.dimensions[0] != embedding) {
        throw std::runtime_error(
            "ZoeDepth encoder layer norm shape mismatch");
    }
    output.resize(input.size());
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float* source =
            input.data() + std::uint64_t(row) * embedding;
        float* destination =
            output.data() + std::uint64_t(row) * embedding;
        float mean = 0.0f;
        for (std::uint32_t channel = 0;
             channel < embedding; ++channel) {
            mean += source[channel];
        }
        mean /= static_cast<float>(embedding);
        float variance = 0.0f;
        for (std::uint32_t channel = 0;
             channel < embedding; ++channel) {
            const float difference = source[channel] - mean;
            variance += difference * difference;
        }
        variance /= static_cast<float>(embedding);
        const float inverse =
            1.0f / std::sqrt(variance + 1.0e-6f);
        for (std::uint32_t channel = 0;
             channel < embedding; ++channel) {
            destination[channel] =
                (source[channel] - mean) * inverse *
                    scale.data[channel] +
                bias.data[channel];
        }
    }
}

float bilinear_table(
    const float* table,
    std::uint32_t head,
    std::uint32_t output_x,
    std::uint32_t output_y,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    constexpr std::uint32_t source_width = 47;
    constexpr std::uint32_t source_height = 47;
    float source_x =
        (static_cast<float>(output_x) + 0.5f) *
            source_width / output_width -
        0.5f;
    float source_y =
        (static_cast<float>(output_y) + 0.5f) *
            source_height / output_height -
        0.5f;
    source_x = std::clamp(
        source_x, 0.0f, static_cast<float>(source_width - 1));
    source_y = std::clamp(
        source_y, 0.0f, static_cast<float>(source_height - 1));
    const std::uint32_t x0 =
        static_cast<std::uint32_t>(std::floor(source_x));
    const std::uint32_t y0 =
        static_cast<std::uint32_t>(std::floor(source_y));
    const std::uint32_t x1 =
        std::min(x0 + 1, source_width - 1);
    const std::uint32_t y1 =
        std::min(y0 + 1, source_height - 1);
    const float wx = source_x - static_cast<float>(x0);
    const float wy = source_y - static_cast<float>(y0);
    const auto at = [&](std::uint32_t x, std::uint32_t y) {
        return table[
            (std::uint64_t(x) * source_height + y) *
                heads +
            head];
    };
    return (at(x0, y0) * (1.0f - wx) + at(x1, y0) * wx) *
               (1.0f - wy) +
           (at(x0, y1) * (1.0f - wx) + at(x1, y1) * wx) *
               wy;
}

std::vector<float> relative_bias(
    const TensorView& table,
    std::uint32_t patch_width,
    std::uint32_t patch_height) {
    if (table.rank != 2 ||
        table.dimensions[0] != 2212 ||
        table.dimensions[1] != heads) {
        throw std::runtime_error(
            "unexpected BEiT relative bias table");
    }
    const std::uint32_t tokens =
        1 + patch_width * patch_height;
    const std::uint32_t relative_width =
        2 * patch_width - 1;
    const std::uint32_t relative_height =
        2 * patch_height - 1;
    const std::uint32_t spatial_count =
        relative_width * relative_height;
    std::vector<float> result(
        std::uint64_t(heads) * tokens * tokens);
    for (std::uint32_t head = 0; head < heads; ++head) {
        for (std::uint32_t query = 0; query < tokens; ++query) {
            for (std::uint32_t key = 0; key < tokens; ++key) {
                std::uint32_t index = 0;
                if (query == 0 && key == 0) {
                    index = spatial_count + 2;
                } else if (query == 0) {
                    index = spatial_count;
                } else if (key == 0) {
                    index = spatial_count + 1;
                } else {
                    const std::uint32_t query_patch = query - 1;
                    const std::uint32_t key_patch = key - 1;
                    const int dy =
                        static_cast<int>(
                            query_patch / patch_width) -
                        static_cast<int>(
                            key_patch / patch_width);
                    const int dx =
                        static_cast<int>(
                            query_patch % patch_width) -
                        static_cast<int>(
                            key_patch % patch_width);
                    index = static_cast<std::uint32_t>(
                        (dy + static_cast<int>(patch_height) - 1) *
                            static_cast<int>(relative_width) +
                        dx + static_cast<int>(patch_width) - 1);
                }
                float value = 0.0f;
                if (index >= spatial_count) {
                    value = table.data[
                        std::uint64_t(2209 + index - spatial_count) *
                            heads +
                        head];
                } else {
                    const std::uint32_t output_x =
                        index / relative_height;
                    const std::uint32_t output_y =
                        index % relative_height;
                    value = bilinear_table(
                        table.data, head, output_x, output_y,
                        relative_width, relative_height);
                }
                result[
                    (std::uint64_t(head) * tokens + query) *
                        tokens +
                    key] = value;
            }
        }
    }
    return result;
}

void attention(
    const ModelFile& model,
    std::uint32_t block,
    std::uint32_t patch_width,
    std::uint32_t patch_height,
    const std::vector<float>& input,
    std::vector<float>& output) {
    const std::uint32_t tokens =
        1 + patch_width * patch_height;
    const std::string base =
        std::string(prefix) + "blocks." +
        std::to_string(block) + ".attn.";
    std::vector<float> qkv;
    linear(
        input, tokens, embedding,
        tensor(model, base + "qkv.weight", 2), nullptr, qkv);
    const TensorView& q_bias =
        tensor(model, base + "q_bias", 1);
    const TensorView& v_bias =
        tensor(model, base + "v_bias", 1);
    for (std::uint32_t token = 0; token < tokens; ++token) {
        float* row =
            qkv.data() + std::uint64_t(token) * 3 * embedding;
        for (std::uint32_t channel = 0;
             channel < embedding; ++channel) {
            row[channel] += q_bias.data[channel];
            row[2 * embedding + channel] +=
                v_bias.data[channel];
        }
    }
    const std::vector<float> bias = relative_bias(
        tensor(model, base + "relative_position_bias_table", 2),
        patch_width, patch_height);
    std::vector<float> attended(
        std::uint64_t(tokens) * embedding);
    std::vector<float> scores(tokens);
    for (std::uint32_t head = 0; head < heads; ++head) {
        for (std::uint32_t query = 0;
             query < tokens; ++query) {
            const float* q = qkv.data() +
                std::uint64_t(query) * 3 * embedding +
                head * head_channels;
            float maximum = -std::numeric_limits<float>::infinity();
            for (std::uint32_t key = 0; key < tokens; ++key) {
                const float* k = qkv.data() +
                    std::uint64_t(key) * 3 * embedding +
                    embedding + head * head_channels;
                float score = bias[
                    (std::uint64_t(head) * tokens + query) *
                        tokens +
                    key];
                for (std::uint32_t channel = 0;
                     channel < head_channels; ++channel) {
                    score +=
                        (q[channel] * 0.125f) * k[channel];
                }
                scores[key] = score;
                maximum = std::max(maximum, score);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            float* destination = attended.data() +
                std::uint64_t(query) * embedding +
                head * head_channels;
            std::fill_n(destination, head_channels, 0.0f);
            for (std::uint32_t key = 0; key < tokens; ++key) {
                const float probability =
                    scores[key] / denominator;
                const float* value = qkv.data() +
                    std::uint64_t(key) * 3 * embedding +
                    2 * embedding + head * head_channels;
                for (std::uint32_t channel = 0;
                     channel < head_channels; ++channel) {
                    destination[channel] +=
                        probability * value[channel];
                }
            }
        }
    }
    linear(
        attended, tokens, embedding,
        tensor(model, base + "proj.weight", 2),
        &tensor(model, base + "proj.bias", 1), output);
}

void transformer_block(
    const ModelFile& model,
    std::uint32_t block,
    std::uint32_t patch_width,
    std::uint32_t patch_height,
    std::vector<float>& state) {
    const std::uint32_t tokens =
        1 + patch_width * patch_height;
    const std::string base =
        std::string(prefix) + "blocks." +
        std::to_string(block) + ".";
    std::vector<float> normalized;
    layer_norm(
        state, tokens,
        tensor(model, base + "norm1.weight", 1),
        tensor(model, base + "norm1.bias", 1), normalized);
    std::vector<float> branch;
    attention(
        model, block, patch_width, patch_height,
        normalized, branch);
    const TensorView& gamma1 =
        tensor(model, base + "gamma_1", 1);
    for (std::uint32_t row = 0; row < tokens; ++row) {
        for (std::uint32_t channel = 0;
             channel < embedding; ++channel) {
            state[std::uint64_t(row) * embedding + channel] +=
                branch[std::uint64_t(row) * embedding + channel] *
                gamma1.data[channel];
        }
    }
    layer_norm(
        state, tokens,
        tensor(model, base + "norm2.weight", 1),
        tensor(model, base + "norm2.bias", 1), normalized);
    linear(
        normalized, tokens, embedding,
        tensor(model, base + "mlp.fc1.weight", 2),
        &tensor(model, base + "mlp.fc1.bias", 1), branch);
    constexpr float inverse_sqrt_two =
        0.7071067811865475244f;
    for (float& value : branch) {
        value = 0.5f * value *
            (1.0f + std::erf(value * inverse_sqrt_two));
    }
    std::vector<float> projected;
    linear(
        branch, tokens, 4096,
        tensor(model, base + "mlp.fc2.weight", 2),
        &tensor(model, base + "mlp.fc2.bias", 1), projected);
    const TensorView& gamma2 =
        tensor(model, base + "gamma_2", 1);
    for (std::uint32_t row = 0; row < tokens; ++row) {
        for (std::uint32_t channel = 0;
             channel < embedding; ++channel) {
            state[std::uint64_t(row) * embedding + channel] +=
                projected[std::uint64_t(row) * embedding + channel] *
                gamma2.data[channel];
        }
    }
}

}  // namespace

EncoderOutput encoder_cpu(
    const ModelFile& model,
    const float* input,
    std::uint32_t width,
    std::uint32_t height) {
    if (!input || width == 0 || height == 0 ||
        width % 16 != 0 || height % 16 != 0) {
        throw std::invalid_argument(
            "ZoeDepth encoder dimensions must be multiples of 16");
    }
    const std::uint32_t patch_width = width / 16;
    const std::uint32_t patch_height = height / 16;
    const std::uint32_t patches =
        patch_width * patch_height;
    const std::uint32_t tokens = 1 + patches;
    const TensorView& weight = tensor(
        model,
        std::string(prefix) + "patch_embed.proj.weight", 4);
    const TensorView& bias = tensor(
        model,
        std::string(prefix) + "patch_embed.proj.bias", 1);
    if (weight.dimensions[0] != embedding ||
        weight.dimensions[1] != 3 ||
        weight.dimensions[2] != 16 ||
        weight.dimensions[3] != 16 ||
        bias.dimensions[0] != embedding) {
        throw std::runtime_error(
            "unexpected ZoeDepth patch embedding shape");
    }
    std::vector<float> state(
        std::uint64_t(tokens) * embedding);
    const TensorView& cls = tensor(
        model, std::string(prefix) + "cls_token", 3);
    std::copy_n(cls.data, embedding, state.data());
    for (std::uint32_t py = 0; py < patch_height; ++py) {
        for (std::uint32_t px = 0; px < patch_width; ++px) {
            float* destination = state.data() +
                (1 + std::uint64_t(py) * patch_width + px) *
                    embedding;
            for (std::uint32_t out = 0;
                 out < embedding; ++out) {
                float value = bias.data[out];
                const float* kernel =
                    weight.data + std::uint64_t(out) * 3 * 16 * 16;
                for (std::uint32_t channel = 0;
                     channel < 3; ++channel) {
                    for (std::uint32_t ky = 0; ky < 16; ++ky) {
                        const float* source = input +
                            std::uint64_t(channel) * width * height +
                            std::uint64_t(py * 16 + ky) * width +
                            px * 16;
                        const float* row = kernel +
                            std::uint64_t(channel) * 16 * 16 +
                            ky * 16;
                        for (std::uint32_t kx = 0; kx < 16; ++kx) {
                            value += source[kx] * row[kx];
                        }
                    }
                }
                destination[out] = value;
            }
        }
    }
    EncoderOutput output;
    output.patch_width = patch_width;
    output.patch_height = patch_height;
    for (std::uint32_t block = 0; block < 24; ++block) {
        transformer_block(
            model, block, patch_width, patch_height, state);
        if (block == 5 || block == 11 ||
            block == 17 || block == 23) {
            output.captures.push_back(state);
        }
    }
    return output;
}

}  // namespace zoe_native
