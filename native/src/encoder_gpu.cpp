#include "encoder_gpu.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zoe_native {
namespace {

constexpr std::uint32_t embedding = 1024;
constexpr std::uint32_t heads = 16;
constexpr const char* prefix = "core.core.pretrained.model.";

const VulkanBuffer& tensor(
    const GpuModel& model, const std::string& name) {
    return model.tensor(name).buffer;
}

bool half_tensor(
    const GpuModel& model, const std::string& name) {
    return model.tensor(name).half_precision;
}

}  // namespace

GpuEncoderOutput encoder_gpu(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& image,
    std::uint32_t width, std::uint32_t height,
    const VulkanBuffer* persistent_zero) {
    if (width == 0 || height == 0 ||
        width % 16 != 0 || height % 16 != 0) {
        throw std::invalid_argument(
            "ZoeDepth GPU encoder dimensions must be multiples of 16");
    }
    const std::uint32_t patch_width = width / 16;
    const std::uint32_t patch_height = height / 16;
    const std::uint32_t tokens =
        patch_width * patch_height + 1;
    const std::uint64_t elements =
        std::uint64_t(tokens) * embedding;
    const VkDeviceSize bytes = elements * sizeof(float);
    VulkanBuffer current = context.create_device_buffer(bytes);
    VulkanBuffer next = context.create_device_buffer(bytes);
    VulkanBuffer normalized = context.create_device_buffer(bytes);
    VulkanBuffer branch = context.create_device_buffer(bytes);
    VulkanBuffer qkv = context.create_device_buffer(bytes * 3);
    VulkanBuffer hidden = context.create_device_buffer(bytes * 4);
    VulkanBuffer scores = context.create_device_buffer(
        std::uint64_t(heads) * tokens * tokens * sizeof(float));
    VulkanBuffer owned_zero;
    if (persistent_zero == nullptr) {
        owned_zero = context.create_device_buffer(
            embedding * 3 * sizeof(float));
        const std::vector<float> zero_values(embedding * 3, 0.0f);
        context.upload(
            owned_zero, zero_values.data(),
            zero_values.size() * sizeof(float));
        persistent_zero = &owned_zero;
    }
    const VulkanBuffer& zero = *persistent_zero;
    context.batch([&] {
        operators.prepare_beit(
            current, image,
            tensor(model, std::string(prefix) + "patch_embed.proj.weight"),
            tensor(model, std::string(prefix) + "patch_embed.proj.bias"),
            tensor(model, std::string(prefix) + "cls_token"),
            width, height);
    });
    GpuEncoderOutput output;
    output.patch_width = patch_width;
    output.patch_height = patch_height;
    output.captures.reserve(4);
    for (std::uint32_t block = 0; block < 24; ++block) {
        const std::string base =
            std::string(prefix) + "blocks." +
            std::to_string(block) + ".";
        context.batch([&] {
            operators.layer_norm(
                normalized, current,
                tensor(model, base + "norm1.weight"),
                tensor(model, base + "norm1.bias"),
                tokens, embedding, 1.0e-6f);
            operators.linear(
                qkv, normalized,
                tensor(model, base + "attn.qkv.weight"),
                zero, tokens, embedding, embedding * 3,
                false, false,
                half_tensor(model, base + "attn.qkv.weight"));
            operators.qv_bias(
                qkv,
                tensor(model, base + "attn.q_bias"),
                tensor(model, base + "attn.v_bias"),
                tokens, embedding);
            operators.attention_head64_relative(
                branch, qkv,
                tensor(
                    model,
                    base + "attn.relative_position_bias_table"),
                scores, patch_width, patch_height, heads);
            operators.linear(
                normalized, branch,
                tensor(model, base + "attn.proj.weight"),
                tensor(model, base + "attn.proj.bias"),
                tokens, embedding, embedding,
                false, false,
                half_tensor(model, base + "attn.proj.weight"));
            operators.add_scaled(
                next, current, normalized,
                tensor(model, base + "gamma_1"),
                static_cast<std::uint32_t>(elements), embedding);
            std::swap(current, next);
            operators.layer_norm(
                normalized, current,
                tensor(model, base + "norm2.weight"),
                tensor(model, base + "norm2.bias"),
                tokens, embedding, 1.0e-6f);
            operators.linear(
                hidden, normalized,
                tensor(model, base + "mlp.fc1.weight"),
                tensor(model, base + "mlp.fc1.bias"),
                tokens, embedding, embedding * 4,
                true, false,
                half_tensor(model, base + "mlp.fc1.weight"));
            operators.linear(
                branch, hidden,
                tensor(model, base + "mlp.fc2.weight"),
                tensor(model, base + "mlp.fc2.bias"),
                tokens, embedding * 4, embedding,
                false, false,
                half_tensor(model, base + "mlp.fc2.weight"));
            operators.add_scaled(
                next, current, branch,
                tensor(model, base + "gamma_2"),
                static_cast<std::uint32_t>(elements), embedding);
            std::swap(current, next);
        });
        if (block == 5 || block == 11 ||
            block == 17 || block == 23) {
            VulkanBuffer capture = context.create_device_buffer(bytes);
            context.copy(capture, 0, current, 0, bytes);
            output.captures.push_back(std::move(capture));
        }
    }
    return output;
}

}  // namespace zoe_native
