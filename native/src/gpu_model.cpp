#include "gpu_model.h"
#include "inferbridge/native_harness_precision.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace zoe_native {
namespace {

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.substr(value.size() - suffix.size()) == suffix;
}

bool use_half_precision(std::string_view name) {
    return ends_with(name, ".attn.qkv.weight") ||
        ends_with(name, ".attn.proj.weight") ||
        ends_with(name, ".mlp.fc1.weight") ||
        ends_with(name, ".mlp.fc2.weight");
}

std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u) {
        const std::uint32_t payload =
            magnitude > 0x7f800000u ? 0x0200u : 0u;
        return static_cast<std::uint16_t>(sign | 0x7c00u | payload);
    }
    const int exponent = static_cast<int>(magnitude >> 23) - 127;
    const std::uint32_t mantissa = magnitude & 0x7fffffu;
    if (exponent > 15) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (exponent >= -14) {
        std::uint32_t rounded =
            mantissa + 0x0fffu + ((mantissa >> 13) & 1u);
        std::uint32_t half_exponent =
            static_cast<std::uint32_t>(exponent + 15);
        if (rounded & 0x800000u) {
            rounded = 0;
            ++half_exponent;
            if (half_exponent >= 31u) {
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            }
        }
        return static_cast<std::uint16_t>(
            sign | (half_exponent << 10) | (rounded >> 13));
    }
    if (exponent < -24) {
        return static_cast<std::uint16_t>(sign);
    }
    const std::uint32_t normalized = mantissa | 0x800000u;
    const std::uint32_t shift =
        static_cast<std::uint32_t>(-exponent - 1);
    const std::uint32_t halfway = 1u << (shift - 1u);
    const std::uint32_t rounded =
        normalized + halfway - 1u + ((normalized >> shift) & 1u);
    return static_cast<std::uint16_t>(sign | (rounded >> shift));
}

}  // namespace

GpuModel::GpuModel(const ModelFile& model, VulkanContext& context) {
    variant_ = model.derivation().variant;
    tensors_.reserve(model.tensor_count());
    constexpr std::uint64_t compact_weight_memory_limit =
        13ull * 1024ull * 1024ull * 1024ull;
    const bool automatic_half = context.float16_storage() &&
        context.device_local_bytes() <= compact_weight_memory_limit;
    precision_ = inferbridge::native::require_supported_precision(
        inferbridge::native::requested_precision(),
        {context.float16_storage(), context.supports_packed_int8_dot()},
        automatic_half ? inferbridge::native::Precision::fp16
                       : inferbridge::native::Precision::fp32);
    const bool compact_weights =
        precision_ == inferbridge::native::Precision::fp16;
    for (std::string_view name : model.tensor_names()) {
        const TensorView& source = model.tensor(name);
        if (source.elements >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error(
                "model tensor is too large: " + std::string(name));
        }
        const bool half_precision =
            compact_weights && use_half_precision(name);
        const std::size_t bytes =
            static_cast<std::size_t>(source.elements) *
            (half_precision ? sizeof(std::uint16_t) : sizeof(float));
        GpuTensor destination{
            context.create_device_buffer(bytes),
            {},
            {},
            source.dimensions,
            source.rank,
            source.elements,
            half_precision,
        };
        if (half_precision) {
            std::vector<std::uint16_t> packed(
                static_cast<std::size_t>(source.elements));
            for (std::size_t index = 0; index < packed.size(); ++index) {
                packed[index] = float_to_half(source.data[index]);
            }
            context.upload(destination.buffer, packed.data(), bytes);
        } else {
            context.upload(destination.buffer, source.data, bytes);
        }
        if (precision_ == inferbridge::native::Precision::int8 &&
            source.rank == 2 && source.dimensions[1] % 4u == 0u) {
            const auto quantized = inferbridge::native::quantize_int8_rows(
                source.data, static_cast<std::size_t>(source.dimensions[0]),
                static_cast<std::size_t>(source.dimensions[1]));
            destination.int8_buffer = context.create_device_buffer(
                quantized.packed.size() * sizeof(std::uint32_t));
            destination.int8_scales = context.create_device_buffer(
                quantized.scales.size() * sizeof(float));
            context.upload(destination.int8_buffer, quantized.packed.data(),
                quantized.packed.size() * sizeof(std::uint32_t));
            context.upload(destination.int8_scales, quantized.scales.data(),
                quantized.scales.size() * sizeof(float));
        }
        if (!tensors_.emplace(name, std::move(destination)).second) {
            throw std::runtime_error(
                "duplicate GPU tensor name: " + std::string(name));
        }
    }
}

const GpuTensor& GpuModel::tensor(std::string_view name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "GPU model is missing tensor: " + std::string(name));
    }
    return found->second;
}

}  // namespace zoe_native
