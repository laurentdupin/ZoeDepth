#pragma once

#include "model.h"
#include "vulkan.h"
#include <inferbridge/native_harness_precision.h>

#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace zoe_native {

struct GpuTensor {
    VulkanBuffer buffer;
    VulkanBuffer int8_buffer;
    VulkanBuffer int8_scales;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
    bool half_precision = false;
};

class GpuModel {
public:
    GpuModel(const ModelFile& model, VulkanContext& context);
    const GpuTensor& tensor(std::string_view name) const;
    std::size_t tensor_count() const { return tensors_.size(); }
    Variant variant() const { return variant_; }
    bool uses_int8_weights() const {
        return precision_ == inferbridge::native::Precision::int8;
    }

private:
    Variant variant_ = Variant::n;
    std::unordered_map<std::string_view, GpuTensor> tensors_;
    inferbridge::native::Precision precision_ =
        inferbridge::native::Precision::automatic;
};

}  // namespace zoe_native
