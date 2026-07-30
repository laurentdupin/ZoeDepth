#pragma once

#include "model.h"
#include "vulkan.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace zoe_native {

struct GpuTensor {
    VulkanBuffer buffer;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

class GpuModel {
public:
    GpuModel(const ModelFile& model, VulkanContext& context);
    const GpuTensor& tensor(std::string_view name) const;
    std::size_t tensor_count() const { return tensors_.size(); }

private:
    std::unordered_map<std::string_view, GpuTensor> tensors_;
};

}  // namespace zoe_native
