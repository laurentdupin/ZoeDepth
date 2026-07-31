#pragma once

#include "vulkan.h"

#include <cstdint>

namespace zoe_native {

class GpuIo {
public:
    explicit GpuIo(VulkanContext& context);
    void preprocess(
        VulkanBuffer& destination,
        const VulkanImage& source,
        std::uint32_t destination_width,
        std::uint32_t destination_height,
        std::uint32_t padded_width,
        std::uint32_t padded_height,
        std::uint32_t pad_width,
        std::uint32_t pad_height,
        bool flip);
    void combine_depth(
        VulkanImage& destination,
        const VulkanBuffer& direct,
        const VulkanBuffer& flipped,
        std::uint32_t network_width,
        std::uint32_t network_height,
        std::uint32_t padded_width,
        std::uint32_t padded_height,
        std::uint32_t pad_width,
        std::uint32_t pad_height);

private:
    VulkanContext& context_;
    VulkanPipeline preprocess_;
    VulkanPipeline combine_depth_;
};

}  // namespace zoe_native
