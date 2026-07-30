#pragma once

#include "encoder_gpu.h"

namespace zoe_native {

struct GpuFeature {
    VulkanBuffer buffer;
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
};

GpuFeature full_graph_gpu(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, GpuEncoderOutput&& encoded);

}  // namespace zoe_native
