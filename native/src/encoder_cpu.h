#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

namespace zoe_native {

struct EncoderOutput {
    std::uint32_t patch_width = 0;
    std::uint32_t patch_height = 0;
    std::vector<std::vector<float>> captures;
};

EncoderOutput encoder_cpu(
    const ModelFile& model,
    const float* normalized_rgb_chw,
    std::uint32_t width,
    std::uint32_t height);

}  // namespace zoe_native

