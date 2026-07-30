#pragma once

#include "encoder_cpu.h"
#include "model.h"

#include <cstdint>
#include <vector>

namespace zoe_native {

struct Image {
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::vector<float> values;
};

struct DecoderOutput {
    Image relative_depth;
    Image out_conv;
    Image bottleneck;
    std::vector<Image> refinement_blocks;
};

DecoderOutput midas_decoder_cpu(
    const ModelFile& model,
    EncoderOutput&& encoded);

}  // namespace zoe_native

