#pragma once

#include "model.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace zoe_native {

class MetalExecutor {
public:
    explicit MetalExecutor(const ModelFile& model);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    std::vector<float> infer(
        const float* normalized_rgb_chw,
        std::uint32_t width,
        std::uint32_t height);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace zoe_native
