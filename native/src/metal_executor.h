#pragma once

#include "model.h"
#include "external_gpu.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace zoe_native {

class MetalExecutor {
public:
    explicit MetalExecutor(const ModelFile& model);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    void set_cache_path(const std::string& cache_path);

    std::vector<float> infer(
        const float* normalized_rgb_chw,
        std::uint32_t width,
        std::uint32_t height);
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace zoe_native
