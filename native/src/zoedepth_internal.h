#pragma once

#include "external_gpu.h"

#include <memory>
#include <string>

struct zoedepth_context;

namespace zoe_native {

std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    zoedepth_context* context, const std::string& cache_path);

}  // namespace zoe_native
