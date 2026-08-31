#pragma once

#include "external_gpu.h"

#include <memory>

struct zoedepth_context;

namespace zoe_native {

std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    zoedepth_context* context);

}  // namespace zoe_native
