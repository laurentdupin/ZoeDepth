#pragma once

#include "zoedepth_native.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace zoe_native {

class GpuSlotsExhausted final : public std::runtime_error {
public:
    GpuSlotsExhausted()
        : std::runtime_error("all ZoeDepth GPU output slots are retained") {}
};

struct ExternalGpuCapabilities {
    bool available = false;
    std::uint64_t adapter_luid = 0;
    std::uint32_t maximum_in_flight_jobs = 0;
};

struct ExternalTextureRequest {
    std::uintptr_t shared_texture_handle = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    bool rgba = false;
    std::uint32_t input_size = 0;
    std::uintptr_t wait_fence_handle = 0;
    std::uint64_t wait_fence_value = 0;
    std::uintptr_t output_texture_handle = 0;
    std::uint32_t output_width = 0;
    std::uint32_t output_height = 0;
    std::uintptr_t signal_fence_handle = 0;
    std::uint64_t signal_fence_value = 0;
    std::uint64_t source_frame_id = 0;
    std::uint64_t timestamp_ns = 0;
};

enum class ExternalJobState { running, complete, cancelled };

class ExternalJob {
public:
    virtual ~ExternalJob() = default;
    virtual ExternalJobState state() const = 0;
    virtual void cancel() = 0;
};

class ExternalGpu : public std::enable_shared_from_this<ExternalGpu> {
public:
    virtual ~ExternalGpu() = default;
    virtual ExternalGpuCapabilities capabilities() const = 0;
    virtual std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) = 0;
    virtual void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const = 0;
};

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& model_path,
    zoedepth_variant variant,
    std::uint32_t device_index);
ExternalGpuCapabilities probe_external_gpu(std::uint32_t device_index);

}  // namespace zoe_native
