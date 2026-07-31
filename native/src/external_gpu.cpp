#include "external_gpu.h"

#include "gpu_io.h"
#include "encoder_gpu.h"
#include "gpu_model.h"
#include "graph_gpu.h"
#include "model.h"
#include "operators.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace zoe_native {
namespace {

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kGpuSlotCount = 3u;

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
    }
}

ComPtr<ID3D12Device> matching_d3d12_device(std::uint64_t luid) {
    if (luid == 0u) return {};
    ComPtr<IDXGIFactory6> factory;
    check_hresult(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        check_hresult(result, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check_hresult(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0u;
        std::memcpy(&candidate, &description.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check_hresult(
            D3D12CreateDevice(
                adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    return {};
}

struct SharedOutput {
    ComPtr<ID3D12Resource> resource;
    ComPtr<ID3D12Fence> fence;
    HANDLE resource_handle = nullptr;
    HANDLE fence_handle = nullptr;
    SharedOutput() = default;
    SharedOutput(const SharedOutput&) = delete;
    SharedOutput& operator=(const SharedOutput&) = delete;
    SharedOutput(SharedOutput&& other) noexcept
        : resource(std::move(other.resource)), fence(std::move(other.fence)),
          resource_handle(std::exchange(other.resource_handle, nullptr)),
          fence_handle(std::exchange(other.fence_handle, nullptr)) {}
    SharedOutput& operator=(SharedOutput&& other) noexcept {
        if (this != &other) {
            if (resource_handle != nullptr) CloseHandle(resource_handle);
            if (fence_handle != nullptr) CloseHandle(fence_handle);
            resource = std::move(other.resource);
            fence = std::move(other.fence);
            resource_handle = std::exchange(other.resource_handle, nullptr);
            fence_handle = std::exchange(other.fence_handle, nullptr);
        }
        return *this;
    }
    ~SharedOutput() {
        if (resource_handle != nullptr) CloseHandle(resource_handle);
        if (fence_handle != nullptr) CloseHandle(fence_handle);
    }
};

struct GpuSlot {
    std::atomic<bool> occupied{false};
    SharedOutput shared;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t fence_value = 0;
};

SharedOutput create_shared_output(
    ID3D12Device* device,
    std::uint32_t width,
    std::uint32_t height) {
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC description{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_R32_FLOAT, {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    SharedOutput output;
    check_hresult(
        device->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_SHARED, &description,
            D3D12_RESOURCE_STATE_COMMON, nullptr,
            IID_PPV_ARGS(&output.resource)),
        "CreateCommittedResource(ZoeDepth output)");
    check_hresult(
        device->CreateFence(
            0, D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&output.fence)),
        "CreateFence(ZoeDepth output)");
    check_hresult(
        device->CreateSharedHandle(
            output.resource.Get(), nullptr, GENERIC_ALL, nullptr,
            &output.resource_handle),
        "CreateSharedHandle(ZoeDepth output)");
    check_hresult(
        device->CreateSharedHandle(
            output.fence.Get(), nullptr, GENERIC_ALL, nullptr,
            &output.fence_handle),
        "CreateSharedHandle(ZoeDepth fence)");
    return output;
}

void validate_input(
    ID3D12Device* device,
    std::uintptr_t handle,
    std::uint32_t width,
    std::uint32_t height) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(
        device->OpenSharedHandle(
            reinterpret_cast<HANDLE>(handle),
            IID_PPV_ARGS(&resource)),
        "OpenSharedHandle(ZoeDepth input)");
    const D3D12_RESOURCE_DESC description = resource->GetDesc();
    if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        description.Width != width || description.Height != height ||
        description.DepthOrArraySize != 1u || description.MipLevels != 1u ||
        description.SampleDesc.Count != 1u ||
        description.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
        throw std::invalid_argument(
            "shared ZoeDepth input is not the declared BGRA8 texture");
    }
}

std::shared_ptr<GpuSlot> acquire_slot(
    const std::array<std::shared_ptr<GpuSlot>, kGpuSlotCount>& slots,
    std::atomic<std::uint32_t>& next) {
    const std::uint32_t first = next.fetch_add(1u) % kGpuSlotCount;
    for (std::uint32_t offset = 0; offset < kGpuSlotCount; ++offset) {
        const auto& slot = slots[(first + offset) % kGpuSlotCount];
        bool expected = false;
        if (slot->occupied.compare_exchange_strong(expected, true)) return slot;
    }
    throw GpuSlotsExhausted();
}

VulkanImage prepare_output(
    GpuSlot& slot,
    ID3D12Device* device,
    VulkanContext& context,
    std::uint32_t width,
    std::uint32_t height) {
    if (slot.shared.resource == nullptr || slot.width != width ||
        slot.height != height) {
        slot.shared = SharedOutput{};
        slot.width = slot.height = 0u;
        slot.fence_value = 0u;
        slot.shared = create_shared_output(device, width, height);
        slot.width = width;
        slot.height = height;
    }
    return context.import_d3d12_image(
        slot.shared.resource_handle, width, height,
        VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}
#endif

class ExternalGpuImpl;

#if defined(_WIN32)
class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(
        std::shared_ptr<ExternalGpu> owner,
        std::shared_ptr<GpuSlot> slot,
        VulkanImage input,
        VulkanImage output,
        VulkanSubmission submission,
        const ExternalTextureRequest& request,
        std::uint64_t fence_value)
        : owner_(std::move(owner)), slot_(std::move(slot)),
          input_(std::move(input)), output_(std::move(output)),
          submission_(std::move(submission)), request_(request),
          fence_value_(fence_value) {}
    ~ExternalJobImpl() override {
        try { submission_.wait(); } catch (...) {}
        submission_ = VulkanSubmission{};
        output_ = VulkanImage{};
        input_ = VulkanImage{};
        slot_->occupied.store(false);
    }
    ExternalJobState state() const override {
        if (cancelled_.load()) return ExternalJobState::cancelled;
        return submission_.ready()
            ? ExternalJobState::complete
            : ExternalJobState::running;
    }
    void cancel() override { cancelled_.store(true); }
    ExternalTextureOutput output() const override {
        if (cancelled_.load()) {
            throw std::runtime_error("ZoeDepth GPU job was cancelled");
        }
        return {
            reinterpret_cast<std::uintptr_t>(slot_->shared.resource_handle),
            request_.width, request_.height,
            reinterpret_cast<std::uintptr_t>(slot_->shared.fence_handle),
            fence_value_, request_.source_frame_id, request_.timestamp_ns};
    }
private:
    std::shared_ptr<ExternalGpu> owner_;
    std::shared_ptr<GpuSlot> slot_;
    VulkanImage input_;
    VulkanImage output_;
    VulkanSubmission submission_;
    ExternalTextureRequest request_{};
    std::uint64_t fence_value_ = 0;
    std::atomic<bool> cancelled_{false};
};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(
        const std::string& path,
        zoedepth_variant variant,
        std::uint32_t index)
        : model_(path, variant == ZOEDEPTH_VARIANT_K ? Variant::k :
              variant == ZOEDEPTH_VARIANT_NK ? Variant::nk : Variant::n),
          context_(index), gpu_model_(model_, context_),
          operators_(context_), io_(context_),
          encoder_zero_(context_.create_device_buffer(1024u * 3u * sizeof(float))),
          graph_zero_(context_.create_device_buffer(1024u * sizeof(float)))
#if defined(_WIN32)
          , d3d12_(matching_d3d12_device(context_.adapter_luid())),
          slots_{std::make_shared<GpuSlot>(), std::make_shared<GpuSlot>(),
                 std::make_shared<GpuSlot>()}
#endif
          {
        const std::vector<float> encoder_zeros(1024u * 3u, 0.0f);
        const std::vector<float> graph_zeros(1024u, 0.0f);
        context_.upload(
            encoder_zero_, encoder_zeros.data(),
            encoder_zeros.size() * sizeof(float));
        context_.upload(
            graph_zero_, graph_zeros.data(),
            graph_zeros.size() * sizeof(float));
    }

    ExternalGpuCapabilities capabilities() const override {
#if defined(_WIN32)
        const auto& capabilities = context_.external_capabilities();
        const bool available = d3d12_ != nullptr &&
            capabilities.d3d12_resource_import &&
            capabilities.d3d12_fence_import &&
            capabilities.d3d12_bgra8_sampled_image_import &&
            capabilities.d3d12_r32_storage_image_import;
        return {available, available ? context_.adapter_luid() : 0u,
                available ? kGpuSlotCount : 0u};
#else
        return {};
#endif
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
#if !defined(_WIN32)
        (void)request;
        throw std::runtime_error("ZoeDepth D3D12 interop is unavailable");
#else
        if (!capabilities().available) throw std::runtime_error(
            "complete ZoeDepth D3D12/Vulkan interop is unavailable");
        if (request.shared_texture_handle == 0u ||
            request.wait_fence_handle == 0u || request.width < 2u ||
            request.height < 2u || request.input_size == 0u ||
            request.input_size % 32u != 0u)
            throw std::invalid_argument("invalid ZoeDepth GPU texture request");
        validate_input(d3d12_.Get(), request.shared_texture_handle,
                       request.width, request.height);
        const std::uint32_t pad_height = static_cast<std::uint32_t>(
            std::sqrt(static_cast<double>(request.height) / 2.0) * 3.0);
        const std::uint32_t pad_width = static_cast<std::uint32_t>(
            std::sqrt(static_cast<double>(request.width) / 2.0) * 3.0);
        if (pad_height >= request.height || pad_width >= request.width)
            throw std::invalid_argument(
                "image is too small for ZoeDepth reflection padding");
        const std::uint32_t padded_width = request.width + 2u * pad_width;
        const std::uint32_t padded_height = request.height + 2u * pad_height;
        double scale_height = static_cast<double>(request.input_size) /
            padded_height;
        double scale_width = static_cast<double>(request.input_size) /
            padded_width;
        if (std::abs(1.0 - scale_width) < std::abs(1.0 - scale_height))
            scale_height = scale_width;
        else
            scale_width = scale_height;
        const auto multiple32 = [](double value) {
            return static_cast<std::uint32_t>(
                std::max(32.0, std::nearbyint(value / 32.0) * 32.0));
        };
        const std::uint32_t network_width =
            multiple32(scale_width * padded_width);
        const std::uint32_t network_height =
            multiple32(scale_height * padded_height);
        auto slot = acquire_slot(slots_, next_slot_);
        try {
            std::lock_guard<std::mutex> lock(record_mutex_);
            VulkanImage output = prepare_output(
                *slot, d3d12_.Get(), context_, request.width, request.height);
            const std::uint64_t signal_value = ++slot->fence_value;
            VulkanImage input = context_.import_d3d12_image(
                reinterpret_cast<void*>(request.shared_texture_handle),
                request.width, request.height, VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                slot->shared.fence_handle, signal_value);
            VulkanSubmission submission = context_.batch_async(
                std::move(wait), std::move(signal), [&] {
                    context_.acquire_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.acquire_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                    const auto run = [&](bool flip) {
                        VulkanBuffer prepared = context_.create_device_buffer(
                            static_cast<std::uint64_t>(network_width) *
                            network_height * 3u * sizeof(float));
                        io_.preprocess(
                            prepared, input, network_width, network_height,
                            padded_width, padded_height,
                            pad_width, pad_height, flip);
                        return full_graph_gpu(
                            context_, gpu_model_, operators_,
                            encoder_gpu(context_, gpu_model_, operators_,
                                        prepared, network_width, network_height,
                                        &encoder_zero_),
                            &graph_zero_);
                    };
                    GpuFeature direct = run(false);
                    GpuFeature flipped = run(true);
                    io_.combine_depth(
                        output, direct.buffer, flipped.buffer,
                        network_width, network_height,
                        padded_width, padded_height,
                        pad_width, pad_height);
                    context_.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_shared<ExternalJobImpl>(
                shared_from_this(), slot, std::move(input),
                std::move(output), std::move(submission), request, signal_value);
        } catch (...) {
            slot->occupied.store(false);
            throw;
        }
#endif
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        context_.transfer_counters(upload_bytes, download_bytes);
    }

private:
    ModelFile model_;
    VulkanContext context_;
    GpuModel gpu_model_;
    VulkanOperators operators_;
    GpuIo io_;
    VulkanBuffer encoder_zero_;
    VulkanBuffer graph_zero_;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_;
    std::array<std::shared_ptr<GpuSlot>, kGpuSlotCount> slots_;
    std::atomic<std::uint32_t> next_slot_{0};
    std::mutex record_mutex_;
#endif
};

}  // namespace

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& path,
    zoedepth_variant variant,
    std::uint32_t index) {
    return std::make_shared<ExternalGpuImpl>(path, variant, index);
}

ExternalGpuCapabilities probe_external_gpu(std::uint32_t index) {
#if defined(_WIN32)
    VulkanContext context(index);
    const auto device = matching_d3d12_device(context.adapter_luid());
    const auto& capabilities = context.external_capabilities();
    const bool available = device != nullptr &&
        capabilities.d3d12_resource_import &&
        capabilities.d3d12_fence_import &&
        capabilities.d3d12_bgra8_sampled_image_import &&
        capabilities.d3d12_r32_storage_image_import;
    return {available, available ? context.adapter_luid() : 0u,
            available ? kGpuSlotCount : 0u};
#else
    (void)index;
    return {};
#endif
}

}  // namespace zoe_native
