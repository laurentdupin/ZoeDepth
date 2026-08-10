
#include "external_gpu.h"

#include "gpu_io.h"
#include "encoder_gpu.h"
#include "gpu_model.h"
#include "graph_gpu.h"
#include "model.h"
#include "operators.h"
#include "inferbridge/native_harness_resource_lifetime.h"

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
constexpr std::uint32_t kMaxInFlightJobs = 3u;

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

void validate_texture(ID3D12Device* device, std::uintptr_t handle,
    std::uint32_t width, std::uint32_t height, DXGI_FORMAT format,
    const char* operation) {
    ComPtr<ID3D12Resource> resource;
    check_hresult(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(handle), IID_PPV_ARGS(&resource)), operation);
    const auto d = resource->GetDesc();
    if (d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        d.Width != width || d.Height != height || d.DepthOrArraySize != 1u ||
        d.MipLevels != 1u || d.SampleDesc.Count != 1u || d.Format != format)
        throw std::invalid_argument("ZoeDepth shared texture descriptor mismatch");
}
class ExternalJobImpl final : public ExternalJob {
public:
    ExternalJobImpl(std::shared_ptr<ExternalGpu> owner, VulkanImage input,
        VulkanImage output, VulkanSubmission submission,
        inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime)
        : owner_(std::move(owner)), input_(std::move(input)),
          output_(std::move(output)), submission_(std::move(submission)),
          lifetime_(std::move(lifetime)) {}
    ~ExternalJobImpl() override {
        inferbridge::native_harness::wait_then_retire(
            lifetime_, submission_, [this] { output_ = {}; input_ = {}; });
    }
    ExternalJobState state() const override {
        if (cancelled_.load()) return ExternalJobState::cancelled;
        return submission_.ready() ? ExternalJobState::complete :
                                     ExternalJobState::running;
    }
    void cancel() override { cancelled_.store(true); }
private:
    std::shared_ptr<ExternalGpu> owner_;
    VulkanImage input_, output_;
    VulkanSubmission submission_;
    inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime_;
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
          , d3d12_(matching_d3d12_device(context_.adapter_luid()))
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
                available ? kMaxInFlightJobs : 0u};
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
        if (!request.shared_texture_handle || !request.wait_fence_handle ||
            !request.output_texture_handle || !request.signal_fence_handle ||
            request.width < 2u || request.height < 2u ||
            request.output_width != request.width ||
            request.output_height != request.height ||
            request.input_size == 0u || request.input_size % 32u != 0u)
            throw std::invalid_argument("invalid ZoeDepth GPU texture request");
        validate_texture(d3d12_.Get(), request.shared_texture_handle,
            request.width, request.height,
            request.rgba ? DXGI_FORMAT_R8G8B8A8_UNORM :
                           DXGI_FORMAT_B8G8R8A8_UNORM,
            "OpenSharedHandle(ZoeDepth input)");
        validate_texture(d3d12_.Get(), request.output_texture_handle,
            request.output_width, request.output_height,
            DXGI_FORMAT_R32_FLOAT, "OpenSharedHandle(ZoeDepth output)");
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
        try {
            auto lifetime_guard = lifetime_->acquire();
            VulkanImage output = context_.import_d3d12_image(
                reinterpret_cast<void*>(request.output_texture_handle),
                request.output_width, request.output_height,
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            VulkanImage input = context_.import_d3d12_image(
                reinterpret_cast<void*>(request.shared_texture_handle),
                request.width, request.height,
                request.rgba ? VK_FORMAT_R8G8B8A8_UNORM :
                               VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.signal_fence_handle),
                request.signal_fence_value);
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
                    const std::uint32_t output_count =
                        request.width * request.height;
                    VulkanBuffer combined = context_.create_device_buffer(
                        static_cast<std::uint64_t>(output_count) * sizeof(float));
                    io_.combine_depth(
                        combined, direct.buffer, flipped.buffer,
                        network_width, network_height,
                        padded_width, padded_height,
                        pad_width, pad_height,
                        request.width, request.height);
                    io_.normalize_inverse(combined, output_count);
                    io_.write_depth(
                        output, combined, request.width, request.height);
                    context_.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
            return std::make_shared<ExternalJobImpl>(
                shared_from_this(), std::move(input), std::move(output),
                std::move(submission), lifetime_);
        } catch (...) { throw; }
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
    inferbridge::native_harness::ResourceLifetimeDomainPtr lifetime_ =
        inferbridge::native_harness::make_resource_lifetime_domain();
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
            available ? kMaxInFlightJobs : 0u};
#else
    (void)index;
    return {};
#endif
}

}  // namespace zoe_native
