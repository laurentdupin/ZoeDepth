#include "gpu_io.h"

#include "preprocess_texture_spv.h"
#include "combine_depth_image_spv.h"

#include <stdexcept>

namespace zoe_native {

GpuIo::GpuIo(VulkanContext& context)
    : context_(context),
      preprocess_(context.create_pipeline(
          zoe_preprocess_texture_spv, zoe_preprocess_texture_spv_size,
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT},
          8u * sizeof(std::uint32_t))),
      combine_depth_(context.create_pipeline(
          zoe_combine_depth_image_spv, zoe_combine_depth_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
           VK_ACCESS_SHADER_READ_BIT},
          8u * sizeof(std::uint32_t))) {
    preprocess_.set_debug_name("zoedepth_preprocess_texture");
    combine_depth_.set_debug_name("zoedepth_combine_depth_image");
}

void GpuIo::preprocess(
    VulkanBuffer& destination, const VulkanImage& source,
    std::uint32_t destination_width, std::uint32_t destination_height,
    std::uint32_t padded_width, std::uint32_t padded_height,
    std::uint32_t pad_width, std::uint32_t pad_height, bool flip) {
    const std::uint64_t required =
        static_cast<std::uint64_t>(destination_width) *
        destination_height * 3u * sizeof(float);
    if (source.width() < 2u || source.height() < 2u ||
        destination_width == 0u || destination_height == 0u ||
        padded_width == 0u || padded_height == 0u ||
        destination.size() < required) {
        throw std::invalid_argument("invalid ZoeDepth GPU preprocess shape");
    }
    struct Parameters {
        std::uint32_t source_width, source_height;
        std::uint32_t destination_width, destination_height;
        std::uint32_t padded_width, padded_height;
        std::uint32_t pad_width, pad_height;
    } parameters{
        source.width(), source.height(),
        destination_width, destination_height,
        padded_width, padded_height,
        pad_width, pad_height};
    if (flip) parameters.pad_height |= 0x80000000u;
    context_.dispatch_image_to_buffer(
        preprocess_, source, destination, &parameters, sizeof(parameters),
        (destination_width + 7u) / 8u,
        (destination_height + 7u) / 8u);
}

void GpuIo::combine_depth(
    VulkanImage& destination,
    const VulkanBuffer& direct,
    const VulkanBuffer& flipped,
    std::uint32_t network_width,
    std::uint32_t network_height,
    std::uint32_t padded_width,
    std::uint32_t padded_height,
    std::uint32_t pad_width,
    std::uint32_t pad_height) {
    if (destination.format() != VK_FORMAT_R32_SFLOAT ||
        destination.width() == 0u || destination.height() == 0u ||
        direct.size() < static_cast<std::uint64_t>(network_width) *
            network_height * sizeof(float) ||
        flipped.size() < static_cast<std::uint64_t>(network_width) *
            network_height * sizeof(float)) {
        throw std::invalid_argument("invalid ZoeDepth GPU output shape");
    }
    struct Parameters {
        std::uint32_t network_width, network_height;
        std::uint32_t padded_width, padded_height;
        std::uint32_t output_width, output_height;
        std::uint32_t pad_width, pad_height;
    } parameters{
        network_width, network_height, padded_width, padded_height,
        destination.width(), destination.height(), pad_width, pad_height};
    context_.dispatch_buffers_to_image(
        combine_depth_, {&direct, &flipped}, destination,
        &parameters, sizeof(parameters),
        (destination.width() + 7u) / 8u,
        (destination.height() + 7u) / 8u);
}

}  // namespace zoe_native
