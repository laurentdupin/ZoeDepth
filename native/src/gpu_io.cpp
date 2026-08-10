#include "gpu_io.h"

#include "preprocess_texture_spv.h"
#include "combine_depth_buffer_spv.h"
#include "reduce_minmax_spv.h"
#include "normalize_inverse_spv.h"
#include "depth_buffer_to_image_spv.h"

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
          zoe_combine_depth_buffer_spv, zoe_combine_depth_buffer_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
           VK_ACCESS_SHADER_READ_BIT},
          8u * sizeof(std::uint32_t))),
      reduce_minmax_(context.create_pipeline(
          zoe_reduce_minmax_spv, zoe_reduce_minmax_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}, 4)),
      normalize_inverse_(context.create_pipeline(
          zoe_normalize_inverse_spv, zoe_normalize_inverse_spv_size, 2, 4)),
      depth_buffer_to_image_(context.create_pipeline(
          zoe_depth_buffer_to_image_spv, zoe_depth_buffer_to_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT}, 8)) {
    preprocess_.set_debug_name("zoedepth_preprocess_texture");
    combine_depth_.set_debug_name("zoedepth_combine_depth_buffer");
    reduce_minmax_.set_debug_name("zoedepth_reduce_minmax");
    normalize_inverse_.set_debug_name("zoedepth_normalize_inverse");
    depth_buffer_to_image_.set_debug_name("zoedepth_depth_buffer_to_image");
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
    VulkanBuffer& destination,
    const VulkanBuffer& direct,
    const VulkanBuffer& flipped,
    std::uint32_t network_width,
    std::uint32_t network_height,
    std::uint32_t padded_width,
    std::uint32_t padded_height,
    std::uint32_t pad_width,
    std::uint32_t pad_height,
    std::uint32_t output_width,
    std::uint32_t output_height) {
    if (output_width == 0u || output_height == 0u ||
        destination.size() < static_cast<std::uint64_t>(output_width) *
            output_height * sizeof(float) ||
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
        output_width, output_height, pad_width, pad_height};
    context_.dispatch(
        combine_depth_, {&destination, &direct, &flipped},
        &parameters, sizeof(parameters),
        (output_width + 7u) / 8u, (output_height + 7u) / 8u);
}

void GpuIo::normalize_inverse(VulkanBuffer& depth, std::uint32_t count) {
    if (count == 0u || depth.size() < static_cast<std::uint64_t>(count) * sizeof(float))
        throw std::invalid_argument("invalid ZoeDepth normalization shape");
    VulkanBuffer range = context_.create_device_buffer(2u * sizeof(float));
    context_.dispatch(reduce_minmax_, {&depth, &range}, &count, sizeof(count), 1);
    context_.dispatch(normalize_inverse_, {&depth, &range}, &count, sizeof(count),
                      (count + 255u) / 256u);
}

void GpuIo::write_depth(
    VulkanImage& destination, const VulkanBuffer& depth,
    std::uint32_t width, std::uint32_t height) {
    if (destination.format() != VK_FORMAT_R32_SFLOAT ||
        destination.width() != width || destination.height() != height)
        throw std::invalid_argument("invalid ZoeDepth output image");
    const std::uint32_t parameters[2] = {width, height};
    context_.dispatch_buffer_to_image(
        depth_buffer_to_image_, depth, destination,
        parameters, sizeof(parameters), (width + 7u) / 8u, (height + 7u) / 8u);
}

}  // namespace zoe_native
