#include "operators.h"

#include "inferbridge/native_harness_environment.h"

#include "add_scaled_spv.h"
#include "add_spv.h"
#include "add_position_spv.h"
#include "bilinear_align_true_spv.h"
#include "bilinear_align_true_image_spv.h"
#include "bmm_spv.h"
#include "bmm_score_half_spv.h"
#include "bmm_value_half_spv.h"
#include "conv2d_spv.h"
#include "conv2d8_spv.h"
#include "conv2d_half_spv.h"
#include "conv2d8_half_spv.h"
#include "conv2d_tiled_spv.h"
#include "conv2d_tiled4_spv.h"
#include "conv_transpose_nonoverlap_spv.h"
#include "conv_transpose_nonoverlap4_spv.h"
#include "conv_transpose_nonoverlap_half_spv.h"
#include "conv_transpose_nonoverlap4_half_spv.h"
#include "gelu_spv.h"
#include "layer_norm_spv.h"
#include "layer_norm_parallel_spv.h"
#include "linear_spv.h"
#include "linear16_spv.h"
#include "linear_half_spv.h"
#include "linear16_half_spv.h"
#include "linear_vec8_spv.h"
#include "linear_vec8_rows24_spv.h"
#include "linear_vec8_rows24_half_spv.h"
#include "linear_int8_tiled_spv.h"
#include "linear_int8_tiled16_spv.h"
#include "quantize_rows_int8_spv.h"
#include "inferbridge/native_harness_precision.h"
#include "prepare_tokens_spv.h"
#include "position_bicubic_spv.h"
#include "project_tokens_spv.h"
#include "project_tokens_half_spv.h"
#include "relu_spv.h"
#include "softmax_lastdim_spv.h"
#include "softmax_lastdim_half_spv.h"
#include "group_norm_spv.h"
#include "silu_spv.h"
#include "nearest_spv.h"
#include "concatenate_spv.h"
#include "add_channel_spv.h"
#include "nchw_tokens_spv.h"
#include "geglu_spv.h"
#include "attention_scores_spv.h"
#include "attention_values_spv.h"
#include "preprocess_rgb_spv.h"
#include "posterior_sample_spv.h"
#include "scale_values_spv.h"
#include "depth_output_spv.h"
#include "prepare_beit_spv.h"
#include "qv_bias_spv.h"
#include "relative_bias_spv.h"
#include "readout_concat_spv.h"
#include "tokens_to_nchw_plain_spv.h"
#include "softplus_spv.h"
#include "router_tokens_spv.h"
#include "qkv_split_spv.h"
#include "seed_centers_spv.h"
#include "attractor_activate_spv.h"
#include "attractor_update_spv.h"
#include "distribution_depth_spv.h"
#include "select_depth_spv.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace zoe_native {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

void require_bytes(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(float) ||
        buffer.size() < elements * sizeof(float)) {
        throw std::invalid_argument(
            std::string(name) + " Vulkan buffer is too small");
    }
}

void require_half_elements(
    const VulkanBuffer& buffer,
    std::uint64_t elements,
    const char* name) {
    const std::uint64_t words = (elements + 1) / 2;
    if (words > std::numeric_limits<std::uint64_t>::max() /
            sizeof(std::uint32_t) ||
        buffer.size() < words * sizeof(std::uint32_t)) {
        throw std::invalid_argument(
            std::string(name) + " packed-half Vulkan buffer is too small");
    }
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      preserve_scalar_layer_norm_order_(
          inferbridge::native::requested_precision() ==
          inferbridge::native::Precision::int8),
      linear_(context.create_pipeline(
          zoe_linear_spv, zoe_linear_spv_size, 4, 16)),
      linear16_(context.create_pipeline(
          zoe_linear16_spv, zoe_linear16_spv_size, 4, 16)),
      linear_half_(context.create_pipeline(
          zoe_linear_half_spv, zoe_linear_half_spv_size, 4, 16)),
      linear16_half_(context.create_pipeline(
          zoe_linear16_half_spv,
          zoe_linear16_half_spv_size,
          4,
          16)),
      linear_vec8_(context.create_pipeline(
          zoe_linear_vec8_spv,
          zoe_linear_vec8_spv_size,
          4,
          16)),
      linear_vec8_rows24_(context.create_pipeline(
          zoe_linear_vec8_rows24_spv,
          zoe_linear_vec8_rows24_spv_size,
          4,
          16)),
      linear_vec8_rows24_half_(
          context.float16_storage()
              ? context.create_pipeline(
                    zoe_linear_vec8_rows24_half_spv,
                    zoe_linear_vec8_rows24_half_spv_size,
                    4,
                    16)
              : VulkanPipeline{}),
      quantize_rows_int8_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(zoe_quantize_rows_int8_spv,
                zoe_quantize_rows_int8_spv_size,
              {
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            },
                            {
                                VK_ACCESS_SHADER_READ_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT,
                            },
                            4)
          : VulkanPipeline{}),
      linear_int8_tiled_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(zoe_linear_int8_tiled_spv,
                zoe_linear_int8_tiled_spv_size, 6, 28)
          : VulkanPipeline{}),
      linear_int8_tiled16_(
          context.supports_packed_int8_dot() &&
              inferbridge::native::requested_precision() ==
                  inferbridge::native::Precision::int8
          ? context.create_pipeline(zoe_linear_int8_tiled16_spv,
                zoe_linear_int8_tiled16_spv_size, 6, 28)
          : VulkanPipeline{}),
      gelu_(context.create_pipeline(
          zoe_gelu_spv, zoe_gelu_spv_size, 2, 4)),
      layer_norm_(context.create_pipeline(
          zoe_layer_norm_spv, zoe_layer_norm_spv_size, 4, 12)),
      layer_norm_parallel_(context.create_pipeline(
          zoe_layer_norm_parallel_spv,
          zoe_layer_norm_parallel_spv_size, 4, 12)),
      add_scaled_(context.create_pipeline(
          zoe_add_scaled_spv, zoe_add_scaled_spv_size, 4, 8)),
      bmm_(context.create_pipeline(
          zoe_bmm_spv, zoe_bmm_spv_size, 3, 44)),
      bmm_score_half_(context.create_pipeline(
          zoe_bmm_score_half_spv,
          zoe_bmm_score_half_spv_size,
          2,
          8)),
      bmm_value_half_(context.create_pipeline(
          zoe_bmm_value_half_spv,
          zoe_bmm_value_half_spv_size,
          3,
          8)),
      softmax_lastdim_(context.create_pipeline(
          zoe_softmax_lastdim_spv,
          zoe_softmax_lastdim_spv_size,
          2,
          8)),
      softmax_lastdim_half_(context.create_pipeline(
          zoe_softmax_lastdim_half_spv,
          zoe_softmax_lastdim_half_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {
              VK_ACCESS_SHADER_READ_BIT |
              VK_ACCESS_SHADER_WRITE_BIT,
          },
          8)),
      prepare_tokens_(context.create_pipeline(
          zoe_prepare_tokens_spv,
          zoe_prepare_tokens_spv_size,
          5,
          24)),
      position_bicubic_(context.create_pipeline(
          zoe_position_bicubic_spv,
          zoe_position_bicubic_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          },
          0)),
      add_position_(context.create_pipeline(
          zoe_add_position_spv,
          zoe_add_position_spv_size,
          4,
          20)),
      add_(context.create_pipeline(
          zoe_add_spv, zoe_add_spv_size, 3, 4)),
      project_tokens_(context.create_pipeline(
          zoe_project_tokens_spv,
          zoe_project_tokens_spv_size,
          4,
          20)),
      project_tokens_half_(context.create_pipeline(
          zoe_project_tokens_half_spv,
          zoe_project_tokens_half_spv_size,
          4,
          20)),
      conv2d_(context.create_pipeline(
          zoe_conv2d_spv, zoe_conv2d_spv_size, 4, 48)),
      conv2d8_(context.create_pipeline(
          zoe_conv2d8_spv, zoe_conv2d8_spv_size, 4, 48)),
      conv2d_half_(context.create_pipeline(
          zoe_conv2d_half_spv, zoe_conv2d_half_spv_size, 4, 48)),
      conv2d8_half_(context.create_pipeline(
          zoe_conv2d8_half_spv,
          zoe_conv2d8_half_spv_size,
          4,
          48)),
      conv2d_tiled_(context.create_pipeline(
          zoe_conv2d_tiled_spv,
          zoe_conv2d_tiled_spv_size,
          4,
          48)),
      conv2d_tiled4_(context.create_pipeline(
          zoe_conv2d_tiled4_spv,
          zoe_conv2d_tiled4_spv_size,
          4,
          48)),
      conv_transpose_nonoverlap_(context.create_pipeline(
          zoe_conv_transpose_nonoverlap_spv,
          zoe_conv_transpose_nonoverlap_spv_size,
          4,
          24)),
      conv_transpose_nonoverlap4_(context.create_pipeline(
          zoe_conv_transpose_nonoverlap4_spv,
          zoe_conv_transpose_nonoverlap4_spv_size,
          4,
          24)),
      conv_transpose_nonoverlap_half_(context.create_pipeline(
          zoe_conv_transpose_nonoverlap_half_spv,
          zoe_conv_transpose_nonoverlap_half_spv_size,
          4,
          24)),
      conv_transpose_nonoverlap4_half_(context.create_pipeline(
          zoe_conv_transpose_nonoverlap4_half_spv,
          zoe_conv_transpose_nonoverlap4_half_spv_size,
          4,
          24)),
      bilinear_align_true_(context.create_pipeline(
          zoe_bilinear_align_true_spv,
          zoe_bilinear_align_true_spv_size,
          2,
          24)),
      bilinear_align_true_image_(context.create_pipeline(
          zoe_bilinear_align_true_image_spv,
          zoe_bilinear_align_true_image_spv_size,
          {
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          },
          16)),
      relu_(context.create_pipeline(
          zoe_relu_spv, zoe_relu_spv_size, 2, 4)),
      group_norm_(context.create_pipeline(
          zoe_group_norm_spv, zoe_group_norm_spv_size, 3, 16)),
      silu_(context.create_pipeline(
          zoe_silu_spv, zoe_silu_spv_size, 1, 4)),
      nearest_(context.create_pipeline(
          zoe_nearest_spv, zoe_nearest_spv_size, 2, 20)),
      concatenate_(context.create_pipeline(
          zoe_concatenate_spv, zoe_concatenate_spv_size, 3, 8)),
      add_channel_(context.create_pipeline(
          zoe_add_channel_spv, zoe_add_channel_spv_size, 2, 8)),
      nchw_tokens_(context.create_pipeline(
          zoe_nchw_tokens_spv, zoe_nchw_tokens_spv_size, 2, 12)),
      geglu_(context.create_pipeline(
          zoe_geglu_spv, zoe_geglu_spv_size, 2, 8)),
      attention_scores_(context.create_pipeline(
          zoe_attention_scores_spv,
          zoe_attention_scores_spv_size, 3, 16)),
      attention_values_(context.create_pipeline(
          zoe_attention_values_spv,
          zoe_attention_values_spv_size, 3, 16)),
      preprocess_rgb_(context.create_pipeline(
          zoe_preprocess_rgb_spv, zoe_preprocess_rgb_spv_size, 2, 8)),
      posterior_sample_(context.create_pipeline(
          zoe_posterior_sample_spv,
          zoe_posterior_sample_spv_size, 3, 4)),
      scale_values_(context.create_pipeline(
          zoe_scale_values_spv, zoe_scale_values_spv_size, 1, 8)),
      depth_output_(context.create_pipeline(
          zoe_depth_output_spv, zoe_depth_output_spv_size, 2, 16)),
      prepare_beit_(context.create_pipeline(
          zoe_prepare_beit_spv, zoe_prepare_beit_spv_size, 5, 16)),
      qv_bias_(context.create_pipeline(
          zoe_qv_bias_spv, zoe_qv_bias_spv_size, 3, 8)),
      relative_bias_(context.create_pipeline(
          zoe_relative_bias_spv, zoe_relative_bias_spv_size, 2, 16)),
      readout_concat_(context.create_pipeline(
          zoe_readout_concat_spv, zoe_readout_concat_spv_size, 2, 8)),
      tokens_to_nchw_plain_(context.create_pipeline(
          zoe_tokens_to_nchw_plain_spv,
          zoe_tokens_to_nchw_plain_spv_size, 2, 8)),
      softplus_(context.create_pipeline(
          zoe_softplus_spv, zoe_softplus_spv_size, 1, 4)),
      router_tokens_(context.create_pipeline(
          zoe_router_tokens_spv, zoe_router_tokens_spv_size, 2, 8)),
      qkv_split_(context.create_pipeline(
          zoe_qkv_split_spv, zoe_qkv_split_spv_size, 4, 8)),
      seed_centers_(context.create_pipeline(
          zoe_seed_centers_spv, zoe_seed_centers_spv_size, 2, 8)),
      attractor_activate_(context.create_pipeline(
          zoe_attractor_activate_spv,
          zoe_attractor_activate_spv_size, 1, 4)),
      attractor_update_(context.create_pipeline(
          zoe_attractor_update_spv,
          zoe_attractor_update_spv_size, 4, 16)),
      distribution_depth_(context.create_pipeline(
          zoe_distribution_depth_spv,
          zoe_distribution_depth_spv_size, 3, 4)),
      select_depth_(context.create_pipeline(
          zoe_select_depth_spv,
          zoe_select_depth_spv_size, 4, 4)) {
    linear_.set_debug_name("linear");
    linear16_.set_debug_name("linear16");
    linear_half_.set_debug_name("linear_half");
    linear16_half_.set_debug_name("linear16_half");
    linear_vec8_.set_debug_name("linear_vec8");
    linear_vec8_rows24_.set_debug_name("linear_vec8_rows24");
    linear_vec8_rows24_half_.set_debug_name(
        "linear_vec8_rows24_half");
    if (context.supports_packed_int8_dot() &&
        inferbridge::native::requested_precision() ==
            inferbridge::native::Precision::int8) {
        quantize_rows_int8_.set_debug_name("quantize_rows_int8");
        linear_int8_tiled_.set_debug_name("linear_int8_tiled");
    }
    gelu_.set_debug_name("gelu");
    layer_norm_.set_debug_name("layer_norm");
    layer_norm_parallel_.set_debug_name("layer_norm_parallel");
    add_scaled_.set_debug_name("add_scaled");
    bmm_.set_debug_name("bmm");
    bmm_score_half_.set_debug_name("bmm_score_half");
    bmm_value_half_.set_debug_name("bmm_value_half");
    softmax_lastdim_.set_debug_name("softmax_lastdim");
    softmax_lastdim_half_.set_debug_name(
        "softmax_lastdim_half");
    prepare_tokens_.set_debug_name("prepare_tokens");
    position_bicubic_.set_debug_name("position_bicubic");
    add_position_.set_debug_name("add_position");
    add_.set_debug_name("add");
    project_tokens_.set_debug_name("project_tokens");
    project_tokens_half_.set_debug_name(
        "project_tokens_half");
    conv2d_.set_debug_name("conv2d");
    conv2d8_.set_debug_name("conv2d8");
    conv2d_half_.set_debug_name("conv2d_half");
    conv2d8_half_.set_debug_name("conv2d8_half");
    conv2d_tiled_.set_debug_name("conv2d_tiled");
    conv2d_tiled4_.set_debug_name("conv2d_tiled4");
    conv_transpose_nonoverlap_.set_debug_name(
        "conv_transpose_nonoverlap");
    conv_transpose_nonoverlap4_.set_debug_name(
        "conv_transpose_nonoverlap4");
    conv_transpose_nonoverlap_half_.set_debug_name(
        "conv_transpose_nonoverlap_half");
    conv_transpose_nonoverlap4_half_.set_debug_name(
        "conv_transpose_nonoverlap4_half");
    bilinear_align_true_.set_debug_name(
        "bilinear_align_true");
    bilinear_align_true_image_.set_debug_name(
        "bilinear_align_true_image");
    relu_.set_debug_name("relu");
    group_norm_.set_debug_name("group_norm");
    silu_.set_debug_name("silu");
    nearest_.set_debug_name("nearest");
    concatenate_.set_debug_name("concatenate");
    add_channel_.set_debug_name("add_channel");
    nchw_tokens_.set_debug_name("nchw_tokens");
    geglu_.set_debug_name("geglu");
    attention_scores_.set_debug_name("attention_scores");
    attention_values_.set_debug_name("attention_values");
    preprocess_rgb_.set_debug_name("preprocess_rgb");
    posterior_sample_.set_debug_name("posterior_sample");
    scale_values_.set_debug_name("scale_values");
    depth_output_.set_debug_name("depth_output");
}

void VulkanOperators::linear_int8(
    VulkanBuffer& output, const VulkanBuffer& input,
    const VulkanBuffer& packed_weight, const VulkanBuffer& weight_scales,
    const VulkanBuffer& bias, std::uint32_t rows,
    std::uint32_t input_columns, std::uint32_t output_columns, bool gelu) {
    if (!context_.supports_packed_int8_dot() || input_columns % 4u != 0u)
        throw std::runtime_error("accelerated packed INT8 linear is unavailable");
    VulkanBuffer& packed_input = int8_workspace_.packed(
        std::uint64_t(rows) * (input_columns / 4u) * sizeof(std::uint32_t),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    VulkanBuffer& input_scales = int8_workspace_.scales(
        std::uint64_t(rows) * sizeof(float),
        [this](std::uint64_t bytes) {
            return context_.create_device_buffer(bytes);
        });
    context_.dispatch(quantize_rows_int8_,
        {&input, &packed_input, &input_scales},
        &input_columns, sizeof(input_columns), rows);
    const std::uint32_t parameters[7] = {
        rows, input_columns, output_columns, 0u, output_columns, 0u, 1u};
    context_.dispatch(rows >= 32u ? linear_int8_tiled16_ : linear_int8_tiled_,
        {&output, &packed_input, &packed_weight, &input_scales,
         &weight_scales, &bias}, parameters, sizeof(parameters),
        divide_up(output_columns, 64u), divide_up(rows, 56u));
    if (gelu) {
        const std::uint32_t count = rows * output_columns;
        context_.dispatch(gelu_, {&output, &output}, &count, sizeof(count),
            divide_up(count, 256u));
    }
}

void VulkanOperators::linear(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t rows,
    std::uint32_t input_columns,
    std::uint32_t output_columns,
    bool gelu,
    bool block16,
    bool half_weight) {
    if (rows == 0 || input_columns == 0 || output_columns == 0) {
        throw std::invalid_argument("linear dimensions cannot be zero");
    }
    require_bytes(input, std::uint64_t(rows) * input_columns, "input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_columns) * input_columns;
    if (half_weight) {
        require_half_elements(weight, weight_elements, "weight");
    } else {
        require_bytes(weight, weight_elements, "weight");
    }
    require_bytes(bias, output_columns, "bias");
    require_bytes(
        output, std::uint64_t(rows) * output_columns, "output");
    struct Parameters {
        std::uint32_t rows;
        std::uint32_t input_columns;
        std::uint32_t output_columns;
        std::uint32_t gelu;
    } parameters{rows, input_columns, output_columns, gelu ? 1u : 0u};
    context_.dispatch(
        context_.subgroup_size() == 32
            ? (half_weight
                ? linear_vec8_rows24_half_
                : rows <= 32 ? linear_vec8_rows24_ : linear_vec8_)
            : (half_weight
            ? (block16 ? linear16_half_ : linear_half_)
            : (block16 ? linear16_ : linear_)),
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        context_.subgroup_size() == 32
            ? divide_up(output_columns, 64)
            : divide_up(divide_up(output_columns, 4), 8),
        context_.subgroup_size() == 32
            ? divide_up(rows, half_weight || rows <= 32 ? 24 : 40)
            : divide_up(divide_up(rows, 4), 8));
}

#include <inferbridge/vulkan/operators/vulkan_operators_layer_norm_variant_2.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_scaled.h>

#include <inferbridge/vulkan/operators/vulkan_operators_attention_head64_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_prepare_tokens_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_project_tokens_variant_1.h>

void VulkanOperators::conv2d(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_channels,
    std::uint32_t kernel,
    std::uint32_t stride,
    std::uint32_t padding,
    bool has_bias,
    bool block8,
    bool half_weight,
    std::uint32_t batches) {
    if (input_width == 0 || input_height == 0 || input_channels == 0 ||
        output_channels == 0 || kernel == 0 || stride == 0 ||
        input_width + 2 * padding < kernel ||
        input_height + 2 * padding < kernel ||
        batches == 0) {
        throw std::invalid_argument("invalid convolution dimensions");
    }
    const std::uint32_t output_width =
        (input_width + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input_height + 2 * padding - kernel) / stride + 1;
    require_bytes(
        input,
        std::uint64_t(batches) * input_width *
            input_height * input_channels,
        "convolution input");
    const std::uint64_t weight_elements =
        std::uint64_t(output_channels) * input_channels * kernel * kernel;
    if (half_weight) {
        require_half_elements(
            weight, weight_elements, "convolution weight");
    } else {
        require_bytes(weight, weight_elements, "convolution weight");
    }
    require_bytes(bias, has_bias ? output_channels : 1, "convolution bias");
    require_bytes(
        output,
        std::uint64_t(batches) * output_width *
            output_height * output_channels,
        "convolution output");
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t output_channels;
        std::uint32_t kernel;
        std::uint32_t stride;
        std::int32_t padding;
        std::uint32_t has_bias;
        std::uint32_t batches;
        std::uint32_t output_channel_blocks;
    };
    const bool tiled =
        !half_weight && !block8 && kernel == 3 && stride == 1 &&
        padding == 1 && input_width == output_width &&
        input_height == output_height &&
        context_.subgroup_size() == 32;
    const bool tiled8 =
        tiled && context_.native_subgroup_size() <= 32;
    const std::uint32_t output_channel_blocks =
        divide_up(output_channels, (block8 || tiled8) ? 8 : 4);
    const Parameters parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel, stride, static_cast<std::int32_t>(padding),
        has_bias ? 1u : 0u,
        batches, output_channel_blocks,
    };
    context_.dispatch(
        tiled
            ? (tiled8 ? conv2d_tiled_ : conv2d_tiled4_)
            : (half_weight
            ? (block8 ? conv2d8_half_ : conv2d_half_)
            : (block8 ? conv2d8_ : conv2d_)),
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, tiled8 ? 16 : 8),
        divide_up(output_height, 8),
        output_channel_blocks * batches);
}

void VulkanOperators::conv2d_asymmetric(
    VulkanBuffer& output, const VulkanBuffer& input,
    const VulkanBuffer& weight, const VulkanBuffer& bias,
    std::uint32_t input_width, std::uint32_t input_height,
    std::uint32_t input_channels, std::uint32_t output_channels,
    std::uint32_t kernel, std::uint32_t stride,
    std::uint32_t pad_before, std::uint32_t pad_after,
    bool has_bias) {
    const std::uint32_t output_width =
        (input_width + pad_before + pad_after - kernel) / stride + 1;
    const std::uint32_t output_height =
        (input_height + pad_before + pad_after - kernel) / stride + 1;
    require_bytes(
        input, std::uint64_t(input_width) * input_height * input_channels,
        "convolution input");
    require_bytes(
        weight, std::uint64_t(output_channels) * input_channels *
            kernel * kernel, "convolution weight");
    require_bytes(bias, has_bias ? output_channels : 1, "convolution bias");
    require_bytes(
        output, std::uint64_t(output_width) * output_height *
            output_channels, "convolution output");
    struct Parameters {
        std::uint32_t input_width, input_height, input_channels;
        std::uint32_t output_width, output_height, output_channels;
        std::uint32_t kernel, stride;
        std::int32_t padding;
        std::uint32_t has_bias, batches, output_channel_blocks;
    };
    const std::uint32_t blocks = divide_up(output_channels, 4);
    const Parameters parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel, stride, static_cast<std::int32_t>(pad_before),
        has_bias ? 1u : 0u, 1u, blocks};
    context_.dispatch(
        conv2d_, {&output, &input, &weight, &bias},
        &parameters, sizeof(parameters),
        divide_up(output_width, 8), divide_up(output_height, 8), blocks);
}

#include <inferbridge/vulkan/operators/vulkan_operators_conv_transpose_nonoverlap_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_variant_1.h>

#include <inferbridge/vulkan/operators/vulkan_operators_bilinear_align_true_image.h>

#include <inferbridge/vulkan/operators/vulkan_operators_relu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add.h>

void VulkanOperators::group_norm(
    VulkanBuffer& values, const VulkanBuffer& scale,
    const VulkanBuffer& bias, std::uint32_t channels,
    std::uint32_t spatial, float epsilon) {
    if (channels == 0 || channels % 32 != 0 || spatial == 0) {
        throw std::invalid_argument("invalid group normalization dimensions");
    }
    require_bytes(values, std::uint64_t(channels) * spatial, "group values");
    require_bytes(scale, channels, "group scale");
    require_bytes(bias, channels, "group bias");
    struct Parameters {
        std::uint32_t channels, spatial, groups;
        float epsilon;
    } parameters{channels, spatial, 32, epsilon};
    context_.dispatch(
        group_norm_, {&values, &scale, &bias},
        &parameters, sizeof(parameters), 32);
}

#include <inferbridge/vulkan/operators/vulkan_operators_silu.h>

#include <inferbridge/vulkan/operators/vulkan_operators_nearest.h>

#include <inferbridge/vulkan/operators/vulkan_operators_concatenate.h>

#include <inferbridge/vulkan/operators/vulkan_operators_add_channel.h>

#include <inferbridge/vulkan/operators/vulkan_operators_nchw_tokens.h>

#include <inferbridge/vulkan/operators/vulkan_operators_geglu.h>

void VulkanOperators::attention_separate(
    VulkanBuffer& output, const VulkanBuffer& query,
    const VulkanBuffer& key, const VulkanBuffer& value,
    std::uint32_t queries, std::uint32_t keys,
    std::uint32_t heads, std::uint32_t head_dimensions) {
    const std::uint32_t dimensions = heads * head_dimensions;
    require_bytes(query, std::uint64_t(queries) * dimensions, "query");
    require_bytes(key, std::uint64_t(keys) * dimensions, "key");
    require_bytes(value, std::uint64_t(keys) * dimensions, "value");
    require_bytes(output, std::uint64_t(queries) * dimensions, "attention");
    VulkanBuffer scores = context_.create_device_buffer(
        std::uint64_t(heads) * queries * keys * sizeof(float));
    struct Parameters {
        std::uint32_t queries, keys, heads, head_dimensions;
    } parameters{queries, keys, heads, head_dimensions};
    context_.dispatch(
        attention_scores_, {&scores, &query, &key},
        &parameters, sizeof(parameters), divide_up(keys, 64),
        queries * heads);
    struct SoftmaxParameters {
        std::uint32_t rows, columns;
    } softmax{heads * queries, keys};
    context_.dispatch(
        softmax_lastdim_, {&scores, &scores},
        &softmax, sizeof(softmax), softmax.rows);
    context_.dispatch(
        attention_values_, {&output, &scores, &value},
        &parameters, sizeof(parameters),
        divide_up(head_dimensions, 64), queries * heads);
}

#include <inferbridge/vulkan/operators/vulkan_operators_preprocess_rgb.h>

#include <inferbridge/vulkan/operators/vulkan_operators_posterior_sample.h>

#include <inferbridge/vulkan/operators/vulkan_operators_scale_values.h>

#include <inferbridge/vulkan/operators/vulkan_operators_depth_output.h>

void VulkanOperators::prepare_beit(
    VulkanBuffer& output, const VulkanBuffer& image,
    const VulkanBuffer& weight, const VulkanBuffer& bias,
    const VulkanBuffer& class_token,
    std::uint32_t width, std::uint32_t height) {
    if (width == 0 || height == 0 ||
        width % 16 != 0 || height % 16 != 0) {
        throw std::invalid_argument("invalid BEiT image dimensions");
    }
    const std::uint32_t patch_width = width / 16;
    const std::uint32_t patch_height = height / 16;
    const std::uint32_t tokens = patch_width * patch_height + 1;
    require_bytes(image, std::uint64_t(3) * width * height, "BEiT image");
    require_bytes(weight, std::uint64_t(1024) * 3 * 16 * 16, "BEiT weight");
    require_bytes(bias, 1024, "BEiT bias");
    require_bytes(class_token, 1024, "BEiT class token");
    require_bytes(output, std::uint64_t(tokens) * 1024, "BEiT output");
    const std::uint32_t parameters[4] = {
        width, height, patch_width, patch_height};
    context_.dispatch(
        prepare_beit_, {&output, &image, &weight, &bias, &class_token},
        parameters, sizeof(parameters), 128, divide_up(tokens, 8));
}

void VulkanOperators::qv_bias(
    VulkanBuffer& qkv, const VulkanBuffer& q_bias,
    const VulkanBuffer& v_bias,
    std::uint32_t tokens, std::uint32_t embedding) {
    require_bytes(qkv, std::uint64_t(tokens) * embedding * 3, "QKV");
    require_bytes(q_bias, embedding, "query bias");
    require_bytes(v_bias, embedding, "value bias");
    const std::uint32_t parameters[2] = {tokens, embedding};
    context_.dispatch(
        qv_bias_, {&qkv, &q_bias, &v_bias},
        parameters, sizeof(parameters),
        divide_up(tokens * embedding, 256));
}

void VulkanOperators::attention_head64_relative(
    VulkanBuffer& output, const VulkanBuffer& qkv,
    const VulkanBuffer& table, VulkanBuffer& scores,
    std::uint32_t patch_width, std::uint32_t patch_height,
    std::uint32_t heads) {
    const std::uint32_t tokens =
        patch_width * patch_height + 1;
    require_bytes(output, std::uint64_t(tokens) * heads * 64, "attention");
    require_bytes(qkv, std::uint64_t(tokens) * heads * 64 * 3, "QKV");
    require_bytes(scores, std::uint64_t(heads) * tokens * tokens, "scores");
    require_bytes(table, std::uint64_t(2212) * heads, "relative table");
    struct BmmParameters {
        std::uint32_t rows, columns, inner, batches;
        std::uint32_t weight_transposed, output_token_major;
        std::uint32_t qkv_embedding, input_qkv_query;
        std::uint32_t weight_qkv_kind, qkv_heads, qkv_tokens;
    } score{
        tokens, tokens, 64, heads, 0, 0, heads * 64,
        1, 1, heads, tokens};
    context_.dispatch(
        bmm_, {&scores, &qkv, &qkv}, &score, sizeof(score),
        divide_up(divide_up(tokens, 4), 8),
        divide_up(divide_up(tokens, 8), 8), heads);
    const std::uint32_t bias_parameters[4] = {
        patch_width, patch_height, tokens, heads};
    context_.dispatch(
        relative_bias_, {&scores, &table},
        bias_parameters, sizeof(bias_parameters),
        divide_up(heads * tokens * tokens, 256));
    const std::uint32_t softmax[2] = {heads * tokens, tokens};
    context_.dispatch(
        softmax_lastdim_, {&scores, &scores},
        softmax, sizeof(softmax), softmax[0]);
    BmmParameters values{
        tokens, 64, tokens, heads, 0, 1, heads * 64,
        0, 2, heads, tokens};
    context_.dispatch(
        bmm_, {&output, &scores, &qkv}, &values, sizeof(values),
        divide_up(divide_up(64, 4), 8),
        divide_up(divide_up(tokens, 8), 8), heads);
}

void VulkanOperators::readout_concat(
    VulkanBuffer& output, const VulkanBuffer& tokens,
    std::uint32_t patches, std::uint32_t embedding) {
    require_bytes(tokens, std::uint64_t(patches + 1) * embedding, "tokens");
    require_bytes(output, std::uint64_t(patches) * embedding * 2, "readout");
    const std::uint32_t parameters[2] = {patches, embedding};
    context_.dispatch(
        readout_concat_, {&output, &tokens},
        parameters, sizeof(parameters),
        divide_up(patches * embedding * 2, 256));
}

void VulkanOperators::tokens_to_nchw_plain(
    VulkanBuffer& output, const VulkanBuffer& tokens,
    std::uint32_t token_count, std::uint32_t channels) {
    require_bytes(tokens, std::uint64_t(token_count) * channels, "tokens");
    require_bytes(output, std::uint64_t(token_count) * channels, "NCHW");
    const std::uint32_t parameters[2] = {token_count, channels};
    context_.dispatch(
        tokens_to_nchw_plain_, {&output, &tokens},
        parameters, sizeof(parameters),
        divide_up(token_count * channels, 256));
}

void VulkanOperators::softplus(
    VulkanBuffer& values, std::uint32_t count) {
    require_bytes(values, count, "softplus");
    context_.dispatch(
        softplus_, {&values}, &count, sizeof(count), divide_up(count, 256));
}

void VulkanOperators::gelu_values(
    VulkanBuffer& values, std::uint32_t count) {
    require_bytes(values, count, "GELU");
    context_.dispatch(
        gelu_, {&values, &values}, &count, sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::router_tokens(
    VulkanBuffer& output, const VulkanBuffer& embedded,
    std::uint32_t spatial, std::uint32_t features) {
    require_bytes(embedded, std::uint64_t(spatial) * features, "router image");
    require_bytes(output, std::uint64_t(spatial + 1) * features, "router tokens");
    const std::uint32_t parameters[2] = {spatial, features};
    context_.dispatch(
        router_tokens_, {&output, &embedded},
        parameters, sizeof(parameters),
        divide_up((spatial + 1) * features, 256));
}

void VulkanOperators::qkv_split(
    VulkanBuffer& query, VulkanBuffer& key, VulkanBuffer& value_buffer,
    const VulkanBuffer& qkv,
    std::uint32_t tokens, std::uint32_t dimensions) {
    const std::uint64_t count = std::uint64_t(tokens) * dimensions;
    require_bytes(qkv, count * 3, "QKV");
    require_bytes(query, count, "query");
    require_bytes(key, count, "key");
    require_bytes(value_buffer, count, "value");
    const std::uint32_t parameters[2] = {tokens, dimensions};
    context_.dispatch(
        qkv_split_, {&query, &key, &value_buffer, &qkv},
        parameters, sizeof(parameters),
        divide_up(static_cast<std::uint32_t>(count), 256));
}

void VulkanOperators::seed_centers(
    VulkanBuffer& output, const VulkanBuffer& input,
    std::uint32_t pixels, std::uint32_t bins) {
    require_bytes(input, std::uint64_t(pixels) * bins, "seed input");
    require_bytes(output, std::uint64_t(pixels) * bins, "seed centers");
    const std::uint32_t parameters[2] = {pixels, bins};
    context_.dispatch(
        seed_centers_, {&output, &input},
        parameters, sizeof(parameters), divide_up(pixels, 256));
}

void VulkanOperators::attractor_activate(
    VulkanBuffer& values, std::uint32_t count) {
    require_bytes(values, count, "attractors");
    context_.dispatch(
        attractor_activate_, {&values},
        &count, sizeof(count), divide_up(count, 256));
}

void VulkanOperators::attractor_update(
    VulkanBuffer& next, VulkanBuffer& centers,
    const VulkanBuffer& previous, const VulkanBuffer& attractors,
    std::uint32_t pixels, std::uint32_t bins,
    std::uint32_t attractor_count, bool normed) {
    require_bytes(previous, std::uint64_t(pixels) * bins, "previous centers");
    require_bytes(next, std::uint64_t(pixels) * bins, "next centers");
    require_bytes(centers, std::uint64_t(pixels) * bins, "output centers");
    require_bytes(
        attractors,
        std::uint64_t(pixels) * attractor_count * (normed ? 2 : 1),
        "attractors");
    const std::uint32_t parameters[4] = {
        pixels, bins, attractor_count, normed ? 1u : 0u};
    context_.dispatch(
        attractor_update_,
        {&next, &centers, &previous, &attractors},
        parameters, sizeof(parameters), divide_up(pixels, 64));
}

void VulkanOperators::distribution_depth(
    VulkanBuffer& depth, const VulkanBuffer& parameters_buffer,
    const VulkanBuffer& centers, std::uint32_t pixels) {
    require_bytes(parameters_buffer, std::uint64_t(pixels) * 4, "distribution");
    require_bytes(centers, std::uint64_t(pixels) * 64, "centers");
    require_bytes(depth, pixels, "depth");
    context_.dispatch(
        distribution_depth_, {&depth, &parameters_buffer, &centers},
        &pixels, sizeof(pixels), divide_up(pixels, 64));
}

void VulkanOperators::select_depth(
    VulkanBuffer& output, const VulkanBuffer& nyu,
    const VulkanBuffer& kitti, const VulkanBuffer& logits,
    std::uint32_t pixels) {
    require_bytes(output, pixels, "selected depth");
    require_bytes(nyu, pixels, "NYU depth");
    require_bytes(kitti, pixels, "KITTI depth");
    require_bytes(logits, 2, "routing logits");
    context_.dispatch(
        select_depth_, {&output, &nyu, &kitti, &logits},
        &pixels, sizeof(pixels), divide_up(pixels, 256));
}

}  // namespace zoe_native
