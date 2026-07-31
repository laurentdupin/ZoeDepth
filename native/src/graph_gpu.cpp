#include "graph_gpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zoe_native {
namespace {

struct GpuDecoderOutput {
    GpuFeature relative_depth;
    GpuFeature out_conv;
    GpuFeature bottleneck;
    std::vector<GpuFeature> refinement_blocks;
};

std::uint64_t elements(const GpuFeature& image) {
    return std::uint64_t(image.channels) * image.height * image.width;
}

const VulkanBuffer& tensor(
    const GpuModel& model, const std::string& name) {
    return model.tensor(name).buffer;
}

GpuFeature clone(
    VulkanContext& context, const GpuFeature& input) {
    GpuFeature output{
        context.create_device_buffer(elements(input) * sizeof(float)),
        input.channels, input.height, input.width};
    context.copy(
        output.buffer, 0, input.buffer, 0,
        elements(input) * sizeof(float));
    return output;
}

GpuFeature conv(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    const GpuFeature& input, const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t stride, std::uint32_t padding) {
    const GpuTensor& shape = model.tensor(weight_name);
    if (shape.rank != 4 || shape.dimensions[1] != input.channels) {
        throw std::runtime_error("GPU convolution shape mismatch: " + weight_name);
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(shape.dimensions[0]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(shape.dimensions[2]);
    const std::uint32_t output_height =
        (input.height + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_width =
        (input.width + 2 * padding - kernel) / stride + 1;
    GpuFeature output{
        context.create_device_buffer(
            std::uint64_t(output_channels) * output_height *
            output_width * sizeof(float)),
        output_channels, output_height, output_width};
    operators.conv2d(
        output.buffer, input.buffer, shape.buffer,
        bias_name.empty() ? zero : tensor(model, bias_name),
        input.width, input.height, input.channels, output_channels,
        kernel, stride, padding, !bias_name.empty(), false, false);
    return output;
}

GpuFeature deconv(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const GpuFeature& input,
    const std::string& weight_name, const std::string& bias_name) {
    const GpuTensor& shape = model.tensor(weight_name);
    if (shape.rank != 4 || shape.dimensions[0] != input.channels) {
        throw std::runtime_error("GPU deconvolution shape mismatch: " + weight_name);
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(shape.dimensions[1]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(shape.dimensions[2]);
    GpuFeature output{
        context.create_device_buffer(
            std::uint64_t(output_channels) * input.height * kernel *
            input.width * kernel * sizeof(float)),
        output_channels, input.height * kernel, input.width * kernel};
    operators.conv_transpose_nonoverlap(
        output.buffer, input.buffer, shape.buffer,
        tensor(model, bias_name), input.width, input.height,
        input.channels, output_channels, kernel);
    return output;
}

GpuFeature resize(
    VulkanContext& context, VulkanOperators& operators,
    const GpuFeature& input,
    std::uint32_t height, std::uint32_t width) {
    GpuFeature output{
        context.create_device_buffer(
            std::uint64_t(input.channels) * height * width * sizeof(float)),
        input.channels, height, width};
    operators.bilinear_align_true(
        output.buffer, input.buffer, input.width, input.height,
        width, height, input.channels);
    return output;
}

void relu(VulkanOperators& operators, GpuFeature& image) {
    operators.relu(
        image.buffer, image.buffer,
        static_cast<std::uint32_t>(elements(image)));
}

GpuFeature add(
    VulkanContext& context, VulkanOperators& operators,
    const GpuFeature& first, const GpuFeature& second) {
    if (first.channels != second.channels ||
        first.height != second.height || first.width != second.width) {
        throw std::runtime_error("GPU feature addition shape mismatch");
    }
    GpuFeature output = clone(context, first);
    operators.add(
        output.buffer, output.buffer, second.buffer,
        static_cast<std::uint32_t>(elements(output)));
    return output;
}

GpuFeature concatenate(
    VulkanContext& context, VulkanOperators& operators,
    const GpuFeature& first, const GpuFeature& second) {
    if (first.height != second.height || first.width != second.width) {
        throw std::runtime_error("GPU concatenation shape mismatch");
    }
    GpuFeature output{
        context.create_device_buffer(
            (elements(first) + elements(second)) * sizeof(float)),
        first.channels + second.channels, first.height, first.width};
    operators.concatenate(
        output.buffer, first.buffer, second.buffer,
        static_cast<std::uint32_t>(elements(first)),
        static_cast<std::uint32_t>(elements(second)));
    return output;
}

GpuFeature residual(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    const GpuFeature& input, const std::string& base) {
    GpuFeature branch = clone(context, input);
    relu(operators, branch);
    branch = conv(
        context, model, operators, zero, branch,
        base + ".conv1.weight", base + ".conv1.bias", 1, 1);
    relu(operators, branch);
    branch = conv(
        context, model, operators, zero, branch,
        base + ".conv2.weight", base + ".conv2.bias", 1, 1);
    operators.add(
        branch.buffer, branch.buffer, input.buffer,
        static_cast<std::uint32_t>(elements(branch)));
    return branch;
}

GpuFeature fusion(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    GpuFeature path, const GpuFeature* skip,
    const std::string& base,
    std::uint32_t output_height, std::uint32_t output_width) {
    if (skip) {
        GpuFeature processed = residual(
            context, model, operators, zero, *skip,
            base + ".resConfUnit1");
        operators.add(
            path.buffer, path.buffer, processed.buffer,
            static_cast<std::uint32_t>(elements(path)));
    }
    path = residual(
        context, model, operators, zero, path,
        base + ".resConfUnit2");
    path = resize(
        context, operators, path, output_height, output_width);
    return conv(
        context, model, operators, zero, path,
        base + ".out_conv.weight", base + ".out_conv.bias", 1, 0);
}

GpuFeature projector(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    const GpuFeature& input, const std::string& base) {
    GpuFeature output = conv(
        context, model, operators, zero, input,
        base + ".0.weight", base + ".0.bias", 1, 0);
    relu(operators, output);
    return conv(
        context, model, operators, zero, output,
        base + ".2.weight", base + ".2.bias", 1, 0);
}

GpuFeature postprocess(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    const VulkanBuffer& capture,
    std::uint32_t patch_height, std::uint32_t patch_width,
    std::uint32_t level) {
    const std::uint32_t patches = patch_height * patch_width;
    const std::string base =
        "core.core.pretrained.act_postprocess" +
        std::to_string(level + 1);
    VulkanBuffer joined = context.create_device_buffer(
        std::uint64_t(patches) * 2048 * sizeof(float));
    VulkanBuffer projected = context.create_device_buffer(
        std::uint64_t(patches) * 1024 * sizeof(float));
    operators.readout_concat(joined, capture, patches, 1024);
    operators.linear(
        projected, joined,
        tensor(model, base + ".0.project.0.weight"),
        tensor(model, base + ".0.project.0.bias"),
        patches, 2048, 1024, true);
    GpuFeature image{
        context.create_device_buffer(
            std::uint64_t(patches) * 1024 * sizeof(float)),
        1024, patch_height, patch_width};
    operators.tokens_to_nchw_plain(
        image.buffer, projected, patches, 1024);
    image = conv(
        context, model, operators, zero, image,
        base + ".3.weight", base + ".3.bias", 1, 0);
    if (level < 2) {
        image = deconv(
            context, model, operators, image,
            base + ".4.weight", base + ".4.bias");
    } else if (level == 3) {
        image = conv(
            context, model, operators, zero, image,
            base + ".4.weight", base + ".4.bias", 2, 1);
    }
    return image;
}

GpuDecoderOutput decode(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    GpuEncoderOutput&& encoded) {
    GpuDecoderOutput output;
    context.batch([&] {
        GpuFeature layers[4];
        GpuFeature refined[4];
        for (std::uint32_t level = 0; level < 4; ++level) {
            layers[level] = postprocess(
                context, model, operators, zero,
                encoded.captures[level],
                encoded.patch_height, encoded.patch_width, level);
            refined[level] = conv(
                context, model, operators, zero, layers[level],
                "core.core.scratch.layer" +
                    std::to_string(level + 1) + "_rn.weight",
                "", 1, 1);
        }
        output.bottleneck = clone(context, refined[3]);
        GpuFeature path = fusion(
            context, model, operators, zero, std::move(refined[3]), nullptr,
            "core.core.scratch.refinenet4",
            refined[2].height, refined[2].width);
        output.refinement_blocks.push_back(clone(context, path));
        path = fusion(
            context, model, operators, zero, std::move(path), &refined[2],
            "core.core.scratch.refinenet3",
            refined[1].height, refined[1].width);
        output.refinement_blocks.push_back(clone(context, path));
        path = fusion(
            context, model, operators, zero, std::move(path), &refined[1],
            "core.core.scratch.refinenet2",
            refined[0].height, refined[0].width);
        output.refinement_blocks.push_back(clone(context, path));
        path = fusion(
            context, model, operators, zero, std::move(path), &refined[0],
            "core.core.scratch.refinenet1",
            refined[0].height * 2, refined[0].width * 2);
        output.refinement_blocks.push_back(clone(context, path));
        path = conv(
            context, model, operators, zero, path,
            "core.core.scratch.output_conv.0.weight",
            "core.core.scratch.output_conv.0.bias", 1, 1);
        path = resize(
            context, operators, path, path.height * 2, path.width * 2);
        path = conv(
            context, model, operators, zero, path,
            "core.core.scratch.output_conv.2.weight",
            "core.core.scratch.output_conv.2.bias", 1, 1);
        relu(operators, path);
        output.out_conv = clone(context, path);
        path = conv(
            context, model, operators, zero, path,
            "core.core.scratch.output_conv.4.weight",
            "core.core.scratch.output_conv.4.bias", 1, 0);
        relu(operators, path);
        output.relative_depth = std::move(path);
    });
    return output;
}

VulkanBuffer route_nk(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    const GpuFeature& bottleneck) {
    VulkanBuffer logits;
    context.batch([&] {
        GpuFeature embedded = conv(
            context, model, operators, zero, bottleneck,
            "patch_transformer.embedding_convPxP.weight",
            "patch_transformer.embedding_convPxP.bias", 1, 0);
        const std::uint32_t spatial = embedded.height * embedded.width;
        const std::uint32_t token_count = spatial + 1;
        const VkDeviceSize token_bytes =
            std::uint64_t(token_count) * 128 * sizeof(float);
        VulkanBuffer tokens = context.create_device_buffer(token_bytes);
        operators.router_tokens(tokens, embedded.buffer, spatial, 128);
        for (std::uint32_t layer = 0; layer < 4; ++layer) {
            const std::string base =
                "patch_transformer.transformer_encoder.layers." +
                std::to_string(layer);
            VulkanBuffer qkv =
                context.create_device_buffer(token_bytes * 3);
            operators.linear(
                qkv, tokens,
                tensor(model, base + ".self_attn.in_proj_weight"),
                tensor(model, base + ".self_attn.in_proj_bias"),
                token_count, 128, 384, false);
            VulkanBuffer query = context.create_device_buffer(token_bytes);
            VulkanBuffer key = context.create_device_buffer(token_bytes);
            VulkanBuffer values = context.create_device_buffer(token_bytes);
            VulkanBuffer attended = context.create_device_buffer(token_bytes);
            operators.qkv_split(
                query, key, values, qkv, token_count, 128);
            operators.attention_separate(
                attended, query, key, values,
                token_count, token_count, 4, 32);
            VulkanBuffer projected = context.create_device_buffer(token_bytes);
            operators.linear(
                projected, attended,
                tensor(model, base + ".self_attn.out_proj.weight"),
                tensor(model, base + ".self_attn.out_proj.bias"),
                token_count, 128, 128, false);
            operators.add(
                projected, projected, tokens,
                token_count * 128);
            VulkanBuffer normalized =
                context.create_device_buffer(token_bytes);
            operators.layer_norm(
                normalized, projected,
                tensor(model, base + ".norm1.weight"),
                tensor(model, base + ".norm1.bias"),
                token_count, 128, 1.0e-5f);
            VulkanBuffer hidden = context.create_device_buffer(
                std::uint64_t(token_count) * 1024 * sizeof(float));
            operators.linear(
                hidden, normalized,
                tensor(model, base + ".linear1.weight"),
                tensor(model, base + ".linear1.bias"),
                token_count, 128, 1024, false);
            operators.relu(hidden, hidden, token_count * 1024);
            VulkanBuffer feed_forward =
                context.create_device_buffer(token_bytes);
            operators.linear(
                feed_forward, hidden,
                tensor(model, base + ".linear2.weight"),
                tensor(model, base + ".linear2.bias"),
                token_count, 1024, 128, false);
            operators.add(
                feed_forward, feed_forward, normalized,
                token_count * 128);
            VulkanBuffer next = context.create_device_buffer(token_bytes);
            operators.layer_norm(
                next, feed_forward,
                tensor(model, base + ".norm2.weight"),
                tensor(model, base + ".norm2.bias"),
                token_count, 128, 1.0e-5f);
            tokens = std::move(next);
        }
        VulkanBuffer hidden =
            context.create_device_buffer(128 * sizeof(float));
        operators.linear(
            hidden, tokens,
            tensor(model, "mlp_classifier.0.weight"),
            tensor(model, "mlp_classifier.0.bias"),
            1, 128, 128, false);
        operators.relu(hidden, hidden, 128);
        logits = context.create_device_buffer(2 * sizeof(float));
        operators.linear(
            logits, hidden,
            tensor(model, "mlp_classifier.2.weight"),
            tensor(model, "mlp_classifier.2.bias"),
            1, 128, 2, false);
    });
    return logits;
}

GpuFeature metric_domain(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    const GpuDecoderOutput& decoded, const GpuFeature& bottleneck,
    const std::string& domain, bool normalized, bool dual) {
    const std::string seed_base = dual
        ? "seed_bin_regressors." + domain : "seed_bin_regressor";
    const std::string attractor_base = dual
        ? "attractors." + domain : "attractors";
    const std::string distribution_base = dual
        ? "conditional_log_binomial." + domain
        : "conditional_log_binomial";
    GpuFeature depth;
    context.batch([&] {
    GpuFeature b_previous = conv(
        context, model, operators, zero, bottleneck,
        seed_base + "._net.0.weight",
        seed_base + "._net.0.bias", 1, 0);
    relu(operators, b_previous);
    b_previous = conv(
        context, model, operators, zero, b_previous,
        seed_base + "._net.2.weight",
        seed_base + "._net.2.bias", 1, 0);
    if (normalized) {
        GpuFeature centers = clone(context, b_previous);
        operators.seed_centers(
            centers.buffer, b_previous.buffer,
            centers.height * centers.width, centers.channels);
        b_previous = std::move(centers);
    } else {
        operators.softplus(
            b_previous.buffer,
            static_cast<std::uint32_t>(elements(b_previous)));
    }
    GpuFeature previous_embedding = projector(
        context, model, operators, zero, bottleneck,
        "seed_projector._net");
    GpuFeature centers;
    GpuFeature embedding;
    const std::uint32_t counts[4] = {
        16, dual ? 16u : 8u, dual ? 16u : 4u, dual ? 16u : 1u};
    for (std::uint32_t level = 0; level < 4; ++level) {
        embedding = projector(
            context, model, operators, zero,
            decoded.refinement_blocks[level],
            "projectors." + std::to_string(level) + "._net");
        GpuFeature previous_resized = resize(
            context, operators, previous_embedding,
            embedding.height, embedding.width);
        GpuFeature attractor_input = add(
            context, operators, embedding, previous_resized);
        GpuFeature attractors = conv(
            context, model, operators, zero, attractor_input,
            attractor_base + "." + std::to_string(level) +
                "._net.0.weight",
            attractor_base + "." + std::to_string(level) +
                "._net.0.bias", 1, 0);
        relu(operators, attractors);
        attractors = conv(
            context, model, operators, zero, attractors,
            attractor_base + "." + std::to_string(level) +
                "._net.2.weight",
            attractor_base + "." + std::to_string(level) +
                "._net.2.bias", 1, 0);
        if (normalized) {
            operators.attractor_activate(
                attractors.buffer,
                static_cast<std::uint32_t>(elements(attractors)));
        } else {
            operators.softplus(
                attractors.buffer,
                static_cast<std::uint32_t>(elements(attractors)));
        }
        GpuFeature previous_centers = resize(
            context, operators, b_previous,
            embedding.height, embedding.width);
        const std::uint32_t pixels =
            embedding.height * embedding.width;
        GpuFeature next{
            context.create_device_buffer(
                std::uint64_t(64) * pixels * sizeof(float)),
            64, embedding.height, embedding.width};
        centers = GpuFeature{
            context.create_device_buffer(
                std::uint64_t(64) * pixels * sizeof(float)),
            64, embedding.height, embedding.width};
        operators.attractor_update(
            next.buffer, centers.buffer,
            previous_centers.buffer, attractors.buffer,
            pixels, 64, counts[level], normalized);
        b_previous = std::move(next);
        previous_embedding = clone(context, embedding);
    }
    GpuFeature last;
    if (dual) {
        last = clone(context, decoded.out_conv);
    } else {
        GpuFeature relative = resize(
            context, operators, decoded.relative_depth,
            decoded.out_conv.height, decoded.out_conv.width);
        last = concatenate(
            context, operators, decoded.out_conv, relative);
    }
    embedding = resize(
        context, operators, embedding, last.height, last.width);
    GpuFeature parameters = concatenate(
        context, operators, last, embedding);
    parameters = conv(
        context, model, operators, zero, parameters,
        distribution_base + ".mlp.0.weight",
        distribution_base + ".mlp.0.bias", 1, 0);
    operators.gelu_values(
        parameters.buffer,
        static_cast<std::uint32_t>(elements(parameters)));
    parameters = conv(
        context, model, operators, zero, parameters,
        distribution_base + ".mlp.2.weight",
        distribution_base + ".mlp.2.bias", 1, 0);
    operators.softplus(
        parameters.buffer,
        static_cast<std::uint32_t>(elements(parameters)));
    centers = resize(
        context, operators, centers, last.height, last.width);
    depth = GpuFeature{
        context.create_device_buffer(
            std::uint64_t(last.height) * last.width * sizeof(float)),
        1, last.height, last.width};
    operators.distribution_depth(
        depth.buffer, parameters.buffer, centers.buffer,
        last.height * last.width);
    });
    return depth;
}

GpuFeature metric(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, const VulkanBuffer& zero,
    GpuDecoderOutput&& decoded) {
    const Variant variant = model.variant();
    const bool dual = variant == Variant::nk;
    GpuFeature bottleneck = conv(
        context, model, operators, zero, decoded.bottleneck,
        "conv2.weight", "conv2.bias", 1, 0);
    if (!dual) {
        return metric_domain(
            context, model, operators, zero, decoded, bottleneck, "",
            variant == Variant::k, false);
    }

    VulkanBuffer logits = route_nk(
        context, model, operators, zero, bottleneck);
    GpuFeature nyu = metric_domain(
        context, model, operators, zero, decoded, bottleneck,
        "nyu", false, true);
    GpuFeature kitti = metric_domain(
        context, model, operators, zero, decoded, bottleneck,
        "kitti", false, true);
    GpuFeature selected{
        context.create_device_buffer(
            std::uint64_t(nyu.height) * nyu.width * sizeof(float)),
        1, nyu.height, nyu.width};
    context.batch([&] {
        operators.select_depth(
            selected.buffer, nyu.buffer, kitti.buffer, logits,
            nyu.height * nyu.width);
    });
    return selected;
}

}  // namespace

GpuFeature full_graph_gpu(
    VulkanContext& context, GpuModel& model,
    VulkanOperators& operators, GpuEncoderOutput&& encoded,
    const VulkanBuffer* persistent_zero) {
    VulkanBuffer owned_zero;
    if (persistent_zero == nullptr) {
        owned_zero = context.create_device_buffer(1024 * sizeof(float));
        const std::vector<float> zeros(1024, 0.0f);
        context.upload(owned_zero, zeros.data(), zeros.size() * sizeof(float));
        persistent_zero = &owned_zero;
    }
    const VulkanBuffer& zero = *persistent_zero;
    GpuDecoderOutput decoded = decode(
        context, model, operators, zero, std::move(encoded));
    GpuFeature result;
    if (model.variant() == Variant::nk) {
        result = metric(
            context, model, operators, zero, std::move(decoded));
    } else {
        context.batch([&] {
            result = metric(
                context, model, operators, zero, std::move(decoded));
        });
    }
    return result;
}

}  // namespace zoe_native
