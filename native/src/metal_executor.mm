#include "metal_executor.h"
#include "inferbridge/native_harness_metal_texture.h"
#include "inferbridge/native_harness_precision.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace zoe_native {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}
MPSShape* shape(const TensorView& value) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:value.rank];
    for (std::uint32_t i = 0; i < value.rank; ++i)
        [result addObject:@(value.dimensions[i])];
    return result;
}
NSString* ns(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

class MetalExternalJob final : public ExternalJob {
public:
    explicit MetalExternalJob(
        std::shared_ptr<inferbridge::native_harness::metal::Submission> value)
        : submission_(std::move(value)) {}
    ExternalJobState state() const override {
        if (submission_->cancelled()) return ExternalJobState::cancelled;
        return submission_->complete() ? ExternalJobState::complete :
            ExternalJobState::running;
    }
    void cancel() override { submission_->cancel(); }
private:
    std::shared_ptr<inferbridge::native_harness::metal::Submission> submission_;
};

struct Image {
    MPSGraphTensor* value = nil;
    int channels = 0;
    int height = 0;
    int width = 0;
};

float bilinear_table(
    const float* table, int head, int output_x, int output_y,
    int output_width, int output_height) {
    constexpr int source = 47;
    float sx = (output_x + 0.5f) * source / output_width - 0.5f;
    float sy = (output_y + 0.5f) * source / output_height - 0.5f;
    sx = std::clamp(sx, 0.0f, 46.0f);
    sy = std::clamp(sy, 0.0f, 46.0f);
    const int x0 = static_cast<int>(sx), y0 = static_cast<int>(sy);
    const int x1 = std::min(x0 + 1, 46), y1 = std::min(y0 + 1, 46);
    const float wx = sx - x0, wy = sy - y0;
    const auto at = [&](int x, int y) {
        return table[(static_cast<std::size_t>(x) * 47 + y) * 16 + head];
    };
    return (at(x0, y0) * (1 - wx) + at(x1, y0) * wx) * (1 - wy) +
           (at(x0, y1) * (1 - wx) + at(x1, y1) * wx) * wy;
}

std::vector<float> relative_bias(
    const TensorView& table, int patch_width, int patch_height) {
    const int tokens = 1 + patch_width * patch_height;
    const int rw = 2 * patch_width - 1, rh = 2 * patch_height - 1;
    const int spatial = rw * rh;
    std::vector<float> result(
        static_cast<std::size_t>(16) * tokens * tokens);
    for (int head = 0; head < 16; ++head)
        for (int query = 0; query < tokens; ++query)
            for (int key = 0; key < tokens; ++key) {
                int index;
                if (query == 0 && key == 0) index = spatial + 2;
                else if (query == 0) index = spatial;
                else if (key == 0) index = spatial + 1;
                else {
                    const int q = query - 1, k = key - 1;
                    const int dy = q / patch_width - k / patch_width;
                    const int dx = q % patch_width - k % patch_width;
                    index = (dy + patch_height - 1) * rw +
                        dx + patch_width - 1;
                }
                float value;
                if (index >= spatial)
                    value = table.data[
                        static_cast<std::size_t>(2209 + index - spatial) * 16 + head];
                else
                    value = bilinear_table(table.data, head,
                        index / rh, index % rh, rw, rh);
                result[(static_cast<std::size_t>(head) * tokens + query) *
                    tokens + key] = value;
            }
    return result;
}

class GraphBuilder {
public:
    GraphBuilder(const ModelFile& model, int width, int height, bool fp16)
        : model_(model), width_(width), height_(height), fp16_(fp16),
          variant_(model.derivation().variant), graph_([MPSGraph new]) {}

    void build() {
        input_ = [graph_ placeholderWithShape:shape({1, 3, height_, width_})
            dataType:MPSDataTypeFloat32 name:@"normalized_rgb"];
        Image image{internal(input_), 3, height_, width_};
        auto captures = encoder(image);
        auto decoded = decoder(captures);
        Image bottleneck = conv(decoded.bottleneck, "conv2", 1, 0);
        MPSGraphTensor* depth;
        if (variant_ == Variant::nk) {
            MPSGraphTensor* logits = router(bottleneck);
            MPSGraphTensor* nyu = metric(decoded, bottleneck, "nyu", false, true);
            MPSGraphTensor* kitti = metric(decoded, bottleneck, "kitti", false, true);
            MPSGraphTensor* left = [graph_ sliceTensor:logits dimension:1
                start:0 length:1 name:nil];
            MPSGraphTensor* right = [graph_ sliceTensor:logits dimension:1
                start:1 length:1 name:nil];
            MPSGraphTensor* condition = [graph_ greaterThanOrEqualToWithPrimaryTensor:
                left secondaryTensor:right name:nil];
            depth = [graph_ selectWithPredicateTensor:condition
                truePredicateTensor:nyu falsePredicateTensor:kitti name:nil];
        } else {
            depth = metric(decoded, bottleneck, "",
                variant_ == Variant::k, false);
        }
        output_ = external(depth);
    }

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input() const { return input_; }
    MPSGraphTensor* output() const { return output_; }

private:
    MPSGraphTensor* internal(MPSGraphTensor* value) {
        return fp16_ && value.dataType != MPSDataTypeFloat16
            ? [graph_ castTensor:value toType:MPSDataTypeFloat16 name:nil]
            : value;
    }
    MPSGraphTensor* external(MPSGraphTensor* value) {
        return value.dataType == MPSDataTypeFloat32 ? value
            : [graph_ castTensor:value toType:MPSDataTypeFloat32 name:nil];
    }
    MPSGraphTensor* constant(const std::string& name) {
        const TensorView& tensor = model_.tensor(name);
        if (fp16_) {
            auto found = half_.find(name);
            if (found == half_.end())
                found = half_.emplace(name, inferbridge::native::pack_fp16(
                    tensor.data, static_cast<std::size_t>(tensor.elements))).first;
            NSData* data = [NSData dataWithBytesNoCopy:found->second.data()
                length:found->second.size() * sizeof(std::uint16_t)
                freeWhenDone:NO];
            return [graph_ constantWithData:data shape:shape(tensor)
                dataType:MPSDataTypeFloat16];
        }
        NSData* data = [NSData dataWithBytesNoCopy:const_cast<float*>(tensor.data)
            length:tensor.elements * sizeof(float) freeWhenDone:NO];
        return [graph_ constantWithData:data shape:shape(tensor)
            dataType:MPSDataTypeFloat32];
    }
    MPSGraphTensor* host_constant(
        const std::string& key, const std::vector<float>& values,
        MPSShape* dimensions) {
        if (fp16_) {
            auto found = generated_half_.find(key);
            if (found == generated_half_.end())
                found = generated_half_.emplace(
                    key, inferbridge::native::pack_fp16(
                        values.data(), values.size())).first;
            NSData* data = [NSData dataWithBytesNoCopy:found->second.data()
                length:found->second.size() * sizeof(std::uint16_t)
                freeWhenDone:NO];
            return [graph_ constantWithData:data shape:dimensions
                dataType:MPSDataTypeFloat16];
        }
        auto found = generated_float_.find(key);
        if (found == generated_float_.end())
            found = generated_float_.emplace(key, values).first;
        NSData* data = [NSData dataWithBytesNoCopy:found->second.data()
            length:found->second.size() * sizeof(float) freeWhenDone:NO];
        return [graph_ constantWithData:data shape:dimensions
            dataType:MPSDataTypeFloat32];
    }
    MPSGraphTensor* scalar(float value) {
        return [graph_ constantWithScalar:value
            dataType:fp16_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32];
    }
    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* sub(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* mul(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* div(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* linear(MPSGraphTensor* value, const std::string& prefix) {
        MPSGraphTensor* weight = [graph_ transposeTensor:
            constant(prefix + ".weight") dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* result = [graph_
            matrixMultiplicationWithPrimaryTensor:value secondaryTensor:weight name:nil];
        return model_.contains(prefix + ".bias")
            ? add(result, constant(prefix + ".bias")) : result;
    }
    MPSGraphTensor* gelu(MPSGraphTensor* value) {
        MPSGraphTensor* error = [graph_ erfWithTensor:mul(
            value, scalar(0.7071067811865475f)) name:nil];
        return mul(mul(value, scalar(0.5f)), add(error, scalar(1.0f)));
    }
    MPSGraphTensor* softplus(MPSGraphTensor* value) {
        MPSGraphTensor* magnitude = [graph_ absoluteWithTensor:value name:nil];
        MPSGraphTensor* exponential = [graph_ exponentWithTensor:
            mul(magnitude, scalar(-1.0f)) name:nil];
        MPSGraphTensor* logarithm = [graph_ logarithmWithTensor:
            add(exponential, scalar(1.0f)) name:nil];
        return add(logarithm, [graph_ maximumWithPrimaryTensor:value
            secondaryTensor:scalar(0.0f) name:nil]);
    }
    MPSGraphTensor* layer_norm(
        MPSGraphTensor* value, const std::string& prefix, float epsilon) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:value axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:value
            meanTensor:mean axes:axes name:nil];
        return [graph_ normalizationWithTensor:value meanTensor:mean
            varianceTensor:variance gammaTensor:constant(prefix + ".weight")
            betaTensor:constant(prefix + ".bias") epsilon:epsilon name:ns(prefix)];
    }
    Image conv(
        const Image& input, const std::string& prefix,
        int stride = 1, int padding = 0, bool bias = true) {
        const TensorView& weight = model_.tensor(prefix + ".weight");
        auto* descriptor = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride dilationRateInX:1
            dilationRateInY:1 groups:1 paddingLeft:padding paddingRight:padding
            paddingTop:padding paddingBottom:padding
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:input.value
            weightsTensor:constant(prefix + ".weight") descriptor:descriptor name:ns(prefix)];
        const int channels = static_cast<int>(weight.dimensions[0]);
        if (bias && model_.contains(prefix + ".bias"))
            result = add(result, [graph_ reshapeTensor:constant(prefix + ".bias")
                withShape:shape({1, channels, 1, 1}) name:nil]);
        const int kernel = static_cast<int>(weight.dimensions[2]);
        return {result, channels,
            (input.height + 2 * padding - kernel) / stride + 1,
            (input.width + 2 * padding - kernel) / stride + 1};
    }
    Image resize(const Image& input, int height, int width) {
        return {[graph_ resizeTensor:input.value size:shape({height, width})
            mode:MPSGraphResizeBilinear centerResult:NO alignCorners:YES
            layout:MPSGraphTensorNamedDataLayoutNCHW name:nil],
            input.channels, height, width};
    }
    Image deconv(const Image& input, const std::string& prefix) {
        const TensorView& weight = model_.tensor(prefix + ".weight");
        const int channels = static_cast<int>(weight.dimensions[1]);
        const int kernel = static_cast<int>(weight.dimensions[2]);
        MPSGraphTensor* w = [graph_ transposeTensor:constant(prefix + ".weight")
            permutation:@[@1, @2, @3, @0] name:nil];
        w = [graph_ reshapeTensor:w
            withShape:shape({channels * kernel * kernel, input.channels, 1, 1}) name:nil];
        auto* descriptor = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1
            groups:1 paddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor* value = [graph_ convolution2DWithSourceTensor:input.value
            weightsTensor:w descriptor:descriptor name:ns(prefix)];
        value = [graph_ reshapeTensor:value withShape:
            shape({1, channels, kernel, kernel, input.height, input.width}) name:nil];
        value = [graph_ transposeTensor:value
            permutation:@[@0, @1, @4, @2, @5, @3] name:nil];
        value = [graph_ reshapeTensor:value withShape:
            shape({1, channels, input.height * kernel, input.width * kernel}) name:nil];
        value = add(value, [graph_ reshapeTensor:constant(prefix + ".bias")
            withShape:shape({1, channels, 1, 1}) name:nil]);
        return {value, channels, input.height * kernel, input.width * kernel};
    }

    std::array<MPSGraphTensor*, 4> encoder(const Image& image) {
        constexpr const char* root = "core.core.pretrained.model.";
        Image patches = conv(image,
            std::string(root) + "patch_embed.proj", 16, 0);
        const int patch_count = patches.height * patches.width;
        const int tokens = patch_count + 1;
        MPSGraphTensor* state = [graph_ reshapeTensor:patches.value
            withShape:shape({1, 1024, patch_count}) name:nil];
        state = [graph_ transposeTensor:state dimension:1 withDimension:2 name:nil];
        state = [graph_ concatTensors:@[
            constant(std::string(root) + "cls_token"), state]
            dimension:1 name:nil];
        std::array<MPSGraphTensor*, 4> captures{};
        int capture = 0;
        for (int block = 0; block < 24; ++block) {
            const std::string base = std::string(root) + "blocks." +
                std::to_string(block) + ".";
            MPSGraphTensor* normalized = layer_norm(
                state, base + "norm1", 1.0e-6f);
            MPSGraphTensor* qkv = linear(normalized, base + "attn.qkv");
            const TensorView& q_bias = model_.tensor(base + "attn.q_bias");
            const TensorView& v_bias = model_.tensor(base + "attn.v_bias");
            std::vector<float> qkv_bias(3072, 0.0f);
            std::copy_n(q_bias.data, 1024, qkv_bias.begin());
            std::copy_n(v_bias.data, 1024, qkv_bias.begin() + 2048);
            qkv = add(qkv, host_constant(base + "qkv_bias", qkv_bias,
                shape({3072})));
            qkv = [graph_ reshapeTensor:qkv
                withShape:shape({1, tokens, 3, 16, 64}) name:nil];
            qkv = [graph_ transposeTensor:qkv
                permutation:@[@2, @0, @3, @1, @4] name:nil];
            MPSGraphTensor* q = [graph_ sliceTensor:qkv dimension:0
                start:0 length:1 name:nil];
            MPSGraphTensor* k = [graph_ sliceTensor:qkv dimension:0
                start:1 length:1 name:nil];
            MPSGraphTensor* v = [graph_ sliceTensor:qkv dimension:0
                start:2 length:1 name:nil];
            q = [graph_ reshapeTensor:q withShape:shape({1, 16, tokens, 64}) name:nil];
            k = [graph_ reshapeTensor:k withShape:shape({1, 16, tokens, 64}) name:nil];
            v = [graph_ reshapeTensor:v withShape:shape({1, 16, tokens, 64}) name:nil];
            k = [graph_ transposeTensor:k dimension:2 withDimension:3 name:nil];
            MPSGraphTensor* scores = [graph_
                matrixMultiplicationWithPrimaryTensor:mul(q, scalar(0.125f))
                secondaryTensor:k name:nil];
            const std::vector<float> bias_values = relative_bias(
                model_.tensor(base + "attn.relative_position_bias_table"),
                patches.width, patches.height);
            scores = add(scores, host_constant(base + "relative_bias",
                bias_values, shape({1, 16, tokens, tokens})));
            scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
            MPSGraphTensor* attended = [graph_
                matrixMultiplicationWithPrimaryTensor:scores
                secondaryTensor:v name:nil];
            attended = [graph_ transposeTensor:attended
                dimension:1 withDimension:2 name:nil];
            attended = [graph_ reshapeTensor:attended
                withShape:shape({1, tokens, 1024}) name:nil];
            attended = linear(attended, base + "attn.proj");
            state = add(state, mul(attended, constant(base + "gamma_1")));
            MPSGraphTensor* hidden = gelu(linear(layer_norm(
                state, base + "norm2", 1.0e-6f), base + "mlp.fc1"));
            state = add(state, mul(linear(hidden, base + "mlp.fc2"),
                constant(base + "gamma_2")));
            if (block == 5 || block == 11 || block == 17 || block == 23)
                captures[capture++] = state;
        }
        return captures;
    }

    Image postprocess(
        MPSGraphTensor* capture, int level, int patch_height, int patch_width) {
        const int patches = patch_height * patch_width;
        const std::string base = "core.core.pretrained.act_postprocess" +
            std::to_string(level + 1);
        MPSGraphTensor* cls = [graph_ sliceTensor:capture dimension:1
            start:0 length:1 name:nil];
        cls = [graph_ tileTensor:cls
            withMultiplier:shape({1, patches, 1}) name:nil];
        MPSGraphTensor* tokens = [graph_ sliceTensor:capture dimension:1
            start:1 length:patches name:nil];
        tokens = [graph_ concatTensors:@[tokens, cls] dimension:2 name:nil];
        tokens = gelu(linear(tokens, base + ".0.project.0"));
        tokens = [graph_ transposeTensor:tokens dimension:1 withDimension:2 name:nil];
        Image result{[graph_ reshapeTensor:tokens
            withShape:shape({1, 1024, patch_height, patch_width}) name:nil],
            1024, patch_height, patch_width};
        result = conv(result, base + ".3", 1, 0);
        if (level < 2) result = deconv(result, base + ".4");
        else if (level == 3) result = conv(result, base + ".4", 2, 1);
        return result;
    }

    Image residual(const Image& input, const std::string& base) {
        Image branch{[graph_ reLUWithTensor:input.value name:nil],
            input.channels, input.height, input.width};
        branch = conv(branch, base + ".conv1", 1, 1);
        branch.value = [graph_ reLUWithTensor:branch.value name:nil];
        branch = conv(branch, base + ".conv2", 1, 1);
        branch.value = add(branch.value, input.value);
        return branch;
    }

    Image fusion(
        Image path, const Image* skip, const std::string& base,
        int output_height, int output_width) {
        if (skip)
            path.value = add(path.value,
                residual(*skip, base + ".resConfUnit1").value);
        path = residual(path, base + ".resConfUnit2");
        path = resize(path, output_height, output_width);
        return conv(path, base + ".out_conv", 1, 0);
    }

    struct Decoder {
        Image relative;
        Image out_conv;
        Image bottleneck;
        std::array<Image, 4> refinement;
    };

    Decoder decoder(const std::array<MPSGraphTensor*, 4>& captures) {
        const int patch_height = height_ / 16, patch_width = width_ / 16;
        std::array<Image, 4> layers;
        std::array<Image, 4> refined;
        for (int level = 0; level < 4; ++level) {
            layers[level] = postprocess(
                captures[level], level, patch_height, patch_width);
            refined[level] = conv(layers[level],
                "core.core.scratch.layer" + std::to_string(level + 1) + "_rn",
                1, 1, false);
        }
        Decoder output;
        output.bottleneck = refined[3];
        Image path = fusion(refined[3], nullptr,
            "core.core.scratch.refinenet4",
            refined[2].height, refined[2].width);
        output.refinement[0] = path;
        path = fusion(path, &refined[2], "core.core.scratch.refinenet3",
            refined[1].height, refined[1].width);
        output.refinement[1] = path;
        path = fusion(path, &refined[1], "core.core.scratch.refinenet2",
            refined[0].height, refined[0].width);
        output.refinement[2] = path;
        path = fusion(path, &refined[0], "core.core.scratch.refinenet1",
            refined[0].height * 2, refined[0].width * 2);
        output.refinement[3] = path;
        path = conv(path, "core.core.scratch.output_conv.0", 1, 1);
        path = resize(path, path.height * 2, path.width * 2);
        path = conv(path, "core.core.scratch.output_conv.2", 1, 1);
        path.value = [graph_ reLUWithTensor:path.value name:nil];
        output.out_conv = path;
        path = conv(path, "core.core.scratch.output_conv.4", 1, 0);
        path.value = [graph_ reLUWithTensor:path.value name:nil];
        output.relative = path;
        return output;
    }

    Image projector(const Image& input, const std::string& base) {
        Image output = conv(input, base + ".0", 1, 0);
        output.value = [graph_ reLUWithTensor:output.value name:nil];
        return conv(output, base + ".2", 1, 0);
    }

    MPSGraphTensor* router(const Image& bottleneck) {
        Image embedded = conv(bottleneck,
            "patch_transformer.embedding_convPxP", 1, 0);
        const int pixels = embedded.height * embedded.width;
        const int tokens_count = pixels + 1;
        MPSGraphTensor* spatial = [graph_ reshapeTensor:embedded.value
            withShape:shape({1, 128, pixels}) name:nil];
        spatial = [graph_ transposeTensor:spatial dimension:1
            withDimension:2 name:nil];
        MPSGraphTensor* zero = [graph_ constantWithScalar:0.0
            shape:shape({1, 1, 128})
            dataType:fp16_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32];
        MPSGraphTensor* tokens = [graph_ concatTensors:@[zero, spatial]
            dimension:1 name:nil];
        std::vector<float> position(
            static_cast<std::size_t>(tokens_count) * 128);
        constexpr float factor = -9.210340371976184f / 128.0f;
        for (int token = 0; token < tokens_count; ++token)
            for (int pair = 0; pair < 64; ++pair) {
                const float angle = token * std::exp(pair * 2.0f * factor);
                position[static_cast<std::size_t>(token) * 128 + pair] =
                    std::sin(angle);
                position[static_cast<std::size_t>(token) * 128 + 64 + pair] =
                    std::cos(angle);
            }
        tokens = add(tokens, host_constant("router_position", position,
            shape({1, tokens_count, 128})));
        for (int layer = 0; layer < 4; ++layer) {
            const std::string base =
                "patch_transformer.transformer_encoder.layers." +
                std::to_string(layer);
            MPSGraphTensor* in_weight = [graph_ transposeTensor:
                constant(base + ".self_attn.in_proj_weight")
                dimension:0 withDimension:1 name:nil];
            MPSGraphTensor* qkv = add([graph_
                matrixMultiplicationWithPrimaryTensor:tokens
                secondaryTensor:in_weight name:nil],
                constant(base + ".self_attn.in_proj_bias"));
            qkv = [graph_ reshapeTensor:qkv
                withShape:shape({1, tokens_count, 3, 4, 32}) name:nil];
            qkv = [graph_ transposeTensor:qkv
                permutation:@[@2, @0, @3, @1, @4] name:nil];
            MPSGraphTensor* q = [graph_ sliceTensor:qkv dimension:0
                start:0 length:1 name:nil];
            MPSGraphTensor* k = [graph_ sliceTensor:qkv dimension:0
                start:1 length:1 name:nil];
            MPSGraphTensor* v = [graph_ sliceTensor:qkv dimension:0
                start:2 length:1 name:nil];
            q = [graph_ reshapeTensor:q
                withShape:shape({1, 4, tokens_count, 32}) name:nil];
            k = [graph_ reshapeTensor:k
                withShape:shape({1, 4, tokens_count, 32}) name:nil];
            v = [graph_ reshapeTensor:v
                withShape:shape({1, 4, tokens_count, 32}) name:nil];
            k = [graph_ transposeTensor:k dimension:2 withDimension:3 name:nil];
            MPSGraphTensor* scores = [graph_
                matrixMultiplicationWithPrimaryTensor:mul(
                    q, scalar(1.0f / std::sqrt(32.0f)))
                secondaryTensor:k name:nil];
            scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
            MPSGraphTensor* attended = [graph_
                matrixMultiplicationWithPrimaryTensor:scores
                secondaryTensor:v name:nil];
            attended = [graph_ transposeTensor:attended
                dimension:1 withDimension:2 name:nil];
            attended = [graph_ reshapeTensor:attended
                withShape:shape({1, tokens_count, 128}) name:nil];
            tokens = layer_norm(add(tokens,
                linear(attended, base + ".self_attn.out_proj")),
                base + ".norm1", 1.0e-5f);
            MPSGraphTensor* hidden = [graph_ reLUWithTensor:
                linear(tokens, base + ".linear1") name:nil];
            tokens = layer_norm(add(tokens,
                linear(hidden, base + ".linear2")),
                base + ".norm2", 1.0e-5f);
        }
        MPSGraphTensor* cls = [graph_ sliceTensor:tokens dimension:1
            start:0 length:1 name:nil];
        cls = [graph_ reshapeTensor:cls withShape:shape({1, 128}) name:nil];
        cls = [graph_ reLUWithTensor:linear(cls, "mlp_classifier.0") name:nil];
        return linear(cls, "mlp_classifier.2");
    }

    MPSGraphTensor* normalized_seed(MPSGraphTensor* widths) {
        widths = add([graph_ reLUWithTensor:widths name:nil], scalar(1.0e-3f));
        MPSGraphTensor* total = [graph_ reductionSumWithTensor:widths
            axis:1 name:nil];
        total = [graph_ reshapeTensor:total
            withShape:shape({1, 1,
                [widths.shape[2] intValue], [widths.shape[3] intValue]}) name:nil];
        widths = div(widths, total);
        MPSGraphTensor* edges = [graph_ cumulativeSumWithTensor:widths
            axis:1 exclusive:NO reverse:NO name:nil];
        return sub(edges, mul(widths, scalar(0.5f)));
    }

    MPSGraphTensor* attractor_update(
        MPSGraphTensor* previous, MPSGraphTensor* attractors,
        int count, bool normed) {
        if (normed) {
            std::vector<std::int32_t> even(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i) even[i] = i * 2;
            NSData* data = [NSData dataWithBytes:even.data()
                length:even.size() * sizeof(std::int32_t)];
            MPSGraphTensor* indices = [graph_ constantWithData:data
                shape:shape({count}) dataType:MPSDataTypeInt32];
            attractors = [graph_ gatherWithUpdatesTensor:attractors
                indicesTensor:indices axis:1 batchDimensions:0 name:nil];
        }
        const int height = [previous.shape[2] intValue];
        const int width = [previous.shape[3] intValue];
        MPSGraphTensor* centers = [graph_ reshapeTensor:previous
            withShape:shape({1, 64, 1, height, width}) name:nil];
        MPSGraphTensor* points = [graph_ reshapeTensor:attractors
            withShape:shape({1, 1, count, height, width}) name:nil];
        MPSGraphTensor* difference = sub(points, centers);
        MPSGraphTensor* denominator = add(scalar(1.0f),
            mul(scalar(1000.0f), mul(difference, difference)));
        MPSGraphTensor* delta = [graph_ reductionSumWithTensor:
            div(difference, denominator) axis:2 name:nil];
        delta = [graph_ reshapeTensor:delta
            withShape:shape({1, 64, height, width}) name:nil];
        return add(previous, mul(delta, scalar(1.0f / count)));
    }

    MPSGraphTensor* distribution(
        const Image& last, const Image& embedding,
        MPSGraphTensor* centers, const std::string& base) {
        MPSGraphTensor* joined = [graph_ concatTensors:@[
            last.value, resize(embedding, last.height, last.width).value]
            dimension:1 name:nil];
        Image parameters{joined, last.channels + embedding.channels,
            last.height, last.width};
        parameters = conv(parameters, base + ".mlp.0", 1, 0);
        parameters.value = gelu(parameters.value);
        parameters = conv(parameters, base + ".mlp.2", 1, 0);
        MPSGraphTensor* p = softplus(parameters.value);
        MPSGraphTensor* p0 = add([graph_ sliceTensor:p dimension:1
            start:0 length:1 name:nil], scalar(1.0e-4f));
        MPSGraphTensor* p1 = add([graph_ sliceTensor:p dimension:1
            start:1 length:1 name:nil], scalar(1.0e-4f));
        MPSGraphTensor* probability = div(p0, add(p0, p1));
        MPSGraphTensor* t0 = add([graph_ sliceTensor:p dimension:1
            start:2 length:1 name:nil], scalar(1.0e-4f));
        MPSGraphTensor* t1 = add([graph_ sliceTensor:p dimension:1
            start:3 length:1 name:nil], scalar(1.0e-4f));
        MPSGraphTensor* temperature = add(mul(div(t0, add(t0, t1)),
            scalar(49.9788f)), scalar(0.0212f));

        std::vector<float> k(64), combination(64);
        constexpr float e = 1.0e-7f;
        const float n = 63.0f + e;
        for (int bin = 0; bin < 64; ++bin) {
            k[bin] = static_cast<float>(bin) + e;
            const float remainder = n - k[bin];
            combination[bin] = n * std::log(n) - k[bin] * std::log(k[bin]) -
                remainder * std::log(remainder + e);
        }
        MPSGraphTensor* bins = host_constant(base + ".bins", k,
            shape({1, 64, 1, 1}));
        MPSGraphTensor* combinations = host_constant(base + ".combination",
            combination, shape({1, 64, 1, 1}));
        MPSGraphTensor* clamped_p = [graph_ clampWithTensor:probability
            minValueTensor:scalar(1.0e-4f) maxValueTensor:scalar(1.0f) name:nil];
        MPSGraphTensor* one_minus = [graph_ clampWithTensor:
            sub(scalar(1.0f), probability) minValueTensor:scalar(1.0e-4f)
            maxValueTensor:scalar(1.0f) name:nil];
        MPSGraphTensor* logits = add(combinations, add(
            mul(bins, [graph_ logarithmWithTensor:clamped_p name:nil]),
            mul(sub(scalar(63.0f), bins),
                [graph_ logarithmWithTensor:one_minus name:nil])));
        logits = div(logits, temperature);
        MPSGraphTensor* probabilities = [graph_ softMaxWithTensor:logits
            axis:1 name:nil];
        return [graph_ reductionSumWithTensor:mul(probabilities, centers)
            axis:1 name:nil];
    }

    MPSGraphTensor* metric(
        const Decoder& decoded, const Image& bottleneck,
        const std::string& domain, bool normed, bool dual) {
        const std::string seed = dual
            ? "seed_bin_regressors." + domain : "seed_bin_regressor";
        const std::string attractor = dual
            ? "attractors." + domain : "attractors";
        const std::string distribution_base = dual
            ? "conditional_log_binomial." + domain
            : "conditional_log_binomial";
        Image previous = conv(bottleneck, seed + "._net.0", 1, 0);
        previous.value = [graph_ reLUWithTensor:previous.value name:nil];
        previous = conv(previous, seed + "._net.2", 1, 0);
        previous.value = normed ? normalized_seed(previous.value)
            : softplus(previous.value);
        Image previous_embedding = projector(
            bottleneck, "seed_projector._net");
        Image embedding;
        MPSGraphTensor* centers = nil;
        const int counts[4] = {
            16, dual ? 16 : 8, dual ? 16 : 4, dual ? 16 : 1};
        for (int level = 0; level < 4; ++level) {
            embedding = projector(decoded.refinement[level],
                "projectors." + std::to_string(level) + "._net");
            Image resized_previous_embedding = resize(
                previous_embedding, embedding.height, embedding.width);
            Image attractor_input{add(embedding.value,
                resized_previous_embedding.value), embedding.channels,
                embedding.height, embedding.width};
            Image points = conv(attractor_input, attractor + "." +
                std::to_string(level) + "._net.0", 1, 0);
            points.value = [graph_ reLUWithTensor:points.value name:nil];
            points = conv(points, attractor + "." +
                std::to_string(level) + "._net.2", 1, 0);
            points.value = normed
                ? add([graph_ reLUWithTensor:points.value name:nil],
                      scalar(1.0e-3f))
                : softplus(points.value);
            previous = resize(previous, embedding.height, embedding.width);
            MPSGraphTensor* updated = attractor_update(
                previous.value, points.value, counts[level], normed);
            previous = {updated, 64, embedding.height, embedding.width};
            centers = updated;
            if (normed) {
                centers = add(mul(centers, scalar(9.999f)), scalar(1.0e-3f));
                centers = [graph_ clampWithTensor:centers
                    minValueTensor:scalar(1.0e-3f)
                    maxValueTensor:scalar(10.0f) name:nil];
                centers = [graph_ sortWithTensor:centers axis:1
                    descending:NO name:nil];
            }
            previous_embedding = embedding;
        }
        Image last = decoded.out_conv;
        if (!dual) {
            Image relative = resize(decoded.relative,
                decoded.out_conv.height, decoded.out_conv.width);
            last = {[graph_ concatTensors:@[last.value, relative.value]
                dimension:1 name:nil], last.channels + 1,
                last.height, last.width};
        }
        Image center_image{centers, 64,
            previous.height, previous.width};
        center_image = resize(center_image, last.height, last.width);
        return distribution(last, embedding, center_image.value,
            distribution_base);
    }

    const ModelFile& model_;
    int width_;
    int height_;
    bool fp16_;
    Variant variant_;
    MPSGraph* graph_;
    MPSGraphTensor* input_ = nil;
    MPSGraphTensor* output_ = nil;
    std::unordered_map<std::string, std::vector<std::uint16_t>> half_;
    std::unordered_map<std::string, std::vector<std::uint16_t>> generated_half_;
    std::unordered_map<std::string, std::vector<float>> generated_float_;
};

struct PlanKey {
    int width;
    int height;
    bool operator==(const PlanKey& other) const {
        return width == other.width && height == other.height;
    }
};
struct PlanHash {
    std::size_t operator()(const PlanKey& key) const {
        return static_cast<std::size_t>(key.width) * 65537u + key.height;
    }
};
struct Plan {
    MPSGraph* graph = nil;
    MPSGraphExecutable* executable = nil;
};

std::string hexadecimal(const std::array<std::uint8_t, 32>& bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string output(bytes.size() * 2, '0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        output[i * 2] = digits[bytes[i] >> 4];
        output[i * 2 + 1] = digits[bytes[i] & 15];
    }
    return output;
}

}  // namespace

class MetalExecutor::Impl {
public:
    explicit Impl(const ModelFile& model) : model_(model) {
        const auto precision = inferbridge::native::requested_precision();
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("ZoeDepth Metal does not support INT8 yet");
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) throw std::runtime_error("Metal is unavailable");
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("could not initialize ZoeDepth Metal");
        create_texture_pipelines();
    }

    void set_cache_path(const std::string& cache_path) {
        std::lock_guard<std::mutex> guard(mutex_);
        cache_path_ = cache_path;
    }

    std::vector<float> infer(
        const float* input, std::uint32_t width, std::uint32_t height) {
        std::lock_guard<std::mutex> lock(mutex_);
        @autoreleasepool {
            const Plan& plan = get_plan(width, height);
            id<MTLBuffer> buffer = [device_ newBufferWithBytes:input
                length:static_cast<NSUInteger>(3) * width * height * sizeof(float)
                options:MTLResourceStorageModeShared];
            if (buffer == nil) throw std::bad_alloc();
            MPSGraphTensorData* data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:buffer shape:shape({1, 3,
                    static_cast<NSInteger>(height), static_cast<NSInteger>(width)})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_ inputsArray:@[data]
                resultsArray:nil executionDescriptor:execution];
            if (results.count != 1)
                throw std::runtime_error("ZoeDepth Metal returned no output");
            std::vector<float> output(static_cast<std::size_t>(width) * height);
            [results[0].mpsndarray readBytes:output.data() strideBytes:nil];
            return output;
        }
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) {
        if (!request.shared_texture_handle || !request.output_texture_handle ||
            !request.signal_fence_handle || !request.signal_fence_value ||
            request.width < 2u || request.height < 2u ||
            request.output_width != request.width ||
            request.output_height != request.height ||
            !request.input_size || request.input_size % 32u != 0u)
            throw std::invalid_argument("invalid ZoeDepth Metal texture request");
        inferbridge::native_harness::metal::Prepared prepared;
        prepared.input_texture=(__bridge id<MTLTexture>)(
            reinterpret_cast<void*>(request.shared_texture_handle));
        prepared.output_texture=(__bridge id<MTLTexture>)(
            reinterpret_cast<void*>(request.output_texture_handle));
        prepared.wait_event=request.wait_fence_handle?
            (__bridge id<MTLSharedEvent>)(reinterpret_cast<void*>(
                request.wait_fence_handle)):nil;
        prepared.signal_event=(__bridge id<MTLSharedEvent>)(
            reinterpret_cast<void*>(request.signal_fence_handle));
        prepared.signal_value=request.signal_fence_value;
        const MTLPixelFormat expected=request.rgba?MTLPixelFormatRGBA8Unorm:
            MTLPixelFormatBGRA8Unorm;
        if(prepared.input_texture.device.registryID!=device_.registryID||
           prepared.output_texture.device.registryID!=device_.registryID||
           prepared.input_texture.textureType!=MTLTextureType2D||
           prepared.input_texture.width!=request.width||
           prepared.input_texture.height!=request.height||
           prepared.input_texture.pixelFormat!=expected||
           prepared.output_texture.textureType!=MTLTextureType2D||
           prepared.output_texture.width!=request.output_width||
           prepared.output_texture.height!=request.output_height||
           prepared.output_texture.pixelFormat!=MTLPixelFormatR32Float)
            throw std::invalid_argument("ZoeDepth Metal texture descriptor mismatch");
        const uint32_t pad_height=static_cast<uint32_t>(
            std::sqrt(static_cast<double>(request.height)/2.0)*3.0);
        const uint32_t pad_width=static_cast<uint32_t>(
            std::sqrt(static_cast<double>(request.width)/2.0)*3.0);
        if(pad_height>=request.height||pad_width>=request.width)
            throw std::invalid_argument(
                "image is too small for ZoeDepth reflection padding");
        const uint32_t padded_width=request.width+2u*pad_width;
        const uint32_t padded_height=request.height+2u*pad_height;
        double scale_height=static_cast<double>(request.input_size)/padded_height;
        double scale_width=static_cast<double>(request.input_size)/padded_width;
        if(std::abs(1.0-scale_width)<std::abs(1.0-scale_height))
            scale_height=scale_width;else scale_width=scale_height;
        const auto multiple32=[](double value){return static_cast<uint32_t>(
            std::max(32.0,std::nearbyint(value/32.0)*32.0));};
        const uint32_t network_width=multiple32(scale_width*padded_width);
        const uint32_t network_height=multiple32(scale_height*padded_height);
        std::lock_guard<std::mutex> lock(mutex_);
        @autoreleasepool {
            id<MTLBuffer> input=[device_ newBufferWithLength:
                static_cast<NSUInteger>(network_width)*network_height*3*sizeof(float)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> network_depth=[device_ newBufferWithLength:
                static_cast<NSUInteger>(network_width)*network_height*sizeof(float)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> combined=[device_ newBufferWithLength:
                static_cast<NSUInteger>(request.width)*request.height*sizeof(float)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> range=[device_ newBufferWithLength:2*sizeof(float)
                options:MTLResourceStorageModePrivate];
            if(!input||!network_depth||!combined||!range)throw std::bad_alloc();
            id<MTLCommandBuffer> preprocess=[queue_ commandBuffer];
            if(prepared.wait_event)[preprocess encodeWaitForEvent:prepared.wait_event
                value:request.wait_fence_value];
            id<MTLComputeCommandEncoder> encoder=[preprocess computeCommandEncoder];
            struct PreprocessParameters{uint32_t sw,sh,dw,dh,pw,ph,padx,pady,rgba;} pp{
                request.width,request.height,network_width,network_height,
                padded_width,padded_height,pad_width,pad_height,request.rgba?1u:0u};
            [encoder setComputePipelineState:preprocess_pipeline_];
            [encoder setTexture:prepared.input_texture atIndex:0];
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBytes:&pp length:sizeof(pp) atIndex:1];
            dispatch(encoder,preprocess_pipeline_,network_width,network_height);
            [encoder endEncoding];[preprocess commit];
            const Plan& plan=get_plan(network_width,network_height);
            MPSGraphTensorData* input_data=[[MPSGraphTensorData alloc]
                initWithMTLBuffer:input shape:shape({1,3,(NSInteger)network_height,
                    (NSInteger)network_width}) dataType:MPSDataTypeFloat32];
            MPSGraphTensorData* output_data=[[MPSGraphTensorData alloc]
                initWithMTLBuffer:network_depth shape:shape({1,1,
                    (NSInteger)network_height,(NSInteger)network_width})
                    dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* execution=
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted=NO;
            NSArray* results=[plan.executable runAsyncWithMTLCommandQueue:queue_
                inputsArray:@[input_data] resultsArray:@[output_data]
                executionDescriptor:execution];
            if(results.count!=1)
                throw std::runtime_error("ZoeDepth Metal output binding failed");
            id<MTLCommandBuffer> completion=[queue_ commandBuffer];
            encoder=[completion computeCommandEncoder];
            struct CombineParameters{uint32_t nw,nh,pw,ph,ow,oh,padx,pady;} cp{
                network_width,network_height,padded_width,padded_height,
                request.width,request.height,pad_width,pad_height};
            [encoder setComputePipelineState:combine_pipeline_];
            [encoder setBuffer:combined offset:0 atIndex:0];
            [encoder setBuffer:network_depth offset:0 atIndex:1];
            [encoder setBytes:&cp length:sizeof(cp) atIndex:2];
            dispatch(encoder,combine_pipeline_,request.width,request.height);
            uint32_t count=request.width*request.height;
            [encoder setComputePipelineState:reduce_pipeline_];
            [encoder setBuffer:combined offset:0 atIndex:0];
            [encoder setBuffer:range offset:0 atIndex:1];
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
            dispatch(encoder,reduce_pipeline_,1,1);
            struct OutputParameters{uint32_t width,height;} op{
                request.width,request.height};
            [encoder setComputePipelineState:output_pipeline_];
            [encoder setBuffer:combined offset:0 atIndex:0];
            [encoder setBuffer:range offset:0 atIndex:1];
            [encoder setTexture:prepared.output_texture atIndex:0];
            [encoder setBytes:&op length:sizeof(op) atIndex:2];
            dispatch(encoder,output_pipeline_,request.width,request.height);
            [encoder endEncoding];
            [completion encodeSignalEvent:prepared.signal_event
                value:prepared.signal_value];[completion commit];
            return std::make_shared<MetalExternalJob>(
                std::make_shared<inferbridge::native_harness::metal::Submission>(
                    prepared,completion));
        }
    }

private:
    static void dispatch(id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline,NSUInteger width,
        NSUInteger height){
        const NSUInteger x=pipeline.threadExecutionWidth;
        const NSUInteger y=std::max<NSUInteger>(1,
            pipeline.maxTotalThreadsPerThreadgroup/x);
        [encoder dispatchThreads:MTLSizeMake(width,height,1)
            threadsPerThreadgroup:MTLSizeMake(x,y,1)];
    }

    void create_texture_pipelines(){
        static constexpr char source_text[]=R"METAL(
#include <metal_stdlib>
using namespace metal;
struct PreprocessParameters{uint sw,sh,dw,dh,pw,ph,padx,pady,rgba;};
int reflect_index(int value,int size){int period=2*(size-1);value%=period;
 if(value<0)value+=period;return value>=size?period-value:value;}
float3 padded(texture2d<float,access::read>src,int x,int y,
 constant PreprocessParameters&p){int sx=reflect_index(x-int(p.padx),int(p.sw));
 int sy=reflect_index(y-int(p.pady),int(p.sh));return src.read(uint2(sx,sy)).rgb;}
kernel void preprocess(texture2d<float,access::read>src[[texture(0)]],
 device float*dst[[buffer(0)]],constant PreprocessParameters&p[[buffer(1)]],
 uint2 q[[thread_position_in_grid]]){if(q.x>=p.dw||q.y>=p.dh)return;
 float sx=p.dw>1?float(q.x)*float(p.pw-1)/float(p.dw-1):0.0f;
 float sy=p.dh>1?float(q.y)*float(p.ph-1)/float(p.dh-1):0.0f;
 int x0=int(floor(sx)),y0=int(floor(sy)),x1=min(x0+1,int(p.pw)-1),
 y1=min(y0+1,int(p.ph)-1);float3 value=mix(mix(padded(src,x0,y0,p),
 padded(src,x1,y0,p),fract(sx)),mix(padded(src,x0,y1,p),
 padded(src,x1,y1,p),fract(sx)),fract(sy))*2.0f-1.0f;
 uint plane=p.dw*p.dh,i=q.y*p.dw+q.x;dst[i]=value.r;
 dst[plane+i]=value.g;dst[2*plane+i]=value.b;}
float cubic(float d){const float a=-0.75f;d=abs(d);if(d<1.0f)
 return ((a+2.0f)*d-(a+3.0f))*d*d+1.0f;if(d<2.0f)
 return ((a*d-5.0f*a)*d+8.0f*a)*d-4.0f*a;return 0.0f;}
struct CombineParameters{uint nw,nh,pw,ph,ow,oh,padx,pady;};
float sample_depth(device const float*src,float px,float py,
 constant CombineParameters&p){float sx=(px+0.5f)*float(p.nw)/float(p.pw)-0.5f;
 float sy=(py+0.5f)*float(p.nh)/float(p.ph)-0.5f;int bx=int(floor(sx)),
 by=int(floor(sy));float result=0.0f;for(int ky=-1;ky<=2;++ky){int iy=clamp(by+ky,0,
 int(p.nh)-1);float wy=cubic(sy-float(by+ky));for(int kx=-1;kx<=2;++kx){
 int ix=clamp(bx+kx,0,int(p.nw)-1);result+=src[uint(iy)*p.nw+uint(ix)]*
 wy*cubic(sx-float(bx+kx));}}return result;}
kernel void combine(device float*dst[[buffer(0)]],device const float*src[[buffer(1)]],
 constant CombineParameters&p[[buffer(2)]],uint2 q[[thread_position_in_grid]]){
 if(q.x>=p.ow||q.y>=p.oh)return;dst[q.y*p.ow+q.x]=sample_depth(src,
 float(q.x+p.padx),float(q.y+p.pady),p);}
kernel void reduce_range(device const float*src[[buffer(0)]],device float*range[[buffer(1)]],
 constant uint&count[[buffer(2)]],uint gid[[thread_position_in_grid]]){if(gid)return;
 float lo=INFINITY,hi=-INFINITY;for(uint i=0;i<count;++i){lo=min(lo,src[i]);
 hi=max(hi,src[i]);}range[0]=lo;range[1]=hi;}
struct OutputParameters{uint width,height;};
kernel void output_depth(device const float*src[[buffer(0)]],device const float*range[[buffer(1)]],
 texture2d<float,access::write>out[[texture(0)]],constant OutputParameters&p[[buffer(2)]],
 uint2 q[[thread_position_in_grid]]){if(q.x>=p.width||q.y>=p.height)return;
 float span=range[1]-range[0];float v=span>1.0e-12f?1.0f-clamp(
 (src[q.y*p.width+q.x]-range[0])/span,0.0f,1.0f):1.0f;
 out.write(float4(v),q);}
)METAL";
        NSError* error=nil;id<MTLLibrary> library=[device_ newLibraryWithSource:
            [NSString stringWithUTF8String:source_text] options:nil error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String?:
            "could not compile ZoeDepth Metal texture kernels");
        auto make=[&](NSString*name){id<MTLComputePipelineState> result=
            [device_ newComputePipelineStateWithFunction:
                [library newFunctionWithName:name] error:&error];
            if(!result)throw std::runtime_error(
                error.localizedDescription.UTF8String?:
                "could not create ZoeDepth Metal texture pipeline");
            return result;
        };
        preprocess_pipeline_=make(@"preprocess");combine_pipeline_=make(@"combine");
        reduce_pipeline_=make(@"reduce_range");output_pipeline_=make(@"output_depth");
    }

    const Plan& get_plan(std::uint32_t width, std::uint32_t height) {
        const PlanKey key{static_cast<int>(width), static_cast<int>(height)};
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        MPSGraphCompilationDescriptor* descriptor =
            [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
        descriptor.waitForCompilationCompletion = YES;
        NSURL* package = cache_url(key);
        MPSGraphExecutable* executable = nil;
        if (@available(macOS 14.0, *)) {
            if (package && [[NSFileManager defaultManager]
                    fileExistsAtPath:package.path]) {
                @try {
                    executable = [[MPSGraphExecutable alloc]
                        initWithMPSGraphPackageAtURL:package
                        compilationDescriptor:descriptor];
                } @catch (NSException*) {
                    [[NSFileManager defaultManager]
                        removeItemAtURL:package error:nil];
                }
            }
        }
        MPSGraph* graph = nil;
        if (executable == nil) {
            GraphBuilder builder(model_, key.width, key.height, fp16_);
            builder.build();
            graph = builder.graph();
            MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc]
                initWithShape:shape({1, 3, key.height, key.width})
                dataType:MPSDataTypeFloat32];
            executable = [builder.graph() compileWithDevice:graph_device_
                feeds:@{builder.input(): input_type}
                targetTensors:@[builder.output()] targetOperations:nil
                compilationDescriptor:descriptor];
            if (@available(macOS 14.0, *)) {
                if (executable && package) {
                    @try {
                        [executable serializeToMPSGraphPackageAtURL:package
                            descriptor:nil];
                    } @catch (NSException*) {
                        [[NSFileManager defaultManager]
                            removeItemAtURL:package error:nil];
                    }
                }
            }
        }
        if (executable == nil)
            throw std::runtime_error("failed to compile ZoeDepth Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(key, Plan{graph, executable}).first->second;
    }

    NSURL* cache_url(const PlanKey& key) const {
        if (@available(macOS 14.0, *)) {
            if (cache_path_.empty()) return nil;
            NSString* directory = [[NSString
                stringWithUTF8String:cache_path_.c_str()]
                stringByAppendingPathComponent:@"ZoeDepthMetalGraphCache-v2"];
            if (![[NSFileManager defaultManager] createDirectoryAtPath:directory
                    withIntermediateDirectories:YES attributes:nil error:nil])
                return nil;
            const NSOperatingSystemVersion os =
                NSProcessInfo.processInfo.operatingSystemVersion;
            const std::string name = hexadecimal(
                model_.derivation().canonical_sha256) + "-" +
                std::to_string(static_cast<int>(model_.derivation().variant)) + "-" +
                std::to_string(key.width) + "x" + std::to_string(key.height) +
                "-" + (fp16_ ? "fp16" : "fp32") + "-" +
                std::to_string(device_.registryID) + "-macos" +
                std::to_string(os.majorVersion) + "." +
                std::to_string(os.minorVersion) + ".mpsgraphpackage";
            return [NSURL fileURLWithPath:[directory
                stringByAppendingPathComponent:ns(name)]];
        }
        return nil;
    }

    const ModelFile& model_;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<PlanKey, Plan, PlanHash> plans_;
    std::mutex mutex_;
    id<MTLComputePipelineState> preprocess_pipeline_=nil;
    id<MTLComputePipelineState> combine_pipeline_=nil;
    id<MTLComputePipelineState> reduce_pipeline_=nil;
    id<MTLComputePipelineState> output_pipeline_=nil;
    std::string cache_path_;
};

MetalExecutor::MetalExecutor(const ModelFile& model)
    : impl_(std::make_unique<Impl>(model)) {}
MetalExecutor::~MetalExecutor() = default;
void MetalExecutor::set_cache_path(const std::string& cache_path) {
    impl_->set_cache_path(cache_path);
}
std::vector<float> MetalExecutor::infer(
    const float* input, std::uint32_t width, std::uint32_t height) {
    return impl_->infer(input, width, height);
}
std::shared_ptr<ExternalJob> MetalExecutor::submit_texture(
    const ExternalTextureRequest& request) {
    return impl_->submit_texture(request);
}

}  // namespace zoe_native
