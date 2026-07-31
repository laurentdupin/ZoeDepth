#version 450 core

layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_width;
    uint output_height;
    uint output_channels;
    uint kernel;
    uint stride;
    int padding;
    uint has_bias;
    uint batches;
    uint channel_blocks;
} parameters;

shared float spatial_tile[8 * 180];
shared float kernel_tile[8 * 8 * 9];

void main() {
    const uint block = gl_GlobalInvocationID.z %
        parameters.channel_blocks;
    const uint batch = gl_GlobalInvocationID.z /
        parameters.channel_blocks;
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel_base = block * 8;
    const bool valid =
        batch < parameters.batches &&
        x < parameters.output_width &&
        y < parameters.output_height &&
        channel_base < parameters.output_channels;
    float sums[8] = float[8](
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    const uint lane =
        gl_LocalInvocationID.y * 16 + gl_LocalInvocationID.x;
    const int origin_x =
        int(gl_WorkGroupID.x * 16) - 1;
    const int origin_y =
        int(gl_WorkGroupID.y * 8) - 1;
    const uint input_batch =
        batch * parameters.input_channels *
        parameters.input_height * parameters.input_width;
    const uint output_batch =
        batch * parameters.output_channels *
        parameters.output_height * parameters.output_width;

    for (uint input_base = 0;
         input_base < parameters.input_channels;
         input_base += 8) {
        for (uint index = lane; index < 8 * 180; index += 128) {
            const uint input_offset = index / 180;
            const uint tile_index = index % 180;
            const uint input_channel = input_base + input_offset;
            const int input_x =
                origin_x + int(tile_index % 18);
            const int input_y =
                origin_y + int(tile_index / 18);
            spatial_tile[index] =
                input_channel < parameters.input_channels &&
                input_x >= 0 &&
                input_x < int(parameters.input_width) &&
                input_y >= 0 &&
                input_y < int(parameters.input_height)
                ? input_buffer.data[
                    input_batch +
                    (input_channel * parameters.input_height +
                     uint(input_y)) * parameters.input_width +
                    uint(input_x)]
                : 0.0;
        }
        for (uint index = lane;
             index < 8 * 8 * 9;
             index += 128) {
            const uint input_offset = index / (8 * 9);
            const uint kernel_index = index % (8 * 9);
            const uint output_offset = kernel_index / 9;
            const uint input_channel = input_base + input_offset;
            const uint output_channel =
                channel_base + output_offset;
            kernel_tile[index] =
                input_channel < parameters.input_channels &&
                output_channel < parameters.output_channels
                ? weight_buffer.data[
                    (output_channel * parameters.input_channels +
                     input_channel) * 9 +
                    kernel_index % 9]
                : 0.0;
        }
        barrier();
        if (valid) {
            for (uint input_offset = 0;
                 input_offset < 8 &&
                 input_base + input_offset <
                     parameters.input_channels;
                 ++input_offset) {
                for (uint ky = 0; ky < 3; ++ky) {
                    for (uint kx = 0; kx < 3; ++kx) {
                        const float value = spatial_tile[
                            input_offset * 180 +
                            (gl_LocalInvocationID.y + ky) * 18 +
                            gl_LocalInvocationID.x + kx];
                        const uint kernel_index = ky * 3 + kx;
                        for (uint output_offset = 0;
                             output_offset < 8;
                             ++output_offset) {
                            sums[output_offset] += value * kernel_tile[
                                input_offset * 72 +
                                output_offset * 9 +
                                kernel_index];
                        }
                    }
                }
            }
        }
        barrier();
    }
    if (!valid) {
        return;
    }
    for (uint output_offset = 0;
         output_offset < 8;
         ++output_offset) {
        const uint output_channel =
            channel_base + output_offset;
        if (output_channel < parameters.output_channels) {
            output_buffer.data[
                output_batch +
                (output_channel * parameters.output_height + y) *
                parameters.output_width + x] =
                sums[output_offset] +
                (parameters.has_bias != 0
                    ? bias_buffer.data[output_channel]
                    : 0.0);
        }
    }
}
