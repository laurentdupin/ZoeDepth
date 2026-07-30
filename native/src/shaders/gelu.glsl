#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;

layout(push_constant) uniform Parameters {
    uint count;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) {
        return;
    }
    const float value = input_buffer.data[index];
    const float x = value * 0.7071067811865475244;
    const float absolute_x = abs(x);
    const float t = 1.0 / (1.0 + 0.3275911 * absolute_x);
    const float polynomial =
        (((((1.061405429 * t - 1.453152027) * t) +
            1.421413741) * t - 0.284496736) * t +
            0.254829592) * t;
    const float erf_value =
        sign(x) * (1.0 - polynomial * exp(-absolute_x * absolute_x));
    output_buffer.data[index] =
        0.5 * value * (1.0 + erf_value);
}
