#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer Depth { float data[]; } depth_data;
layout(std430, binding = 1) readonly buffer Range { float data[]; } range_data;
layout(push_constant) uniform Parameters { uint count; } parameters;

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) return;
    float minimum = range_data.data[0];
    float span = range_data.data[1] - minimum;
    depth_data.data[index] = span > 1.0e-12
        ? 1.0 - clamp((depth_data.data[index] - minimum) / span, 0.0, 1.0)
        : 1.0;
}
