#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) readonly buffer Input { float data[]; } input_data;
layout(std430, binding = 1) buffer Range { uint data[]; } range_data;
layout(push_constant) uniform Parameters { uint count; } parameters;

void main() {
    if (gl_GlobalInvocationID.x == 0) {
        uint minimum = floatBitsToUint(input_data.data[0]);
        uint maximum = minimum;
        for (uint index = 1; index < parameters.count; ++index) {
            uint value = floatBitsToUint(input_data.data[index]);
            minimum = min(minimum, value);
            maximum = max(maximum, value);
        }
        range_data.data[0] = minimum;
        range_data.data[1] = maximum;
    }
}
