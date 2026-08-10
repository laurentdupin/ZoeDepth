#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(binding = 0, r32f) uniform writeonly image2D output_image;
layout(std430, binding = 1) readonly buffer Input { float data[]; } input_data;
layout(push_constant) uniform Parameters { uint width; uint height; } parameters;

void main() {
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= parameters.width || pixel.y >= parameters.height) return;
    imageStore(output_image, ivec2(pixel), vec4(input_data.data[pixel.y * parameters.width + pixel.x]));
}
