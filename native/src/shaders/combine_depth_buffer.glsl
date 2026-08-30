#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(std430, binding = 0) writeonly buffer Output { float values[]; } output_data;
layout(std430, binding = 1) readonly buffer Direct { float values[]; } direct_data;
layout(std430, binding = 2) readonly buffer Flipped { float values[]; } flipped_data;
layout(push_constant) uniform Parameters {
    uint network_width, network_height;
    uint padded_width, padded_height;
    uint output_width, output_height;
    uint pad_width, pad_height;
} p;

float cubic(float distance) {
    const float a = -0.75;
    distance = abs(distance);
    if (distance < 1.0) return ((a + 2.0) * distance - (a + 3.0)) * distance * distance + 1.0;
    if (distance < 2.0) return ((a * distance - 5.0 * a) * distance + 8.0 * a) * distance - 4.0 * a;
    return 0.0;
}
float sample_depth(bool flipped, float px, float py) {
    const float sx = (px + 0.5) * float(p.network_width) / float(p.padded_width) - 0.5;
    const float sy = (py + 0.5) * float(p.network_height) / float(p.padded_height) - 0.5;
    const int bx = int(floor(sx));
    const int by = int(floor(sy));
    float result = 0.0;
    for (int ky = -1; ky <= 2; ++ky) {
        const int iy = clamp(by + ky, 0, int(p.network_height) - 1);
        const float wy = cubic(sy - float(by + ky));
        for (int kx = -1; kx <= 2; ++kx) {
            const int ix = clamp(bx + kx, 0, int(p.network_width) - 1);
            const uint index = uint(iy) * p.network_width + uint(ix);
            result += (flipped ? flipped_data.values[index] : direct_data.values[index]) *
                wy * cubic(sx - float(bx + kx));
        }
    }
    return result;
}
void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    if (x >= p.output_width || y >= p.output_height) return;
    const bool blend_flipped = (p.pad_height & 0x80000000u) != 0u;
    const uint pad_height = p.pad_height & 0x7fffffffu;
    const float direct_x = float(x + p.pad_width);
    const float flipped_x = float(p.output_width - 1u - x + p.pad_width);
    const float padded_y = float(y + pad_height);
    const float direct = sample_depth(false, direct_x, padded_y);
    output_data.values[y * p.output_width + x] = blend_flipped
        ? 0.5 * (direct + sample_depth(true, flipped_x, padded_y))
        : direct;
}
