#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(binding = 0) uniform sampler2D source_texture;
layout(std430, binding = 1) writeonly buffer Destination { float values[]; } output_data;
layout(push_constant) uniform Parameters {
    uint source_width, source_height;
    uint destination_width, destination_height;
    uint padded_width, padded_height;
    uint pad_width, pad_height_and_flip;
} p;

int reflect_index(int value, int size) {
    const int period = 2 * (size - 1);
    value = value % period;
    if (value < 0) value += period;
    return value >= size ? period - value : value;
}

vec3 padded_pixel(int x, int y, bool flip) {
    int sx = reflect_index(x - int(p.pad_width), int(p.source_width));
    const int sy = reflect_index(y - int(p.pad_height_and_flip & 0x7fffffffu),
                                 int(p.source_height));
    if (flip) sx = int(p.source_width) - 1 - sx;
    return texelFetch(source_texture, ivec2(sx, sy), 0).rgb;
}

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    if (x >= p.destination_width || y >= p.destination_height) return;
    const bool flip = (p.pad_height_and_flip & 0x80000000u) != 0u;
    const float sx = p.destination_width > 1u
        ? float(x) * float(p.padded_width - 1u) / float(p.destination_width - 1u)
        : 0.0;
    const float sy = p.destination_height > 1u
        ? float(y) * float(p.padded_height - 1u) / float(p.destination_height - 1u)
        : 0.0;
    const int x0 = int(floor(sx));
    const int y0 = int(floor(sy));
    const int x1 = min(x0 + 1, int(p.padded_width) - 1);
    const int y1 = min(y0 + 1, int(p.padded_height) - 1);
    const vec3 top = mix(padded_pixel(x0, y0, flip), padded_pixel(x1, y0, flip), fract(sx));
    const vec3 bottom = mix(padded_pixel(x0, y1, flip), padded_pixel(x1, y1, flip), fract(sx));
    const vec3 value = mix(top, bottom, fract(sy)) * 2.0 - 1.0;
    const uint plane = p.destination_width * p.destination_height;
    const uint index = y * p.destination_width + x;
    output_data.values[index] = value.r;
    output_data.values[plane + index] = value.g;
    output_data.values[2u * plane + index] = value.b;
}
