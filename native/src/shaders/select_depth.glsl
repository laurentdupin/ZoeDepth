#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) writeonly buffer OutputBuffer {
    float output_values[];
};
layout(set = 0, binding = 1, std430) readonly buffer NyuBuffer {
    float nyu_values[];
};
layout(set = 0, binding = 2, std430) readonly buffer KittiBuffer {
    float kitti_values[];
};
layout(set = 0, binding = 3, std430) readonly buffer LogitsBuffer {
    float logits[];
};

layout(push_constant) uniform PushConstants {
    uint count;
};

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index < count) {
        output_values[index] =
            logits[0] >= logits[1] ? nyu_values[index] : kitti_values[index];
    }
}
