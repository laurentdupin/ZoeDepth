#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct zoedepth_context;
ibr_linux_capture_capabilities
zoedepth_linux_capture_capabilities(zoedepth_context *);
void zoedepth_infer_linux_capture(
    zoedepth_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint32_t, float *);
#endif
