#if defined(__linux__) && !defined(__ANDROID__)
extern "C" zoedepth_status ZOEDEPTH_CALL zoedepth_infer_bgra8_f32(
    zoedepth_context *context, const uint8_t *bgra, uint64_t stride,
    int32_t width, int32_t height, int32_t size, float *depth, uint64_t count) {
  return zoedepth_infer_bgra8_f32_linux_impl(
      context, bgra, stride, width, height, size, depth, count, nullptr);
}
ibr_linux_capture_capabilities
zoedepth_linux_capture_capabilities(zoedepth_context *context) {
#if defined(ZOEDEPTH_WITH_VULKAN)
  if (context->vulkan)
    return context->vulkan->linux_capture_capabilities();
#endif
  return {};
}
void zoedepth_infer_linux_capture(
    zoedepth_context *context,
    const inferbridge::linux_capture::LinuxDmaBufImage &source, uint32_t size,
    float *output) {
  const auto status = zoedepth_infer_bgra8_f32_linux_impl(
      context, nullptr, source.row_stride, source.width, source.height, size,
      output, uint64_t(source.width) * source.height, &source);
  if (status != 0)
    throw std::runtime_error(zoedepth_last_error());
}
#endif
