#ifndef ZOEDEPTH_NATIVE_H
#define ZOEDEPTH_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(ZOEDEPTH_BUILD_DLL)
#    define ZOEDEPTH_API __declspec(dllexport)
#  else
#    define ZOEDEPTH_API __declspec(dllimport)
#  endif
#  define ZOEDEPTH_CALL __cdecl
#else
#  define ZOEDEPTH_API __attribute__((visibility("default")))
#  define ZOEDEPTH_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ZOEDEPTH_ABI_VERSION 1u

typedef struct zoedepth_context zoedepth_context;

typedef enum zoedepth_status {
    ZOEDEPTH_STATUS_OK = 0,
    ZOEDEPTH_STATUS_INVALID_ARGUMENT = 1,
    ZOEDEPTH_STATUS_UNSUPPORTED = 2,
    ZOEDEPTH_STATUS_OUT_OF_MEMORY = 3,
    ZOEDEPTH_STATUS_INTERNAL_ERROR = 4
} zoedepth_status;

typedef enum zoedepth_variant {
    ZOEDEPTH_VARIANT_N = 0,
    ZOEDEPTH_VARIANT_K = 1,
    ZOEDEPTH_VARIANT_NK = 2
} zoedepth_variant;

ZOEDEPTH_API uint32_t ZOEDEPTH_CALL zoedepth_abi_version(void);
ZOEDEPTH_API const char* ZOEDEPTH_CALL zoedepth_version_string(void);
ZOEDEPTH_API const char* ZOEDEPTH_CALL zoedepth_last_error(void);

ZOEDEPTH_API zoedepth_status ZOEDEPTH_CALL zoedepth_create(
    const char* native_model_path_utf8,
    zoedepth_variant variant,
    zoedepth_context** context);
ZOEDEPTH_API void ZOEDEPTH_CALL zoedepth_destroy(
    zoedepth_context* context);

/*
 * Executes ZoeD-M12-N. Input is contiguous planar RGB FP32 in [0,1].
 * Width and height must be positive multiples of 32. Output is contiguous
 * metric depth HW FP32 at the same dimensions.
 */
ZOEDEPTH_API zoedepth_status ZOEDEPTH_CALL
zoedepth_infer_rgb_f32(
    zoedepth_context* context,
    const float* rgb_chw,
    int32_t width,
    int32_t height,
    float* depth_hw,
    uint64_t depth_elements);

#ifdef __cplusplus
}
#endif

#endif

