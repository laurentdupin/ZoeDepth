#include "zoedepth_native.h"

#include <assert.h>
#include <string.h>

int main(void) {
    zoedepth_context* context = 0;
    assert(zoedepth_abi_version() == ZOEDEPTH_ABI_VERSION);
    assert(zoedepth_version_string() != 0);
    assert(zoedepth_last_error() != 0);
    assert(
        zoedepth_create(
            0, ZOEDEPTH_VARIANT_N, &context) ==
        ZOEDEPTH_STATUS_INVALID_ARGUMENT);
    assert(context == 0);
    assert(
        zoedepth_create_vulkan(
            0, ZOEDEPTH_VARIANT_N, 0, &context) ==
        ZOEDEPTH_STATUS_INVALID_ARGUMENT);
    assert(context == 0);
    assert(
        zoedepth_infer_bgra8_f32(
            0, 0, 0, 0, 0, 0, 0, 0) ==
        ZOEDEPTH_STATUS_INVALID_ARGUMENT);
    zoedepth_destroy(0);
    return 0;
}
