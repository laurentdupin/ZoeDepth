#include "zoedepth_native.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
zoedepth_variant parse_variant(const std::string& value) {
    if (value == "n") return ZOEDEPTH_VARIANT_N;
    if (value == "k") return ZOEDEPTH_VARIANT_K;
    if (value == "nk") return ZOEDEPTH_VARIANT_NK;
    throw std::invalid_argument("variant must be n, k, or nk");
}

std::vector<float> make_input(std::uint32_t size) {
    const std::size_t plane = static_cast<std::size_t>(size) * size;
    std::vector<float> input(plane * 3);
    for (std::uint32_t y = 0; y < size; ++y) {
        for (std::uint32_t x = 0; x < size; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * size + x;
            input[index] = static_cast<float>((x * 17 + y * 3) % 251) / 250.0f;
            input[plane + index] = static_cast<float>((x * 5 + y * 19) % 241) / 240.0f;
            input[plane * 2 + index] = static_cast<float>((x * 11 + y * 7) % 239) / 238.0f;
        }
    }
    return input;
}

double infer(
    zoedepth_context* context,
    const std::vector<float>& input,
    std::uint32_t size,
    std::vector<float>& output) {
    const auto start = std::chrono::steady_clock::now();
    const zoedepth_status status = zoedepth_infer_rgb_f32(
        context, input.data(), static_cast<std::int32_t>(size),
        static_cast<std::int32_t>(size), output.data(), output.size());
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (status != ZOEDEPTH_STATUS_OK) {
        throw std::runtime_error(zoedepth_last_error());
    }
    return seconds;
}
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: zoedepth_metal_parity_probe model.zoe "
                     "variant size tolerance\n";
        return 2;
    }
    zoedepth_context* cpu = nullptr;
    zoedepth_context* metal = nullptr;
    try {
        const zoedepth_variant variant = parse_variant(argv[2]);
        const std::uint32_t size = static_cast<std::uint32_t>(std::stoul(argv[3]));
        const double tolerance = std::stod(argv[4]);
        if (size == 0 || size % 32 != 0) {
            throw std::invalid_argument("size must be a positive multiple of 32");
        }
        if (zoedepth_create(argv[1], variant, &cpu) != ZOEDEPTH_STATUS_OK) {
            throw std::runtime_error(zoedepth_last_error());
        }
        if (zoedepth_create_vulkan(argv[1], variant, 0, &metal) != ZOEDEPTH_STATUS_OK) {
            throw std::runtime_error(zoedepth_last_error());
        }
        const std::vector<float> input = make_input(size);
        std::vector<float> reference(static_cast<std::size_t>(size) * size);
        std::vector<float> output(reference.size());
        const double cpu_seconds = infer(cpu, input, size, reference);
        const double first_metal_seconds = infer(metal, input, size, output);
        const double warm_metal_seconds = infer(metal, input, size, output);

        double absolute_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        bool finite = true;
        for (std::size_t index = 0; index < reference.size(); ++index) {
            finite = finite && std::isfinite(reference[index]) && std::isfinite(output[index]);
            const float difference = std::abs(output[index] - reference[index]);
            maximum = std::max(maximum, difference);
            absolute_sum += difference;
            reference_sum += std::abs(reference[index]);
        }
        const double relative = absolute_sum / std::max(reference_sum, 1.0e-30);
        std::cout << "cpu_seconds=" << cpu_seconds
                  << "\nmetal_first_seconds=" << first_metal_seconds
                  << "\nmetal_warm_seconds=" << warm_metal_seconds
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative
                  << "\nfinite=" << (finite ? "true" : "false") << "\n";
        zoedepth_destroy(metal);
        zoedepth_destroy(cpu);
        return finite && relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        zoedepth_destroy(metal);
        zoedepth_destroy(cpu);
        std::cerr << error.what() << "\n";
        return 1;
    }
}
