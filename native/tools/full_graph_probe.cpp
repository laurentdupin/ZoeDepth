#include "zoedepth_native.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::vector<float> read_floats(
    const std::string& path,
    std::uint64_t count) {
    std::vector<float> result(static_cast<std::size_t>(count));
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!input ||
        input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid raw tensor: " + path);
    }
    return result;
}
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr
            << "usage: zoedepth_full_graph_probe model.zoe variant size "
               "rgb.bin depth.bin tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const double tolerance = std::stod(argv[6]);
        const std::vector<float> input = read_floats(
            argv[4], std::uint64_t(3) * size * size);
        const std::vector<float> reference = read_floats(
            argv[5], std::uint64_t(size) * size);
        zoedepth_variant variant = ZOEDEPTH_VARIANT_N;
        const std::string variant_name = argv[2];
        if (variant_name == "k") {
            variant = ZOEDEPTH_VARIANT_K;
        } else if (variant_name == "nk") {
            variant = ZOEDEPTH_VARIANT_NK;
        } else if (variant_name != "n") {
            throw std::invalid_argument("invalid variant");
        }
        zoedepth_context* context = nullptr;
        const zoedepth_status create_status = zoedepth_create(
            argv[1], variant, &context);
        if (create_status != ZOEDEPTH_STATUS_OK) {
            throw std::runtime_error(zoedepth_last_error());
        }
        std::vector<float> output(
            static_cast<std::size_t>(size) * size);
        const auto start = std::chrono::steady_clock::now();
        const zoedepth_status inference_status =
            zoedepth_infer_rgb_f32(
                context, input.data(),
                static_cast<std::int32_t>(size),
                static_cast<std::int32_t>(size),
                output.data(), output.size());
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (inference_status != ZOEDEPTH_STATUS_OK) {
            const std::string error = zoedepth_last_error();
            zoedepth_destroy(context);
            throw std::runtime_error(error);
        }
        zoedepth_destroy(context);
        double absolute_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        for (std::size_t index = 0;
             index < reference.size(); ++index) {
            const float difference =
                std::abs(output[index] - reference[index]);
            maximum = std::max(maximum, difference);
            absolute_sum += difference;
            reference_sum += std::abs(reference[index]);
        }
        const double relative =
            absolute_sum / std::max(reference_sum, 1.0e-30);
        std::cout << "seconds=" << seconds
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative << "\n";
        return relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
