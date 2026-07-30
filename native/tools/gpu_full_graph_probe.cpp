#include "zoedepth_native.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> read(const std::string& path, std::uint64_t count) {
    std::vector<float> result(static_cast<std::size_t>(count));
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid tensor file: " + path);
    }
    return result;
}
}

int main(int argc, char** argv) {
    if (argc != 8 && argc != 9) {
        std::cerr << "usage: gpu_probe model variant device size rgb depth "
                     "tolerance [iterations]\n";
        return 2;
    }
    zoedepth_context* context = nullptr;
    try {
        const std::string name = argv[2];
        const zoedepth_variant variant =
            name == "k" ? ZOEDEPTH_VARIANT_K :
            name == "nk" ? ZOEDEPTH_VARIANT_NK :
            ZOEDEPTH_VARIANT_N;
        if (name != "n" && name != "k" && name != "nk") {
            throw std::invalid_argument("invalid variant");
        }
        const std::uint32_t device =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[4]));
        const double tolerance = std::stod(argv[7]);
        const std::uint32_t iterations =
            argc == 9 ? static_cast<std::uint32_t>(std::stoul(argv[8])) : 1u;
        if (iterations == 0u) {
            throw std::invalid_argument("iterations must be positive");
        }
        const std::vector<float> input =
            read(argv[5], std::uint64_t(3) * size * size);
        const std::vector<float> reference =
            read(argv[6], std::uint64_t(size) * size);
        const auto create_start = std::chrono::steady_clock::now();
        zoedepth_status status = zoedepth_create_vulkan(
            argv[1], variant, device, &context);
        if (status != ZOEDEPTH_STATUS_OK) {
            throw std::runtime_error(zoedepth_last_error());
        }
        const double create_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - create_start).count();
        std::vector<float> output(reference.size());
        std::vector<double> samples;
        samples.reserve(iterations);
        for (std::uint32_t iteration = 0u;
             iteration < iterations; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            status = zoedepth_infer_rgb_f32(
                context, input.data(), static_cast<int32_t>(size),
                static_cast<int32_t>(size), output.data(), output.size());
            samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            if (status != ZOEDEPTH_STATUS_OK) {
                throw std::runtime_error(zoedepth_last_error());
            }
        }
        std::sort(samples.begin(), samples.end());
        double difference = 0.0;
        double magnitude = 0.0;
        float maximum = 0.0f;
        for (std::size_t index = 0; index < output.size(); ++index) {
            const float delta = std::abs(output[index] - reference[index]);
            difference += delta;
            magnitude += std::abs(reference[index]);
            maximum = std::max(maximum, delta);
        }
        const double relative = difference / magnitude;
        const auto bounds =
            std::minmax_element(output.begin(), output.end());
        double sum = 0.0;
        for (float value : output) {
            sum += value;
        }
        std::cout << "create_seconds=" << create_seconds
                  << "\niterations=" << iterations
                  << "\nmedian_ms=" << samples[samples.size() / 2u]
                  << "\nminimum=" << *bounds.first
                  << "\nmaximum=" << *bounds.second
                  << "\nmean=" << sum / output.size()
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative << "\n";
        zoedepth_destroy(context);
        return relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        zoedepth_destroy(context);
        std::cerr << error.what() << "\n";
        return 1;
    }
}
