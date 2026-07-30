#include "encoder_gpu.h"
#include "gpu_model.h"
#include "model.h"
#include "operators.h"
#include "vulkan.h"

#include <algorithm>
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
    if (argc != 7) {
        std::cerr << "usage: probe model device size input prefix tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t device =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const double tolerance = std::stod(argv[6]);
        const std::vector<float> input =
            read(argv[4], std::uint64_t(3) * size * size);
        zoe_native::ModelFile file(argv[1], zoe_native::Variant::n);
        zoe_native::VulkanContext context(device);
        zoe_native::GpuModel model(file, context);
        zoe_native::VulkanOperators operators(context);
        zoe_native::VulkanBuffer image =
            context.create_device_buffer(input.size() * sizeof(float));
        context.upload(image, input.data(), input.size() * sizeof(float));
        zoe_native::GpuEncoderOutput output =
            zoe_native::encoder_gpu(
                context, model, operators, image, size, size);
        const std::uint64_t count =
            std::uint64_t(output.patch_width * output.patch_height + 1) *
            1024;
        double worst = 0.0;
        const std::uint32_t blocks[4] = {5, 11, 17, 23};
        for (std::uint32_t level = 0; level < 4; ++level) {
            std::vector<float> actual(count);
            context.download(
                output.captures[level], actual.data(),
                actual.size() * sizeof(float));
            const std::vector<float> reference = read(
                std::string(argv[5]) + ".block" +
                    std::to_string(blocks[level]) + ".bin", count);
            double difference = 0.0;
            double magnitude = 0.0;
            float maximum = 0.0f;
            for (std::size_t index = 0; index < actual.size(); ++index) {
                const float delta =
                    std::abs(actual[index] - reference[index]);
                difference += delta;
                magnitude += std::abs(reference[index]);
                maximum = std::max(maximum, delta);
            }
            const double relative = difference / magnitude;
            worst = std::max(worst, relative);
            std::cout << "block" << blocks[level]
                      << "_relative_l1=" << relative
                      << "\nblock" << blocks[level]
                      << "_maximum_absolute=" << maximum << "\n";
        }
        return worst <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
