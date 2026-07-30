#include "encoder_cpu.h"
#include "model.h"

#include <algorithm>
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
    if (argc != 8) {
        std::cerr
            << "usage: zoedepth_encoder_probe model.zoe size "
               "input.bin block5.bin block11.bin block17.bin "
               "block23.bin\n";
        return 2;
    }
    try {
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const std::uint32_t tokens =
            1 + (size / 16) * (size / 16);
        const std::uint64_t elements =
            std::uint64_t(tokens) * 1024;
        const std::vector<float> input = read_floats(
            argv[3], std::uint64_t(3) * size * size);
        zoe_native::ModelFile model(
            argv[1], zoe_native::Variant::n);
        const zoe_native::EncoderOutput output =
            zoe_native::encoder_cpu(
                model, input.data(), size, size);
        bool passed = output.captures.size() == 4;
        for (std::size_t capture = 0;
             capture < output.captures.size(); ++capture) {
            const std::vector<float> reference =
                read_floats(argv[4 + capture], elements);
            double absolute_sum = 0.0;
            double reference_sum = 0.0;
            float maximum = 0.0f;
            for (std::size_t index = 0;
                 index < reference.size(); ++index) {
                const float difference = std::abs(
                    output.captures[capture][index] -
                    reference[index]);
                maximum = std::max(maximum, difference);
                absolute_sum += difference;
                reference_sum += std::abs(reference[index]);
            }
            const double relative =
                absolute_sum /
                std::max(reference_sum, 1.0e-30);
            std::cout << "capture" << capture
                      << "_maximum_absolute=" << maximum
                      << "\ncapture" << capture
                      << "_relative_l1=" << relative << "\n";
            passed = passed && relative < 1.0e-4;
        }
        return passed ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
