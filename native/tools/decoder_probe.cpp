#include "encoder_cpu.h"
#include "graph_cpu.h"
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

bool compare(
    const char* name,
    const zoe_native::Image& output,
    const std::string& path,
    double tolerance) {
    const std::vector<float> reference =
        read_floats(path, output.values.size());
    double absolute_sum = 0.0;
    double reference_sum = 0.0;
    float maximum = 0.0f;
    for (std::size_t index = 0;
         index < reference.size(); ++index) {
        const float difference =
            std::abs(output.values[index] - reference[index]);
        maximum = std::max(maximum, difference);
        absolute_sum += difference;
        reference_sum += std::abs(reference[index]);
    }
    const double relative =
        absolute_sum / std::max(reference_sum, 1.0e-30);
    std::cout << name << "_maximum_absolute=" << maximum
              << "\n" << name << "_relative_l1="
              << relative << "\n";
    return relative <= tolerance;
}
}

int main(int argc, char** argv) {
    if (argc != 11) {
        std::cerr
            << "usage: zoedepth_decoder_probe model.zoe size "
               "prepared.bin relative.bin out_conv.bin l4_rn.bin "
               "r4.bin r3.bin r2.bin r1.bin\n";
        return 2;
    }
    try {
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const std::vector<float> input = read_floats(
            argv[3], std::uint64_t(3) * size * size);
        zoe_native::ModelFile model(
            argv[1], zoe_native::Variant::n);
        const zoe_native::DecoderOutput output =
            zoe_native::midas_decoder_cpu(
                model, zoe_native::encoder_cpu(
                    model, input.data(), size, size));
        bool passed = true;
        passed &= compare(
            "relative", output.relative_depth, argv[4], 2.0e-4);
        passed &= compare(
            "out_conv", output.out_conv, argv[5], 2.0e-4);
        passed &= compare(
            "l4_rn", output.bottleneck, argv[6], 2.0e-4);
        for (std::size_t index = 0; index < 4; ++index) {
            passed &= compare(
                ("r" + std::to_string(4 - index)).c_str(),
                output.refinement_blocks[index],
                argv[7 + index], 2.0e-4);
        }
        return passed ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
