#include "model.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: zoedepth_model_probe model.zoe\n";
        return 2;
    }
    try {
        zoe_native::ModelFile model(
            argv[1], zoe_native::Variant::n);
        const auto& patch = model.tensor(
            "core.core.pretrained.model.patch_embed.proj.weight");
        const auto& derivation = model.derivation();
        const bool expected =
            model.tensor_count() == 510 &&
            patch.rank == 4 &&
            patch.dimensions[0] == 1024 &&
            patch.dimensions[1] == 3 &&
            patch.dimensions[2] == 16 &&
            patch.dimensions[3] == 16 &&
            model.contains("conditional_log_binomial.mlp.2.weight") &&
            derivation.converter ==
                "zoedepth-export-pytorch-v1";
        std::cout << "tensors=" << model.tensor_count()
                  << "\npatch_shape="
                  << patch.dimensions[0] << "x"
                  << patch.dimensions[1] << "x"
                  << patch.dimensions[2] << "x"
                  << patch.dimensions[3]
                  << "\ncanonical_sha256=";
        for (const std::uint8_t value :
             derivation.canonical_sha256) {
            std::cout << std::hex << std::setw(2)
                      << std::setfill('0')
                      << static_cast<unsigned>(value);
        }
        std::cout << "\n";
        return expected ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}

