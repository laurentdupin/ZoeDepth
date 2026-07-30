#include "gpu_model.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace zoe_native {

GpuModel::GpuModel(const ModelFile& model, VulkanContext& context) {
    variant_ = model.derivation().variant;
    tensors_.reserve(model.tensor_count());
    for (std::string_view name : model.tensor_names()) {
        const TensorView& source = model.tensor(name);
        if (source.elements >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error(
                "model tensor is too large: " + std::string(name));
        }
        const std::size_t bytes =
            static_cast<std::size_t>(source.elements) * sizeof(float);
        GpuTensor destination{
            context.create_device_buffer(bytes),
            source.dimensions,
            source.rank,
            source.elements,
        };
        context.upload(destination.buffer, source.data, bytes);
        if (!tensors_.emplace(name, std::move(destination)).second) {
            throw std::runtime_error(
                "duplicate GPU tensor name: " + std::string(name));
        }
    }
}

const GpuTensor& GpuModel::tensor(std::string_view name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "GPU model is missing tensor: " + std::string(name));
    }
    return found->second;
}

}  // namespace zoe_native
