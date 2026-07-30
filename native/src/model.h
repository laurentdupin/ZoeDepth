#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

namespace zoe_native {

enum class Variant : std::uint32_t { n = 0, k = 1, nk = 2 };

struct TensorView {
    const float* data = nullptr;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

struct Derivation {
    std::array<std::uint8_t, 32> canonical_sha256{};
    std::string converter;
    std::uint32_t format_version = 0;
    Variant variant = Variant::n;
};

class ModelFile {
public:
    ModelFile(const std::string& path_utf8, Variant expected_variant);
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;
    ~ModelFile();

    const TensorView& tensor(std::string_view name) const;
    bool contains(std::string_view name) const;
    std::size_t tensor_count() const { return tensors_.size(); }
    const Derivation& derivation() const { return derivation_; }

private:
    void close() noexcept;

#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int file_descriptor_ = -1;
#endif
    const std::byte* view_ = nullptr;
    std::uint64_t size_ = 0;
    std::unordered_map<std::string, TensorView> tensors_;
    Derivation derivation_;
};

}  // namespace zoe_native

