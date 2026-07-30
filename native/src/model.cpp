#include "model.h"

#include <cstring>
#include <limits>
#include <stdexcept>

#if !defined(_WIN32)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace zoe_native {
namespace {

constexpr char model_magic[8] = {
    'Z', 'O', 'E', 'N', 'M', 'O', 'D', '\0'};
constexpr char metadata_magic[8] = {
    'Z', 'O', 'E', 'N', 'M', 'E', 'T', 'A'};
constexpr std::uint32_t format_version = 1;
constexpr std::uint32_t endian_tag = 0x01020304;
constexpr std::uint32_t dtype_float32 = 1;
constexpr std::uint64_t alignment = 64;

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t endian;
    std::uint32_t variant;
    std::uint32_t tensor_count;
    std::uint64_t directory_offset;
    std::uint64_t directory_bytes;
    std::uint64_t data_offset;
    std::uint64_t file_bytes;
    std::uint64_t metadata_offset;
};

struct TensorRecord {
    char name[112];
    std::uint32_t dtype;
    std::uint32_t rank;
    std::uint64_t dimensions[4];
    std::uint64_t data_offset;
    std::uint64_t data_bytes;
    std::uint64_t element_count;
    std::uint32_t crc32;
    std::uint32_t flags;
    std::uint64_t reserved;
};

struct Metadata {
    char magic[8];
    std::uint32_t version;
    std::uint32_t bytes;
    std::uint32_t model_format_version;
    std::uint32_t variant;
    std::uint32_t flags;
    std::uint32_t reserved;
    std::uint8_t canonical_sha256[32];
    char converter[64];
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 64);
static_assert(sizeof(TensorRecord) == 192);
static_assert(sizeof(Metadata) == 128);

bool range_valid(
    std::uint64_t offset,
    std::uint64_t bytes,
    std::uint64_t limit) {
    return offset <= limit && bytes <= limit - offset;
}

#if defined(_WIN32)
std::wstring utf8_to_wide(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("model path is empty");
    }
    const int count = MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
        static_cast<int>(text.size()), nullptr, 0);
    if (count <= 0) {
        throw std::invalid_argument("model path is not valid UTF-8");
    }
    std::wstring result(static_cast<std::size_t>(count), L'\0');
    if (MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
            static_cast<int>(text.size()), result.data(), count) != count) {
        throw std::invalid_argument("failed to decode model path");
    }
    return result;
}
#endif

}  // namespace

ModelFile::ModelFile(
    const std::string& path_utf8,
    Variant expected_variant) {
    try {
#if defined(_WIN32)
        const std::wstring path = utf8_to_wide(path_utf8);
        file_ = CreateFileW(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("failed to open model file");
        }
        LARGE_INTEGER file_size{};
        if (!GetFileSizeEx(file_, &file_size) ||
            file_size.QuadPart < 0) {
            throw std::runtime_error("failed to query model file size");
        }
        size_ = static_cast<std::uint64_t>(file_size.QuadPart);
        mapping_ = CreateFileMappingW(
            file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) {
            throw std::runtime_error("failed to map model file");
        }
        view_ = static_cast<const std::byte*>(
            MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!view_) {
            throw std::runtime_error("failed to view model file");
        }
#else
        if (path_utf8.empty()) {
            throw std::invalid_argument("model path is empty");
        }
        file_descriptor_ = open(
            path_utf8.c_str(), O_RDONLY | O_CLOEXEC);
        if (file_descriptor_ < 0) {
            throw std::runtime_error("failed to open model file");
        }
        struct stat status {};
        if (fstat(file_descriptor_, &status) != 0 ||
            status.st_size < 0) {
            throw std::runtime_error("failed to query model file size");
        }
        size_ = static_cast<std::uint64_t>(status.st_size);
        if (size_ >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("model is too large to map");
        }
        void* mapped = mmap(
            nullptr, static_cast<std::size_t>(size_), PROT_READ,
            MAP_PRIVATE, file_descriptor_, 0);
        if (mapped == MAP_FAILED) {
            throw std::runtime_error("failed to map model file");
        }
        view_ = static_cast<const std::byte*>(mapped);
#endif
        if (size_ < sizeof(FileHeader)) {
            throw std::runtime_error("model file is truncated");
        }
        const auto& header =
            *reinterpret_cast<const FileHeader*>(view_);
        if (std::memcmp(
                header.magic, model_magic, sizeof(model_magic)) != 0 ||
            header.version != format_version ||
            header.endian != endian_tag ||
            header.variant !=
                static_cast<std::uint32_t>(expected_variant) ||
            header.tensor_count == 0 ||
            header.tensor_count > 2048 ||
            header.file_bytes != size_) {
            throw std::runtime_error("invalid ZoeDepth model header");
        }
        const std::uint64_t expected_directory =
            std::uint64_t(header.tensor_count) * sizeof(TensorRecord);
        if (header.directory_offset != sizeof(FileHeader) ||
            header.directory_bytes != expected_directory ||
            !range_valid(
                header.directory_offset, header.directory_bytes, size_) ||
            header.data_offset % alignment != 0 ||
            header.data_offset > size_ ||
            header.metadata_offset !=
                header.directory_offset + header.directory_bytes ||
            !range_valid(
                header.metadata_offset, sizeof(Metadata),
                header.data_offset)) {
            throw std::runtime_error(
                "invalid ZoeDepth model directory");
        }
        const auto& metadata =
            *reinterpret_cast<const Metadata*>(
                view_ + header.metadata_offset);
        if (std::memcmp(
                metadata.magic, metadata_magic,
                sizeof(metadata_magic)) != 0 ||
            metadata.version != 1 ||
            metadata.bytes != sizeof(Metadata) ||
            metadata.model_format_version != format_version ||
            metadata.variant != header.variant ||
            metadata.flags != 0 || metadata.reserved != 0 ||
            std::memchr(
                metadata.converter, '\0',
                sizeof(metadata.converter)) == nullptr) {
            throw std::runtime_error(
                "invalid ZoeDepth derivation metadata");
        }
        std::memcpy(
            derivation_.canonical_sha256.data(),
            metadata.canonical_sha256,
            derivation_.canonical_sha256.size());
        derivation_.converter = metadata.converter;
        derivation_.format_version =
            metadata.model_format_version;
        derivation_.variant =
            static_cast<Variant>(metadata.variant);

        tensors_.reserve(header.tensor_count);
        const auto* records = reinterpret_cast<const TensorRecord*>(
            view_ + header.directory_offset);
        std::uint64_t previous_end = header.data_offset;
        for (std::uint32_t index = 0;
             index < header.tensor_count; ++index) {
            const TensorRecord& record = records[index];
            const auto* end = static_cast<const char*>(
                std::memchr(record.name, '\0', sizeof(record.name)));
            if (!end || record.name[0] == '\0' ||
                record.dtype != dtype_float32 ||
                record.rank == 0 || record.rank > 4 ||
                record.flags != 0 || record.reserved != 0 ||
                record.data_offset < previous_end ||
                record.data_offset % alignment != 0 ||
                !range_valid(
                    record.data_offset, record.data_bytes, size_)) {
                throw std::runtime_error(
                    "invalid ZoeDepth tensor record");
            }
            std::uint64_t elements = 1;
            for (std::uint32_t dimension = 0;
                 dimension < 4; ++dimension) {
                const std::uint64_t value =
                    record.dimensions[dimension];
                if (dimension < record.rank) {
                    if (value == 0 ||
                        elements >
                            std::numeric_limits<std::uint64_t>::max() /
                                value) {
                        throw std::runtime_error(
                            "invalid ZoeDepth tensor dimensions");
                    }
                    elements *= value;
                } else if (value != 0) {
                    throw std::runtime_error(
                        "invalid unused tensor dimension");
                }
            }
            if (elements != record.element_count ||
                elements >
                    std::numeric_limits<std::uint64_t>::max() /
                        sizeof(float) ||
                elements * sizeof(float) != record.data_bytes) {
                throw std::runtime_error(
                    "invalid ZoeDepth tensor byte count");
            }
            const std::string name(
                record.name,
                static_cast<std::size_t>(end - record.name));
            TensorView tensor{
                reinterpret_cast<const float*>(
                    view_ + record.data_offset),
                {record.dimensions[0], record.dimensions[1],
                 record.dimensions[2], record.dimensions[3]},
                record.rank, record.element_count};
            if (!tensors_.emplace(name, tensor).second) {
                throw std::runtime_error(
                    "duplicate ZoeDepth tensor name");
            }
            previous_end = record.data_offset + record.data_bytes;
        }
    } catch (...) {
        close();
        throw;
    }
}

ModelFile::~ModelFile() {
    close();
}

void ModelFile::close() noexcept {
    tensors_.clear();
#if defined(_WIN32)
    if (view_) {
        UnmapViewOfFile(view_);
        view_ = nullptr;
    }
    if (mapping_) {
        CloseHandle(mapping_);
        mapping_ = nullptr;
    }
    if (file_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
    }
#else
    if (view_) {
        munmap(
            const_cast<std::byte*>(view_),
            static_cast<std::size_t>(size_));
        view_ = nullptr;
    }
    if (file_descriptor_ >= 0) {
        ::close(file_descriptor_);
        file_descriptor_ = -1;
    }
#endif
    size_ = 0;
}

const TensorView& ModelFile::tensor(std::string_view name) const {
    const auto found = tensors_.find(std::string(name));
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "model is missing tensor: " + std::string(name));
    }
    return found->second;
}

bool ModelFile::contains(std::string_view name) const {
    return tensors_.find(std::string(name)) != tensors_.end();
}

std::vector<std::string_view> ModelFile::tensor_names() const {
    std::vector<std::string_view> result;
    result.reserve(tensors_.size());
    for (const auto& entry : tensors_) {
        result.emplace_back(entry.first);
    }
    return result;
}

}  // namespace zoe_native
