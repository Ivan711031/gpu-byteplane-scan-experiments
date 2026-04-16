#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace buff {

struct EncodedSegment {
    std::uint64_t value_count = 0;
    std::uint32_t fractional_bits = 0;
    std::uint32_t integer_offset_bits = 0;
    std::vector<std::uint8_t> integer_base_le;
    std::vector<std::vector<std::uint8_t>> byte_planes;
};

struct EncodedFileHeader {
    std::uint64_t total_values = 0;
    std::uint64_t segment_size = 0;
    std::uint64_t segment_count = 0;
};

EncodedSegment encode_segment(std::span<const double> values);
std::vector<double> decode_segment(const EncodedSegment& segment);
std::vector<double> decode_segment_top_k(const EncodedSegment& segment, std::size_t top_k_planes);
double segment_max_abs_error_bound(const EncodedSegment& segment, std::size_t top_k_planes);
std::size_t segment_plane_count(const EncodedSegment& segment);

void encode_file(const std::filesystem::path& input_path,
                 const std::filesystem::path& output_path,
                 std::uint64_t segment_size,
                 std::uint64_t max_values = 0);

void decode_file(const std::filesystem::path& input_path,
                 const std::filesystem::path& output_path);

EncodedFileHeader read_file_header(const std::filesystem::path& input_path);

bool files_have_identical_fp64_payload(const std::filesystem::path& lhs_path,
                                       const std::filesystem::path& rhs_path);

std::string summarize_segment(const EncodedSegment& segment);

}  // namespace buff
