#include "buff_codec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace buff {
namespace {

constexpr std::array<char, 8> kMagic = {'B', 'U', 'F', 'F', '6', '4', '\0', '\2'};

struct DyadicValue {
    bool is_zero = true;
    std::uint64_t significand = 0;
    std::int32_t exponent = 0;
};

template <typename UInt>
void write_pod(std::ostream& out, UInt value) {
    static_assert(std::is_integral_v<UInt>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!out) {
        throw std::runtime_error("failed to write output stream");
    }
}

template <typename UInt>
UInt read_pod(std::istream& in) {
    static_assert(std::is_integral_v<UInt>);
    UInt value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("failed to read input stream");
    }
    return value;
}

void normalize_le(std::vector<std::uint8_t>& bytes) {
    while (!bytes.empty() && bytes.back() == 0) {
        bytes.pop_back();
    }
}

bool is_zero_le(const std::vector<std::uint8_t>& bytes) {
    return std::all_of(bytes.begin(), bytes.end(), [](std::uint8_t byte) { return byte == 0; });
}

std::uint64_t bit_width_le(const std::vector<std::uint8_t>& bytes) {
    for (std::size_t index = bytes.size(); index > 0; --index) {
        std::uint8_t byte = bytes[index - 1];
        if (byte != 0) {
            return static_cast<std::uint64_t>((index - 1) * 8U + (8U - std::countl_zero(byte)));
        }
    }
    return 0;
}

int compare_le(const std::vector<std::uint8_t>& lhs, const std::vector<std::uint8_t>& rhs) {
    std::size_t lhs_size = lhs.size();
    while (lhs_size > 0 && lhs[lhs_size - 1] == 0) {
        --lhs_size;
    }

    std::size_t rhs_size = rhs.size();
    while (rhs_size > 0 && rhs[rhs_size - 1] == 0) {
        --rhs_size;
    }

    if (lhs_size != rhs_size) {
        return lhs_size < rhs_size ? -1 : 1;
    }

    for (std::size_t index = lhs_size; index > 0; --index) {
        if (lhs[index - 1] != rhs[index - 1]) {
            return lhs[index - 1] < rhs[index - 1] ? -1 : 1;
        }
    }

    return 0;
}

void add_le_into(const std::vector<std::uint8_t>& lhs,
                 const std::vector<std::uint8_t>& rhs,
                 std::vector<std::uint8_t>& sum) {
    std::size_t max_size = std::max(lhs.size(), rhs.size());
    sum.assign(max_size + 1, 0);

    std::uint16_t carry = 0;
    for (std::size_t index = 0; index < max_size; ++index) {
        std::uint16_t total = carry;
        if (index < lhs.size()) {
            total = static_cast<std::uint16_t>(total + lhs[index]);
        }
        if (index < rhs.size()) {
            total = static_cast<std::uint16_t>(total + rhs[index]);
        }
        sum[index] = static_cast<std::uint8_t>(total & 0xffU);
        carry = static_cast<std::uint16_t>(total >> 8U);
    }

    sum[max_size] = static_cast<std::uint8_t>(carry);
    normalize_le(sum);
}

std::vector<std::uint8_t> subtract_le(const std::vector<std::uint8_t>& lhs,
                                      const std::vector<std::uint8_t>& rhs) {
    if (compare_le(lhs, rhs) < 0) {
        throw std::runtime_error("attempted negative subtraction in unsigned code space");
    }

    std::vector<std::uint8_t> diff(lhs.size(), 0);
    int borrow = 0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        int value = static_cast<int>(lhs[index]) - borrow;
        if (index < rhs.size()) {
            value -= static_cast<int>(rhs[index]);
        }
        if (value < 0) {
            value += 256;
            borrow = 1;
        } else {
            borrow = 0;
        }
        diff[index] = static_cast<std::uint8_t>(value);
    }

    if (borrow != 0) {
        throw std::runtime_error("subtraction underflow in unsigned code space");
    }

    normalize_le(diff);
    return diff;
}

std::vector<std::uint8_t> shift_significand_to_le(std::uint64_t significand, std::uint64_t shift_bits) {
    if (significand == 0) {
        return {};
    }

    std::uint64_t byte_shift = shift_bits / 8U;
    std::uint32_t bit_shift = static_cast<std::uint32_t>(shift_bits % 8U);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(byte_shift) + sizeof(significand) + 1U, 0);

    std::uint16_t carry = 0;
    for (std::size_t byte_index = 0; byte_index < sizeof(significand); ++byte_index) {
        std::uint16_t chunk = static_cast<std::uint16_t>((significand >> (byte_index * 8U)) & 0xffU);
        std::uint16_t shifted = static_cast<std::uint16_t>((chunk << bit_shift) | carry);
        bytes[static_cast<std::size_t>(byte_shift) + byte_index] = static_cast<std::uint8_t>(shifted & 0xffU);
        carry = static_cast<std::uint16_t>(shifted >> 8U);
    }
    bytes[static_cast<std::size_t>(byte_shift) + sizeof(significand)] = static_cast<std::uint8_t>(carry);
    normalize_le(bytes);
    return bytes;
}

std::vector<std::uint8_t> left_shift_le(const std::vector<std::uint8_t>& bytes, std::uint64_t shift_bits) {
    if (bytes.empty() || shift_bits == 0) {
        return bytes;
    }

    std::uint64_t byte_shift = shift_bits / 8U;
    std::uint32_t bit_shift = static_cast<std::uint32_t>(shift_bits % 8U);
    std::vector<std::uint8_t> shifted(bytes.size() + static_cast<std::size_t>(byte_shift) + 1U, 0);

    std::uint16_t carry = 0;
    for (std::size_t byte_index = 0; byte_index < bytes.size(); ++byte_index) {
        std::uint16_t chunk = bytes[byte_index];
        std::uint16_t value = static_cast<std::uint16_t>((chunk << bit_shift) | carry);
        shifted[static_cast<std::size_t>(byte_shift) + byte_index] = static_cast<std::uint8_t>(value & 0xffU);
        carry = static_cast<std::uint16_t>(value >> 8U);
    }
    shifted[static_cast<std::size_t>(byte_shift) + bytes.size()] = static_cast<std::uint8_t>(carry);
    normalize_le(shifted);
    return shifted;
}

std::vector<std::uint8_t> right_shift_le(const std::vector<std::uint8_t>& bytes, std::uint64_t shift_bits) {
    if (bytes.empty()) {
        return {};
    }
    if (shift_bits == 0) {
        return bytes;
    }

    std::uint64_t byte_shift = shift_bits / 8U;
    std::uint32_t bit_shift = static_cast<std::uint32_t>(shift_bits % 8U);
    if (byte_shift >= bytes.size()) {
        return {};
    }

    std::vector<std::uint8_t> shifted(bytes.size() - static_cast<std::size_t>(byte_shift), 0);
    for (std::size_t out_index = 0; out_index < shifted.size(); ++out_index) {
        std::size_t source_index = out_index + static_cast<std::size_t>(byte_shift);
        std::uint16_t value = static_cast<std::uint16_t>(bytes[source_index] >> bit_shift);
        if (bit_shift != 0 && source_index + 1 < bytes.size()) {
            value = static_cast<std::uint16_t>(
                value | static_cast<std::uint16_t>(bytes[source_index + 1] << (8U - bit_shift)));
        }
        shifted[out_index] = static_cast<std::uint8_t>(value & 0xffU);
    }
    normalize_le(shifted);
    return shifted;
}

std::vector<std::uint8_t> mask_low_bits_le(const std::vector<std::uint8_t>& bytes, std::uint64_t bit_count) {
    if (bit_count == 0 || bytes.empty()) {
        return {};
    }

    std::size_t full_bytes = static_cast<std::size_t>(bit_count / 8U);
    std::uint32_t remainder = static_cast<std::uint32_t>(bit_count % 8U);
    std::size_t copy_bytes = full_bytes + (remainder == 0 ? 0U : 1U);

    std::vector<std::uint8_t> masked(copy_bytes, 0);
    for (std::size_t index = 0; index < std::min(copy_bytes, bytes.size()); ++index) {
        masked[index] = bytes[index];
    }
    if (remainder != 0 && !masked.empty()) {
        std::uint8_t keep_mask = static_cast<std::uint8_t>((UINT8_C(1) << remainder) - 1U);
        masked.back() = static_cast<std::uint8_t>(masked.back() & keep_mask);
    }
    normalize_le(masked);
    return masked;
}

bool bit_is_set_le(const std::vector<std::uint8_t>& bytes, std::uint64_t bit_index) {
    std::size_t byte_index = static_cast<std::size_t>(bit_index / 8U);
    if (byte_index >= bytes.size()) {
        return false;
    }
    return ((bytes[byte_index] >> (bit_index % 8U)) & 1U) != 0;
}

void set_bit_le(std::vector<std::uint8_t>& bytes, std::uint64_t bit_index, bool value) {
    std::size_t byte_index = static_cast<std::size_t>(bit_index / 8U);
    if (byte_index >= bytes.size()) {
        throw std::runtime_error("bit write exceeded destination width");
    }
    std::uint8_t mask = static_cast<std::uint8_t>(UINT8_C(1) << (bit_index % 8U));
    if (value) {
        bytes[byte_index] = static_cast<std::uint8_t>(bytes[byte_index] | mask);
    } else {
        bytes[byte_index] = static_cast<std::uint8_t>(bytes[byte_index] & ~mask);
    }
}

bool any_bits_below_le(const std::vector<std::uint8_t>& bytes, std::uint64_t bit_limit) {
    std::size_t whole_bytes = static_cast<std::size_t>(bit_limit / 8U);
    for (std::size_t index = 0; index < std::min(whole_bytes, bytes.size()); ++index) {
        if (bytes[index] != 0) {
            return true;
        }
    }

    std::uint32_t remainder = static_cast<std::uint32_t>(bit_limit % 8U);
    if (remainder == 0 || whole_bytes >= bytes.size()) {
        return false;
    }

    std::uint8_t mask = static_cast<std::uint8_t>((UINT8_C(1) << remainder) - 1U);
    return (bytes[whole_bytes] & mask) != 0;
}

std::uint64_t extract_u64_window_le(const std::vector<std::uint8_t>& bytes,
                                    std::uint64_t start_bit,
                                    std::uint32_t width_bits) {
    if (width_bits > 64) {
        throw std::runtime_error("cannot extract more than 64 bits into u64");
    }

    std::uint64_t value = 0;
    for (std::uint32_t offset = 0; offset < width_bits; ++offset) {
        if (bit_is_set_le(bytes, start_bit + offset)) {
            value |= (UINT64_C(1) << offset);
        }
    }
    return value;
}

std::uint64_t bytes_to_u64_le(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() > sizeof(std::uint64_t)) {
        throw std::runtime_error("integer magnitude does not fit in u64");
    }

    std::uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }
    return value;
}

std::uint64_t trailing_zero_bits_le(const std::vector<std::uint8_t>& bytes) {
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        if (bytes[index] != 0) {
            return static_cast<std::uint64_t>(index * 8U + std::countr_zero(bytes[index]));
        }
    }
    return 0;
}

std::size_t plane_count_for_bits(std::size_t total_bits) {
    return total_bits == 0 ? 0 : (total_bits + 7U) / 8U;
}

std::size_t plane_bit_width(std::size_t total_bits, std::size_t plane_index) {
    std::size_t plane_count = plane_count_for_bits(total_bits);
    if (plane_index >= plane_count) {
        throw std::runtime_error("plane index is out of range");
    }
    if (plane_index + 1U == plane_count) {
        std::size_t trailing = total_bits - 8U * (plane_count - 1U);
        return trailing == 0 ? 8U : trailing;
    }
    return 8U;
}

std::size_t plane_lsb_start(std::size_t total_bits, std::size_t plane_index) {
    std::size_t width = plane_bit_width(total_bits, plane_index);
    return total_bits - 8U * plane_index - width;
}

std::size_t segment_total_bit_width(const EncodedSegment& segment) {
    return static_cast<std::size_t>(segment.fractional_bits) +
           static_cast<std::size_t>(segment.integer_offset_bits);
}

bool combined_bit_is_set(const std::vector<std::uint8_t>& integer_offset_le,
                         const std::vector<std::uint8_t>& fractional_le,
                         std::uint32_t fractional_bits,
                         std::uint64_t bit_index) {
    if (bit_index < fractional_bits) {
        return bit_is_set_le(fractional_le, bit_index);
    }
    return bit_is_set_le(integer_offset_le, bit_index - fractional_bits);
}

std::uint8_t extract_combined_chunk(const std::vector<std::uint8_t>& integer_offset_le,
                                    const std::vector<std::uint8_t>& fractional_le,
                                    std::uint32_t fractional_bits,
                                    std::uint64_t start_bit,
                                    std::uint32_t width_bits) {
    std::uint8_t value = 0;
    for (std::uint32_t offset = 0; offset < width_bits; ++offset) {
        if (combined_bit_is_set(integer_offset_le, fractional_le, fractional_bits, start_bit + offset)) {
            value = static_cast<std::uint8_t>(value | (UINT8_C(1) << offset));
        }
    }
    return value;
}

std::vector<std::uint8_t> assemble_combined_from_planes(const EncodedSegment& segment,
                                                        std::uint64_t row,
                                                        std::size_t keep_planes) {
    std::size_t total_bits = segment_total_bit_width(segment);
    std::vector<std::uint8_t> combined((total_bits + 7U) / 8U, 0);
    std::size_t keep = std::min<std::size_t>(keep_planes, segment.byte_planes.size());

    for (std::size_t plane = 0; plane < keep; ++plane) {
        if (segment.byte_planes[plane].size() != segment.value_count) {
            throw std::runtime_error("corrupt segment: plane length mismatch");
        }
        std::uint8_t value = segment.byte_planes[plane][static_cast<std::size_t>(row)];
        std::uint32_t width = static_cast<std::uint32_t>(plane_bit_width(total_bits, plane));
        if (width < 8U && (value >> width) != 0U) {
            throw std::runtime_error("corrupt segment: trailing plane uses bits beyond its declared width");
        }
        std::uint64_t start_bit = plane_lsb_start(total_bits, plane);
        for (std::uint32_t bit = 0; bit < width; ++bit) {
            if (((value >> bit) & 1U) != 0U) {
                set_bit_le(combined, start_bit + bit, true);
            }
        }
    }

    return combined;
}

DyadicValue decompose_non_negative(double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("encoder only supports finite FP64 values");
    }

    std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    if ((bits >> 63U) != 0) {
        throw std::runtime_error("encoder currently supports unsigned/non-negative FP64 values only");
    }

    std::uint64_t exponent_bits = (bits >> 52U) & 0x7ffU;
    std::uint64_t mantissa_bits = bits & ((UINT64_C(1) << 52U) - 1U);

    if (exponent_bits == 0) {
        if (mantissa_bits == 0) {
            return {};
        }
        return DyadicValue{
            .is_zero = false,
            .significand = mantissa_bits,
            .exponent = -1074,
        };
    }

    return DyadicValue{
        .is_zero = false,
        .significand = (UINT64_C(1) << 52U) | mantissa_bits,
        .exponent = static_cast<std::int32_t>(exponent_bits) - 1023 - 52,
    };
}

std::vector<std::uint8_t> encode_value_to_code(double value, std::int32_t scale_exponent) {
    DyadicValue dyadic = decompose_non_negative(value);
    if (dyadic.is_zero) {
        return {};
    }

    std::int64_t shift_bits = static_cast<std::int64_t>(dyadic.exponent) - scale_exponent;
    if (shift_bits < 0) {
        throw std::runtime_error("value exponent fell below segment scale");
    }

    return shift_significand_to_le(dyadic.significand, static_cast<std::uint64_t>(shift_bits));
}

double code_to_rounded_double(const std::vector<std::uint8_t>& code_le, std::int32_t scale_exponent) {
    if (is_zero_le(code_le)) {
        return 0.0;
    }

    std::uint64_t trailing_zeros = trailing_zero_bits_le(code_le);
    std::vector<std::uint8_t> reduced = right_shift_le(code_le, trailing_zeros);
    std::int32_t reduced_scale = static_cast<std::int32_t>(scale_exponent + trailing_zeros);
    std::uint64_t width = bit_width_le(reduced);
    std::int64_t total_exponent = static_cast<std::int64_t>(reduced_scale) + static_cast<std::int64_t>(width) - 1;

    if (total_exponent > 1023) {
        return std::numeric_limits<double>::infinity();
    }

    if (total_exponent >= -1022) {
        std::uint64_t significand = 0;
        if (width <= 53U) {
            significand = bytes_to_u64_le(reduced) << static_cast<std::uint32_t>(53U - width);
        } else {
            std::uint64_t shift = width - 53U;
            significand = extract_u64_window_le(reduced, shift, 53);
            bool round_up = false;
            bool round_bit = bit_is_set_le(reduced, shift - 1U);
            bool sticky = any_bits_below_le(reduced, shift - 1U);
            if (round_bit && (sticky || (significand & 1U) != 0U)) {
                round_up = true;
            }
            if (round_up) {
                significand += 1U;
                if (significand == (UINT64_C(1) << 53U)) {
                    significand >>= 1U;
                    total_exponent += 1;
                    if (total_exponent > 1023) {
                        return std::numeric_limits<double>::infinity();
                    }
                }
            }
        }

        std::uint64_t biased_exponent = static_cast<std::uint64_t>(total_exponent + 1023);
        std::uint64_t mantissa = significand - (UINT64_C(1) << 52U);
        std::uint64_t bits = (biased_exponent << 52U) | mantissa;
        return std::bit_cast<double>(bits);
    }

    std::int64_t shift = static_cast<std::int64_t>(reduced_scale) + 1074;
    std::uint64_t mantissa = 0;
    if (shift >= 0) {
        if (width + static_cast<std::uint64_t>(shift) > 52U) {
            std::uint64_t bits = UINT64_C(1) << 52U;
            return std::bit_cast<double>(bits);
        }
        mantissa = bytes_to_u64_le(reduced) << static_cast<std::uint32_t>(shift);
    } else {
        std::uint64_t right_shift = static_cast<std::uint64_t>(-shift);
        if (right_shift >= width + 2U) {
            return 0.0;
        }
        std::uint64_t kept_width = width > right_shift ? width - right_shift : 0U;
        mantissa = kept_width == 0 ? 0 : extract_u64_window_le(
                                          reduced,
                                          right_shift,
                                          static_cast<std::uint32_t>(std::min<std::uint64_t>(kept_width, 52U)));
        bool round_up = false;
        if (right_shift > 0) {
            bool round_bit = bit_is_set_le(reduced, right_shift - 1U);
            bool sticky = any_bits_below_le(reduced, right_shift - 1U);
            if (round_bit && (sticky || (mantissa & 1U) != 0U)) {
                round_up = true;
            }
        }
        if (round_up) {
            mantissa += 1U;
        }
    }

    if (mantissa == 0) {
        return 0.0;
    }
    if (mantissa >= (UINT64_C(1) << 52U)) {
        std::uint64_t bits = UINT64_C(1) << 52U;
        return std::bit_cast<double>(bits);
    }

    std::uint64_t bits = mantissa;
    return std::bit_cast<double>(bits);
}

double code_to_double(const std::vector<std::uint8_t>& code_le, std::int32_t scale_exponent) {
    if (is_zero_le(code_le)) {
        return 0.0;
    }

    std::uint64_t trailing_zeros = trailing_zero_bits_le(code_le);
    std::vector<std::uint8_t> reduced = right_shift_le(code_le, trailing_zeros);
    std::int32_t reduced_scale = static_cast<std::int32_t>(scale_exponent + trailing_zeros);
    std::uint64_t width = bit_width_le(reduced);
    if (width == 0) {
        return 0.0;
    }
    if (width > 53U) {
        throw std::runtime_error("decoded significand exceeded exact FP64 precision");
    }

    std::uint64_t significand = bytes_to_u64_le(reduced);
    std::int64_t total_exponent = static_cast<std::int64_t>(reduced_scale) + static_cast<std::int64_t>(width) - 1;

    if (total_exponent >= -1022) {
        significand <<= static_cast<std::uint32_t>(53U - width);
        std::uint64_t biased_exponent = static_cast<std::uint64_t>(total_exponent + 1023);
        std::uint64_t mantissa = significand - (UINT64_C(1) << 52U);
        std::uint64_t bits = (biased_exponent << 52U) | mantissa;
        return std::bit_cast<double>(bits);
    }

    std::int64_t shift = static_cast<std::int64_t>(reduced_scale) + 1074;
    if (shift < 0) {
        throw std::runtime_error("decoded subnormal underflowed below FP64 range");
    }

    std::uint64_t mantissa = significand << static_cast<std::uint32_t>(shift);
    if (mantissa == 0 || mantissa >= (UINT64_C(1) << 52U)) {
        throw std::runtime_error("decoded subnormal mantissa is out of range");
    }

    std::uint64_t bits = mantissa;
    return std::bit_cast<double>(bits);
}

std::vector<double> materialize_segment(const EncodedSegment& segment, std::size_t keep_planes, bool exact) {
    std::vector<double> values(segment.value_count, 0.0);
    if (segment.value_count == 0) {
        return values;
    }

    std::int32_t scale_exponent = -static_cast<std::int32_t>(segment.fractional_bits);
    for (std::uint64_t row = 0; row < segment.value_count; ++row) {
        std::vector<std::uint8_t> combined = assemble_combined_from_planes(segment, row, keep_planes);
        std::vector<std::uint8_t> integer_offset = right_shift_le(combined, segment.fractional_bits);
        std::vector<std::uint8_t> fractional = mask_low_bits_le(combined, segment.fractional_bits);
        std::vector<std::uint8_t> integer_value;
        add_le_into(segment.integer_base_le, integer_offset, integer_value);
        std::vector<std::uint8_t> integer_code = left_shift_le(integer_value, segment.fractional_bits);
        std::vector<std::uint8_t> full_code;
        add_le_into(integer_code, fractional, full_code);
        values[static_cast<std::size_t>(row)] = exact ? code_to_double(full_code, scale_exponent)
                                                      : code_to_rounded_double(full_code, scale_exponent);
    }

    return values;
}

void write_bytes(std::ostream& out, const std::vector<std::uint8_t>& bytes) {
    if (!bytes.empty()) {
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            throw std::runtime_error("failed to write byte payload");
        }
    }
}

std::vector<std::uint8_t> read_bytes(std::istream& in, std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0);
    if (size == 0) {
        return bytes;
    }

    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("failed to read byte payload");
    }
    return bytes;
}

EncodedFileHeader read_stream_header(std::istream& input) {
    std::array<char, kMagic.size()> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!input) {
        throw std::runtime_error("failed to read BUFF header");
    }
    if (magic != kMagic) {
        throw std::runtime_error("input is not a supported BUFF64 file");
    }

    return EncodedFileHeader{
        .total_values = read_pod<std::uint64_t>(input),
        .segment_size = read_pod<std::uint64_t>(input),
        .segment_count = read_pod<std::uint64_t>(input),
    };
}

void write_stream_header(std::ostream& output, const EncodedFileHeader& header) {
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    if (!output) {
        throw std::runtime_error("failed to write BUFF header");
    }
    write_pod(output, header.total_values);
    write_pod(output, header.segment_size);
    write_pod(output, header.segment_count);
}

void write_segment(std::ostream& out, const EncodedSegment& segment) {
    write_pod(out, segment.value_count);
    write_pod(out, segment.fractional_bits);
    write_pod(out, segment.integer_offset_bits);
    write_pod(out, static_cast<std::uint32_t>(segment.integer_base_le.size()));
    write_pod(out, static_cast<std::uint32_t>(segment.byte_planes.size()));
    write_bytes(out, segment.integer_base_le);

    for (const auto& plane : segment.byte_planes) {
        if (plane.size() != segment.value_count) {
            throw std::runtime_error("byte plane length does not match segment length");
        }
        write_bytes(out, plane);
    }
}

EncodedSegment read_segment(std::istream& in) {
    EncodedSegment segment;
    segment.value_count = read_pod<std::uint64_t>(in);
    segment.fractional_bits = read_pod<std::uint32_t>(in);
    segment.integer_offset_bits = read_pod<std::uint32_t>(in);
    std::uint32_t base_width = read_pod<std::uint32_t>(in);
    std::uint32_t plane_count = read_pod<std::uint32_t>(in);
    segment.integer_base_le = read_bytes(in, base_width);
    segment.byte_planes.reserve(plane_count);
    for (std::uint32_t plane = 0; plane < plane_count; ++plane) {
        segment.byte_planes.push_back(read_bytes(in, static_cast<std::size_t>(segment.value_count)));
    }
    return segment;
}

std::uint64_t count_values_in_fp64_file(const std::filesystem::path& input_path) {
    std::uint64_t file_size = std::filesystem::file_size(input_path);
    if (file_size % sizeof(double) != 0) {
        throw std::runtime_error("input file size is not aligned to FP64 rows");
    }
    return file_size / sizeof(double);
}

}  // namespace

EncodedSegment encode_segment(std::span<const double> values) {
    EncodedSegment segment;
    segment.value_count = values.size();
    if (values.empty()) {
        return segment;
    }

    std::uint32_t fractional_bits = 0;
    std::vector<std::vector<std::uint8_t>> integer_parts;
    std::vector<std::vector<std::uint8_t>> fractional_parts;
    integer_parts.reserve(values.size());
    fractional_parts.reserve(values.size());

    for (double value : values) {
        DyadicValue dyadic = decompose_non_negative(value);
        if (!dyadic.is_zero && dyadic.exponent < 0) {
            fractional_bits = std::max<std::uint32_t>(
                fractional_bits, static_cast<std::uint32_t>(-dyadic.exponent));
        }
    }
    segment.fractional_bits = fractional_bits;

    std::int32_t scale_exponent = -static_cast<std::int32_t>(fractional_bits);
    bool base_set = false;
    std::vector<std::uint8_t> integer_base;

    for (double value : values) {
        std::vector<std::uint8_t> code = encode_value_to_code(value, scale_exponent);
        std::vector<std::uint8_t> integer_part = right_shift_le(code, fractional_bits);
        std::vector<std::uint8_t> fractional_part = mask_low_bits_le(code, fractional_bits);
        if (!base_set || compare_le(integer_part, integer_base) < 0) {
            integer_base = integer_part;
            base_set = true;
        }
        integer_parts.push_back(std::move(integer_part));
        fractional_parts.push_back(std::move(fractional_part));
    }

    segment.integer_base_le = integer_base;

    std::vector<std::vector<std::uint8_t>> integer_offsets;
    integer_offsets.reserve(values.size());
    std::uint32_t max_integer_offset_bits = 0;
    for (const auto& integer_part : integer_parts) {
        std::vector<std::uint8_t> offset = subtract_le(integer_part, segment.integer_base_le);
        max_integer_offset_bits = std::max<std::uint32_t>(
            max_integer_offset_bits, static_cast<std::uint32_t>(bit_width_le(offset)));
        integer_offsets.push_back(std::move(offset));
    }
    segment.integer_offset_bits = max_integer_offset_bits;

    std::size_t total_bits = segment_total_bit_width(segment);
    std::size_t plane_count = plane_count_for_bits(total_bits);
    segment.byte_planes.assign(plane_count, std::vector<std::uint8_t>(values.size(), 0));

    for (std::size_t row = 0; row < values.size(); ++row) {
        const auto& integer_offset = integer_offsets[row];
        const auto& fractional_part = fractional_parts[row];
        for (std::size_t plane = 0; plane < plane_count; ++plane) {
            std::uint64_t start_bit = plane_lsb_start(total_bits, plane);
            std::uint32_t width = static_cast<std::uint32_t>(plane_bit_width(total_bits, plane));
            segment.byte_planes[plane][row] = extract_combined_chunk(
                integer_offset, fractional_part, segment.fractional_bits, start_bit, width);
        }
    }

    return segment;
}

std::vector<double> decode_segment(const EncodedSegment& segment) {
    return materialize_segment(segment, segment.byte_planes.size(), true);
}

std::vector<double> decode_segment_top_k(const EncodedSegment& segment, std::size_t top_k_planes) {
    return materialize_segment(segment, top_k_planes, false);
}

double segment_max_abs_error_bound(const EncodedSegment& segment, std::size_t top_k_planes) {
    std::size_t plane_count = segment.byte_planes.size();
    if (top_k_planes >= plane_count) {
        return 0.0;
    }

    std::size_t total_bits = segment_total_bit_width(segment);
    std::size_t kept_bits = 0;
    for (std::size_t plane = 0; plane < std::min(top_k_planes, plane_count); ++plane) {
        kept_bits += plane_bit_width(total_bits, plane);
    }
    std::size_t omitted_bits = total_bits - kept_bits;
    if (omitted_bits == 0) {
        return 0.0;
    }

    int omitted_minus_fractional = static_cast<int>(omitted_bits) - static_cast<int>(segment.fractional_bits);
    long double bound = std::ldexp(1.0L, omitted_minus_fractional) -
                        std::ldexp(1.0L, -static_cast<int>(segment.fractional_bits));
    return static_cast<double>(bound);
}

std::size_t segment_plane_count(const EncodedSegment& segment) {
    return segment.byte_planes.size();
}

void encode_file(const std::filesystem::path& input_path,
                 const std::filesystem::path& output_path,
                 std::uint64_t segment_size,
                 std::uint64_t max_values) {
    if (segment_size == 0) {
        throw std::runtime_error("segment size must be greater than zero");
    }

    std::uint64_t total_values = count_values_in_fp64_file(input_path);
    if (max_values != 0) {
        total_values = std::min(total_values, max_values);
    }
    std::uint64_t segment_count = (total_values + segment_size - 1) / segment_size;

    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open input file for encoding: " + input_path.string());
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open output file for encoding: " + output_path.string());
    }

    write_stream_header(output, EncodedFileHeader{
                                   .total_values = total_values,
                                   .segment_size = segment_size,
                                   .segment_count = segment_count,
                               });

    std::vector<double> buffer(static_cast<std::size_t>(segment_size));
    std::uint64_t remaining = total_values;
    while (remaining > 0) {
        std::uint64_t current = std::min(remaining, segment_size);
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(current * sizeof(double)));
        if (!input) {
            throw std::runtime_error("failed to read input FP64 payload");
        }

        EncodedSegment segment = encode_segment(std::span<const double>(buffer.data(), static_cast<std::size_t>(current)));
        write_segment(output, segment);
        remaining -= current;
    }
}

void decode_file(const std::filesystem::path& input_path,
                 const std::filesystem::path& output_path) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open encoded file for decoding: " + input_path.string());
    }

    EncodedFileHeader header = read_stream_header(input);

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to open decoded file path: " + output_path.string());
    }

    std::uint64_t written = 0;
    for (std::uint64_t segment_index = 0; segment_index < header.segment_count; ++segment_index) {
        EncodedSegment segment = read_segment(input);
        std::vector<double> values = decode_segment(segment);
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(double)));
        if (!output) {
            throw std::runtime_error("failed to write decoded FP64 payload");
        }
        written += segment.value_count;
    }

    if (written != header.total_values) {
        throw std::runtime_error("decoded row count did not match BUFF header");
    }
}

EncodedFileHeader read_file_header(const std::filesystem::path& input_path) {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open encoded file header: " + input_path.string());
    }
    return read_stream_header(input);
}

bool files_have_identical_fp64_payload(const std::filesystem::path& lhs_path,
                                       const std::filesystem::path& rhs_path) {
    if (std::filesystem::file_size(lhs_path) != std::filesystem::file_size(rhs_path)) {
        return false;
    }

    std::ifstream lhs(lhs_path, std::ios::binary);
    std::ifstream rhs(rhs_path, std::ios::binary);
    if (!lhs || !rhs) {
        throw std::runtime_error("failed to open files for bitwise comparison");
    }

    constexpr std::size_t kChunkBytes = 1U << 20U;
    std::vector<char> lhs_buffer(kChunkBytes);
    std::vector<char> rhs_buffer(kChunkBytes);

    while (lhs && rhs) {
        lhs.read(lhs_buffer.data(), static_cast<std::streamsize>(lhs_buffer.size()));
        rhs.read(rhs_buffer.data(), static_cast<std::streamsize>(rhs_buffer.size()));
        std::streamsize lhs_read = lhs.gcount();
        std::streamsize rhs_read = rhs.gcount();
        if (lhs_read != rhs_read) {
            return false;
        }
        if (lhs_read == 0) {
            break;
        }
        if (std::memcmp(lhs_buffer.data(), rhs_buffer.data(), static_cast<std::size_t>(lhs_read)) != 0) {
            return false;
        }
    }

    return true;
}

std::string summarize_segment(const EncodedSegment& segment) {
    std::ostringstream builder;
    builder << "rows=" << segment.value_count
            << " int_base_bytes=" << segment.integer_base_le.size()
            << " int_offset_bits=" << segment.integer_offset_bits
            << " frac_bits=" << segment.fractional_bits
            << " total_bits=" << segment_total_bit_width(segment)
            << " planes=" << segment.byte_planes.size();
    return builder.str();
}

}  // namespace buff
