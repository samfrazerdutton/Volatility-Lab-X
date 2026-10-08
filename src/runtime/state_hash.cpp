// SPDX-License-Identifier: MIT
#include "volatility_lab/runtime/state_hash.hpp"

#include <cstdio>

namespace vl {

std::string StateHash::to_hex() const {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v_));
    return std::string(buf, 16);
}

StateHasher& StateHasher::combine_bytes(const void* data, std::size_t n) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    std::uint64_t h = state_;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= 0x100000001b3ULL;  // FNV-1a 64-bit prime
    }
    state_ = h;
    return *this;
}

StateHasher& StateHasher::combine(std::uint64_t v) noexcept {
    return combine_bytes(&v, sizeof(v));
}

StateHasher& StateHasher::combine(std::int64_t v) noexcept {
    return combine_bytes(&v, sizeof(v));
}

StateHasher& StateHasher::combine(double v) noexcept {
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return combine_bytes(&bits, sizeof(bits));
}

StateHasher& StateHasher::combine(std::string_view s) noexcept {
    // The length is folded in too, not just the bytes: without it,
    // combine("ab").combine("c") and combine("a").combine("bc") would hash
    // identically (a classic ambiguous-concatenation hash bug), silently
    // merging what should be two distinct field sequences.
    combine(static_cast<std::uint64_t>(s.size()));
    return combine_bytes(s.data(), s.size());
}

}  // namespace vl
