// SPDX-License-Identifier: MIT
#pragma once
/// \file state_hash.hpp
/// \brief A portable, explicit content fingerprint for deterministic-replay
///        verification (directive Phase 5).
///
/// ## Why not `std::hash`
///
/// `std::hash<T>`'s implementation is not specified by the standard beyond
/// "equal inputs hash equal" -- different standard library versions,
/// different build configurations, even different optimisation levels are
/// free to hash the same value differently. That is completely fine for
/// `std::unordered_map`'s own internal use, and completely wrong for a
/// fingerprint whose entire purpose is "prove two runs produced the same
/// state": a divergence caused by the hash function itself, rather than by
/// the state it is hashing, is exactly the kind of accidental, invisible
/// nondeterminism replay testing exists to rule out. `StateHasher` is
/// therefore a small, explicit, fully-specified FNV-1a (64-bit) accumulator
/// -- auditable in a few lines, and identical on every platform this
/// project builds for.
///
/// ## What a hash match proves, and what it does not
///
/// Two states that fingerprint equal are not *proven* bit-identical --
/// hashing is lossy, a collision is possible in principle. What this
/// actually buys is the practically useful direction: two states produced
/// by a genuinely deterministic computation from the same inputs are
/// *guaranteed* to fingerprint equal, so a replay that produces a
/// *different* fingerprint from an earlier run has, with overwhelming
/// probability, actually diverged -- which is the failure mode (a
/// nondeterministic reduction, a race, hidden time-of-day dependence) this
/// exists to catch.

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace vl {

/// A 64-bit content fingerprint.
class StateHash {
  public:
    StateHash() = default;
    constexpr explicit StateHash(std::uint64_t v) noexcept : v_(v) {}
    [[nodiscard]] constexpr std::uint64_t value() const noexcept { return v_; }

    friend constexpr bool operator==(StateHash a, StateHash b) noexcept { return a.v_ == b.v_; }

    /// 16 lowercase hex digits, e.g. "9d8e3f1a2b4c5d6e" -- the form the
    /// directive's own example output (`state_hash = ...`) uses.
    [[nodiscard]] std::string to_hex() const;

  private:
    std::uint64_t v_ = 0;
};

/// Incremental FNV-1a (64-bit) accumulator. `combine` overloads feed raw
/// bytes into the running hash; order matters (as it must for a
/// fingerprint of an *ordered* sequence of fields), so callers combine
/// fields in a fixed, documented order at each call site.
class StateHasher {
  public:
    StateHasher() = default;

    StateHasher& combine_bytes(const void* data, std::size_t n) noexcept;
    StateHasher& combine(std::uint64_t v) noexcept;
    StateHasher& combine(std::int64_t v) noexcept;
    /// Hashes the IEEE-754 bit pattern directly -- +0.0 and -0.0 hash
    /// differently, deliberately: this is a bit-exact "did the computation
    /// produce literally the same floating-point result" check, stricter
    /// than value equality on purpose.
    StateHasher& combine(double v) noexcept;
    StateHasher& combine(std::string_view s) noexcept;

    [[nodiscard]] StateHash finish() const noexcept { return StateHash{state_}; }

  private:
    // FNV-1a 64-bit offset basis and prime -- the standard, published
    // constants, not chosen here.
    std::uint64_t state_ = 0xcbf29ce484222325ULL;
};

}  // namespace vl
