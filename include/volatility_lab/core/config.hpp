// SPDX-License-Identifier: MIT
#pragma once
/// \file config.hpp
/// \brief Compiler/platform abstraction for the hot paths.
///
/// Everything in this header is a *portability* concern, not an abstraction
/// layer.  The rule for this file: no macro that merely renames a keyword, and
/// no macro whose effect cannot be explained in terms of generated code.

#include <cstddef>
#include <cstdint>
#include <new>

// ---------------------------------------------------------------------------
// Inlining control.
//
// The pricing kernels are small leaf functions called from inside tight loops.
// Whether they inline determines whether the loop vectorises at all, so the
// decision is forced rather than left to a heuristic that sees the function in
// isolation.  VL_NEVER_INLINE exists for the reference implementations, where
// inlining a slow path into a fast one would distort benchmarks.
// ---------------------------------------------------------------------------
#if defined(_MSC_VER) && !defined(__clang__)
#  define VL_FORCE_INLINE  __forceinline
#  define VL_NEVER_INLINE  __declspec(noinline)
#  define VL_RESTRICT      __restrict
#  define VL_ASSUME(x)     __assume(x)
#elif defined(__GNUC__) || defined(__clang__)
#  define VL_FORCE_INLINE  inline __attribute__((always_inline))
#  define VL_NEVER_INLINE  __attribute__((noinline))
#  define VL_RESTRICT      __restrict__
#  if defined(__clang__)
#    define VL_ASSUME(x)   __builtin_assume(x)
#  else
#    define VL_ASSUME(x)   do { if (!(x)) __builtin_unreachable(); } while (0)
#  endif
#else
#  define VL_FORCE_INLINE  inline
#  define VL_NEVER_INLINE
#  define VL_RESTRICT
#  define VL_ASSUME(x)     ((void)0)
#endif

// ---------------------------------------------------------------------------
// Branch hints.  Used *only* where a branch is provably rare and sits in a
// loop body: input validation in batch kernels, solver fallback paths.  The
// measured effect is documented in docs/performance-report.md.
// ---------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
#  define VL_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define VL_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#  define VL_LIKELY(x)   (x)
#  define VL_UNLIKELY(x) (x)
#endif

namespace vl {

/// Cache-line size used for alignment and false-sharing padding.
///
/// std::hardware_destructive_interference_size is the standard spelling but is
/// a compile-time constant that libstdc++ warns about using in an ABI-visible
/// way, and it reports 64 on every target we support.  Both x86-64 and AArch64
/// cores of interest use 64-byte lines (Apple M-series uses 128, hence the
/// conservative 64 is still a correct *divisor* of the line size, which is all
/// the padding logic requires).
inline constexpr std::size_t kCacheLine = 64;

/// Alignment for SoA columns.  32 bytes is the AVX2 natural alignment and the
/// largest vector register width we target; 64 is used instead so that a
/// column start also begins a cache line, which makes the SoA-vs-AoS
/// measurement in benchmarks/pricing comparable across batch sizes.
inline constexpr std::size_t kSimdAlign = 64;

/// Widest vector register in `double` lanes for the compiled SIMD level.
/// This is a *build-time* constant used for buffer padding only; the runtime
/// dispatcher in kernels/simd decides what actually executes.
#if defined(__AVX512F__)
inline constexpr std::size_t kMaxVectorLanesF64 = 8;
#elif defined(__AVX__) || defined(__AVX2__) || defined(_M_AVX2)
inline constexpr std::size_t kMaxVectorLanesF64 = 4;
#else
inline constexpr std::size_t kMaxVectorLanesF64 = 2;
#endif

/// Round `n` up to the next multiple of `m` (m must be a power of two).
constexpr std::size_t round_up(std::size_t n, std::size_t m) noexcept {
    return (n + m - 1) & ~(m - 1);
}

}  // namespace vl
