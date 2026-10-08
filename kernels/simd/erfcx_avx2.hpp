// SPDX-License-Identifier: MIT
#pragma once
/// \file erfcx_avx2.hpp
/// \brief AVX2+FMA batch erfcx -- the SIMD step of the directive's
///        REFERENCE -> SCALAR OPTIMIZED -> SIMD ladder, built on
///        `kernels/scalar/erfcx_poly.hpp`'s already-validated formula.
///
/// ## What this vectorises, and the one real obstacle
///
/// `kernels::scalar::erfcx_poly`'s formula (`t = 1/(1+x/2)`,
/// `erfcx(x) = t*exp(poly(t))` for `x >= 0`, one reflection for `x < 0`) is
/// entirely straight-line arithmetic -- no branches, no iteration -- which
/// is exactly the shape that vectorises cleanly, with one exception: AVX2
/// has no instruction for `exp`. That is provided by Intel's SVML, which
/// is not portable, or by a vector math library (SLEEF, Agner Fog's
/// vectorclass), neither of which this project depends on. So this file
/// also contains `exp_avx2_bounded`, a small internal vectorised `exp` --
/// deliberately **not** offered as a general-purpose primitive, see its
/// own comment for the exact domain it is valid over and why.
///
/// ## The numerical contract
///
/// Two separate, independently-justified claims, with two separate
/// tolerances (`math/compare.hpp`):
///
///  * **AVX2 vs the scalar polynomial kernel** (`math::tol::kSimdEquivalence`,
///    ~4 ulps): both compute the *same* formula, so any difference is only
///    FMA contraction / instruction-level rounding, not a different
///    approximation. This is checked over the full differential sweep in
///    `tests/kernels/erfcx_avx2.cpp`.
///  * **The formula vs the true value** (`math::tol::kErfcxPoly`, ~1e-7):
///    already established for the scalar kernel; the AVX2 kernel inherits
///    it exactly because it computes the same formula, not a separate
///    claim to re-derive.
///
/// Reporting "AVX2 matches scalar to 4 ulps" and separately "the formula
/// itself is accurate to 1e-7" is deliberate, not an oversight: conflating
/// them into one number would hide which of the two questions a given
/// error actually answers.

#include <span>

namespace vl::kernels::simd {

/// Batch erfcx over 4-wide AVX2 lanes, with a scalar-polynomial tail for
/// any remainder not divisible by 4 -- the tail calls
/// `kernels::scalar::erfcx_poly` directly, so it is not merely "close to"
/// the vectorised lanes' result, it is computed by the identical formula.
///
/// `x` and `out` must have equal size; `out` is overwritten completely.
void erfcx_avx2_batch(std::span<const double> x, std::span<double> out) noexcept;

}  // namespace vl::kernels::simd
