// SPDX-License-Identifier: MIT
#pragma once
/// \file erfcx_poly.hpp
/// \brief A branch-light, SIMD-shaped `erfcx` approximation -- the scalar
///        tier between the double-double reference and the AVX2 kernel
///        (directive's REFERENCE -> SCALAR OPTIMIZED -> SIMD ladder).
///
/// ## Why this exists alongside the already-correct `math::erfcx`
///
/// `math::erfcx` (`math/special.hpp`) is excellent and is not being
/// replaced: it dispatches on `|x| < 8` to `exp_sq(x) * std::erfc(x)`
/// (effectively machine precision, since `std::erfc` is a tuned library
/// routine) and a Legendre continued fraction above that. Both paths
/// involve data-dependent branching and, for the continued fraction, a
/// variable number of iterations -- neither vectorises as one SIMD lane
/// computation without either diverging lanes or masking off whichever
/// lanes took the other path, which defeats much of the point of
/// vectorising in the first place.
///
/// ## The formula, and the identity that makes it work for erfcx specifically
///
/// This is W. J. Press et al.'s classic rational/polynomial approximation
/// (*Numerical Recipes*, `erfcc`), valid for every `x >= 0` with no domain
/// switch at all:
///
///     t = 1 / (1 + x/2)
///     erfc(x) = t * exp(-x^2 + poly(t))
///
/// where `poly` is a fixed 9th-degree polynomial in `t` (coefficients
/// below). The published claim is fractional error under 1.2e-7 across the
/// whole domain -- independently re-measured against this project's own
/// double-double reference (`erfcx_ref`) in `tests/kernels/erfcx_poly.cpp`
/// rather than trusted on citation alone, exactly per this project's
/// "verify, don't assume" discipline.
///
/// The useful trick for *this* function specifically: `erfcx(x) =
/// exp(x^2)*erfc(x)`, so substituting the formula above,
///
///     erfcx(x) = exp(x^2) * t * exp(-x^2 + poly(t)) = t * exp(poly(t))
///
/// -- the `exp(x^2)` and `exp(-x^2)` cancel *exactly*, so `erfcx` is
/// computed here **without ever forming `exp(x^2)` or `exp(-x^2)`
/// separately**. That is strictly better-behaved than computing `erfc(x)`
/// and multiplying by `exp(x^2)` afterwards, which overflows for `x >
/// 26.6416` (`kExpSqMax`) long before `erfcx(x)` itself does (`erfcx`
/// decays like `1/(x*sqrt(pi))` for large `x`, it does not blow up).
///
/// `x < 0` still needs one reflection, `erfcx(x) = 2*exp(x^2) - erfcx(-x)`,
/// which *does* form `exp(x^2)` -- unavoidably, since erfcx itself is
/// `O(exp(x^2))` there -- and is guarded the same way `math::exp_sq` is
/// guarded (`kExpSqMax`).

namespace vl::kernels::scalar {

/// Approximate erfcx via the branch-light Numerical Recipes formula. See
/// the file comment for the accuracy claim and where it is independently
/// re-measured.
[[nodiscard]] double erfcx_poly(double x) noexcept;

}  // namespace vl::kernels::scalar
