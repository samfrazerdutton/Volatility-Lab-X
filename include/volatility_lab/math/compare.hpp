// SPDX-License-Identifier: MIT
#pragma once
/// \file compare.hpp
/// \brief Floating-point comparison with an explicit error budget.
///
/// `EXPECT_TRUE(a == b)` on doubles is banned in this project (see
/// docs/numerical-validation.md).  Every comparison must name the tolerance it
/// is asserting, and the tolerance must be justified by the algorithm's error
/// analysis rather than tuned until the test passes.
///
/// Three comparison modes are offered because they answer different questions:
///
///  * `close_abs`  -- "the absolute error is within budget".  Correct for
///    quantities with a natural scale and a meaningful zero: a price in a
///    currency, a probability, a volatility in absolute vol points.
///
///  * `close_rel`  -- "the relative error is within budget".  Correct for
///    quantities spanning many decades: deep-OTM option values, vegas,
///    residual norms.  Undefined at zero, hence `close_mixed`.
///
///  * `close_ulp`  -- "the results differ by at most N representable steps".
///    The only meaningful statement for a bitwise-equivalence claim such as
///    "the SIMD kernel reproduces the scalar kernel"; it is scale-free and
///    does not need a hand-picked epsilon.
///
/// `close_mixed(a, b, atol, rtol)` is the workhorse: `|a-b| <= atol + rtol*|b|`,
/// which degrades gracefully to the absolute test near zero.  It is the same
/// predicate numpy.allclose uses, deliberately, so that the Python research
/// layer and the C++ tests agree about what "equal" means.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace vl::math {

/// Reinterpret a double as a sign-magnitude-ordered integer so that integer
/// subtraction counts representable steps.  This maps the IEEE-754 ordering
/// (which is lexicographic within a sign) onto a monotone integer line.
[[nodiscard]] inline std::int64_t ordered_bits(double x) noexcept {
    std::int64_t i;
    std::memcpy(&i, &x, sizeof(i));
    // Flip the whole magnitude for negatives so that -0.0 and +0.0 are
    // adjacent and the ordering is monotone across zero.
    return (i < 0) ? (std::numeric_limits<std::int64_t>::min() - i) : i;
}

/// Number of representable doubles between a and b (0 == bitwise identical).
/// Returns INT64_MAX if either is NaN.
[[nodiscard]] inline std::int64_t ulp_distance(double a, double b) noexcept {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<std::int64_t>::max();
    if (a == b) return 0;
    if (std::isinf(a) || std::isinf(b)) return std::numeric_limits<std::int64_t>::max();
    const std::int64_t ia = ordered_bits(a);
    const std::int64_t ib = ordered_bits(b);
    return (ia > ib) ? (ia - ib) : (ib - ia);
}

[[nodiscard]] inline bool close_ulp(double a, double b, std::int64_t max_ulps) noexcept {
    return ulp_distance(a, b) <= max_ulps;
}

[[nodiscard]] inline bool close_abs(double a, double b, double atol) noexcept {
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= atol;
}

/// Relative difference against |b| (the second argument is treated as the
/// reference).  Returns false when b == 0 and a != 0.
[[nodiscard]] inline bool close_rel(double a, double b, double rtol) noexcept {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (b == 0.0) return a == 0.0;
    return std::abs(a - b) <= rtol * std::abs(b);
}

/// |a-b| <= atol + rtol*|b|.  The default tolerances are deliberately *not*
/// provided: every call site must state its budget.
[[nodiscard]] inline bool close_mixed(double a, double b, double atol, double rtol) noexcept {
    if (std::isnan(a) || std::isnan(b)) return false;
    if (std::isinf(a) || std::isinf(b)) return a == b;
    return std::abs(a - b) <= atol + rtol * std::abs(b);
}

/// Relative error, with the absolute error substituted when the reference is
/// zero.  Used for reporting, not for assertions.
[[nodiscard]] inline double rel_error(double value, double reference) noexcept {
    const double d = std::abs(value - reference);
    const double r = std::abs(reference);
    return (r > 0.0) ? d / r : d;
}

/// Error in units of the reference's own ulp -- the scale-free way to report
/// "how good is this approximation".
[[nodiscard]] inline double error_in_ulps(double value, double reference) noexcept {
    if (value == reference) return 0.0;
    const double u = std::nextafter(std::abs(reference),
                                    std::numeric_limits<double>::infinity()) -
                     std::abs(reference);
    if (u == 0.0) return 0.0;
    return std::abs(value - reference) / u;
}

// ---------------------------------------------------------------------------
// Error-budget records
//
// Each numerical routine publishes its budget as a named constant so that the
// tests, the benchmarks, and the validation document all read the same number.
// A tolerance that appears only inside a test is a tolerance nobody reviewed.
// ---------------------------------------------------------------------------

struct Tolerance {
    double atol;
    double rtol;
    const char* rationale;

    [[nodiscard]] constexpr Tolerance scaled(double k) const noexcept {
        return {atol * k, rtol * k, rationale};
    }
    [[nodiscard]] bool accepts(double value, double reference) const noexcept {
        return close_mixed(value, reference, atol, rtol);
    }
};

inline constexpr double kEps = std::numeric_limits<double>::epsilon();  // 2.22e-16

namespace tol {

/// Normal CDF against the double-double reference.  The double implementation
/// is 0.5*erfc(-x/sqrt(2)); the sqrt(2) division and the libm erfc each
/// contribute <=1 ulp, and the erfc argument carries a further 0.5 ulp that
/// erfc amplifies by its own condition number (bounded by ~x^2/2 for the tail,
/// which is why the budget is relative rather than absolute).
inline constexpr Tolerance kNormCdf{0.0, 8.0 * kEps,
                                    "2 libm ops + 1 division, each <=1 ulp, "
                                    "with tail condition-number headroom"};

/// Inverse normal CDF after Newton polish.  One Newton step on a 1e-9-accurate
/// seed squares the error to ~1e-18, below the representable resolution, so
/// the limit is the conditioning of the CDF near the tails.
inline constexpr Tolerance kNormInv{0.0, 16.0 * kEps,
                                    "Newton-polished; limited by CDF conditioning"};

/// Black price against the double-double reference.  Dominated by the two CDF
/// evaluations and the subtraction between them; for OTM options the
/// subtraction is benign because the library always prices the OTM side.
inline constexpr Tolerance kBlackPrice{1e-300, 32.0 * kEps,
                                       "2 CDF evals + benign subtraction (OTM side priced)"};

/// Implied volatility round-trip: price -> IV -> price.  The budget is stated
/// on the *volatility*, in absolute vol points, because that is what a trader
/// would read.  2e-15 is roughly 10 ulps at a 1.0 vol.
inline constexpr Tolerance kImpliedVol{2e-15, 1e-14,
                                       "Householder iteration to machine precision; "
                                       "limit is the conditioning of dPrice/dVol"};

/// SIMD vs scalar kernel equivalence.  Not bitwise: the SIMD path uses FMA
/// contraction and its own vectorised `exp` (`kernels/simd/erfcx_avx2.hpp`'s
/// `exp_avx2_bounded`, a single-ln2-constant range reduction + degree-13
/// Taylor series, not libm's `std::exp` that the scalar kernel calls
/// directly) -- two independent approximations of `exp`, not one shared
/// primitive, so their difference is not purely FMA-rounding noise.
///
/// This was budgeted at "within 4 ulps" before any SIMD kernel existed to
/// measure against; the real number, swept exhaustively in
/// `tests/kernels/erfcx_avx2.cpp` over x in [-26, 100], is up to 690 ulps
/// (worst case near x = -25.7, where the reflection branch's `exp(x^2)`
/// term is enormous and amplifies the two exp implementations' absolute
/// disagreement before the subtraction). 690 ulps is `690 * kEps ~=
/// 1.5e-13` relative error -- six orders of magnitude inside the erfcx
/// formula's own ~1e-7 accuracy ceiling (`kErfcxPoly`), so it changes
/// nothing about whether the kernel is fit for purpose. The budget below
/// is the measured number with roughly 3x headroom, not the original
/// guess.
inline constexpr Tolerance kSimdEquivalence{0.0, 2048.0 * kEps,
                                            "two independent exp approximations (libm vs "
                                            "this project's own vectorised exp), not pure "
                                            "FMA-rounding noise; measured max 690 ulps"};

/// `kernels::scalar::erfcx_poly` (the branch-light Numerical Recipes
/// rational approximation, `kernels/scalar/erfcx_poly.hpp`) against the
/// double-double reference. The published claim is fractional error under
/// 1.2e-7; independently re-measured in `tests/kernels/erfcx_poly.cpp` at
/// ~1.045e-7 across [-10, 10] plus extreme/boundary points, which is what
/// this budget is set from -- not the citation.  This is orders of
/// magnitude looser than `kSimdEquivalence` deliberately: that tolerance is
/// for "does the AVX2 kernel reproduce *this same* polynomial", this one is
/// for "how close is the polynomial itself to the true value" -- two
/// different questions with two different, independently justified
/// budgets.
inline constexpr Tolerance kErfcxPoly{1e-300, 1.5e-7,
                                      "Numerical Recipes rational approximation; "
                                      "measured ~1.045e-7 max relative error"};

/// Parallel vs serial reduction.  This one *is* bitwise, by construction: the
/// chunk decomposition is fixed by input size, not thread count, and partials
/// combine in index order.  Stated as a tolerance of exactly zero so that a
/// regression is impossible to paper over.
inline constexpr Tolerance kParallelReduction{0.0, 0.0,
                                              "fixed chunking + in-order combine => bitwise"};

/// Analytic vs central-difference Greeks.  Central differences with step
/// h = eps^(1/3)*scale have error O(h^2) ~ 4e-11 relative, which is the real
/// limit here -- the analytic value is the accurate one.
inline constexpr Tolerance kFiniteDifferenceGreek{1e-7, 1e-6,
                                                  "central difference truncation O(h^2) "
                                                  "with h = cbrt(eps)*scale"};

}  // namespace tol
}  // namespace vl::math
