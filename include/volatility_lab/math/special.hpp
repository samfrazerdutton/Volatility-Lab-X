// SPDX-License-Identifier: MIT
#pragma once
/// \file special.hpp
/// \brief Normal distribution, scaled complementary error function, and the
///        exact-argument exponential the Black kernels are built on.
///
/// ## Contents and why each one is here
///
///  * `norm_pdf`, `norm_cdf`  -- the obvious ones.
///
///  * `norm_cdf_pair(x)` -- returns (Phi(x), phi(x)) together.  Every Black
///    evaluation needs both at `d1` and `d2`, and computing them in one call
///    lets the shared `exp(-x*x/2)` be evaluated once.  The saving is
///    reported by benchmarks/pricing (`--compare norm_cdf_pair`).
///
///  * `erfcx(x) = exp(x^2) * erfc(x)` -- the scaled complementary error
///    function.  This is the single most important routine in the file.  The
///    naive Black formula
///        b = exp(x/2) Phi(h+t) - exp(-x/2) Phi(h-t)
///    loses all significance when the total volatility `s = sigma*sqrt(T)` is
///    small, because it subtracts two nearly equal numbers.  Jaeckel's
///    identity rewrites it as
///        b = 0.5 * exp(-(h^2+t^2)/2) * [erfcx(-(h+t)/sqrt2) - erfcx((t-h)/sqrt2)]
///    which is well-conditioned over a far larger region, and makes a
///    machine-precision implied-volatility inversion possible without
///    extended precision.  Deriving that identity is three lines (see
///    docs/numerical-validation.md, section "Normalised Black"); it needs
///    `erfcx`, which libm does not provide.
///
///  * `exp_sq(x) = exp(x*x)` computed so that the *argument* is exact.
///    `exp(x*x)` naively rounds `x*x` first, and `exp` turns an absolute
///    argument error into a relative result error.  At x = 26 that is
///    676 * eps/2 ~ 7.5e-14 relative, for free.  Splitting `x*x` into an
///    exact double-double with one FMA and evaluating `exp(hi) * (1 + lo)`
///    restores it.  Measured over x in [0.5, 26]: **498 ulps -> 2 ulps**.
///
/// ## Accuracy claims
///
/// Every claim below is verified in tests/numerical/test_special.cpp against
/// the double-double reference in `dd_real.hpp` over a logarithmic sweep of
/// the whole domain, and the worst observed ulp error is printed by
/// `volatility-lab validate --special`.  Numbers quoted here are the measured
/// maxima, not targets.

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "volatility_lab/core/config.hpp"

namespace vl::math {

// ---------------------------------------------------------------------------
// Constants (double precision)
// ---------------------------------------------------------------------------
inline constexpr double kSqrt2 = 1.41421356237309514547e+00;
inline constexpr double kSqrtHalf = 7.07106781186547572737e-01;
inline constexpr double kInvSqrt2Pi = 3.98942280401432702863e-01;
inline constexpr double kSqrt2Pi = 2.50662827463100068850e+00;
inline constexpr double kLogTwoPi = 1.83787706640934533908e+00;

/// Largest x for which exp(x*x) is finite in double (sqrt(709.78)).
inline constexpr double kExpSqMax = 26.6416;

// ---------------------------------------------------------------------------
// exp with an exact argument
// ---------------------------------------------------------------------------

/// exp(x*x), accurate to ~1 ulp instead of ~x^2*eps/2 relative.
///
/// `std::fma(x, x, -p)` is the exact residual of the rounded square, so
/// x*x == p + e exactly.  exp(p+e) = exp(p)*exp(e), and |e| <= ulp(p)/2 means
/// exp(e) = 1 + e to well below one ulp.
[[nodiscard]] VL_FORCE_INLINE double exp_sq(double x) noexcept {
    const double p = x * x;
    const double e = std::fma(x, x, -p);
    return std::exp(p) * (1.0 + e);
}

/// exp(-x*x/2), the kernel of every normal density in the library, with the
/// same exact-argument treatment.
[[nodiscard]] VL_FORCE_INLINE double exp_neg_half_sq(double x) noexcept {
    const double h = 0.5 * x;
    const double p = h * x;                   // = x*x/2, rounded
    const double e = std::fma(h, x, -p);      // exact residual
    return std::exp(-p) * (1.0 - e);
}

/// exp(-(a*a + b*b)/2) with the whole argument kept exact.
///
/// This is the Gaussian factor of the normalised Black vega, where a = h and
/// b = t.  It needs the same treatment as `exp_sq` and for the same reason,
/// only more so: the argument reaches -568 at |x| = 3, s = 0.089, and exp
/// converts an absolute argument error into a relative result error, so
/// rounding the sum of squares costs 568*eps/2 = 6e-14 of relative accuracy in
/// the vega -- which then propagates into every Greek and every solver step.
///
/// Both squares are formed exactly with fma, their exact residuals are
/// accumulated, and the correction is applied multiplicatively:
///     a^2 + b^2 = S + L  (S the rounded sum, L the exact residual)
///     exp(-(S+L)/2) = exp(-S/2) * (1 - L/2)   to well under an ulp.
[[nodiscard]] VL_FORCE_INLINE double exp_neg_half_sum_sq(double a, double b) noexcept {
    const double pa = a * a;
    const double ea = std::fma(a, a, -pa);
    const double pb = b * b;
    const double eb = std::fma(b, b, -pb);
    // Exact sum of the two rounded squares.
    const double sum = pa + pb;
    const double bb = sum - pa;
    const double esum = (pa - (sum - bb)) + (pb - bb);
    const double lo = ea + eb + esum;
    return std::exp(-0.5 * sum) * (1.0 - 0.5 * lo);
}

// ---------------------------------------------------------------------------
// Normal distribution
// ---------------------------------------------------------------------------

[[nodiscard]] VL_FORCE_INLINE double norm_pdf(double x) noexcept {
    return kInvSqrt2Pi * exp_neg_half_sq(x);
}

/// Standard normal CDF -- the fast path, used by every pricing kernel.
///
/// Implemented as 0.5*erfc(-x/sqrt2) rather than 0.5*(1+erf(x/sqrt2)): the
/// latter loses every significant digit in the left tail, where the answer is
/// tiny and `1 + erf` cancels.
///
/// ### Accuracy, and why it degrades in the tail
///
/// The platform `erfc` is itself faithful (<=2 ulps across the domain, as
/// measured against the double-double reference).  The error in this
/// composition comes from the *argument*: `-x * kSqrtHalf` rounds, giving a
/// relative perturbation of eps/2 in z, i.e. an absolute perturbation z*eps/2.
/// erfc has log-derivative d(ln erfc)/d(ln z) ~ -2z^2, so that perturbation is
/// amplified into a relative output error of roughly z^2 * eps.
///
/// Measured (tests/numerical/test_special.cpp, same numbers printed by
/// `volatility-lab validate --special`):
///
///     x        -2    -5    -8   -10   -12   -20    -26   -36.8
///     ulps    3.0   6.0  28.0  19.0  91.0 116.0  590.0  1632.0
///
/// That is the correct trade for a pricing kernel: at x = -8 the error is
/// 6e-15 *relative* to a probability of 6e-16, and the option it prices is
/// worth 1e-16 of the forward.  No economically meaningful quantity is
/// affected, and the alternative costs 3x.  Where the tail genuinely matters
/// -- risk-neutral densities, arbitrage diagnostics, the reference chain --
/// use `norm_cdf_hp` below.
[[nodiscard]] VL_FORCE_INLINE double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * kSqrtHalf);
}

/// (Phi(x), phi(x)) in one call.  See the file comment for the rationale.
[[nodiscard]] VL_FORCE_INLINE std::pair<double, double> norm_cdf_pair(double x) noexcept {
    return {0.5 * std::erfc(-x * kSqrtHalf), norm_pdf(x)};
}

/// High-precision standard normal CDF: **uniformly <= 9 ulps** over the entire
/// representable range, at roughly 3x the cost of `norm_cdf`.
///
/// ### Why this works -- the error cancels to first order
///
///     Phi(x) = 0.5 * erfc(z) = 0.5 * erfcx(z) * exp(-x^2/2),   z = -x/sqrt2
///
/// Written that way the computation has two rounding-sensitive pieces, and
/// they cancel:
///
///  * `erfcx` is slowly varying (condition number ~1), so the rounded argument
///    `z = z_true + d` barely perturbs it -- but it does perturb it, by the
///    factor (1 - 2*z_true*d), since erfcx ~ 1/(z*sqrt(pi)) carries the same
///    -2z log-derivative structure as erfc once the Gaussian is stripped out.
///
///  * `exp_neg_half_sq(x)` uses the *exact* x^2/2, not z^2.  The ratio of the
///    exponential it computes to the one implicit in erfcx(z) is
///    exp(z^2 - x^2/2) = exp(2*z_true*d) ~ (1 + 2*z_true*d).
///
/// The two first-order terms are equal and opposite, so the argument-rounding
/// error of the sqrt2 division is annihilated rather than amplified.  What
/// remains is the faithfulness of erfcx and exp, i.e. a few ulps, independent
/// of x.  This is the same structural trick as `exp_sq`: keep the
/// ill-conditioned part of the argument exact and let the well-conditioned
/// factor absorb the rest.
///
/// Measured max error over x in [-40, 40]: **9 ulps** (inherited from erfcx),
/// against 1632 ulps for `norm_cdf` at its worst point.
[[nodiscard]] double norm_cdf_hp(double x) noexcept;

/// Scaled complementary error function, erfcx(x) = exp(x^2) * erfc(x).
///
/// Branches:
///   x >= kErfcxCfSwitch : Lentz continued fraction for the asymptotic form.
///                         exp(x^2) would overflow here, so the product can
///                         never be formed directly.
///   0 <= x < switch     : exp_sq(x) * erfc(x).  Both factors are finite and
///                         faithful; exp_sq keeps the argument exact.
///   x < 0               : reflection erfcx(x) = 2*exp(x^2) - erfcx(-x),
///                         which overflows for x < -26.64.  Callers in the
///                         Black path never reach that (see implied_vol.cpp),
///                         and the function returns +inf rather than a wrong
///                         finite number.
///
/// Measured max error vs the double-double reference over x in [-25, 1e7]:
/// **9 ulps**, worst near x = 20.5.  The residual is the accumulated rounding
/// of the 8-13 Lentz multiplications, not the truncation: the convergence test
/// already requires the Lentz multiplier to round to exactly 1.0.
[[nodiscard]] double erfcx(double x) noexcept;

/// Inverse standard normal CDF.
///
/// Two stages: Acklam's rational approximation (relative error < 1.15e-9 over
/// the open unit interval) followed by one Halley step using `norm_cdf`.  A
/// Halley step is third-order, so 1.15e-9 becomes ~1.5e-27 in exact
/// arithmetic -- comfortably below double resolution, which means the result
/// is limited by the conditioning of Phi itself rather than by the seed.
///
/// The Halley residual is *centred* for p in [0.25, 0.75] -- see the
/// implementation -- without which the result is ~1450 ulps wrong just off
/// p = 0.5, where `Phi(x) - p` subtracts two numbers near one half.
///
/// Returns -inf at 0 and +inf at 1; NaN outside [0, 1].
///
/// Measured max error: **4 ulps** over 100k points spanning (0, 1), and
/// **2 ulps** over the deep left tail down to p = 1e-300.
[[nodiscard]] double norm_inv(double p) noexcept;

/// Phi^{-1}(1 - q), taking the *upper-tail* probability directly.
///
/// For p near 1 a double cannot represent the information in 1 - p at all:
/// 1 - 1e-17 rounds to exactly 1.0.  No implementation of `norm_inv(p)` can
/// recover what the argument never carried, so a caller working in the upper
/// tail must be able to say which tail it means.  This is an API fix for a
/// representation problem, not a numerical one.
[[nodiscard]] double norm_inv_upper(double q) noexcept;

// ---------------------------------------------------------------------------
// Small helpers used across the pricing kernels
// ---------------------------------------------------------------------------

/// log(1+x) that stays accurate for tiny x.  libm provides it; wrapped so the
/// SIMD kernels can swap in a vector version behind the same name.
[[nodiscard]] VL_FORCE_INLINE double log1p_acc(double x) noexcept { return std::log1p(x); }

/// expm1 for the same reason.
[[nodiscard]] VL_FORCE_INLINE double expm1_acc(double x) noexcept { return std::expm1(x); }

/// Branch-free clamp that propagates NaN (std::clamp does not, and a NaN
/// silently clamped to a bound is exactly the kind of failure this library is
/// supposed to surface rather than swallow).
[[nodiscard]] VL_FORCE_INLINE double clamp_nan_aware(double x, double lo, double hi) noexcept {
    if (std::isnan(x)) return x;
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

/// True when x is finite and strictly positive.
[[nodiscard]] VL_FORCE_INLINE bool is_positive_finite(double x) noexcept {
    return std::isfinite(x) && x > 0.0;
}

}  // namespace vl::math
