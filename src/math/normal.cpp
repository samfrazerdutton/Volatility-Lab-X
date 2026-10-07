// SPDX-License-Identifier: MIT
/// \file normal.cpp
/// \brief Implementations of erfcx, norm_inv, and the double-double references.

#include "volatility_lab/math/special.hpp"

#include <algorithm>
#include <array>

#include "volatility_lab/math/dd_real.hpp"
#include "volatility_lab/math/reference_special.hpp"

namespace vl::math {

// ===========================================================================
// erfcx
// ===========================================================================

namespace {

/// Below this the direct product exp(x^2)*erfc(x) is used; above it, the
/// continued fraction.  The switch point is chosen as the largest x where
/// exp(x^2) is comfortably finite (overflow is at 26.6416) *and* the continued
/// fraction already converges in few iterations.  At x = 8 the CF needs ~12
/// terms; at x = 4 it would need ~40, which is why the switch is not lower.
constexpr double kErfcxCfSwitch = 8.0;

/// 1/sqrt(pi).
constexpr double kInvSqrtPi = 5.64189583547756286948e-01;

/// Modified Lentz evaluation of the Legendre continued fraction
///
///   erfcx(x) = 1/(x*sqrt(pi)) * 1/K,   K = 1 + v/(1 + 2v/(1 + 3v/(1 + ...)))
///   with v = 1/(2 x^2)
///
/// Note carefully what Lentz returns.  With b0 = 1, b_n = 1, a_n = n*v it
/// evaluates `b0 + a1/(b1 + a2/(b2 + ...))`, i.e. **K itself**, not 1/K.  The
/// caller must take the reciprocal.  (Getting this backwards produces a result
/// wrong by a factor of 1 + 1/(2x^2) -- 1.5% at x = 8, 1e-4 at x = 100 --
/// which looks plausible enough to survive a casual eyeball and is exactly the
/// kind of error the double-double reference exists to catch.)
///
/// This CF is convergent for every x > 0, not merely asymptotic: dividing the
/// A&S 7.1.14 form `1/(x + (1/2)/(x + 1/(x + (3/2)/(x + ...))))` through by x
/// yields precisely a_n = n*v.  Lentz is used rather than a fixed-depth
/// backward recurrence because the depth needed varies by an order of
/// magnitude across the domain, and a convergence test is both faster and
/// safer than a worst-case constant.
///
/// Returns K via `k_out` so the caller can also build erfc without forming
/// exp(x^2).
double erfcx_cf_denominator(double x, int& iters_out) noexcept {
    constexpr double kTiny = 1e-300;
    // Stop when the Lentz multiplier is 1 to within half an ulp.  Tightening
    // this further cannot help: delta is a double.
    constexpr double kTol = 1.0e-16;
    constexpr int kMaxIter = 512;

    const double v = 0.5 / (x * x);

    double f = 1.0;  // b0
    double C = 1.0;  // = f
    double D = 0.0;

    int n = 1;
    for (; n <= kMaxIter; ++n) {
        const double a = static_cast<double>(n) * v;
        D = 1.0 + a * D;
        if (std::abs(D) < kTiny) D = kTiny;
        D = 1.0 / D;
        C = 1.0 + a / C;
        if (std::abs(C) < kTiny) C = kTiny;
        const double delta = C * D;
        f *= delta;
        if (std::abs(delta - 1.0) <= kTol) break;
    }
    iters_out = n;
    return f;
}

double erfcx_cf(double x) noexcept {
    int iters = 0;
    const double k = erfcx_cf_denominator(x, iters);
    return kInvSqrtPi / (x * k);
}

}  // namespace

double erfcx(double x) noexcept {
    if (std::isnan(x)) return x;

    if (x >= 0.0) {
        if (x < kErfcxCfSwitch) {
            // Both factors finite; exp_sq keeps the argument exact.
            return exp_sq(x) * std::erfc(x);
        }
        if (std::isinf(x)) return 0.0;
        return erfcx_cf(x);
    }

    // Reflection: erfc(-y) = 2 - erfc(y)  =>  erfcx(-y) = 2 exp(y^2) - erfcx(y).
    const double y = -x;
    if (y > kExpSqMax) return std::numeric_limits<double>::infinity();
    if (y < kErfcxCfSwitch) {
        // Direct is both cheaper and better conditioned here: erfc(x) for
        // x < 0 is O(1), so no cancellation.
        return exp_sq(x) * std::erfc(x);
    }
    return 2.0 * exp_sq(y) - erfcx_cf(y);
}

double norm_cdf_hp(double x) noexcept {
    if (std::isnan(x)) return x;
    // Phi(x) = 0.5 * erfcx(z) * exp(-x^2/2) with z = -x/sqrt2.  See the
    // header for why the argument-rounding error cancels here.
    //
    // For x > 0 the identity is used on the left tail and complemented, so
    // that the small quantity is always the one computed directly:
    //   x >= 0 : Phi(x)  = 1 - 0.5*erfcx(x/sqrt2)*exp(-x^2/2)
    //   x <  0 : Phi(x)  =     0.5*erfcx(-x/sqrt2)*exp(-x^2/2)
    // For x >= 0 the subtraction from 1 is benign: the subtrahend is <= 0.5.
    const double z = std::abs(x) * kSqrtHalf;
    const double tail = 0.5 * erfcx(z) * exp_neg_half_sq(x);
    return (x < 0.0) ? tail : 1.0 - tail;
}

// ===========================================================================
// norm_inv
// ===========================================================================

namespace {

/// Acklam's rational approximation to Phi^{-1}.  Relative error < 1.15e-9.
/// Coefficients are quoted from the published algorithm; they are regression
/// coefficients, not derived constants, hence the opaque values.
double norm_inv_acklam(double p) noexcept {
    static constexpr std::array<double, 6> a{-3.969683028665376e+01, 2.209460984245205e+02,
                                             -2.759285104469687e+02, 1.383577518672690e+02,
                                             -3.066479806614716e+01, 2.506628277459239e+00};
    static constexpr std::array<double, 5> b{-5.447609879822406e+01, 1.615858368580409e+02,
                                             -1.556989798598866e+02, 6.680131188771972e+01,
                                             -1.328068155288572e+01};
    static constexpr std::array<double, 6> c{-7.784894002430293e-03, -3.223964580411365e-01,
                                             -2.400758277161838e+00, -2.549732539343734e+00,
                                             4.374664141464968e+00,  2.938163982698783e+00};
    static constexpr std::array<double, 4> d{7.784695709041462e-03, 3.224671290700398e-01,
                                             2.445134137142996e+00, 3.754408661907416e+00};

    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;

    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > p_high) {
        const double q = std::sqrt(-2.0 * std::log1p(-p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

}  // namespace

double norm_inv(double p) noexcept {
    if (std::isnan(p) || p < 0.0 || p > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (p == 0.0) return -std::numeric_limits<double>::infinity();
    if (p == 1.0) return std::numeric_limits<double>::infinity();

    double x = norm_inv_acklam(p);

    // One Halley step on f(x) = Phi(x) - p.
    //   f'  = phi(x),  f'' = -x phi(x)
    //   x <- x - (f/phi) / (1 + x*(f/phi)/2)
    //
    // The whole accuracy of the polish rests on computing the residual f
    // without cancellation, and the obvious `norm_cdf(x) - p` fails to do so
    // in the central region: both terms are ~0.5, so f carries an absolute
    // error of ~eps/2 regardless of how small f actually is.  Dividing by
    // phi ~ 0.4 leaves ~2.8e-16 of absolute error in x -- which at p = 0.5001
    // (x ~ 2.5e-4) is a *relative* error of 1.1e-12, about 1450 ulps.
    //
    // The fix is to arrange, in every regime, for both operands of the
    // subtraction to be O(f) rather than O(1).  Three cases, each relying on
    // an exactness property of IEEE-754 subtraction (Sterbenz's lemma: a - b
    // is exact whenever b/2 <= a <= 2b):
    //
    //   p < 0.25      f = Phi(x) - p
    //                 Phi(x) < 0.25 and is computed with good relative
    //                 accuracy there; p is exact.  No cancellation.
    //
    //   0.25<=p<=0.75 f = 0.5*erf(x/sqrt2) - (p - 0.5)
    //                 p - 0.5 is exact (Sterbenz, since 0.25 <= p <= 1), and
    //                 Phi(x) - 0.5 == 0.5*erf(x/sqrt2) is computed with full
    //                 *relative* accuracy by libm rather than as a difference.
    //
    //   p > 0.75      f = (1 - p) - Phi(-x)
    //                 1 - p is exact (Sterbenz, since p >= 0.5), and Phi(-x)
    //                 is the small tail value, accurate relatively.  This is
    //                 algebraically Phi(x) - p but numerically 1000x better:
    //                 the residual error drops from ~1e-16 absolute to ~1e-19.
    const double pdf = norm_pdf(x);
    if (pdf > 1e-300) {
        double f;
        if (p < 0.25) {
            f = norm_cdf(x) - p;
        } else if (p <= 0.75) {
            f = 0.5 * std::erf(x * kSqrtHalf) - (p - 0.5);
        } else {
            f = (1.0 - p) - norm_cdf(-x);
        }
        const double u = f / pdf;
        x -= u / (1.0 + 0.5 * x * u);
    }
    return x;
}

double norm_inv_upper(double q) noexcept {
    // Phi^{-1}(1-q), taking q directly.
    //
    // This exists because of an *input representation* limit that no amount of
    // care inside norm_inv can fix: for p close to 1, the double p simply does
    // not carry the information in 1-p.  p = 1 - 1e-17 is not representable;
    // it rounds to 1.0, and Phi^{-1} of that is +inf.  A caller who knows the
    // upper-tail probability must be able to say so.
    //
    // Implementation: Phi^{-1}(1-q) = -Phi^{-1}(q) by symmetry, exactly, and
    // Phi^{-1}(q) for small q is the well-conditioned left-tail problem.
    return -norm_inv(q);
}

// ===========================================================================
// Double-double references (slow, used only for validation)
// ===========================================================================

namespace reference {

DDouble norm_pdf_dd(DDouble x) noexcept {
    return dd_const::kInvSqrt2Pi * exp_dd(DDouble(-0.5) * x * x);
}

/// erf by Taylor series: erf(z) = 2/sqrt(pi) * sum (-1)^n z^(2n+1) / (n!(2n+1)).
///
/// Converges for all z but needs ~z^2 terms; used only for |z| <= 3 where that
/// is under 30 terms.  Alternating signs cause cancellation that grows with z,
/// which is the other reason for the cutoff.
static DDouble erf_series_dd(DDouble z) noexcept {
    const DDouble z2 = z * z;
    DDouble term = z;          // z^(2n+1)/n!
    DDouble sum = z;           // n = 0 contributes z/1
    for (int n = 1; n <= 120; ++n) {
        term = term * z2 / static_cast<double>(n);
        const DDouble contrib = term / static_cast<double>(2 * n + 1);
        const DDouble signed_contrib = (n % 2 == 0) ? contrib : -contrib;
        sum += signed_contrib;
        if (std::abs(contrib.hi()) < 1e-36 * std::abs(sum.hi())) break;
    }
    return dd_const::kTwoOverSqrtPi * sum;
}

/// erfc by the asymptotic continued fraction, evaluated bottom-up in
/// double-double.  Fixed depth (chosen from the convergence rate at the
/// switch point) because a reference may be slow but must be deterministic.
static DDouble erfc_cf_dd(DDouble z) noexcept {
    const DDouble v = DDouble(0.5) / (z * z);
    constexpr int kDepth = 300;
    DDouble frac(0.0);
    for (int n = kDepth; n >= 1; --n) {
        frac = (v * static_cast<double>(n)) / (DDouble(1.0) + frac);
    }
    const DDouble mills = DDouble(1.0) / (DDouble(1.0) + frac);
    // erfc(z) = exp(-z^2)/(z sqrt(pi)) * mills
    const DDouble inv_sqrt_pi = dd_const::kTwoOverSqrtPi * 0.5;
    return exp_dd(-(z * z)) * inv_sqrt_pi / z * mills;
}

DDouble erfc_dd(DDouble z) noexcept {
    if (z.hi() < 0.0) return DDouble(2.0) - erfc_dd(-z);
    if (z.hi() < 3.0) return DDouble(1.0) - erf_series_dd(z);
    return erfc_cf_dd(z);
}

/// erfcx reference.  For z >= 3 this is built straight from the continued
/// fraction, with no exp(z^2) anywhere -- otherwise the reference would
/// return inf*0 = NaN for z >= 26.65, i.e. it would be *less* robust than the
/// implementation it is meant to validate.
static DDouble erfcx_cf_dd(DDouble z) noexcept {
    const DDouble v = DDouble(0.5) / (z * z);
    constexpr int kDepth = 400;
    DDouble frac(0.0);
    for (int n = kDepth; n >= 1; --n) {
        frac = (v * static_cast<double>(n)) / (DDouble(1.0) + frac);
    }
    const DDouble inv_sqrt_pi = dd_const::kTwoOverSqrtPi * 0.5;
    return inv_sqrt_pi / z / (DDouble(1.0) + frac);
}

DDouble erf_dd(DDouble z) noexcept {
    if (std::abs(z.hi()) < 3.0) return erf_series_dd(z);
    return DDouble(1.0) - erfc_dd(z);
}

DDouble norm_cdf_dd(DDouble x) noexcept {
    return DDouble(0.5) * erfc_dd(-(x * dd_const::kSqrtHalf));
}

DDouble erfcx_dd(DDouble x) noexcept {
    if (x.hi() >= 3.0) return erfcx_cf_dd(x);
    if (x.hi() > -3.0) return exp_dd(x * x) * erfc_dd(x);
    // erfcx(-y) = 2 exp(y^2) - erfcx(y).  Overflows for y > 26.65; the caller
    // is told so by an infinity rather than a quiet wrong answer.
    const DDouble y = -x;
    return DDouble(2.0) * exp_dd(y * y) - erfcx_cf_dd(y);
}

double norm_cdf_ref(double x) noexcept { return norm_cdf_dd(DDouble(x)).to_double(); }
double norm_pdf_ref(double x) noexcept { return norm_pdf_dd(DDouble(x)).to_double(); }
double erfcx_ref(double x) noexcept { return erfcx_dd(DDouble(x)).to_double(); }

double norm_inv_ref(double p) noexcept {
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();
    // Newton in double-double, seeded with the double result.  Three steps
    // from a 3-ulp seed is far more than needed; it is cheap and removes any
    // doubt about the reference being seed-limited.
    DDouble x(norm_inv(p));
    const DDouble target(p);
    for (int i = 0; i < 3; ++i) {
        const DDouble f = norm_cdf_dd(x) - target;
        const DDouble fp = norm_pdf_dd(x);
        if (fp.hi() == 0.0) break;
        x -= f / fp;
    }
    return x.to_double();
}

}  // namespace reference
}  // namespace vl::math
