// SPDX-License-Identifier: MIT
#include "volatility_lab/pricing/black.hpp"

#include <algorithm>

namespace vl {

using math::erfcx;
using math::kInvSqrt2Pi;
using math::kSqrtHalf;

namespace {

constexpr double kTwoOverSqrtPi = 1.12837916709551257390e+00;

// ---------------------------------------------------------------------------
// The midpoint series for the erfcx bracket
// ---------------------------------------------------------------------------
//
// Branch A of `normalised_black` needs
//     Delta(zbar, d) = erfcx(zbar - d) - erfcx(zbar + d)
// where the two arguments are symmetric about zbar = |h|/sqrt2 with
// half-separation d = t/sqrt2.  Expanding both about zbar, the even-order
// terms cancel identically and the odd ones double:
//
//     Delta = -2 * sum_{m>=0} d^(2m+1) E_{2m+1}(zbar) / (2m+1)!
//
// where E_n is the n-th derivative of erfcx.  Those derivatives need no new
// special function: erfcx satisfies E' = 2 z E - 2/sqrt(pi), and
// differentiating n times by Leibniz gives a two-term recursion
//
//     E_0 = erfcx(z)
//     E_1 = 2 z E_0 - 2/sqrt(pi)
//     E_{n+1} = 2 (z E_n + n E_{n-1})
//
// so the series costs one erfcx call plus a short fma chain -- cheaper than
// the two erfcx calls it replaces.
//
// **Why it is cancellation-free.**  Every odd derivative of erfcx is negative
// (asymptotically E_n ~ (-1)^n n!/(z^(n+1) sqrt(pi)); at z = 0 the recursion
// gives E_{2m+1}(0) = -2^(m+1) m!/sqrt(pi)), so every term of -2*sum(...) is
// positive and the sum accumulates monotonically.  That is the entire point:
// the quantity the direct subtraction obtains by cancelling 1e-9 out of two
// O(1) numbers is here obtained by adding a handful of positive numbers.
//
// **Why it is term-capped.**  The recursion is a Miller recurrence.  Its two
// solution families are the one we want, which decays like n!/z^(n+1), and a
// parasite; forward evaluation seeds the parasite with rounding error and it
// eventually dominates.  Measured against the double-double reference over a
// (zbar, d/zbar) grid, the breakdown tracks the *term count* closely:
//
//     terms used    1-8     10      13      19      21      33      64
//     worst rel     1e-13   2e-12   5e-13   4e-12   1e-8    1e-6    2e+08
//
// So rather than deriving a stability bound -- the obvious (2 zbar)^n estimate
// is far too pessimistic and would reject regions that measure clean -- the
// series is capped at a small number of terms and the caller falls back to the
// direct subtraction if it has not converged by then.  Inside the cap the
// series is uniformly <= 1e-12 at every zbar tested from 0.01 to 35.  The full
// table is in docs/numerical-validation.md; the reasoning is D-05 in
// docs/design-decisions.md.

/// Term cap, from the measurement above.
constexpr int kMaxBracketTerms = 8;

/// Cheap predicate for "the series will converge inside the cap", so that the
/// hot path does not do eight terms of work and then discard them.
///
/// Successive terms shrink by about max(2 d^2/(2m+3), (d/zbar)^2): the first
/// factor governs small zbar, the second large zbar.  Either `d` being small
/// in absolute terms or `d` being small relative to `zbar` is therefore
/// sufficient, and both constants were read off the same measured grid as the
/// term cap.
///
/// The first clause covers the great majority of real quotes: d <= 0.06 means
/// sigma*sqrt(T) <= 0.17, which includes every ATM option out to roughly nine
/// months at 20% volatility.
[[nodiscard]] VL_FORCE_INLINE bool series_will_converge(double zbar, double d) noexcept {
    return d <= 0.06 || d <= 0.01 * zbar;
}

/// Evaluates Delta(zbar, d).  Returns false if the cap was reached or the term
/// sequence started growing, in which case `out` must be ignored.
[[nodiscard]] bool erfcx_symmetric_difference(double zbar, double d, double& out,
                                              int& terms) noexcept {
    double e_prev = erfcx(zbar);                                  // E_0
    double e_cur = std::fma(2.0 * zbar, e_prev, -kTwoOverSqrtPi);  // E_1

    const double d2 = d * d;
    double d_pow = d;       // d^(2m+1)
    double inv_fact = 1.0;  // 1/(2m+1)!
    double sum = d_pow * e_cur;
    double prev_mag = std::abs(sum);

    int n = 1;  // index of e_cur
    int m = 1;
    bool converged = false;
    for (; m <= kMaxBracketTerms; ++m) {
        // Advance the derivative recursion two steps: E_{n+1}, E_{n+2}.
        for (int k = 0; k < 2; ++k) {
            const double e_next = 2.0 * std::fma(zbar, e_cur, static_cast<double>(n) * e_prev);
            e_prev = e_cur;
            e_cur = e_next;
            ++n;
        }
        d_pow *= d2;
        // 1/(2m+1)! from 1/(2m-1)!: divide by (2m)(2m+1).
        inv_fact /= static_cast<double>(2 * m) * static_cast<double>(2 * m + 1);
        const double term = d_pow * e_cur * inv_fact;
        const double mag = std::abs(term);
        sum += term;

        if (mag <= 1e-18 * std::abs(sum)) {
            converged = true;
            break;
        }
        if (m >= 2 && mag > prev_mag) {
            terms = m;
            return false;  // diverging: the Miller parasite has taken over
        }
        prev_mag = mag;
    }
    terms = m;
    if (!converged) return false;

    out = -2.0 * sum;
    // All odd derivatives are negative, so the sum is negative and the result
    // positive.  A non-positive result means something above went wrong.
    return out > 0.0;
}

}  // namespace

// ===========================================================================
// Normalised Black
// ===========================================================================

NormalisedBlack normalised_black(double x, double s) noexcept {
    NormalisedBlack out{0.0, 0.0, BlackBranch::ZeroVariance};

    if (!std::isfinite(x) || !std::isfinite(s) || s < 0.0) {
        out.value = std::numeric_limits<double>::quiet_NaN();
        out.dv_ds = out.value;
        return out;
    }
    const double ax = std::abs(x);  // x <= 0 by contract; folded defensively

    if (s == 0.0) {
        // The OTM option is worthless at zero volatility.  Its vega vanishes
        // for x < 0 but tends to 1/sqrt(2pi) at x == 0; report the limit so
        // that an ATM caller at s = 0 still sees a usable derivative.
        out.value = 0.0;
        out.dv_ds = (ax == 0.0) ? kInvSqrt2Pi : 0.0;
        return out;
    }

    const double t = 0.5 * s;

    if (ax == 0.0) {
        // b = 2 Phi(t) - 1 = erf(t/sqrt2): one libm call, no cancellation.
        out.value = std::erf(t * kSqrtHalf);
        out.dv_ds = kInvSqrt2Pi * std::exp(-0.5 * t * t);
        out.branch = BlackBranch::Atm;
        return out;
    }

    const double abs_h = ax / s;

    // Normalised vega, (1/sqrt(2pi)) * G with G = exp(-(h^2+t^2)/2).  Shared
    // by every branch, so computed once.  The exponent is kept *exact* rather
    // than merely fma-fused: it reaches -568 in the deep-OTM corner, and since
    // exp turns an absolute argument error into a relative result error, a
    // single rounding of h^2 + t^2 there costs 6e-14 of relative accuracy in
    // the vega -- measured, and visible against the double-double reference.
    const double gaussian = math::exp_neg_half_sum_sq(abs_h, t);
    out.dv_ds = kInvSqrt2Pi * gaussian;

    const double zbar = abs_h * kSqrtHalf;
    const double d = t * kSqrtHalf;

    // The midpoint series is preferred wherever it converges quickly: it is
    // more accurate than the direct subtraction, cheaper than it, and -- since
    // it never evaluates erfcx at a negative argument -- it is the only one of
    // the three forms that is well behaved on both sides of t = |h|.
    bool done = false;
    if (series_will_converge(zbar, d)) {
        double bracket = 0.0;
        int terms = 0;
        if (erfcx_symmetric_difference(zbar, d, bracket, terms)) {
            out.value = 0.5 * gaussian * bracket;
            out.branch = BlackBranch::Series;
            done = true;
        }
    }

    if (!done) {
        if (t <= abs_h) {
            // ---- form (2), direct -----------------------------------------
            // Loses log10(|h|/s) digits, which the measurement bounds at ~3
            // over the region the series declines to handle.
            out.value = 0.5 * gaussian * (erfcx(zbar - d) - erfcx(zbar + d));
            out.branch = BlackBranch::ErfcxDirect;
        } else {
            // ---- form (3), reflected --------------------------------------
            // b = exp(x/2) - (G/2)[erfcx(d-zbar) + erfcx(d+zbar)]
            // Both arguments are positive, so nothing overflows, and the
            // leading term is the exact s -> infinity limit rather than a
            // cancellation.
            // exp(x/2).  Computed from ax directly, NOT as exp(-abs_h*t):
            // abs_h*t is (ax/s)*(s/2), two roundings whose product differs
            // from ax/2 by an ulp, and that ulp made `b` non-monotone in s
            // across the saturation boundary and let it exceed its own upper
            // bound.  A pricer that is 1 ulp above the no-arbitrage bound is a
            // pricer whose output the implied-vol inverter rejects.
            const double cap = std::exp(-0.5 * ax);
            out.value = cap - 0.5 * gaussian * (erfcx(d - zbar) + erfcx(d + zbar));
            out.branch = BlackBranch::HighVariance;
            if (out.value >= cap) {
                // The correction has underflowed relative to the limit: the
                // option has saturated at its s -> infinity bound.
                out.value = cap;
                out.branch = BlackBranch::Saturated;
            }
        }
    }

    // A negative option value is never correct.  It can only arise from total
    // cancellation at the edge of the representable region, where the true
    // value is below the smallest denormal anyway.
    if (!(out.value > 0.0)) out.value = 0.0;
    return out;
}

double normalised_black_value(double x, double s) noexcept {
    return normalised_black(x, s).value;
}

double normalised_black_vega(double x, double s) noexcept {
    if (!std::isfinite(x) || !std::isfinite(s) || s <= 0.0) {
        return (s == 0.0 && x == 0.0) ? kInvSqrt2Pi : 0.0;
    }
    const double h = x / s;
    const double t = 0.5 * s;
    return kInvSqrt2Pi * math::exp_neg_half_sum_sq(h, t);
}

double normalised_black_d2(double x, double s) noexcept {
    if (!std::isfinite(x) || !std::isfinite(s) || s <= 0.0) return 0.0;
    // d2b/ds2 = vega * (x^2/s^3 - s/4).  The bracket changes sign exactly once,
    // at s = sqrt(2|x|); that single inflection is what makes the implied
    // volatility iteration provably convergent.
    const double vega = normalised_black_vega(x, s);
    return vega * (x * x / (s * s * s) - 0.25 * s);
}

BlackInflection normalised_black_inflection(double x) noexcept {
    const double s_c = std::sqrt(2.0 * std::abs(x));
    if (s_c == 0.0) return {0.0, 0.0};
    return {s_c, normalised_black_value(-std::abs(x), s_c)};
}

// ===========================================================================
// Prices
// ===========================================================================

double black_undiscounted(double forward, double strike, double vol, double years,
                          OptionType type) noexcept {
    if (!(forward > 0.0) || !(strike > 0.0) || !std::isfinite(vol) || !std::isfinite(years) ||
        years < 0.0 || vol < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double intrinsic = forward_intrinsic(forward, strike, type);
    if (years == 0.0 || vol == 0.0) return intrinsic;

    // Always price the OTM side (x <= 0) and add the exact intrinsic.  This is
    // what makes put-call parity hold to the last bit rather than to within a
    // tolerance: both legs share the same normalised value.
    const double x = -std::abs(std::log(forward / strike));
    const double s = vol * std::sqrt(years);
    const double sqrt_fk = std::sqrt(forward) * std::sqrt(strike);

    const double price = std::fma(sqrt_fk, normalised_black_value(x, s), intrinsic);

    // Clamp to the no-arbitrage interval.  At saturation the exact identity
    // sqrt(F*K) * exp(x/2) == F holds in real arithmetic but not in floating
    // point, so the product can land one ulp above F.  Downstream that is not
    // cosmetic: `implied_volatility` rejects any price at or above the upper
    // bound, so an unclamped pricer can produce a quote its own inverter
    // refuses.
    const PriceBounds bounds = forward_price_bounds(forward, strike, type);
    return (price > bounds.upper) ? bounds.upper
                                  : ((price < bounds.lower) ? bounds.lower : price);
}

double black_scholes_price(double spot, double strike, double vol, double years, double rate,
                           double carry, OptionType type) noexcept {
    if (!(spot > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    const double forward = spot * std::exp((rate - carry) * years);
    const double discount = std::exp(-rate * years);
    return discount * black_undiscounted(forward, strike, vol, years, type);
}

}  // namespace vl
