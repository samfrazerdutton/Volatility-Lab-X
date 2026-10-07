// SPDX-License-Identifier: MIT
#include "volatility_lab/pricing/reference.hpp"

#include <cmath>
#include <limits>

#include "volatility_lab/math/reference_special.hpp"
#include "volatility_lab/pricing/black.hpp"

namespace vl::reference {

using math::dd_const::kInvSqrt2Pi;
using math::exp_dd;
using math::reference::norm_cdf_dd;
using math::reference::norm_pdf_dd;
using math::sqrt_dd;

namespace {

/// Below this w/|x| even 106 bits cannot absorb the cancellation in the
/// textbook form, and the reference switches to the dd Taylor series.  The
/// production code switches at 1e-2, eight decades earlier, so there is a wide
/// band in which the double series is checked against the dd textbook form.
constexpr double kDdTextbookFloor = 1.0e-10;

/// Taylor series in t, in double-double.  Same recursion as the production
/// version (see pricing/black.hpp); the duplication is intentional -- this one
/// may be slow and run to dd convergence.
DDouble normalised_black_series_dd(DDouble x, DDouble s) noexcept {
    const DDouble h = x / s;
    const DDouble t = s * 0.5;

    DDouble g_prev = norm_pdf_dd(h);
    DDouble g_cur(0.0);
    DDouble b_cur(0.0);
    DDouble sum(0.0);
    DDouble t_pow_over_fact(1.0);

    for (int n = 0; n < 400; ++n) {
        const DDouble b_next = h * b_cur + (n == 0 ? g_prev : g_cur) * 2.0;
        const DDouble g_next = (n == 0) ? DDouble(0.0) : g_prev * (-static_cast<double>(n));

        t_pow_over_fact = t_pow_over_fact / static_cast<double>(n + 1) * t;
        const DDouble term = b_next * t_pow_over_fact;
        sum += term;

        if (n >= 2 && std::abs(term.hi()) <= 1e-36 * std::abs(sum.hi())) break;

        if (n != 0) g_prev = g_cur;
        g_cur = g_next;
        b_cur = b_next;
    }
    return sum;
}

}  // namespace

DDouble normalised_black_dd(DDouble x, DDouble s) noexcept {
    if (s.hi() <= 0.0) return DDouble(0.0);
    if (x.hi() > 0.0) x = -x;  // fold to the OTM side

    const double w = s.hi() * s.hi();
    const double ax = std::abs(x.hi());

    if (ax == 0.0) {
        // ATM: b = 2 Phi(s/2) - 1.  In dd the subtraction of 1 costs nothing
        // because Phi(s/2) is known to 31 digits.
        return norm_cdf_dd(s * 0.5) * 2.0 - 1.0;
    }
    if (w < kDdTextbookFloor * ax) {
        return normalised_black_series_dd(x, s);
    }

    // Textbook form (1), evaluated at 106 bits:
    //     b = exp(x/2) Phi(h+t) - exp(-x/2) Phi(h-t)
    const DDouble h = x / s;
    const DDouble t = s * 0.5;
    const DDouble half_x = x * 0.5;
    const DDouble term1 = exp_dd(half_x) * norm_cdf_dd(h + t);
    const DDouble term2 = exp_dd(-half_x) * norm_cdf_dd(h - t);
    const DDouble v = term1 - term2;
    return (v.hi() < 0.0) ? DDouble(0.0) : v;
}

double normalised_black_ref(double x, double s) noexcept {
    return normalised_black_dd(DDouble(x), DDouble(s)).to_double();
}

double black_undiscounted_ref(double forward, double strike, double vol, double years,
                              OptionType type) noexcept {
    if (!(forward > 0.0) || !(strike > 0.0) || vol < 0.0 || years < 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (years == 0.0 || vol == 0.0) return forward_intrinsic(forward, strike, type);

    // Everything in dd, including the log and the sqrt, so that the only
    // double rounding is the final collapse.
    const DDouble F(forward);
    const DDouble K(strike);
    const DDouble x = math::log_dd(F / K);
    const DDouble s = DDouble(vol) * sqrt_dd(DDouble(years));
    const DDouble sqrt_fk = sqrt_dd(F * K);

    const DDouble otm = sqrt_fk * normalised_black_dd(x, s);
    // Intrinsic in dd as well: F - K is exact in double for ordinary inputs,
    // but not when F and K differ in exponent, and the reference should not
    // inherit that.
    const DDouble intrinsic = (type == OptionType::Call) ? (F - K) : (K - F);
    const DDouble total = (intrinsic.hi() > 0.0) ? otm + intrinsic : otm;
    return total.to_double();
}

double normalised_black_vega_ref(double x, double s) noexcept {
    if (!(s > 0.0)) return 0.0;
    const DDouble xd(x);
    const DDouble sd(s);
    const DDouble h = sd * 1.0e-12;
    const DDouble up = normalised_black_dd(xd, sd + h);
    const DDouble dn = normalised_black_dd(xd, sd - h);
    return ((up - dn) / (h * 2.0)).to_double();
}

ReferenceGreeks black_greeks_ref(double forward, double strike, double vol, double years,
                                 OptionType type) noexcept {
    // Central-difference steps.  In dd the working epsilon is ~1e-31, so the
    // round-off term eps/h and the truncation term h^2 balance at
    // h ~ eps^(1/3) ~ 2e-11.  1e-9 is used instead: slightly larger than
    // optimal, giving truncation ~1e-18 and round-off ~1e-22, which keeps the
    // result comfortably better than the 1e-16 quantity under test while
    // staying far away from any risk of the step underflowing the dd
    // representation of F.
    const DDouble F(forward);
    const DDouble hF = F * 1.0e-9;
    const DDouble hV = DDouble(vol) * 1.0e-9;
    const DDouble hT = DDouble(years) * 1.0e-9;

    auto price = [&](DDouble f, DDouble v, DDouble tt) {
        const DDouble x = math::log_dd(f / DDouble(strike));
        const DDouble s = v * sqrt_dd(tt);
        const DDouble sqrt_fk = sqrt_dd(f * DDouble(strike));
        const DDouble otm = sqrt_fk * normalised_black_dd(x, s);
        const DDouble intr = (type == OptionType::Call) ? (f - DDouble(strike))
                                                        : (DDouble(strike) - f);
        return (intr.hi() > 0.0) ? otm + intr : otm;
    };

    const DDouble V(vol);
    const DDouble T(years);

    const DDouble p0 = price(F, V, T);
    const DDouble pFp = price(F + hF, V, T);
    const DDouble pFm = price(F - hF, V, T);
    const DDouble pVp = price(F, V + hV, T);
    const DDouble pVm = price(F, V - hV, T);
    const DDouble pTp = price(F, V, T + hT);
    const DDouble pTm = price(F, V, T - hT);
    const DDouble pFpVp = price(F + hF, V + hV, T);
    const DDouble pFpVm = price(F + hF, V - hV, T);
    const DDouble pFmVp = price(F - hF, V + hV, T);
    const DDouble pFmVm = price(F - hF, V - hV, T);

    ReferenceGreeks g{};
    g.price = p0.to_double();
    g.delta = ((pFp - pFm) / (hF * 2.0)).to_double();
    g.gamma = ((pFp - p0 * 2.0 + pFm) / (hF * hF)).to_double();
    g.vega = ((pVp - pVm) / (hV * 2.0)).to_double();
    g.volga = ((pVp - p0 * 2.0 + pVm) / (hV * hV)).to_double();
    g.vanna = ((pFpVp - pFpVm - pFmVp + pFmVm) / (hF * hV * 4.0)).to_double();
    g.theta = ((pTp - pTm) / (hT * 2.0)).to_double();
    return g;
}

SpotGreeks black_scholes_greeks_ref(double spot, double strike, double vol, double years,
                                    double rate, double carry, OptionType type) noexcept {
    // Price as a function of the four free axes (S, sigma, T, r), with K, q
    // fixed.  Built from the same dd primitives as the rest of this file --
    // log_dd, exp_dd, sqrt_dd, normalised_black_dd -- but driven from the spot
    // measure outward, so F and DF appear as *derived* quantities inside the
    // lambda rather than as the free variables, exactly as they are in a real
    // pricer.
    const DDouble K(strike);
    auto price = [&](DDouble S, DDouble sig, DDouble T, DDouble r) {
        const DDouble F = S * exp_dd((r - DDouble(carry)) * T);
        const DDouble DF = exp_dd(-r * T);
        DDouble x = math::log_dd(F / K);
        if (x.hi() > 0.0) x = -x;  // fold to the OTM side, as black_undiscounted does
        const DDouble s = sig * sqrt_dd(T);
        const DDouble sqrt_fk = sqrt_dd(F * K);
        const DDouble otm = sqrt_fk * normalised_black_dd(x, s);
        const DDouble intr = (type == OptionType::Call) ? (F - K) : (K - F);
        const DDouble U = (intr.hi() > 0.0) ? otm + intr : otm;
        return DF * U;
    };

    const DDouble S(spot);
    const DDouble V(vol);
    const DDouble T(years);
    const DDouble R(rate);

    // Steps as fractions of each axis's own scale; see black_greeks_ref above
    // for why 1e-9 is the chosen balance between truncation and round-off at
    // dd precision.  `years` can be very small (the vol-crush regime goes
    // down to a few hours), so hT additionally has an absolute floor.
    const DDouble hS = S * 1.0e-9;
    const DDouble hV = V * 1.0e-9;
    const DDouble hT = DDouble(std::max(years * 1.0e-9, 1.0e-13));
    const DDouble hR = DDouble(std::max(std::abs(rate), 0.01)) * 1.0e-9;

    // Speed is a *third* derivative, estimated from a 4-point stencil whose
    // error has a round-off term scaling as eps_dd/h^3 and a truncation term
    // scaling as h^2.  At the same h = 1.0e-9*S used for the first and second
    // derivatives above, h^3 ~= 1.0e-27*S^3 and dividing a dd-precision
    // (~1.0e-31) numerator by that leaves only ~1.0e-4 relative digits --
    // exactly the "BAD" discrepancy an early version of the Greeks test saw
    // against an otherwise-correct production formula.  A larger step
    // rebalances the two error terms: at h ~= 1.0e-6*S, round-off is
    // ~1.0e-31/1.0e-18 ~= 1.0e-13 and truncation is still ~1.0e-12, both far
    // below the double-precision quantity this is meant to validate.
    const DDouble hS3 = S * 1.0e-6;

    const DDouble p0 = price(S, V, T, R);
    const DDouble pSp = price(S + hS, V, T, R);
    const DDouble pSm = price(S - hS, V, T, R);
    const DDouble pSp3 = price(S + hS3, V, T, R);
    const DDouble pSm3 = price(S - hS3, V, T, R);
    const DDouble pSp2x3 = price(S + hS3 * 2.0, V, T, R);
    const DDouble pSm2x3 = price(S - hS3 * 2.0, V, T, R);
    const DDouble pVp = price(S, V + hV, T, R);
    const DDouble pVm = price(S, V - hV, T, R);
    const DDouble pTp = price(S, V, T + hT, R);
    const DDouble pTm = price(S, V, T - hT, R);
    const DDouble pRp = price(S, V, T, R + hR);
    const DDouble pRm = price(S, V, T, R - hR);
    const DDouble pSpVp = price(S + hS, V + hV, T, R);
    const DDouble pSpVm = price(S + hS, V - hV, T, R);
    const DDouble pSmVp = price(S - hS, V + hV, T, R);
    const DDouble pSmVm = price(S - hS, V - hV, T, R);
    const DDouble pSpTp = price(S + hS, V, T + hT, R);
    const DDouble pSpTm = price(S + hS, V, T - hT, R);
    const DDouble pSmTp = price(S - hS, V, T + hT, R);
    const DDouble pSmTm = price(S - hS, V, T - hT, R);

    SpotGreeks g{};
    g.price = p0.to_double();
    g.delta = ((pSp - pSm) / (hS * 2.0)).to_double();
    g.gamma = ((pSp - p0 * 2.0 + pSm) / (hS * hS)).to_double();
    g.vega = ((pVp - pVm) / (hV * 2.0)).to_double();
    g.volga = ((pVp - p0 * 2.0 + pVm) / (hV * hV)).to_double();
    g.vanna = ((pSpVp - pSpVm - pSmVp + pSmVm) / (hS * hV * 4.0)).to_double();
    // Theta in the calendar-decay convention: d/dt = -d/dT.
    g.theta = (-(pTp - pTm) / (hT * 2.0)).to_double();
    g.rho = ((pRp - pRm) / (hR * 2.0)).to_double();
    // Charm = dDelta/dt = -d2Price/(dS dT).
    g.charm = (-(pSpTp - pSpTm - pSmTp + pSmTm) / (hS * hT * 4.0)).to_double();
    // Speed = d3Price/dS3, standard central third-derivative stencil, at the
    // wider hS3 step derived above.
    g.speed =
        ((pSp2x3 - pSp3 * 2.0 + pSm3 * 2.0 - pSm2x3) / (hS3 * hS3 * hS3 * 2.0)).to_double();
    return g;
}

double implied_vol_ref(double normalised_price, double x) noexcept {
    // Bisection in double-double on b(x, s) - beta.  No derivatives, no
    // initial guess, no branch logic: nothing shared with the production
    // solver, so agreement is evidence rather than coincidence.
    const double ax = std::abs(x);
    const double upper_bound = std::exp(-0.5 * ax);
    if (!(normalised_price > 0.0)) return 0.0;
    if (normalised_price >= upper_bound) {
        return std::numeric_limits<double>::infinity();
    }

    const DDouble xd(-ax);
    const DDouble beta(normalised_price);

    DDouble lo(0.0);
    // Grow the upper end until it brackets.  b is increasing in s, so this
    // terminates; the cap of 2^20 corresponds to a total volatility of a
    // million, far outside any admissible input.
    DDouble hi(1.0);
    for (int i = 0; i < 20 && normalised_black_dd(xd, hi) < beta; ++i) {
        lo = hi;
        hi = hi * 2.0;
    }

    // 120 halvings resolves 2^-120 of the initial width, i.e. below the dd
    // epsilon -- the loop is bounded by precision, not by a tolerance guess.
    for (int i = 0; i < 120; ++i) {
        const DDouble mid = (lo + hi) * 0.5;
        if (normalised_black_dd(xd, mid) < beta) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return ((lo + hi) * 0.5).to_double();
}

}  // namespace vl::reference
