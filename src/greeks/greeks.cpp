// SPDX-License-Identifier: MIT
#include "volatility_lab/greeks/greeks.hpp"

#include <algorithm>
#include <cmath>

#include "volatility_lab/math/special.hpp"
#include "volatility_lab/pricing/black.hpp"

namespace vl {

using math::norm_cdf_hp;
using math::norm_pdf;

namespace {

/// Every field NaN, not just price: a caller that reads `g.delta` after an
/// invalid input seeing a plausible-looking `0.0` (the default-constructed
/// value) rather than NaN would have no signal that anything was wrong.
/// Found by fuzzing (tests/fuzz/numerical_fuzz.cpp) against the *other*
/// gap this helper also fixes -- see its call sites below.
OptionGreeks nan_greeks() noexcept {
    constexpr double n = std::numeric_limits<double>::quiet_NaN();
    return OptionGreeks{n, n, n, n, n, n, n, n, n, n};
}

}  // namespace

OptionGreeks black_scholes_greeks(double spot, double strike, double vol, double years,
                                  double rate, double carry, OptionType type) noexcept {
    OptionGreeks g{};
    if (!(spot > 0.0) || !(strike > 0.0) || !std::isfinite(rate) || !std::isfinite(carry)) {
        return nan_greeks();
    }
    // NaN vol/years must propagate as "unknown", not be folded into the
    // T->0/sigma->0 *limit* handling just below: that limit has a genuine,
    // well-defined answer (the option is worth its intrinsic), which is a
    // specific, confident claim NaN does not entitle this function to
    // make. The `!(x > 0.0)` idiom just below is written that way
    // specifically so it *also* catches NaN (a deliberate, common idiom
    // for "reject non-positive-or-NaN"), which is exactly how a NaN vol
    // ended up silently answering "the price is the discounted intrinsic"
    // instead of "I don't know" -- found by fuzzing, not assumed; fixed by
    // checking NaN explicitly, first.
    if (std::isnan(vol) || std::isnan(years)) {
        return nan_greeks();
    }

    const double w = payoff_sign(type);
    const double forward = spot * std::exp((rate - carry) * years);
    const double df = std::exp(-rate * years);
    const double dq = std::exp(-carry * years);  // the "growth-adjusted" discount, exp(-qT)

    // Degenerate axes: T<=0 or sigma<=0 collapse the whole smooth-Greek
    // machinery (d1/d2 divide by s=sigma*sqrt(T)).  The option is worth its
    // intrinsic there and delta is a step function; every higher derivative
    // is zero almost everywhere, which is the honest answer rather than a
    // division-by-zero NaN.
    if (!(years > 0.0) || !(vol > 0.0)) {
        g.price = df * forward_intrinsic(forward, strike, type);
        g.delta = (w * (forward - strike) > 0.0) ? w * dq : 0.0;
        return g;
    }

    const double s = vol * std::sqrt(years);
    const double xi = std::log(forward / strike);
    const double d1 = xi / s + 0.5 * s;
    const double d2 = d1 - s;

    // norm_cdf_hp rather than the fast norm_cdf: Greeks are read directly off
    // d1/d2 with no OTM folding, so unlike the price path there is no
    // complementary cancellation to absorb the fast path's tail degradation
    // (see math/special.hpp).  A deep-wing option is exactly where a risk
    // system most needs Phi(d1) to still be meaningful.
    const double n_w_d1 = norm_cdf_hp(w * d1);
    const double n_w_d2 = norm_cdf_hp(w * d2);
    const double pdf_d1 = norm_pdf(d1);

    const double sqrt_t = std::sqrt(years);

    g.price = w * df * (forward * n_w_d1 - strike * n_w_d2);
    g.delta = w * dq * n_w_d1;
    g.gamma = dq * pdf_d1 / (spot * vol * sqrt_t);
    g.vega = spot * dq * pdf_d1 * sqrt_t;
    g.rho = w * strike * years * df * n_w_d2;
    g.theta = -spot * dq * pdf_d1 * vol / (2.0 * sqrt_t) - w * rate * strike * df * n_w_d2 +
              w * carry * spot * dq * n_w_d1;
    g.vanna = -dq * pdf_d1 * d2 / vol;
    g.volga = g.vega * d1 * d2 / vol;
    // Derived as dDelta/dT (differentiating Delta = w*dq*Phi(w*d1) w.r.t. the
    // *time-to-expiry* axis holding S, r, sigma fixed), then negated to match
    // the calendar-decay convention charm = dDelta/dt = -dDelta/dT that every
    // other time-sensitive Greek in this file uses (see theta above).  The
    // negation was missing in the first version -- caught by validating
    // against reference::black_scholes_greeks_ref, which disagreed in sign
    // (relative error exactly 2.0, i.e. equal magnitude, opposite sign) at
    // every point tested.
    g.charm = -dq * (-w * carry * n_w_d1 +
                     pdf_d1 * ((rate - carry) / (vol * sqrt_t) - d1 / (2.0 * years) +
                              vol / (2.0 * sqrt_t)));
    g.speed = -g.gamma / spot * (d1 / (vol * sqrt_t) + 1.0);
    return g;
}

OptionGreeks black_scholes_greeks_forward(double forward, double strike, double vol,
                                          double years, double rate, double discount,
                                          OptionType type) noexcept {
    // Recover (spot, carry) from (forward, discount) so the single
    // implementation above still applies.  discount = exp(-rate*years), so
    // rate is already fully determined by `rate` itself (passed explicitly,
    // since discount alone cannot separate rate from years at years==0); carry
    // is recovered from forward = spot*exp((rate-carry)*years).  This overload
    // exists for callers who already have F and DF from a surface query and
    // would otherwise have to invert back to a spot -- here that inversion is
    // done once, centrally, instead of ad hoc at every call site.
    if (!(forward > 0.0) || !(discount > 0.0) || !(years > 0.0)) {
        return nan_greeks();
    }
    // spot is a free choice here -- only F and DF are observable -- so fix
    // spot := forward and solve carry accordingly; this makes carry whatever
    // it needs to be to reproduce the given forward from that spot, and all
    // the *economic* quantities (price, Greeks) are invariant to that choice
    // since they depend on F and DF only, never on spot and carry separately
    // except through dPrice/dS in the chain rule below.
    //
    // Concretely: with spot := forward, (rate - carry)*years == 0, so
    // carry = rate.  Delta/gamma/vanna/speed as computed by
    // black_scholes_greeks are dPrice/dS at *that* spot choice, which equals
    // dPrice/dF (the forward delta) because dF/dS = exp((rate-carry)*years) = 1
    // at this particular spot.  That is exactly what a caller who only has F
    // and DF can mean by "delta" without an independent spot/carry split.
    return black_scholes_greeks(forward, strike, vol, years, rate, rate, type);
}

void black_scholes_greeks_batch(std::span<const double> spot, std::span<const double> strike,
                                std::span<const double> vol, std::span<const double> years,
                                std::span<const double> rate, std::span<const double> carry,
                                std::span<const std::int8_t> type_sign,
                                std::span<OptionGreeks> out) noexcept {
    const std::size_t n = std::min({spot.size(), strike.size(), vol.size(), years.size(),
                                    rate.size(), carry.size(), type_sign.size(), out.size()});
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = black_scholes_greeks(spot[i], strike[i], vol[i], years[i], rate[i], carry[i],
                                      static_cast<OptionType>(type_sign[i]));
    }
}

PortfolioGreeks aggregate_greeks(std::span<const OptionGreeks> greeks,
                                 std::span<const double> quantity,
                                 std::span<const double> multiplier) noexcept {
    PortfolioGreeks out{};
    const std::size_t n = std::min({greeks.size(), quantity.size(), multiplier.size()});
    // Accumulated in index order -- part of the library-wide determinism
    // contract: the same positions in the same order must give a bit-
    // identical aggregate regardless of thread count, which the parallel
    // reduction phase will rely on by fixing chunk boundaries rather than
    // changing this loop.
    for (std::size_t i = 0; i < n; ++i) {
        const double w = quantity[i] * multiplier[i];
        if (w == 0.0) continue;
        const OptionGreeks& g = greeks[i];
        out.value += w * g.price;
        out.delta += w * g.delta;
        out.gamma += w * g.gamma;
        out.vega += w * g.vega;
        out.theta += w * g.theta;
        out.rho += w * g.rho;
        out.vanna += w * g.vanna;
        out.volga += w * g.volga;
        out.charm += w * g.charm;
        out.speed += w * g.speed;
    }
    return out;
}

double TaylorVsExact::order1_relative_error() const noexcept {
    const double denom = std::max(std::abs(exact_pnl), 1e-12 * std::max(std::abs(base_price), 1.0));
    return order1_error() / denom;
}

double TaylorVsExact::order2_relative_error() const noexcept {
    const double denom = std::max(std::abs(exact_pnl), 1e-12 * std::max(std::abs(base_price), 1.0));
    return order2_error() / denom;
}

TaylorVsExact taylor_vs_exact_reprice(const OptionGreeks& greeks, double spot, double strike,
                                      double vol, double years, double rate, double carry,
                                      OptionType type, const GreekBump& bump) noexcept {
    TaylorVsExact out{};
    // base_price is recomputed from black_scholes_price rather than taken
    // from greeks.price: OptionGreeks.price (direct d1/d2) and
    // black_scholes_price (the normalised-Black erfcx form) are two
    // independently-derived formulas for the same quantity, and while both
    // are separately validated, they are not guaranteed to be *bit*-identical.
    // The comparison this function exists to make -- Taylor estimate vs
    // exact reprice -- is only honest if "exact at zero bump" reproduces
    // exactly the number the Taylor series was built from, so both base and
    // bumped prices are taken from the same code path.
    out.base_price = black_scholes_price(spot, strike, vol, years, rate, carry, type);

    const double new_spot = spot + bump.d_spot;
    const double new_vol = std::max(vol + bump.d_vol, 0.0);
    const double new_rate = rate + bump.d_rate;
    const double new_years = std::max(years + bump.d_years, 0.0);

    out.exact_price = black_scholes_price(new_spot, strike, new_vol, new_years, new_rate,
                                          carry, type);
    out.exact_pnl = out.exact_price - out.base_price;

    const double dS = bump.d_spot;
    const double dV = bump.d_vol;
    const double dR = bump.d_rate;
    // The "time" axis of the Taylor expansion is calendar time t, not T, and
    // theta is already stated in that convention (d/dt = -d/dT), so the
    // natural bump for the Taylor series is dt = -d_years (time passing
    // *shrinks* years-to-expiry).  A caller handing in a positive d_years
    // (more time to expiry, e.g. comparing against a longer-dated option) is
    // still handled correctly: the sign simply flips, matching theta*dt with
    // dt = -dT exactly as the field comment states.
    const double dt = -bump.d_years;

    out.order1_pnl = greeks.delta * dS + greeks.vega * dV + greeks.theta * dt + greeks.rho * dR;
    out.order2_pnl = out.order1_pnl + 0.5 * greeks.gamma * dS * dS +
                     0.5 * greeks.volga * dV * dV + greeks.vanna * dS * dV +
                     greeks.charm * dS * dt;
    return out;
}

}  // namespace vl
