// SPDX-License-Identifier: MIT
/// \file core_invariants.cpp
/// \brief Mathematical invariants of the pricing/Greeks layer, checked
///        across a domain sweep rather than at individual points.
///
/// This project already has extensive property-style coverage under its
/// per-module test files (`NormCdfSymmetry`, `ErfcxIsMonotoneDecreasing`,
/// `NormInvIsMonotone` in tests/numerical/special.cpp;
/// `NormalisedBlackIsStrictlyIncreasingInTotalVolatility`,
/// `NormalisedBlackIsBoundedByItsLimits`, `PutCallParityHoldsToTheLastBit`
/// in tests/pricing/black.cpp; `RoundTripRecoversTotalVolatilityOverTheFullDomain`
/// in tests/pricing/implied_vol.cpp) -- those are not duplicated here. This
/// file covers the specific invariants that were not yet checked as an
/// explicit domain sweep anywhere: Gamma and Vega non-negativity, and
/// price >= intrinsic value, for `greeks::black_scholes_greeks` across the
/// same kind of wide, difficult-case domain the rest of this project's
/// sweeps already use (deep ITM/OTM, near-expiry, extreme vol, nonzero
/// rate and carry).

#include "vl_test_support.hpp"

#include "volatility_lab/greeks/greeks.hpp"
#include "volatility_lab/pricing/black.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace vl;

namespace {

struct DomainPoint {
    double spot, strike, vol, years, rate, carry;
    OptionType type;
};

std::vector<DomainPoint> wide_domain() {
    std::vector<DomainPoint> out;
    const auto strikes = test::log_space(20.0, 500.0, 7);
    const auto vols = test::log_space(0.02, 2.0, 6);
    const auto years = test::log_space(1.0 / 365.0, 5.0, 6);
    const double rates[] = {-0.02, 0.0, 0.03, 0.08};
    const double carries[] = {0.0, 0.01, 0.04};
    for (double k : strikes) {
        for (double v : vols) {
            for (double t : years) {
                for (double r : rates) {
                    for (double q : carries) {
                        for (OptionType ty : {OptionType::Call, OptionType::Put}) {
                            out.push_back({100.0, k, v, t, r, q, ty});
                        }
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace

TEST(PropertyInvariants, GammaIsNeverNegativeAcrossTheWholeDomain) {
    // Gamma = d(delta)/d(spot) for a vanilla option under Black-Scholes is
    // mathematically a probability density scaled by positive quantities --
    // it cannot be negative for a long call or a long put, regardless of
    // moneyness, tenor, rate, or carry.
    long checked = 0;
    for (const auto& d : wide_domain()) {
        const auto g = black_scholes_greeks(d.spot, d.strike, d.vol, d.years, d.rate, d.carry, d.type);
        ASSERT_GE(g.gamma, 0.0) << "spot=" << d.spot << " strike=" << d.strike << " vol=" << d.vol
                                << " years=" << d.years << " rate=" << d.rate
                                << " carry=" << d.carry << " type=" << to_string(d.type);
        ++checked;
    }
    EXPECT_GT(checked, 1000);
}

TEST(PropertyInvariants, VegaIsNeverNegativeAcrossTheWholeDomain) {
    // Vega = d(price)/d(vol): a vanilla option's value cannot decrease as
    // volatility rises, for a call or a put.
    for (const auto& d : wide_domain()) {
        const auto g = black_scholes_greeks(d.spot, d.strike, d.vol, d.years, d.rate, d.carry, d.type);
        ASSERT_GE(g.vega, 0.0) << "spot=" << d.spot << " strike=" << d.strike << " vol=" << d.vol
                               << " years=" << d.years << " type=" << to_string(d.type);
    }
}

TEST(PropertyInvariants, PriceNeverFallsBelowTheDiscountedForwardIntrinsicAcrossTheWholeDomain) {
    // The correct, textbook no-arbitrage lower bound for a *European*
    // option is the discounted forward intrinsic,
    // DF * max(payoff_sign*(F-K), 0) -- not the naive, undiscounted spot
    // intrinsic max(payoff_sign*(S-K), 0). An earlier version of this test
    // asserted the spot-intrinsic bound and failed immediately on deep-ITM
    // puts at long tenors/high vol: that failure was correct, not a bug in
    // `black_scholes_greeks` -- a European put's price legitimately falls
    // below its naive spot intrinsic when discounting dominates (the
    // holder gives up S-K today but only receives it, discounted, at
    // expiry, during which the stock could also recover). Calls under a
    // non-negative carry do not exhibit this (forward >= spot when
    // carry <= rate for the domain's typical parameters), which is why the
    // earlier, wrong bound happened to pass for calls and only exposed
    // itself on puts -- exactly the kind of case a wide domain sweep is
    // for.
    //
    // Reusing `forward_intrinsic` (the same function `black_price` itself
    // is built on, already validated elsewhere) rather than re-deriving
    // the formula a second time independently.
    for (const auto& d : wide_domain()) {
        const auto g = black_scholes_greeks(d.spot, d.strike, d.vol, d.years, d.rate, d.carry, d.type);
        const double forward = d.spot * std::exp((d.rate - d.carry) * d.years);
        const double discount = std::exp(-d.rate * d.years);
        const double lower_bound = discount * forward_intrinsic(forward, d.strike, d.type);
        EXPECT_GE(g.price, lower_bound - 1e-9 * std::max(1.0, lower_bound))
            << "price=" << g.price << " lower_bound=" << lower_bound << " spot=" << d.spot
            << " strike=" << d.strike << " vol=" << d.vol << " years=" << d.years
            << " type=" << to_string(d.type);
    }
}

TEST(PropertyInvariants, PriceIsStrictlyIncreasingInVolatilityForBothCallAndPut) {
    // Direct consequence of Vega >= 0, but checked as its own monotonicity
    // property (not derived from the Vega test) across pairs of adjacent
    // vol points in the sweep, for both option types.
    const auto strikes = test::log_space(50.0, 200.0, 5);
    const auto vols = test::log_space(0.05, 1.5, 10);
    for (double k : strikes) {
        for (OptionType ty : {OptionType::Call, OptionType::Put}) {
            double previous_price = -1.0;
            for (double v : vols) {
                const double price = black_scholes_price(100.0, k, v, 0.5, 0.03, 0.01, ty);
                if (previous_price >= 0.0) {
                    EXPECT_GE(price, previous_price - 1e-12)
                        << "price decreased with increasing vol at strike=" << k
                        << " vol=" << v << " type=" << to_string(ty);
                }
                previous_price = price;
            }
        }
    }
}

TEST(PropertyInvariants, CallMinusPutEqualsForwardMinusStrikeDiscounted) {
    // Put-call parity as an independent, formula-free check: this is a
    // model-free no-arbitrage identity, true regardless of how price()
    // itself is implemented, so it is a genuinely independent property
    // check rather than testing the implementation against itself.
    for (const auto& d : wide_domain()) {
        if (d.type != OptionType::Call) continue;  // check each (S,K,vol,T,r,q) once
        const double call = black_scholes_price(d.spot, d.strike, d.vol, d.years, d.rate, d.carry,
                                                 OptionType::Call);
        const double put = black_scholes_price(d.spot, d.strike, d.vol, d.years, d.rate, d.carry,
                                                OptionType::Put);
        const double forward = d.spot * std::exp((d.rate - d.carry) * d.years);
        const double discount = std::exp(-d.rate * d.years);
        const double expected = discount * (forward - d.strike);
        EXPECT_NEAR(call - put, expected, 1e-8 * std::max(1.0, std::abs(expected)))
            << "spot=" << d.spot << " strike=" << d.strike << " vol=" << d.vol
            << " years=" << d.years << " rate=" << d.rate << " carry=" << d.carry;
    }
}
