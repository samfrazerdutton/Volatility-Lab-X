// SPDX-License-Identifier: MIT
/// Validates uncertainty propagation (directive section 16): the delta-
/// method Greek propagation is checked against its own closed-form
/// definition exactly; the portfolio aggregation's same-expiry-correlated
/// / different-expiry-independent rule is checked directly by comparing
/// same-direction vs offsetting exposures within one expiry against two
/// separate expiries, which is the property the whole design exists for.

#include "vl_test_support.hpp"

#include "volatility_lab/risk/uncertainty_propagation.hpp"

#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/options/normalize.hpp"

#include <cmath>

using namespace vl;

namespace {

VolSurface make_surface(double atm_vol = 0.20, double skew = -0.4) {
    std::vector<SliceVariant> slices;
    for (double T : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        SviParams p;
        p.years = T;
        p.b = 0.08;
        p.rho = skew;
        p.m = 0.0;
        p.sigma = 0.13;
        p.a = atm_vol * atm_vol * T - p.b * std::sqrt(p.m * p.m + p.sigma * p.sigma);
        slices.emplace_back(svi_project_to_admissible(p));
    }
    return VolSurface(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
}

struct Fixture {
    VolSurface surface;
    std::vector<OptionQuote> quotes;
    MarketPoint market{100.0, 0.03, 0.0};
};

Fixture make_fixture() {
    Fixture f;
    f.surface = make_surface();
    auto market = generate_market(MarketRegime::Normal);
    auto norm = normalize(market.snapshot);
    (void)assign_weights_by_slice(norm.quotes);
    f.quotes = std::move(norm.quotes);
    return f;
}

}  // namespace

// ---------------------------------------------------------------------------
// propagate_greek_uncertainty: exact delta-method formula
// ---------------------------------------------------------------------------

TEST(UncertaintyPropagation, GreekUncertaintyMatchesTheClosedFormExactly) {
    OptionGreeks g;
    g.vega = 19.79;
    g.vanna = -3.5;
    g.volga = 7.2;
    const auto result = propagate_greek_uncertainty(g, 0.01);
    EXPECT_DOUBLE_EQ(result.price_std_error, std::abs(g.vega) * 0.01);
    EXPECT_DOUBLE_EQ(result.delta_std_error, std::abs(g.vanna) * 0.01);
    EXPECT_DOUBLE_EQ(result.vega_std_error, std::abs(g.volga) * 0.01);
    EXPECT_DOUBLE_EQ(result.vol_std_error, 0.01);
}

TEST(UncertaintyPropagation, NegativeGreeksStillProduceNonNegativeUncertainty) {
    OptionGreeks g;
    g.vega = -19.79;
    g.vanna = -3.5;
    g.volga = -7.2;
    const auto result = propagate_greek_uncertainty(g, 0.01);
    EXPECT_GE(result.price_std_error, 0.0);
    EXPECT_GE(result.delta_std_error, 0.0);
    EXPECT_GE(result.vega_std_error, 0.0);
}

TEST(UncertaintyPropagation, ZeroVolStdErrorGivesZeroGreekUncertainty) {
    OptionGreeks g;
    g.vega = 19.79;
    g.vanna = -3.5;
    g.volga = 7.2;
    const auto result = propagate_greek_uncertainty(g, 0.0);
    EXPECT_EQ(result.price_std_error, 0.0);
    EXPECT_EQ(result.delta_std_error, 0.0);
    EXPECT_EQ(result.vega_std_error, 0.0);
}

// ---------------------------------------------------------------------------
// propagate_portfolio_uncertainty: single leg matches the per-leg formula
// ---------------------------------------------------------------------------

TEST(UncertaintyPropagation, SingleLegPortfolioMatchesItsOwnDollarScaledGreekUncertainty) {
    const auto f = make_fixture();
    const std::vector<Position> book = {{"atm", 10.0, 100.0, 100.0, 0.25, OptionType::Call}};
    std::vector<PositionValuation> vals(1);
    value_portfolio(book, f.surface, f.market, vals);

    const auto pu = propagate_portfolio_uncertainty(book, vals, f.surface, f.quotes, {});
    const auto pu_point = estimate_point_uncertainty(f.surface, f.quotes, vals[0].log_moneyness,
                                                      book[0].years, {});
    ASSERT_FALSE(pu_point.no_local_coverage);

    const double expected_price_std_err =
        std::abs(book[0].signed_notional() * vals[0].greeks.vega) * pu_point.vol_std_error;
    EXPECT_NEAR(pu.price_std_error, expected_price_std_err, 1e-9);
    EXPECT_TRUE(pu.all_positions_had_coverage);
    ASSERT_EQ(pu.price_contribution_by_expiry.size(), 1u);
    EXPECT_NEAR(pu.price_contribution_by_expiry[0].second, expected_price_std_err, 1e-9);
}

// ---------------------------------------------------------------------------
// The central design claim: same-expiry correlated, cross-expiry independent
// ---------------------------------------------------------------------------

TEST(UncertaintyPropagation, SameExpiryOffsettingVegaReducesUncertaintyBelowEitherLegAlone) {
    // Two legs at the SAME expiry with opposite-sign dollar vega: treated
    // as fully correlated within the bucket, so their contributions should
    // partially cancel (|sum| < sum of |each|) -- the whole point of
    // summing before taking the absolute value within a bucket, rather
    // than combining every leg independently.
    const auto f = make_fixture();
    const std::vector<Position> book = {
        {"long_call", 10.0, 100.0, 100.0, 0.25, OptionType::Call},
        {"short_call_otm", -6.0, 100.0, 110.0, 0.25, OptionType::Call},  // smaller, offsetting vega
    };
    std::vector<PositionValuation> vals(book.size());
    value_portfolio(book, f.surface, f.market, vals);
    const auto pu = propagate_portfolio_uncertainty(book, vals, f.surface, f.quotes, {});

    // Compare against treating the two legs as independent (naive quadrature).
    const auto pu_a = propagate_portfolio_uncertainty(
        std::vector<Position>{book[0]}, std::vector<PositionValuation>{vals[0]}, f.surface,
        f.quotes, {});
    const auto pu_b = propagate_portfolio_uncertainty(
        std::vector<Position>{book[1]}, std::vector<PositionValuation>{vals[1]}, f.surface,
        f.quotes, {});
    const double naive_independent =
        std::sqrt(pu_a.price_std_error * pu_a.price_std_error +
                 pu_b.price_std_error * pu_b.price_std_error);

    ASSERT_EQ(pu.price_contribution_by_expiry.size(), 1u) << "both legs share one expiry";
    EXPECT_LT(pu.price_std_error, naive_independent)
        << "same-expiry offsetting vega must cancel partially, not add in quadrature";
}

TEST(UncertaintyPropagation, DifferentExpiriesCombineInQuadratureNotLinearly) {
    const auto f = make_fixture();
    const std::vector<Position> book = {
        {"short_tenor", 10.0, 100.0, 100.0, 0.0833, OptionType::Call},
        {"long_tenor", 10.0, 100.0, 100.0, 1.0, OptionType::Call},
    };
    std::vector<PositionValuation> vals(book.size());
    value_portfolio(book, f.surface, f.market, vals);
    const auto pu = propagate_portfolio_uncertainty(book, vals, f.surface, f.quotes, {});

    ASSERT_EQ(pu.price_contribution_by_expiry.size(), 2u);
    const double c0 = pu.price_contribution_by_expiry[0].second;
    const double c1 = pu.price_contribution_by_expiry[1].second;
    const double linear_sum = c0 + c1;
    EXPECT_NEAR(pu.price_std_error, std::sqrt(c0 * c0 + c1 * c1), 1e-9);
    EXPECT_LT(pu.price_std_error, linear_sum) << "independent buckets must not add linearly";
}

// ---------------------------------------------------------------------------
// Coverage and degenerate inputs
// ---------------------------------------------------------------------------

TEST(UncertaintyPropagation, NoLocalCoverageIsFlaggedRatherThanProducingInfinity) {
    // An empty quote book: every position's estimate_point_uncertainty call
    // has nothing to draw on, exactly like
    // tests/risk/uncertainty.cpp's own EmptyBookReportsInfiniteUncertainty
    // test -- the direct, robust way to force no_local_coverage, rather
    // than relying on an extreme-but-technically-still-nonzero-weight
    // query point.
    const auto f = make_fixture();
    const std::vector<Position> book = {{"atm", 10.0, 100.0, 100.0, 0.25, OptionType::Call}};
    std::vector<PositionValuation> vals(1);
    value_portfolio(book, f.surface, f.market, vals);
    const auto pu = propagate_portfolio_uncertainty(book, vals, f.surface,
                                                     std::span<const OptionQuote>{}, {});
    EXPECT_FALSE(pu.all_positions_had_coverage);
    EXPECT_EQ(pu.price_std_error, 0.0) << "the uncovered leg must be excluded, not NaN/inf";
}

TEST(UncertaintyPropagation, EmptyPortfolioGivesZeroUncertaintyNotNaN) {
    const auto f = make_fixture();
    const auto pu = propagate_portfolio_uncertainty(std::span<const Position>{},
                                                     std::span<const PositionValuation>{},
                                                     f.surface, f.quotes, {});
    EXPECT_EQ(pu.price_std_error, 0.0);
    EXPECT_EQ(pu.delta_std_error, 0.0);
    EXPECT_TRUE(pu.price_contribution_by_expiry.empty());
}

TEST(UncertaintyPropagation, MismatchedPositionsAndValuationsSizesIsHandledSafely) {
    const auto f = make_fixture();
    const std::vector<Position> book = {{"a", 1.0, 100.0, 100.0, 0.25, OptionType::Call}};
    const std::vector<PositionValuation> vals;  // deliberately empty
    const auto pu = propagate_portfolio_uncertainty(book, vals, f.surface, f.quotes, {});
    EXPECT_EQ(pu.price_std_error, 0.0);
}
