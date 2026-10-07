// SPDX-License-Identifier: MIT
/// Validates the portfolio and PnL attribution layer: the exact
/// reconciliation identity (brief section 10 forbids a silently dropped
/// residual), aggregation consistency with the Greeks module it builds on,
/// and the research-query helper that ranks positions by PnL contribution.

#include "vl_test_support.hpp"

#include "volatility_lab/portfolio/portfolio.hpp"
#include "volatility_lab/pricing/black.hpp"

#include <cmath>
#include <vector>

using namespace vl;
using vl::math::rel_error;

namespace {

/// A small two-slice surface, deliberately with a skew and a term structure,
/// so a spot or vol move genuinely has different effects on different legs
/// of a book -- a flat surface would let several bugs hide behind every leg
/// agreeing by coincidence.
VolSurface make_surface(double atm_short = 0.22, double atm_long = 0.20,
                        double skew = -0.4) {
    std::vector<SliceVariant> slices;
    SviParams s1;
    s1.years = 0.25;
    s1.b = 0.08;
    s1.rho = skew;
    s1.m = 0.0;
    s1.sigma = 0.12;
    s1.a = atm_short * atm_short * s1.years -
           s1.b * (s1.rho * (0.0 - s1.m) + std::sqrt(s1.m * s1.m + s1.sigma * s1.sigma));
    slices.emplace_back(svi_project_to_admissible(s1));

    SviParams s2;
    s2.years = 1.0;
    s2.b = 0.07;
    s2.rho = skew;
    s2.m = 0.0;
    s2.sigma = 0.15;
    s2.a = atm_long * atm_long * s2.years -
           s2.b * (s2.rho * (0.0 - s2.m) + std::sqrt(s2.m * s2.m + s2.sigma * s2.sigma));
    slices.emplace_back(svi_project_to_admissible(s2));

    return VolSurface(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
}

std::vector<Position> make_book() {
    return {
        Position{"long_call_atm", 10.0, 100.0, 100.0, 0.25, OptionType::Call},
        Position{"short_put_otm", -5.0, 100.0, 90.0, 0.25, OptionType::Put},
        Position{"long_call_1y", 3.0, 100.0, 110.0, 1.0, OptionType::Call},
        Position{"short_call_1y", -3.0, 100.0, 130.0, 1.0, OptionType::Call},
    };
}

}  // namespace

// ===========================================================================
// Valuation
// ===========================================================================

TEST(Portfolio, ValuePositionMatchesDirectBlackScholesGreeks) {
    const auto surface = make_surface();
    const MarketPoint market{100.0, 0.03, 0.01};
    const Position p{"x", 1.0, 1.0, 105.0, 0.5, OptionType::Call};

    const auto val = value_position(p, surface, market);
    const double forward = market.spot * std::exp((market.rate - market.carry) * p.years);
    const double k = log_moneyness(forward, p.strike);
    const double expected_vol = surface.vol(k, p.years);
    EXPECT_NEAR(val.vol_used, expected_vol, 1e-15);

    const auto direct = black_scholes_greeks(market.spot, p.strike, expected_vol, p.years,
                                             market.rate, market.carry, p.type);
    EXPECT_EQ(val.greeks.price, direct.price);
    EXPECT_EQ(val.greeks.delta, direct.delta);
    EXPECT_EQ(val.greeks.vega, direct.vega);
}

TEST(Portfolio, PortfolioGreeksMatchManualAggregation) {
    const auto surface = make_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const auto book = make_book();

    const auto agg = portfolio_greeks(book, surface, market);

    double manual_value = 0.0, manual_delta = 0.0;
    for (const auto& p : book) {
        const auto v = value_position(p, surface, market);
        manual_value += p.signed_notional() * v.greeks.price;
        manual_delta += p.signed_notional() * v.greeks.delta;
    }
    EXPECT_NEAR(agg.value, manual_value, 1e-9 * std::abs(manual_value));
    EXPECT_NEAR(agg.delta, manual_delta, 1e-9 * std::max(1.0, std::abs(manual_delta)));
}

TEST(Portfolio, InvalidMarketOrContractReportsNaNRatherThanAPlausibleNumber) {
    const auto surface = make_surface();
    EXPECT_TRUE(std::isnan(
        value_position(Position{"x", 1, 1, 100, 1, OptionType::Call}, surface,
                       MarketPoint{0.0, 0.03, 0.0})
            .greeks.price));
    EXPECT_TRUE(std::isnan(
        value_position(Position{"x", 1, 1, -5, 1, OptionType::Call}, surface,
                       MarketPoint{100, 0.03, 0.0})
            .greeks.price));
}

// ===========================================================================
// PnL attribution: the exact reconciliation identity
// ===========================================================================

TEST(Portfolio, AttributionReconcilesExactlyForOrdinaryMarketMoves) {
    // Section 10's hard requirement: Total PnL == sum(components) + residual,
    // *exactly*, by construction.  Checked across a range of move sizes,
    // including ones large enough that the residual is not negligible, so
    // this is a check of the bookkeeping identity, not merely of Taylor
    // accuracy (that is a separate claim, tested below).
    const auto base_surface = make_surface();
    const MarketPoint base_market{100.0, 0.03, 0.01};
    const auto book = make_book();

    struct Move {
        double dSpot, dVolLevel, dRate;
    };
    for (Move mv : {Move{2.0, 0.0, 0.0}, Move{-15.0, 0.08, 0.0}, Move{5.0, -0.03, 0.005},
                   Move{-30.0, 0.20, -0.01}}) {
        const auto new_surface = make_surface(0.22 + mv.dVolLevel, 0.20 + mv.dVolLevel);
        const MarketPoint new_market{base_market.spot + mv.dSpot, base_market.rate + mv.dRate,
                                     base_market.carry};
        const auto attr =
            compute_pnl_attribution(book, base_surface, base_market, new_surface, new_market);

        const double attributed = attr.spot_pnl + attr.vol_pnl + attr.rate_pnl +
                                  attr.theta_pnl + attr.gamma_pnl + attr.vanna_pnl +
                                  attr.volga_pnl + attr.charm_pnl;
        EXPECT_NEAR(attr.total_exact_pnl, attributed + attr.residual,
                    1e-9 * std::max(1.0, std::abs(attr.total_exact_pnl)))
            << "dSpot=" << mv.dSpot << " dVol=" << mv.dVolLevel;

        // And new_value - base_value must equal the exact PnL -- the other
        // half of "the numbers actually reconcile".
        EXPECT_NEAR(attr.new_value - attr.base_value, attr.total_exact_pnl,
                    1e-9 * std::max(1.0, std::abs(attr.total_exact_pnl)));
    }
}

TEST(Portfolio, PerLegResidualsSumToThePortfolioResidual) {
    const auto base_surface = make_surface();
    const MarketPoint base_market{100.0, 0.03, 0.0};
    const auto new_surface = make_surface(0.30, 0.24);
    const MarketPoint new_market{80.0, 0.035, 0.0};
    const auto book = make_book();

    const auto attr = compute_pnl_attribution(book, base_surface, base_market, new_surface,
                                              new_market);
    ASSERT_EQ(attr.leg_residual.size(), book.size());
    double sum_leg_residual = 0.0;
    for (double r : attr.leg_residual) sum_leg_residual += r;
    EXPECT_NEAR(sum_leg_residual, attr.residual,
                1e-9 * std::max(1.0, std::abs(attr.residual)));

    double sum_leg_exact = 0.0;
    for (double p : attr.leg_exact_pnl) sum_leg_exact += p;
    EXPECT_NEAR(sum_leg_exact, attr.total_exact_pnl,
                1e-9 * std::max(1.0, std::abs(attr.total_exact_pnl)));
}

TEST(Portfolio, ResidualShrinksAsTheMoveShrinks) {
    // Consistency with Taylor's own error behaviour: the residual (what the
    // second-order expansion missed) must shrink as the bump shrinks, since
    // it is built from a genuine Taylor-remainder term.  This is a property
    // of the construction, not an independent claim, but it is exactly the
    // kind of sanity check that would catch an attribution wired to the
    // wrong Greek or the wrong sign.
    const auto base_surface = make_surface();
    const MarketPoint base_market{100.0, 0.03, 0.0};
    const auto book = make_book();

    double prev_abs_residual = 1e300;
    for (double scale : {1.0, 0.5, 0.25, 0.1, 0.01}) {
        const auto new_surface = make_surface(0.22 + 0.10 * scale, 0.20 + 0.10 * scale);
        const MarketPoint new_market{base_market.spot - 20.0 * scale, base_market.rate,
                                     base_market.carry};
        const auto attr = compute_pnl_attribution(book, base_surface, base_market, new_surface,
                                                  new_market);
        EXPECT_LE(std::abs(attr.residual), prev_abs_residual + 1e-9)
            << "scale=" << scale;
        prev_abs_residual = std::abs(attr.residual);
    }
    // And at the smallest scale the attribution should explain almost all of
    // the move.
    const auto tiny_surface = make_surface(0.221, 0.201);
    const MarketPoint tiny_market{99.8, base_market.rate, base_market.carry};
    const auto tiny_attr =
        compute_pnl_attribution(book, base_surface, base_market, tiny_surface, tiny_market);
    EXPECT_GT(tiny_attr.explained_fraction(), 0.95) << "explained_fraction = "
                                                    << tiny_attr.explained_fraction();
}

TEST(Portfolio, ZeroMoveGivesExactlyZeroEverywhere) {
    const auto surface = make_surface();
    const MarketPoint market{100.0, 0.03, 0.01};
    const auto book = make_book();
    const auto attr = compute_pnl_attribution(book, surface, market, surface, market);

    EXPECT_EQ(attr.total_exact_pnl, 0.0);
    EXPECT_EQ(attr.spot_pnl, 0.0);
    EXPECT_EQ(attr.vol_pnl, 0.0);
    EXPECT_EQ(attr.rate_pnl, 0.0);
    EXPECT_EQ(attr.gamma_pnl, 0.0);
    EXPECT_EQ(attr.residual, 0.0);
    for (double r : attr.leg_exact_pnl) EXPECT_EQ(r, 0.0);
}

TEST(Portfolio, OnlyMovedAxesContributeToTheMatchingBucket) {
    // A pure spot move with everything else (rate, carry, each leg's own
    // years, and the VolSurface object itself) held fixed.
    //
    // Three buckets are exactly zero unconditionally, because their own
    // delta is exactly zero regardless of any Greek's value: dR == 0 kills
    // rate_pnl, dt == 0 (no time passed) kills theta_pnl, and dt == 0 also
    // kills charm_pnl's dS*dt cross term even though dS != 0.
    //
    // vol_pnl, by contrast, is a genuine economic effect here and was wrong
    // to assume zero in an earlier version of this test: this surface keys
    // volatility by log-moneyness k = log(K/F) ("sticky moneyness"), and a
    // spot move changes the forward F for every fixed-strike position, which
    // moves each leg to a different point on the *same*, literally
    // unchanged, smile.  With the skew this surface carries (rho = -0.4),
    // that is a real, nonzero vol_pnl -- it is the mechanism by which
    // "spot-vol correlation" and skew delta-hedging effects arise, not a
    // bug, and the attribution is doing its job by surfacing it rather than
    // hiding it inside spot_pnl.
    const auto surface = make_surface();
    const MarketPoint base_market{100.0, 0.03, 0.0};
    const MarketPoint spot_only{95.0, 0.03, 0.0};
    const auto book = make_book();

    const auto attr =
        compute_pnl_attribution(book, surface, base_market, surface, spot_only);
    EXPECT_EQ(attr.rate_pnl, 0.0);
    EXPECT_EQ(attr.theta_pnl, 0.0);
    EXPECT_EQ(attr.charm_pnl, 0.0);  // charm's cross term is dS*dt; dt == 0 here
    EXPECT_NE(attr.spot_pnl, 0.0);
    EXPECT_NE(attr.gamma_pnl, 0.0);
    EXPECT_NE(attr.vol_pnl, 0.0) << "sticky-moneyness vol_pnl from the spot move should not "
                                   "vanish on a skewed surface";

    // Reconciliation must still hold with vol_pnl correctly included.
    const double attributed = attr.spot_pnl + attr.vol_pnl + attr.rate_pnl + attr.theta_pnl +
                              attr.gamma_pnl + attr.vanna_pnl + attr.volga_pnl +
                              attr.charm_pnl;
    EXPECT_NEAR(attr.total_exact_pnl, attributed + attr.residual,
                1e-9 * std::max(1.0, std::abs(attr.total_exact_pnl)));
}

TEST(Portfolio, PureVolMoveIsolatesTheVolAndVolgaBuckets) {
    const auto base_surface = make_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const auto vol_only_surface = make_surface(0.30, 0.28);
    const auto book = make_book();

    const auto attr =
        compute_pnl_attribution(book, base_surface, market, vol_only_surface, market);
    EXPECT_EQ(attr.spot_pnl, 0.0);
    EXPECT_EQ(attr.rate_pnl, 0.0);
    EXPECT_EQ(attr.gamma_pnl, 0.0);  // gamma's term is 0.5*gamma*dS^2; dS == 0
    EXPECT_EQ(attr.vanna_pnl, 0.0);  // vanna's term is vanna*dS*dvol; dS == 0
    EXPECT_NE(attr.vol_pnl, 0.0);
    EXPECT_NE(attr.volga_pnl, 0.0);
}

// ===========================================================================
// Largest contributors
// ===========================================================================

TEST(Portfolio, LargestContributorsAreRankedByAbsolutePnlDescending) {
    const auto base_surface = make_surface();
    const MarketPoint base_market{100.0, 0.03, 0.0};
    const auto new_surface = make_surface(0.26, 0.22);
    const MarketPoint new_market{85.0, 0.03, 0.0};
    const auto book = make_book();

    const auto attr =
        compute_pnl_attribution(book, base_surface, base_market, new_surface, new_market);
    const auto top2 = largest_pnl_contributors(attr, 2);
    ASSERT_EQ(top2.size(), 2u);
    EXPECT_GE(std::abs(attr.leg_exact_pnl[top2[0]]), std::abs(attr.leg_exact_pnl[top2[1]]));
    for (std::size_t i = 0; i < attr.leg_exact_pnl.size(); ++i) {
        if (i == top2[0] || i == top2[1]) continue;
        EXPECT_LE(std::abs(attr.leg_exact_pnl[i]), std::abs(attr.leg_exact_pnl[top2[1]]) + 1e-9)
            << "position " << i << " (" << book[i].label << ") should not outrank the top 2";
    }

    // Asking for more than the book has must not crash or duplicate.
    const auto all = largest_pnl_contributors(attr, 100);
    EXPECT_EQ(all.size(), book.size());
}

TEST(Portfolio, EmptyBookIsHarmless) {
    const auto surface = make_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const std::vector<Position> empty;
    const auto attr = compute_pnl_attribution(empty, surface, market, surface, market);
    EXPECT_EQ(attr.total_exact_pnl, 0.0);
    EXPECT_TRUE(attr.leg_exact_pnl.empty());
    EXPECT_TRUE(largest_pnl_contributors(attr, 5).empty());
}

TEST(Portfolio, TimeDecayBetweenSnapshotsFeedsThroughTheThetaBucket) {
    // Same market, same VolSurface object, but five calendar days have
    // passed (years shrinks for every leg).  spot_pnl, rate_pnl, gamma_pnl
    // and vanna_pnl are exactly zero: each has a dS factor, and dS == 0
    // unconditionally here since the market point is unchanged.
    //
    // vol_pnl is, once again, *not* necessarily zero, for two independent
    // sticky-moneyness reasons: (1) with a nonzero rate, the forward
    // F = S*exp((r-q)T) itself depends on T, so letting T shrink moves k at
    // fixed strike even though S hasn't; and (2) this surface has a real
    // term structure (ATM vol 0.22 at 3 months vs 0.20 at 1 year), so even
    // at fixed k, aging five days changes which point of the term structure
    // a position reads its vol from.  Both are genuine, named effects --
    // the second is exactly what the surface differential engine (a later
    // phase) decomposes explicitly into its own "term structure" bucket --
    // and asserting vol_pnl == 0 here was simply wrong about what "pure
    // time decay" means on a surface with any term structure at all.
    const auto surface = make_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const auto book = make_book();

    std::vector<double> new_years;
    for (const auto& p : book) new_years.push_back(p.years - 5.0 / 365.0);

    const auto attr =
        compute_pnl_attribution(book, surface, market, surface, market, new_years);
    EXPECT_EQ(attr.spot_pnl, 0.0);
    EXPECT_EQ(attr.rate_pnl, 0.0);
    EXPECT_EQ(attr.gamma_pnl, 0.0);
    EXPECT_EQ(attr.vanna_pnl, 0.0);
    EXPECT_NE(attr.theta_pnl, 0.0);
    // Reconciliation must still hold.  Summing all eight components -- not a
    // hand-picked subset -- is deliberate: production defines residual as
    // total_exact_pnl minus the sum of *all* eight, so reconstructing that
    // same full sum here is what actually tests the identity rather than
    // risking the test itself silently dropping a term that turns out not
    // to be zero (volga_pnl, in an earlier version of this test, since
    // dvol != 0 here for the term-structure reasons explained above).
    const double attributed = attr.spot_pnl + attr.vol_pnl + attr.rate_pnl + attr.theta_pnl +
                              attr.gamma_pnl + attr.vanna_pnl + attr.volga_pnl +
                              attr.charm_pnl;
    EXPECT_NEAR(attr.total_exact_pnl, attributed + attr.residual,
                1e-9 * std::max(1.0, std::abs(attr.total_exact_pnl)));
}
