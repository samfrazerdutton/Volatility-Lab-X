// SPDX-License-Identifier: MIT
/// Validates the scenario graph: that a constructed shock's effect, as
/// independently measured by the surface differential engine, matches what
/// was requested; that forward-only shocks stay orthogonal to the shape
/// coefficients (same guarantee `diagnostics/surface_differential.hpp`
/// already proves, now exercised through shock construction rather than
/// hand-built surfaces); and that the PnL attribution for every scenario
/// still reconciles exactly.

#include "vl_test_support.hpp"

#include "volatility_lab/scenarios/scenario.hpp"

#include <cmath>
#include <vector>

using namespace vl;

namespace {

/// A degenerate (b = 0) SVI slice is exactly flat in k -- see
/// tests/diagnostics/surface_differential.cpp for why that makes it the
/// right fixture for an *exact* recovery test, free of the GridSlice
/// cubic-spline-vs-closed-form-SVI mismatch a smiling surface has even at
/// zero shock (measured directly: ~1.3e-4 vol points off the reconstruction
/// grid, tiny but nonzero).
VolSurface make_flat_surface(double atm_vol = 0.20) {
    std::vector<SliceVariant> slices;
    for (double T : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        SviParams p;
        p.years = T;
        p.b = 0.0;
        p.rho = 0.0;
        p.m = 0.0;
        p.sigma = 0.1;
        p.a = atm_vol * atm_vol * T;
        slices.emplace_back(svi_project_to_admissible(p));
    }
    return VolSurface(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
}

VolSurface make_smile_surface(double atm_vol = 0.20, double skew = -0.4) {
    std::vector<SliceVariant> slices;
    for (double T : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        SviParams p;
        p.years = T;
        p.b = 0.08;
        p.rho = skew;
        p.m = 0.0;
        p.sigma = 0.13;
        p.a = atm_vol * atm_vol * T -
              p.b * (p.rho * (0.0 - p.m) + std::sqrt(p.m * p.m + p.sigma * p.sigma));
        slices.emplace_back(svi_project_to_admissible(p));
    }
    return VolSurface(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
}

std::vector<Position> make_book() {
    return {
        {"long_call_atm", 10.0, 100.0, 100.0, 0.25, OptionType::Call},
        {"short_put_otm", -5.0, 100.0, 90.0, 0.25, OptionType::Put},
        {"long_call_1y", 8.0, 100.0, 110.0, 1.0, OptionType::Call},
    };
}

double attributed_sum(const PnLAttribution& a) {
    return a.spot_pnl + a.vol_pnl + a.rate_pnl + a.theta_pnl + a.gamma_pnl + a.vanna_pnl +
          a.volga_pnl + a.charm_pnl;
}

}  // namespace

// ---------------------------------------------------------------------------
// Realized shock matches the requested shock
// ---------------------------------------------------------------------------

TEST(Scenario, PureLevelShockIsRecoveredExactlyOnAFlatSurface) {
    const auto base = make_flat_surface();
    const ShockSpec shock{"level", 0.03, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto shocked = apply_shock_to_surface(base, shock);
    const auto d = compute_surface_differential(base, shocked);

    EXPECT_NEAR(d.level_shift, 0.03, 1e-9);
    EXPECT_NEAR(d.skew_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.curvature_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.term_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.residual_rms, 0.0, 1e-7);
}

TEST(Scenario, PureSkewShockIsRecoveredExactlyOnAFlatSurface) {
    const auto base = make_flat_surface();
    const ShockSpec shock{"skew", 0.0, -0.05, 0.0, 0.0, 0.0, 0.0};
    const auto shocked = apply_shock_to_surface(base, shock);
    const auto d = compute_surface_differential(base, shocked);

    EXPECT_NEAR(d.skew_shift, -0.05, 1e-9);
    EXPECT_NEAR(d.level_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.curvature_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.term_shift, 0.0, 1e-9);
}

TEST(Scenario, CombinedShapeShockIsRecoveredWithAllFourCoefficientsSimultaneously) {
    const auto base = make_flat_surface();
    const ShockSpec shock{"combo", 0.05, -0.08, 0.02, 0.01, 0.0, 0.0};
    const auto shocked = apply_shock_to_surface(base, shock);
    const auto d = compute_surface_differential(base, shocked);

    EXPECT_NEAR(d.level_shift, 0.05, 1e-9);
    EXPECT_NEAR(d.skew_shift, -0.08, 1e-9);
    EXPECT_NEAR(d.curvature_shift, 0.02, 1e-9);
    EXPECT_NEAR(d.term_shift, 0.01, 1e-9);
}

TEST(Scenario, ShapeShocksAreRecoveredApproximatelyOnARealisticSmilingSurface) {
    // A smile introduces a small GridSlice-reconstruction mismatch (the new
    // slice is a cubic spline through sampled points, not the original
    // closed-form SVI formula) -- measured directly at ~1e-4 vol points for
    // a zero shock, so the tolerance here is set from that measurement, not
    // guessed.
    const auto base = make_smile_surface();
    const ShockSpec shock{"level", 0.03, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto shocked = apply_shock_to_surface(base, shock);
    const auto d = compute_surface_differential(base, shocked);

    EXPECT_NEAR(d.level_shift, 0.03, 1e-3);
    EXPECT_NEAR(d.skew_shift, 0.0, 1e-3);
    EXPECT_NEAR(d.curvature_shift, 0.0, 1e-3);
}

// ---------------------------------------------------------------------------
// Forward / shape orthogonality, through shock construction
// ---------------------------------------------------------------------------

TEST(Scenario, SpotOnlyShockMovesTheForwardAndNothingElseInShapeSpace) {
    const auto base = make_smile_surface();
    const ShockSpec shock{"spot-10pct", 0.0, 0.0, 0.0, 0.0, -0.10, 0.0};
    const auto shocked = apply_shock_to_surface(base, shock);
    const auto d = compute_surface_differential(base, shocked);

    EXPECT_NEAR(d.forward_shift, std::log(0.9), 1e-9);
    EXPECT_NEAR(d.level_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.skew_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.curvature_shift, 0.0, 1e-9);
    EXPECT_NEAR(d.term_shift, 0.0, 1e-9);
}

TEST(Scenario, SpotShockScalesTheMarketPointBySameFactorAsTheSurfaceForward) {
    const MarketPoint base{100.0, 0.03, 0.01};
    const ShockSpec shock{"spot-10pct", 0.0, 0.0, 0.0, 0.0, -0.10, 0.0};
    const auto shocked = apply_shock_to_market(base, shock);
    EXPECT_NEAR(shocked.spot, 90.0, 1e-9);
    EXPECT_EQ(shocked.rate, base.rate);
    EXPECT_EQ(shocked.carry, base.carry);
}

// ---------------------------------------------------------------------------
// PnL attribution still reconciles exactly for every scenario
// ---------------------------------------------------------------------------

TEST(Scenario, AttributionReconcilesExactlyForAnOrdinaryShock) {
    const auto base = make_smile_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const auto book = make_book();
    const ShockSpec shock{"crash-combo", 0.05, -0.08, 0.02, 0.0, -0.15, 0.0};

    const auto result = evaluate_scenario(book, base, market, shock);
    const double attributed = attributed_sum(result.attribution);
    EXPECT_NEAR(result.attribution.total_exact_pnl, attributed + result.attribution.residual,
                1e-9 * std::max(1.0, std::abs(result.attribution.total_exact_pnl)));
}

TEST(Scenario, LargerShocksHaveALargerFractionOfUnexplainedTaylorResidual) {
    // Consistent with the Portfolio module's own finding: the Taylor
    // approximation's error grows with move size, so a small shock should
    // explain a larger fraction of its own PnL than a large one.
    const auto base = make_smile_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const auto book = make_book();

    const auto small = evaluate_scenario(book, base, market,
                                         ShockSpec{"small", 0.01, 0.0, 0.0, 0.0, -0.02, 0.0});
    const auto large = evaluate_scenario(book, base, market,
                                         ShockSpec{"large", 0.05, -0.08, 0.02, 0.0, -0.15, 0.0});
    EXPECT_GT(small.attribution.explained_fraction(), large.attribution.explained_fraction());
}

// ---------------------------------------------------------------------------
// Time decay
// ---------------------------------------------------------------------------

TEST(Scenario, TimeDecayAloneLeavesTheRealizedShockAtEffectivelyZero) {
    const auto base = make_smile_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const auto book = make_book();
    const ShockSpec shock{"time+30d", 0.0, 0.0, 0.0, 0.0, 0.0, 30.0};

    const auto result = evaluate_scenario(book, base, market, shock);
    EXPECT_NEAR(result.realized_shock.level_shift, 0.0, 1e-3);
    EXPECT_NEAR(result.realized_shock.forward_shift, 0.0, 1e-9);
    // theta-driven: this book is net long optionality, so time decay alone
    // should be a loss.
    EXPECT_LT(result.attribution.theta_pnl, 0.0);
}

// ---------------------------------------------------------------------------
// The graph: many scenarios against the same book
// ---------------------------------------------------------------------------

TEST(Scenario, BatchEvaluationMatchesIndividualCallsExactly) {
    const auto base = make_smile_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const auto book = make_book();
    const std::vector<ShockSpec> shocks = {
        ShockSpec{"a", 0.02, 0.0, 0.0, 0.0, 0.0, 0.0},
        ShockSpec{"b", 0.0, -0.03, 0.0, 0.0, 0.05, 0.0},
        ShockSpec{"c", 0.0, 0.0, 0.0, 0.0, 0.0, 10.0},
    };
    const auto batch = evaluate_scenarios(book, base, market, shocks);
    ASSERT_EQ(batch.size(), shocks.size());
    for (std::size_t i = 0; i < shocks.size(); ++i) {
        const auto single = evaluate_scenario(book, base, market, shocks[i]);
        EXPECT_EQ(batch[i].label, shocks[i].label);
        EXPECT_EQ(batch[i].attribution.total_exact_pnl, single.attribution.total_exact_pnl);
        EXPECT_EQ(batch[i].realized_shock.level_shift, single.realized_shock.level_shift);
    }
}

TEST(Scenario, EmptyShockListProducesAnEmptyResultList) {
    const auto base = make_smile_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const auto book = make_book();
    const auto batch = evaluate_scenarios(book, base, market, std::span<const ShockSpec>{});
    EXPECT_TRUE(batch.empty());
}

TEST(Scenario, EmptyBookStillProducesAWellDefinedZeroAttribution) {
    const auto base = make_smile_surface();
    const MarketPoint market{100.0, 0.03, 0.0};
    const ShockSpec shock{"level", 0.03, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto result = evaluate_scenario(std::span<const Position>{}, base, market, shock);
    EXPECT_EQ(result.attribution.total_exact_pnl, 0.0);
    // The surface still moved -- an empty book just has nothing to reprice.
    EXPECT_NEAR(result.realized_shock.level_shift, 0.03, 1e-3);
}
