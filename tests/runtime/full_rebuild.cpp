// SPDX-License-Identifier: MIT
/// Phase 2 section 9: a differential test that compares `IncrementalEngine`
/// against the deliberately independent `full_rebuild` reference -- which
/// goes through no `DependencyGraph` at all -- for the same quotes.
/// Correctness (this file) and performance (benchmarks/incremental_scaling.cpp)
/// are checked separately, per the directive: a timing comparison proves
/// nothing about correctness, and a correctness comparison proves nothing
/// about speed.

#include "vl_test_support.hpp"

#include "volatility_lab/runtime/full_rebuild.hpp"

#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/options/normalize.hpp"

#include <cmath>

using namespace vl;

namespace {

VolSurface make_baseline_surface() {
    std::vector<SliceVariant> slices;
    for (double T : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        SviParams p;
        p.years = T;
        p.b = 0.08;
        p.rho = -0.4;
        p.m = 0.0;
        p.sigma = 0.13;
        p.a = 0.20 * 0.20 * T - p.b * std::sqrt(p.m * p.m + p.sigma * p.sigma);
        slices.emplace_back(svi_project_to_admissible(p));
    }
    return VolSurface(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
}

struct Fixture {
    std::vector<OptionQuote> quotes;
    IncrementalEngine::Config config;
};

Fixture make_fixture() {
    Fixture f;
    auto market = generate_market(MarketRegime::Normal);
    auto norm = normalize(market.snapshot);
    (void)assign_weights_by_slice(norm.quotes);
    f.quotes = std::move(norm.quotes);

    f.config.baseline_surface = make_baseline_surface();
    f.config.baseline_market = MarketPoint{market.snapshot.spot, 0.03, 0.0};
    f.config.positions = {
        {"long_call_atm", 10.0, 100.0, market.snapshot.spot, 0.25, OptionType::Call},
        {"short_put_otm", -5.0, 100.0, market.snapshot.spot * 0.9, 0.25, OptionType::Put},
        {"long_call_1y", 8.0, 100.0, market.snapshot.spot * 1.1, 1.0, OptionType::Call},
    };
    return f;
}

}  // namespace

TEST(FullRebuild, MatchesIncrementalEngineConstructionExactly) {
    const auto f = make_fixture();

    IncrementalEngine engine(f.quotes, f.config);
    const auto result = full_rebuild(f.quotes, f.config);

    // Same final state, computed by two code paths that share no
    // dependency-graph or indexing machinery -- only the lowest-level,
    // separately-tested pipeline functions.
    EXPECT_NEAR(engine.pnl().total_exact_pnl, result.pnl.total_exact_pnl, 1e-9);
    EXPECT_NEAR(engine.portfolio().delta, result.portfolio.delta, 1e-9);
    EXPECT_NEAR(engine.portfolio().vega, result.portfolio.vega, 1e-9);
    EXPECT_NEAR(engine.differential().level_shift, result.differential.level_shift, 1e-9);
    for (double years : {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0}) {
        EXPECT_NEAR(engine.surface().vol(0.0, years), result.surface.vol(0.0, years), 1e-12)
            << "mismatch at years=" << years;
    }
}

TEST(FullRebuild, MatchesIncrementalEngineAfterAnIncrementalUpdate) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);

    const auto& q0 = f.quotes[f.quotes.size() / 3];
    const MarketEvent evt{
        SequenceNumber{1},      Timestamp{1},
        "SPX",                   "instr",
        Years{q0.years},         Strike{q0.strike},
        q0.type,                 MarketEventType::Quote,
        Money{q0.bid + 0.05},    Money{q0.ask + 0.05},
        Money{q0.mid + 0.05},    10.0,
    };
    ASSERT_TRUE(engine.apply_event(evt).has_value());
    engine.recompute();

    std::vector<OptionQuote> rebuilt_quotes = f.quotes;
    for (auto& q : rebuilt_quotes) {
        if (std::abs(q.years - q0.years) < 1e-9 && std::abs(q.strike - q0.strike) < 1e-6 &&
            q.type == q0.type) {
            q.bid += 0.05;
            q.ask += 0.05;
            q.mid += 0.05;
        }
    }
    const auto result = full_rebuild(rebuilt_quotes, f.config);

    EXPECT_NEAR(engine.pnl().total_exact_pnl, result.pnl.total_exact_pnl, 1e-9);
    EXPECT_NEAR(engine.surface().vol(0.0, q0.years), result.surface.vol(0.0, q0.years), 1e-12);
}
