// SPDX-License-Identifier: MIT
/// Validates the runtime dependency graph (directive Phase 2, section 6):
/// that a quote event dirties exactly its own expiry slice and everything
/// downstream of the surface (and nothing else), that the reported
/// recompute counts are real (not fabricated), that an incrementally
/// updated engine matches a freshly-built one from the same final quotes
/// *exactly*, and that a market-point-only change never recalibrates
/// anything (the sticky-moneyness/forward-orthogonality finding this
/// project has already made three times elsewhere, reused here).

#include "vl_test_support.hpp"

#include "volatility_lab/runtime/incremental_engine.hpp"

#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/options/normalize.hpp"

#include <algorithm>
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
    SyntheticMarket market;
    std::vector<OptionQuote> quotes;
    IncrementalEngine::Config config;
};

Fixture make_fixture() {
    Fixture f;
    f.market = generate_market(MarketRegime::Normal);
    auto norm = normalize(f.market.snapshot);
    (void)assign_weights_by_slice(norm.quotes);
    f.quotes = std::move(norm.quotes);

    f.config.baseline_surface = make_baseline_surface();
    f.config.baseline_market = MarketPoint{f.market.snapshot.spot, 0.03, 0.0};
    f.config.positions = {
        {"long_call_atm", 10.0, 100.0, f.market.snapshot.spot, 0.25, OptionType::Call},
        {"short_put_otm", -5.0, 100.0, f.market.snapshot.spot * 0.9, 0.25, OptionType::Put},
        {"long_call_1y", 8.0, 100.0, f.market.snapshot.spot * 1.1, 1.0, OptionType::Call},
    };
    return f;
}

MarketEvent make_tick(const OptionQuote& q, double bump) {
    return MarketEvent{
        SequenceNumber{1},  Timestamp{1000},
        "SPX",              "instr",
        Years{q.years},      Strike{q.strike},
        q.type,              MarketEventType::Quote,
        Money{q.bid + bump}, Money{q.ask + bump},
        Money{q.mid + bump}, 10.0,
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(IncrementalEngine, ConstructionFullyRecomputesAndEndsClean) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);
    EXPECT_TRUE(engine.all_clean());
    EXPECT_GT(engine.surface().num_slices(), 0u);
    EXPECT_EQ(engine.position_valuations().size(), f.config.positions.size());
}

TEST(IncrementalEngine, TotalNodeCountAccountsForEveryExpiryPlusFixedNodes) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);
    std::size_t distinct_years = 0;
    {
        std::vector<double> seen;
        for (const auto& q : f.quotes) {
            bool found = false;
            for (double y : seen) {
                if (std::abs(y - q.years) < 1e-9) { found = true; break; }
            }
            if (!found) seen.push_back(q.years);
        }
        distinct_years = seen.size();
    }
    // expiries + surface + differential + uncertainty + positions + portfolio + pnl
    const std::size_t expected = distinct_years + 1 + 1 + 1 + f.config.positions.size() + 1 + 1;
    EXPECT_EQ(engine.node_count(), expected);
}

// ---------------------------------------------------------------------------
// Minimal invalidation: the central claim
// ---------------------------------------------------------------------------

TEST(IncrementalEngine, UnknownExpiryIsRejectedAndChangesNothing) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);
    ASSERT_TRUE(engine.all_clean());

    MarketEvent evt{
        SequenceNumber{1}, Timestamp{1}, "SPX", "x", Years{99.0},  // no such expiry
        Strike{100.0},     OptionType::Call,       MarketEventType::Quote,
        Money{1.0},        Money{1.1},              Money{1.05},   1.0,
    };
    const auto result = engine.apply_event(evt);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), RuntimeError::UnknownExpiry);
    EXPECT_TRUE(engine.all_clean()) << "a rejected event must not dirty anything";
}

TEST(IncrementalEngine, OneQuoteTickDirtiesExactlyItsExpiryAndEverythingDownstreamOfSurface) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);
    ASSERT_TRUE(engine.all_clean());

    const auto& q0 = f.quotes[10];
    ASSERT_TRUE(engine.apply_event(make_tick(q0, 0.5)).has_value());

    const auto report = engine.recompute();
    EXPECT_TRUE(engine.all_clean());

    // Dirty set must be: 1 expiry-slice + surface + differential +
    // uncertainty + every position + portfolio + pnl. Every OTHER expiry
    // slice must be reused.
    const std::size_t expected_recomputed = 1 + 1 + 1 + 1 + f.config.positions.size() + 1 + 1;
    EXPECT_EQ(report.recomputed_nodes, expected_recomputed);
    EXPECT_EQ(report.total_nodes, engine.node_count());
    EXPECT_EQ(report.reused_nodes, report.total_nodes - report.recomputed_nodes);
    EXPECT_GT(report.reused_nodes, 0u) << "at least one other expiry must have been reused";
    EXPECT_NEAR(report.fraction_avoided(),
                static_cast<double>(report.reused_nodes) / static_cast<double>(report.total_nodes),
                1e-12);
}

TEST(IncrementalEngine, RecomputeWithNothingDirtyDoesZeroWork) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);
    const auto report = engine.recompute();  // nothing changed since construction's own recompute
    EXPECT_EQ(report.recomputed_nodes, 0u);
    EXPECT_EQ(report.reused_nodes, report.total_nodes);
    EXPECT_DOUBLE_EQ(report.fraction_avoided(), 1.0);
}

// ---------------------------------------------------------------------------
// Observable work reduction (Phase 2 section 7): the engine must not just
// be fast, it must report evidence of *why*. This is also the permanent
// regression test for the indexed-quote-store fix (Phase 2 section 3/5):
// before that fix, `quotes_examined` would have equalled `quotes_total`
// on every single-quote update, because `recompute_node`'s `ExpirySlice`
// case scanned the whole book to find one expiry's own quotes.
// ---------------------------------------------------------------------------

TEST(IncrementalEngine, SingleQuoteUpdateExaminesOnlyItsOwnExpiryNotTheWholeBook) {
    // A book with many expiries, each with many quotes, specifically so
    // "examined one expiry's worth, not the whole book" is a large,
    // unmistakable gap rather than a coincidence of a small fixture.
    SyntheticMarketConfig cfg;
    cfg.regime = MarketRegime::Normal;
    cfg.strikes_per_expiry = 50;
    cfg.strike_increment = 0.0;
    cfg.expiries.clear();
    for (int i = 0; i < 100; ++i) cfg.expiries.push_back(0.02 + 0.001 * i);
    auto market = generate_market(cfg);
    auto norm = normalize(market.snapshot);
    (void)assign_weights_by_slice(norm.quotes);

    std::vector<double> distinct_years;
    for (const auto& q : norm.quotes) {
        if (std::find_if(distinct_years.begin(), distinct_years.end(), [&](double y) {
                return std::abs(y - q.years) < 1e-9;
            }) == distinct_years.end()) {
            distinct_years.push_back(q.years);
        }
    }

    IncrementalEngine::Config config;
    config.baseline_surface = make_baseline_surface();  // tenor-agnostic: only vol() is queried
    config.baseline_market = MarketPoint{market.snapshot.spot, 0.03, 0.0};
    IncrementalEngine engine(norm.quotes, config);

    const auto& q0 = norm.quotes[norm.quotes.size() / 2];
    ASSERT_TRUE(engine.apply_event(make_tick(q0, 0.01)).has_value());
    const auto report = engine.recompute();

    EXPECT_EQ(report.calibrations_run, 1u) << "exactly one expiry should have been recalibrated";
    EXPECT_EQ(report.quotes_total, norm.quotes.size());
    // The touched expiry has ~50 quotes; the whole book has thousands.
    // This is the direct, measured evidence -- not an inferred one -- that
    // a single-quote update does not scan the whole market.
    EXPECT_LT(report.quotes_examined, report.quotes_total / 10)
        << "quotes_examined=" << report.quotes_examined
        << " quotes_total=" << report.quotes_total;
    EXPECT_GT(report.quotes_examined, 0u);
}

// ---------------------------------------------------------------------------
// Correctness: incremental must match a fresh full rebuild, exactly
// ---------------------------------------------------------------------------

TEST(IncrementalEngine, IncrementalUpdateMatchesAFreshFullRebuildExactly) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);

    const auto& q0 = f.quotes[10];
    ASSERT_TRUE(engine.apply_event(make_tick(q0, 0.5)).has_value());
    engine.recompute();

    // A second engine, built from scratch, with the same quote already
    // bumped in the initial book -- no incremental path involved at all.
    std::vector<OptionQuote> bumped_quotes = f.quotes;
    for (auto& q : bumped_quotes) {
        if (std::abs(q.years - q0.years) < 1e-9 && std::abs(q.strike - q0.strike) < 1e-6 &&
            q.type == q0.type) {
            q.bid += 0.5;
            q.ask += 0.5;
            q.mid += 0.5;
        }
    }
    IncrementalEngine fresh(bumped_quotes, f.config);

    EXPECT_NEAR(engine.pnl().total_exact_pnl, fresh.pnl().total_exact_pnl, 1e-9);
    EXPECT_NEAR(engine.portfolio().delta, fresh.portfolio().delta, 1e-9);
    EXPECT_NEAR(engine.portfolio().vega, fresh.portfolio().vega, 1e-9);
    EXPECT_NEAR(engine.surface().vol(0.0, 0.25), fresh.surface().vol(0.0, 0.25), 1e-12);
    EXPECT_NEAR(engine.differential().level_shift, fresh.differential().level_shift, 1e-9);
}

// A real bug, found by the engineering demo (`apps/cli/main.cpp`), not by
// this file: `apply_event` used to build a brand-new, default-constructed
// `OptionQuote` for an instrument that already existed, which silently
// zeroed `volume`/`open_interest`/`age_seconds` -- fields a `MarketEvent`
// has no opinion on at all -- even though the slot already held real
// values for them. `assign_weights_by_slice`'s liquidity factor reads
// volume/open_interest directly, so the touched quote's calibration weight
// differed from a fresh rebuild of the same final quotes, and the surface
// for that one expiry (and everything downstream of it) came out wrong by
// more than float noise.
//
// `IncrementalUpdateMatchesAFreshFullRebuildExactly` above did not catch
// this: `f.quotes[10]` happens to have zero volume/open_interest in this
// fixture's synthetic market, so resetting "zero" to "zero" is invisible.
// This test picks a quote with nonzero volume and open_interest
// specifically so the bug class cannot hide behind an unlucky index again.
TEST(IncrementalEngine, ApplyEventOnAnExistingQuotePreservesVolumeAndOpenInterest) {
    const auto f = make_fixture();

    const auto liquid_it = std::find_if(f.quotes.begin(), f.quotes.end(), [](const OptionQuote& q) {
        return q.volume > 0.0 && q.open_interest > 0.0;
    });
    ASSERT_NE(liquid_it, f.quotes.end())
        << "fixture must contain at least one quote with real volume/open_interest "
           "for this test to exercise anything";
    const OptionQuote q0 = *liquid_it;

    IncrementalEngine engine(f.quotes, f.config);
    ASSERT_TRUE(engine.apply_event(make_tick(q0, 0.05)).has_value());
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
    IncrementalEngine fresh(rebuilt_quotes, f.config);

    EXPECT_NEAR(engine.pnl().total_exact_pnl, fresh.pnl().total_exact_pnl, 1e-9);
    EXPECT_NEAR(engine.surface().vol(0.0, q0.years), fresh.surface().vol(0.0, q0.years), 1e-12);
}

TEST(IncrementalEngine, UntouchedExpirySlicesAreBitIdenticalAfterAnUnrelatedUpdate) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);
    const auto before = engine.surface().slice(0);  // whichever is first (shortest expiry)

    const auto& q0 = f.quotes[10];
    const double q0_years = q0.years;
    ASSERT_TRUE(engine.apply_event(make_tick(q0, 0.5)).has_value());
    engine.recompute();

    // Find the same expiry in the (possibly larger) surface again.
    const auto expiries = engine.surface().expiries();
    ASSERT_FALSE(expiries.empty());
    if (std::abs(expiries[0] - q0_years) > 1e-9) {
        // The touched expiry was not the first slice -- confirm the first
        // slice's params are untouched, bit for bit.
        const auto& after = engine.surface().slice(0);
        const auto& p_before = std::get<SviParams>(before);
        const auto& p_after = std::get<SviParams>(after);
        EXPECT_EQ(p_before.a, p_after.a);
        EXPECT_EQ(p_before.b, p_after.b);
        EXPECT_EQ(p_before.rho, p_after.rho);
        EXPECT_EQ(p_before.m, p_after.m);
        EXPECT_EQ(p_before.sigma, p_after.sigma);
    }
}

// ---------------------------------------------------------------------------
// Market-point updates never recalibrate (forward-orthogonality, reused)
// ---------------------------------------------------------------------------

TEST(IncrementalEngine, MarketPointUpdateDirtiesSurfaceButNeverAnExpirySlice) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);
    ASSERT_TRUE(engine.all_clean());

    const auto before = engine.surface().slice(0);
    MarketPoint bumped = engine.market();
    bumped.spot *= 1.05;
    engine.update_market_point(bumped);

    const auto report = engine.recompute();
    // surface + differential + uncertainty + positions + portfolio + pnl --
    // no expiry slice in that count.
    const std::size_t expected = 1 + 1 + 1 + f.config.positions.size() + 1 + 1;
    EXPECT_EQ(report.recomputed_nodes, expected);

    const auto& after = engine.surface().slice(0);
    const auto& p_before = std::get<SviParams>(before);
    const auto& p_after = std::get<SviParams>(after);
    EXPECT_EQ(p_before.a, p_after.a);
    EXPECT_EQ(p_before.b, p_after.b);
    EXPECT_EQ(p_before.rho, p_after.rho);
    EXPECT_EQ(p_before.m, p_after.m);
    EXPECT_EQ(p_before.sigma, p_after.sigma);
}

TEST(IncrementalEngine, MarketPointUpdateMovesTheForwardCurve) {
    const auto f = make_fixture();
    IncrementalEngine engine(f.quotes, f.config);
    const double old_forward = engine.surface().forwards()(0.25);

    MarketPoint bumped = engine.market();
    bumped.spot *= 1.10;
    engine.update_market_point(bumped);
    engine.recompute();

    const double new_forward = engine.surface().forwards()(0.25);
    EXPECT_NEAR(new_forward / old_forward, 1.10, 1e-9);
}
