// SPDX-License-Identifier: MIT
/// \file incremental_scaling.cpp
/// \brief Phase 2 section 6/7: how `IncrementalEngine`'s costs change as
///        the market grows, measured directly -- not assumed.
///
/// Dataset sizes are built with a FIXED quotes-per-expiry (50) and a
/// GROWING expiry count, specifically so the benchmark isolates the claim
/// the directive asks for: a single quote update's cost should depend on
/// the size of *its own* expiry, not on the total size of the book. A
/// scheme that instead grew quotes-per-expiry at a fixed expiry count
/// would conflate the two and make the comparison meaningless.
///
/// Every number below is measured with `std::chrono::steady_clock` around
/// the actual call; this file injects no delay, assumption, or
/// extrapolation except where explicitly labelled EXTRAPOLATED.

#include "alloc_counter.hpp"

#include "volatility_lab/core/build_info.hpp"
#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/options/normalize.hpp"
#include "volatility_lab/runtime/incremental_engine.hpp"
#include "volatility_lab/runtime/market_event.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace vl;
using vl::benchutil::AllocScope;

namespace {

VolSurface make_baseline_surface(std::span<const double> tenors) {
    std::vector<SliceVariant> slices;
    for (double T : tenors) {
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

struct Dataset {
    std::vector<OptionQuote> quotes;
    IncrementalEngine::Config config;
    std::vector<double> distinct_years;
    double spot = 100.0;
};

/// `n_expiries` expiries of exactly `strikes_per_expiry` quotes each
/// (`strike_increment = 0` so no two generated strikes round to the same
/// value and silently collapse -- the default config rounds, which is
/// realistic for a listed market but would make the quote count here
/// unpredictable).
Dataset make_dataset(int n_expiries, int strikes_per_expiry) {
    Dataset d;
    SyntheticMarketConfig cfg;
    cfg.regime = MarketRegime::Normal;
    cfg.strikes_per_expiry = strikes_per_expiry;
    cfg.strike_increment = 0.0;
    cfg.expiries.clear();
    cfg.expiries.reserve(static_cast<std::size_t>(n_expiries));
    for (int i = 0; i < n_expiries; ++i) cfg.expiries.push_back(0.02 + 0.001 * i);

    auto market = generate_market(cfg);
    auto norm = normalize(market.snapshot);
    (void)assign_weights_by_slice(norm.quotes);
    d.quotes = std::move(norm.quotes);
    d.spot = market.snapshot.spot;

    for (const auto& q : d.quotes) {
        if (std::find_if(d.distinct_years.begin(), d.distinct_years.end(), [&](double y) {
                return std::abs(y - q.years) < 1e-9;
            }) == d.distinct_years.end()) {
            d.distinct_years.push_back(q.years);
        }
    }
    d.config.baseline_surface = make_baseline_surface(d.distinct_years);
    d.config.baseline_market = MarketPoint{d.spot, 0.03, 0.0};
    return d;
}

MarketEvent make_tick(const OptionQuote& q, double bump) {
    return MarketEvent{
        SequenceNumber{1},   Timestamp{1},
        "SPX",               "instr",
        Years{q.years},       Strike{q.strike},
        q.type,               MarketEventType::Quote,
        Money{q.bid + bump}, Money{q.ask + bump},
        Money{q.mid + bump}, 10.0,
    };
}

struct Timed {
    double ms = 0.0;
    std::uint64_t allocations = 0;
    std::uint64_t alloc_bytes = 0;
    RecomputeReport report;
};

template <class Fn>
Timed time_it(Fn&& fn) {
    Timed t;
    AllocScope scope;
    const auto t0 = std::chrono::steady_clock::now();
    t.report = fn();
    const auto t1 = std::chrono::steady_clock::now();
    scope.stop();
    t.ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    t.allocations = scope.allocations();
    t.alloc_bytes = scope.bytes();
    return t;
}

void run_size(int n_expiries, int strikes_per_expiry) {
    Dataset d = make_dataset(n_expiries, strikes_per_expiry);
    const std::size_t n_quotes = d.quotes.size();

    // FULL REBUILD: engine construction does one full recompute of every
    // node (every node starts dirty) -- the deliberately simple reference
    // cost this benchmark compares everything else against.
    AllocScope build_scope;
    const auto build_t0 = std::chrono::steady_clock::now();
    IncrementalEngine engine(d.quotes, d.config);
    const auto build_t1 = std::chrono::steady_clock::now();
    build_scope.stop();
    const double full_rebuild_ms =
        std::chrono::duration<double, std::milli>(build_t1 - build_t0).count();

    // SINGLE QUOTE UPDATE: one instrument, in one expiry, ticked once.
    const auto& q0 = d.quotes[d.quotes.size() / 2];
    const auto single_quote = time_it([&] {
        (void)engine.apply_event(make_tick(q0, 0.01));
        return engine.recompute();
    });

    // SINGLE EXPIRY UPDATE: every quote belonging to ONE expiry ticked,
    // then one recompute() -- still dirties exactly one ExpirySlice node
    // (plus everything downstream of Surface), regardless of how many of
    // that expiry's own quotes moved.
    const double touched_years = d.distinct_years.back();
    std::vector<OptionQuote> one_expiry_quotes;
    for (const auto& q : d.quotes) {
        if (std::abs(q.years - touched_years) < 1e-9) one_expiry_quotes.push_back(q);
    }
    const auto single_expiry = time_it([&] {
        for (const auto& q : one_expiry_quotes) (void)engine.apply_event(make_tick(q, 0.01));
        return engine.recompute();
    });

    // LARGE MARKET UPDATE: one quote in EVERY expiry ticked, then one
    // recompute() -- dirties every ExpirySlice node, i.e. the same total
    // work as a full rebuild. Included specifically to show where the
    // incremental architecture does *not* help, honestly, rather than
    // only showing the cases that flatter it.
    std::vector<OptionQuote> one_per_expiry;
    one_per_expiry.reserve(d.distinct_years.size());
    for (double y : d.distinct_years) {
        auto it = std::find_if(d.quotes.begin(), d.quotes.end(),
                               [&](const OptionQuote& q) { return std::abs(q.years - y) < 1e-9; });
        if (it != d.quotes.end()) one_per_expiry.push_back(*it);
    }
    const auto large_market = time_it([&] {
        for (const auto& q : one_per_expiry) (void)engine.apply_event(make_tick(q, 0.01));
        return engine.recompute();
    });

    std::printf(
        "n_expiries=%-6d quotes=%-9zu | full_rebuild=%9.3f ms | "
        "single_quote=%8.4f ms (alloc=%llu/%llubyte) | "
        "single_expiry=%8.4f ms | large_market=%9.3f ms | speedup(single/full)=%7.1fx\n",
        n_expiries, n_quotes, full_rebuild_ms, single_quote.ms,
        static_cast<unsigned long long>(single_quote.allocations),
        static_cast<unsigned long long>(single_quote.alloc_bytes), single_expiry.ms,
        large_market.ms, full_rebuild_ms / single_quote.ms);
}

}  // namespace

int main(int argc, char** argv) {
    const auto& bi = build_info();
    std::printf("incremental_scaling\n");
    std::printf("host: %s | compiler: %s %s | build: %s | SIMD: %s\n\n", bi.host_cpu,
               bi.compiler_id, bi.compiler_version, bi.build_type, bi.simd_build_level);

    bool run_million = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--million") == 0) run_million = true;
    }

    // Fixed 50 quotes/expiry; expiry count grows to scale total quotes.
    run_size(2, 50);       // ~100 quotes
    run_size(20, 50);      // ~1,000 quotes
    run_size(200, 50);     // ~10,000 quotes
    run_size(2000, 50);    // ~100,000 quotes
    if (run_million) {
        std::printf("\n--million requested: this one takes tens of seconds (20,000 expiries, "
                    "one full calibration per expiry at construction).\n");
        run_size(20000, 50);  // ~1,000,000 quotes
    } else {
        std::printf(
            "\n(1,000,000-quote case skipped by default -- pass --million to run it; "
            "it takes on the order of 10s of seconds because the full-rebuild reference "
            "calibrates 20,000 expiries once at construction.)\n");
    }
    return 0;
}
