// SPDX-License-Identifier: MIT
/// \file million_state.cpp
/// \brief The flagship benchmark (directive section 12): BASELINE (full
///        rebuild every event) vs INCREMENTAL (the real dependency-graph
///        runtime), processing a realistic market-event pipeline --
///        quote -> expiry slice -> surface -> {differential, uncertainty}
///        -> Greeks -> portfolio -> PnL -- via `IncrementalEngine`.
///
/// ## Scope, stated honestly before any numbers are shown
///
/// The directive's full matrix also asks for SIMD and multithreaded tiers
/// of this same end-to-end pipeline. Those are **not included here**: the
/// AVX2 erfcx kernel and the thread pool exist and are independently
/// benchmarked at the kernel level (`bench_simd_erfcx`,
/// `tests/core/thread_pool.cpp`), but neither is wired into
/// `IncrementalEngine`'s own calibration/Greeks/PnL path yet. Reporting a
/// "SIMD" or "threaded" column for this benchmark without that wiring
/// would be exactly the fabricated-number problem the directive forbids
/// (section 38). What is reported is BASELINE and INCREMENTAL, both real,
/// both measured on this run.
///
/// ## Why BASELINE is not measured at the full 1,000,000-state scale
///
/// A full rebuild at this benchmark's scale costs ~5ms (measured below,
/// not assumed) -- extrapolated, 1,000,000 of them would take on the
/// order of 90 minutes, which is not a benchmark anyone re-runs to check
/// a regression. BASELINE is therefore measured directly on a smaller,
/// explicitly-labelled sample (`--baseline-n`, default 2000) and its
/// per-state cost is reported both as measured *and*, separately and
/// clearly marked, extrapolated to 1,000,000 states -- the measured
/// number and the extrapolated one are never presented as the same kind
/// of claim.
///
/// INCREMENTAL is measured for the full requested state count (default
/// 1,000,000, `--n`), because it is actually fast enough to run that many
/// times.

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
#include <fstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <psapi.h>
#endif

using namespace vl;

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

struct Setup {
    std::vector<OptionQuote> quotes;
    IncrementalEngine::Config config;
    std::vector<double> distinct_years;
};

Setup make_setup(int n_positions) {
    Setup s;
    auto market = generate_market(MarketRegime::Normal);
    auto norm = normalize(market.snapshot);
    (void)assign_weights_by_slice(norm.quotes);
    s.quotes = std::move(norm.quotes);

    for (const auto& q : s.quotes) {
        if (std::find_if(s.distinct_years.begin(), s.distinct_years.end(), [&](double y) {
                return std::abs(y - q.years) < 1e-9;
            }) == s.distinct_years.end()) {
            s.distinct_years.push_back(q.years);
        }
    }
    std::sort(s.distinct_years.begin(), s.distinct_years.end());

    s.config.baseline_surface = make_baseline_surface(s.distinct_years);
    s.config.baseline_market = MarketPoint{market.snapshot.spot, 0.03, 0.0};
    for (int i = 0; i < n_positions; ++i) {
        const double strike =
            market.snapshot.spot * (0.8 + 0.4 * (static_cast<double>(i) / n_positions));
        const double years = s.distinct_years[static_cast<std::size_t>(i) % s.distinct_years.size()];
        s.config.positions.push_back(Position{
            "pos" + std::to_string(i), (i % 2 == 0) ? 10.0 : -5.0, 100.0, strike, years,
            (i % 2 == 0) ? OptionType::Call : OptionType::Put});
    }
    return s;
}

MarketEvent make_tick(const OptionQuote& q, std::uint64_t seq, double bump) {
    return MarketEvent{
        SequenceNumber{seq},   Timestamp{static_cast<std::int64_t>(seq)},
        "SPX",                 "instr",
        Years{q.years},        Strike{q.strike},
        q.type,                MarketEventType::Quote,
        Money{q.bid + bump},   Money{q.ask + bump},
        Money{q.mid + bump},   10.0,
    };
}

struct LatencyStats {
    double p50 = 0, p95 = 0, p99 = 0, mean = 0;
};

LatencyStats compute_stats(std::vector<double> ns_samples) {
    LatencyStats out;
    if (ns_samples.empty()) return out;
    std::sort(ns_samples.begin(), ns_samples.end());
    const std::size_t n = ns_samples.size();
    auto pct = [&](double p) {
        return ns_samples[std::min(n - 1, static_cast<std::size_t>(p * static_cast<double>(n)))];
    };
    out.p50 = pct(0.50);
    out.p95 = pct(0.95);
    out.p99 = pct(0.99);
    double sum = 0.0;
    for (double v : ns_samples) sum += v;
    out.mean = sum / static_cast<double>(n);
    return out;
}

std::size_t peak_working_set_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS info{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
        return info.PeakWorkingSetSize;
    }
    return 0;
#else
    return 0;  // not measured on this platform
#endif
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t n_incremental = 1'000'000;
    std::size_t n_baseline = 2'000;
    int n_positions = 200;
    std::string json_path;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n_incremental = static_cast<std::size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "--baseline-n") == 0 && i + 1 < argc) {
            n_baseline = static_cast<std::size_t>(std::atoll(argv[++i]));
        } else if (std::strcmp(argv[i], "--positions") == 0 && i + 1 < argc) {
            n_positions = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        }
    }

    const Setup setup = make_setup(n_positions);
    const auto& bi = build_info();

    std::printf("million_state benchmark\n");
    std::printf("host: %s, compiler %s %s, build %s\n", bi.host_cpu, bi.compiler_id,
               bi.compiler_version, bi.build_type);
    std::printf("scale: %zu expiries, %d positions\n\n", setup.distinct_years.size(),
               n_positions);

    // --- INCREMENTAL: the full requested state count -----------------------
    IncrementalEngine engine(setup.quotes, setup.config);
    const std::size_t total_nodes = engine.node_count();

    std::vector<double> incr_latency_ns;
    incr_latency_ns.reserve(n_incremental);
    double sum_recomputed = 0.0, sum_reused = 0.0;

    const auto incr_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < n_incremental; ++i) {
        const auto& q = setup.quotes[i % setup.quotes.size()];
        const double bump = 0.001 * static_cast<double>((i % 7) - 3);
        const MarketEvent evt = make_tick(q, i + 1, bump);

        const auto t0 = std::chrono::steady_clock::now();
        const auto applied = engine.apply_event(evt);
        if (applied.has_value()) {
            const auto report = engine.recompute();
            sum_recomputed += static_cast<double>(report.recomputed_nodes);
            sum_reused += static_cast<double>(report.reused_nodes);
        }
        const auto t1 = std::chrono::steady_clock::now();
        incr_latency_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    const auto incr_end = std::chrono::steady_clock::now();
    const double incr_total_s = std::chrono::duration<double>(incr_end - incr_start).count();
    const LatencyStats incr_stats = compute_stats(incr_latency_ns);
    const double avg_recomputed = sum_recomputed / static_cast<double>(n_incremental);
    const double avg_reused = sum_reused / static_cast<double>(n_incremental);
    const double fraction_avoided =
        (total_nodes > 0) ? avg_reused / static_cast<double>(total_nodes) : 0.0;
    const std::size_t peak_mem_incremental = peak_working_set_bytes();

    // --- BASELINE: full rebuild, measured on a smaller sample ---------------
    std::vector<double> base_latency_ns;
    base_latency_ns.reserve(n_baseline);
    std::vector<OptionQuote> mutable_quotes = setup.quotes;

    const auto base_start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < n_baseline; ++i) {
        auto& q = mutable_quotes[i % mutable_quotes.size()];
        const double bump = 0.001 * static_cast<double>((i % 7) - 3);
        q.bid += bump;
        q.ask += bump;
        q.mid += bump;

        const auto t0 = std::chrono::steady_clock::now();
        IncrementalEngine fresh(mutable_quotes, setup.config);  // full rebuild, every state
        const auto t1 = std::chrono::steady_clock::now();
        base_latency_ns.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
        (void)fresh;
    }
    const auto base_end = std::chrono::steady_clock::now();
    const double base_total_s = std::chrono::duration<double>(base_end - base_start).count();
    const LatencyStats base_stats = compute_stats(base_latency_ns);
    const double base_extrapolated_1m_s = (base_stats.mean / 1e9) * 1'000'000.0;

    // --- report --------------------------------------------------------------
    std::printf("INCREMENTAL (measured, n=%zu states, %zu dependency graph nodes):\n", n_incremental,
               total_nodes);
    std::printf("  total runtime:       %.3f s\n", incr_total_s);
    std::printf("  throughput:          %.0f states/sec\n",
               static_cast<double>(n_incremental) / incr_total_s);
    std::printf("  latency p50/p95/p99: %.1f / %.1f / %.1f us\n", incr_stats.p50 / 1e3,
               incr_stats.p95 / 1e3, incr_stats.p99 / 1e3);
    std::printf("  latency mean:        %.1f us\n", incr_stats.mean / 1e3);
    std::printf("  avg nodes recomputed/reused (of %zu total): %.1f / %.1f\n", total_nodes,
               avg_recomputed, avg_reused);
    std::printf("  computation avoided: %.2f%%\n", fraction_avoided * 100.0);
    std::printf("  peak working set:    %.1f MiB\n\n",
               static_cast<double>(peak_mem_incremental) / (1024.0 * 1024.0));

    std::printf("BASELINE / full rebuild (measured, n=%zu states -- NOT the full %zu; "
               "see file comment for why):\n",
               n_baseline, n_incremental);
    std::printf("  total runtime:       %.3f s\n", base_total_s);
    std::printf("  throughput:          %.1f states/sec\n",
               static_cast<double>(n_baseline) / base_total_s);
    std::printf("  latency p50/p95/p99: %.2f / %.2f / %.2f ms\n", base_stats.p50 / 1e6,
               base_stats.p95 / 1e6, base_stats.p99 / 1e6);
    std::printf("  latency mean:        %.2f ms\n", base_stats.mean / 1e6);
    std::printf("  EXTRAPOLATED (not measured) time for %zu states: %.1f s (%.2f min)\n\n",
               n_incremental, base_extrapolated_1m_s, base_extrapolated_1m_s / 60.0);

    const double speedup = base_stats.mean / incr_stats.mean;
    std::printf("Per-state speedup (baseline mean / incremental mean): %.1fx\n", speedup);
    std::printf("(This number is real: both means are measured latencies on this run. The\n"
               " *extrapolated* 1M-state baseline total above is a separate, clearly-labelled\n"
               " projection from that same measured per-state cost, not a second measurement.)\n");

    if (!json_path.empty()) {
        std::ofstream out(json_path);
        out << "{\n"
            << "  \"benchmark\": \"million_state\",\n"
            << "  \"host_cpu\": \"" << bi.host_cpu << "\",\n"
            << "  \"total_nodes\": " << total_nodes << ",\n"
            << "  \"incremental\": {\n"
            << "    \"n\": " << n_incremental << ",\n"
            << "    \"measured\": true,\n"
            << "    \"total_runtime_s\": " << incr_total_s << ",\n"
            << "    \"throughput_states_per_second\": "
            << (static_cast<double>(n_incremental) / incr_total_s) << ",\n"
            << "    \"p50_ns\": " << incr_stats.p50 << ",\n"
            << "    \"p95_ns\": " << incr_stats.p95 << ",\n"
            << "    \"p99_ns\": " << incr_stats.p99 << ",\n"
            << "    \"mean_ns\": " << incr_stats.mean << ",\n"
            << "    \"avg_nodes_recomputed\": " << avg_recomputed << ",\n"
            << "    \"avg_nodes_reused\": " << avg_reused << ",\n"
            << "    \"computation_avoided_fraction\": " << fraction_avoided << ",\n"
            << "    \"peak_working_set_bytes\": " << peak_mem_incremental << "\n"
            << "  },\n"
            << "  \"baseline_full_rebuild\": {\n"
            << "    \"n\": " << n_baseline << ",\n"
            << "    \"measured\": true,\n"
            << "    \"note\": \"measured on n_baseline states, NOT the full incremental n -- see "
               "benchmarks/million_state.cpp's file comment\",\n"
            << "    \"total_runtime_s\": " << base_total_s << ",\n"
            << "    \"mean_ns\": " << base_stats.mean << ",\n"
            << "    \"p50_ns\": " << base_stats.p50 << ",\n"
            << "    \"p95_ns\": " << base_stats.p95 << ",\n"
            << "    \"p99_ns\": " << base_stats.p99 << ",\n"
            << "    \"extrapolated_1m_state_runtime_s\": " << base_extrapolated_1m_s << ",\n"
            << "    \"extrapolation_is_not_a_measurement\": true\n"
            << "  },\n"
            << "  \"per_state_speedup\": " << speedup << "\n"
            << "}\n";
        std::printf("\nwrote %s\n", json_path.c_str());
    }
    return 0;
}
