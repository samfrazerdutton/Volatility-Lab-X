// SPDX-License-Identifier: MIT
/// \file scalar_baseline.cpp
/// \brief Scalar performance profiling (directive implementation-order
///        item 9), measured *before* any SIMD or threading work -- the
///        baseline that work will later be compared against, and the
///        evidence for which kernels are actually worth vectorising.
///
/// A hand-rolled std::chrono timing loop, not Google Benchmark: no new
/// dependency, a fixed warm-up pass, and a fixed, deterministic,
/// reproducible input domain (not random data -- re-running this must give
/// the same input set every time, so a regression is attributable to the
/// code, not to which random inputs happened to be drawn).
///
/// Prints a human-readable table to stdout and, if given a path argument,
/// also writes the machine-readable JSON form the directive's performance-
/// regression system (section 13) consumes.

#include "volatility_lab/core/build_info.hpp"
#include "volatility_lab/greeks/greeks.hpp"
#include "volatility_lab/math/special.hpp"
#include "volatility_lab/pricing/black.hpp"
#include "volatility_lab/pricing/implied_vol.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace vl;

namespace {

struct BenchResult {
    std::string name;
    std::size_t n = 0;
    double total_ns = 0.0;
    [[nodiscard]] double ns_per_op() const noexcept { return total_ns / static_cast<double>(n); }
    [[nodiscard]] double ops_per_sec() const noexcept {
        return static_cast<double>(n) / (total_ns * 1e-9);
    }
};

/// A fixed, deterministic sweep: strikes x vols x years, the same domain
/// shape `tests/greeks/greeks.cpp`'s own validation sweep uses (so these
/// numbers are measured over inputs this project has already independently
/// confirmed are numerically well-behaved across, not a cherry-picked
/// friendly subset).
struct Inputs {
    std::vector<double> spot, strike, vol, years, rate, carry;
    std::vector<OptionType> type;
};

Inputs make_inputs() {
    Inputs in;
    const double strikes[] = {20, 40, 60, 80, 100, 120, 160, 250, 500};
    const double vols[] = {0.02, 0.06, 0.15, 0.30, 0.60, 1.0, 2.0};
    const double tenors[] = {1.0 / 365.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 5.0};
    const double rates[] = {-0.02, 0.0, 0.03, 0.08};
    for (double k : strikes) {
        for (double v : vols) {
            for (double t : tenors) {
                for (double r : rates) {
                    for (OptionType ty : {OptionType::Call, OptionType::Put}) {
                        in.spot.push_back(100.0);
                        in.strike.push_back(k);
                        in.vol.push_back(v);
                        in.years.push_back(t);
                        in.rate.push_back(r);
                        in.carry.push_back(0.01);
                        in.type.push_back(ty);
                    }
                }
            }
        }
    }
    return in;
}

template <class Fn>
BenchResult time_it(const std::string& name, std::size_t n, Fn&& fn) {
    // Warm-up: excluded from every timed trial, so page faults,
    // branch-predictor warm-up and cold instruction-cache effects don't
    // bias the number.
    fn();

    // Repeat the whole sweep enough times that one trial's wall time is
    // comfortably above the OS scheduler's own quantum, and take the
    // MINIMUM across several trials (standard microbenchmark practice):
    // a trial can only be slowed down by scheduling noise, a cache-cold
    // neighbour process, or a page fault, never sped up below what the
    // code actually costs -- so the minimum is the best estimate of the
    // code's own achievable cost, and the trials above it are evidence of
    // noise, not of the code being slower.
    constexpr int kRepeats = 200;
    constexpr int kTrials = 7;
    double best_ns = -1.0;
    for (int trial = 0; trial < kTrials; ++trial) {
        const auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < kRepeats; ++r) fn();
        const auto end = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(end - start).count();
        if (best_ns < 0.0 || ns < best_ns) best_ns = ns;
    }
    return BenchResult{name, n * static_cast<std::size_t>(kRepeats), best_ns};
}

}  // namespace

int main(int argc, char** argv) {
    const Inputs in = make_inputs();
    const std::size_t n = in.spot.size();
    volatile double sink = 0.0;  // prevents the optimiser from deleting the loop entirely

    std::vector<BenchResult> results;

    results.push_back(time_it("black_scholes_price", n, [&] {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            s += black_scholes_price(in.spot[i], in.strike[i], in.vol[i], in.years[i], in.rate[i],
                                     in.carry[i], in.type[i]);
        }
        sink = s;
    }));

    results.push_back(time_it("black_scholes_greeks", n, [&] {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto g = black_scholes_greeks(in.spot[i], in.strike[i], in.vol[i], in.years[i],
                                                in.rate[i], in.carry[i], in.type[i]);
            s += g.delta + g.vega;
        }
        sink = s;
    }));

    results.push_back(time_it("erfcx", n, [&] {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            s += math::erfcx(in.vol[i] - 1.0 + 0.01 * static_cast<double>(i % 7));
        }
        sink = s;
    }));

    results.push_back(time_it("norm_cdf_hp", n, [&] {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) s += math::norm_cdf_hp(in.vol[i] - 1.0);
        sink = s;
    }));

    results.push_back(time_it("implied_volatility (round-trip)", n, [&] {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double forward = in.spot[i] * std::exp((in.rate[i] - in.carry[i]) * in.years[i]);
            const double discount = std::exp(-in.rate[i] * in.years[i]);
            const double price =
                black_price(forward, in.strike[i], in.vol[i], in.years[i], discount, in.type[i]);
            const auto r = implied_volatility(price, forward, in.strike[i], in.years[i], discount,
                                              in.type[i]);
            s += r.total_volatility;
        }
        sink = s;
    }));

    const auto& bi = build_info();
    std::printf("scalar_baseline  (n=%zu per kernel)\n", n);
    std::printf("host: %s (%d physical / %d logical cores), compiler %s %s, build %s\n\n",
               bi.host_cpu, bi.physical_cores, bi.logical_cores, bi.compiler_id,
               bi.compiler_version, bi.build_type);
    std::printf("%-36s %12s %12s %16s\n", "kernel", "ns/op", "ops/sec", "total_ms");
    for (const auto& r : results) {
        std::printf("%-36s %12.2f %12.0f %16.3f\n", r.name.c_str(), r.ns_per_op(),
                   r.ops_per_sec(), r.total_ns / 1.0e6);
    }
    std::printf("\n(sink=%g, prevents the optimiser from discarding the loops)\n",
               static_cast<double>(sink));

    if (argc > 1) {
        std::ofstream out(argv[1]);
        out << "{\n  \"benchmark\": \"scalar_baseline\",\n  \"n\": " << n << ",\n"
            << "  \"host_cpu\": \"" << bi.host_cpu << "\",\n"
            << "  \"compiler\": \"" << bi.compiler_id << " " << bi.compiler_version << "\",\n"
            << "  \"build_type\": \"" << bi.build_type << "\",\n"
            << "  \"kernels\": [\n";
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            out << "    {\"name\": \"" << r.name << "\", \"ns_per_op\": " << r.ns_per_op()
                << ", \"ops_per_sec\": " << r.ops_per_sec() << "}" << (i + 1 < results.size() ? "," : "")
                << "\n";
        }
        out << "  ]\n}\n";
        std::printf("\nwrote %s\n", argv[1]);
    }
    return 0;
}
