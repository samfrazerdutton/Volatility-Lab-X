// SPDX-License-Identifier: MIT
/// \file simd_erfcx.cpp
/// \brief Scalar vs AVX2 throughput for erfcx -- the directive's explicit
///        "do not claim SIMD speedup without a benchmark" requirement
///        (section 8), measured here rather than assumed from the vector
///        width.

#include "scalar/erfcx_poly.hpp"
#include "simd/erfcx_avx2.hpp"
#include "volatility_lab/core/build_info.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace vl;

namespace {

std::vector<double> make_inputs(std::size_t n) {
    // Deterministic, not random: spans the nonneg branch, the reflection
    // branch, and the region near the overflow guard -- the same practical
    // domain tests/kernels/erfcx_avx2.cpp validates correctness over.
    std::vector<double> xs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double u = static_cast<double>(i % 10000) / 10000.0;
        xs[i] = -26.0 + 52.0 * u;  // sweeps [-26, 26) repeatedly
    }
    return xs;
}

double min_trial_ns(int trials, int repeats, void (*fn)(const std::vector<double>&)) {
    const std::vector<double> xs = make_inputs(100000);
    fn(xs);  // warm-up
    double best = -1.0;
    for (int t = 0; t < trials; ++t) {
        const auto start = std::chrono::steady_clock::now();
        for (int r = 0; r < repeats; ++r) fn(xs);
        const auto end = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(end - start).count();
        if (best < 0.0 || ns < best) best = ns;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    constexpr std::size_t n = 100000;
    constexpr int kRepeats = 50;
    constexpr int kTrials = 7;

    volatile double sink = 0.0;

    static volatile double leak = 0.0;  // prevents the optimiser from deleting either loop

    const double scalar_ns = min_trial_ns(kTrials, kRepeats, [](const std::vector<double>& xs) {
        double s = 0.0;
        for (double x : xs) s += kernels::scalar::erfcx_poly(x);
        leak = s;
    });

    std::vector<double> out(n);
    const double avx2_ns = min_trial_ns(kTrials, kRepeats, [](const std::vector<double>& xs) {
        static std::vector<double> o(xs.size());
        kernels::simd::erfcx_avx2_batch(xs, o);
        leak = o[0];
    });
    (void)sink;
    (void)leak;

    const double scalar_total_ops = static_cast<double>(n) * kRepeats;
    const double avx2_total_ops = static_cast<double>(n) * kRepeats;
    const double scalar_ns_per_op = scalar_ns / scalar_total_ops;
    const double avx2_ns_per_op = avx2_ns / avx2_total_ops;
    const double speedup = scalar_ns_per_op / avx2_ns_per_op;

    const auto& bi = build_info();
    std::printf("simd_erfcx  (n=%zu, %d repeats, %d trials, best-of)\n", n, kRepeats, kTrials);
    std::printf("host: %s, compiler %s %s, build %s, SIMD: %s\n\n", bi.host_cpu, bi.compiler_id,
               bi.compiler_version, bi.build_type, bi.simd_build_level);
    std::printf("%-30s %12s %12s\n", "kernel", "ns/op", "ops/sec");
    std::printf("%-30s %12.3f %12.0f\n", "scalar erfcx_poly", scalar_ns_per_op,
               1e9 / scalar_ns_per_op);
    std::printf("%-30s %12.3f %12.0f\n", "AVX2 erfcx_avx2_batch", avx2_ns_per_op,
               1e9 / avx2_ns_per_op);
    std::printf("\nmeasured speedup: %.2fx  (AVX2 processes 4 lanes/instruction; "
               "an exact 4x would assume zero overhead)\n",
               speedup);

    if (argc > 1) {
        std::ofstream o(argv[1]);
        o << "{\n  \"benchmark\": \"simd_erfcx\",\n  \"n\": " << n << ",\n"
          << "  \"host_cpu\": \"" << bi.host_cpu << "\",\n"
          << "  \"simd_build_level\": \"" << bi.simd_build_level << "\",\n"
          << "  \"scalar_ns_per_op\": " << scalar_ns_per_op << ",\n"
          << "  \"avx2_ns_per_op\": " << avx2_ns_per_op << ",\n"
          << "  \"measured_speedup\": " << speedup << "\n}\n";
        std::printf("\nwrote %s\n", argv[1]);
    }
    return 0;
}
