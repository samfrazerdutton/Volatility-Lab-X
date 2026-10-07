// SPDX-License-Identifier: MIT
#pragma once
/// \file vl_test_support.hpp
/// \brief Shared assertions and samplers for the numerical test suite.
///
/// ## The one rule
///
/// `EXPECT_EQ` on a floating-point value is forbidden in this suite, with
/// exactly two exceptions: a value that is *constructed* to be exact (an
/// intrinsic, a payoff sign) and a claim of *bitwise* reproducibility, where
/// equality is the whole point.  Everything else goes through
/// `EXPECT_CLOSE_TOL`, which requires a named `vl::math::Tolerance` -- so a
/// test cannot assert accuracy without stating, in a reviewable place, what
/// accuracy it is asserting and why.
///
/// ## Sweeps, not points
///
/// A pricing routine that is right at (F=100, K=100, sigma=0.2, T=1) and wrong
/// at (K=400, T=0.08) has passed a point test and failed its job.  The
/// samplers below generate logarithmic sweeps over the normalised (x, s)
/// domain, which -- because the Black normalisation reduces five inputs to two
/// -- is small enough to cover densely.  Tests report the *worst* case over a
/// sweep and where it occurred, so a failure message names the input rather
/// than just the discrepancy.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "volatility_lab/math/compare.hpp"

namespace vl::test {

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

/// GoogleTest predicate for `vl::math::Tolerance`.  Failure messages include
/// the tolerance's own rationale string, so a red test explains the budget it
/// broke without the reader going to look it up.
inline ::testing::AssertionResult CheckClose(const char* a_expr, const char* b_expr,
                                            const char* /*tol_expr*/, double a, double b,
                                            const math::Tolerance& tol) {
    if (tol.accepts(a, b)) return ::testing::AssertionSuccess();
    const double abs_err = std::abs(a - b);
    return ::testing::AssertionFailure()
           << a_expr << " = " << std::to_string(a) << "\n"
           << b_expr << " = " << std::to_string(b) << "\n"
           << "  absolute error  " << abs_err << "\n"
           << "  relative error  " << math::rel_error(a, b) << "\n"
           << "  error in ulps   " << math::error_in_ulps(a, b) << "\n"
           << "  budget          atol=" << tol.atol << " rtol=" << tol.rtol << "\n"
           << "  budget rationale: " << tol.rationale;
}

#define EXPECT_CLOSE_TOL(a, b, tol) EXPECT_PRED_FORMAT3(::vl::test::CheckClose, a, b, tol)
#define ASSERT_CLOSE_TOL(a, b, tol) ASSERT_PRED_FORMAT3(::vl::test::CheckClose, a, b, tol)

/// Bitwise identity.  Spelled out rather than EXPECT_EQ so that grepping the
/// suite for intentional exact comparisons finds every one of them.
#define EXPECT_BITWISE_EQ(a, b)                                                  \
    EXPECT_EQ(::vl::math::ulp_distance((a), (b)), 0) << "values differ by "       \
                                                     << ::vl::math::ulp_distance((a), (b)) \
                                                     << " ulps"

// ---------------------------------------------------------------------------
// Worst-case accumulator
// ---------------------------------------------------------------------------

/// Tracks the worst error over a sweep together with the inputs that produced
/// it, so a failure reports a reproducible case.
struct WorstCase {
    double error = 0.0;
    double arg0 = 0.0;
    double arg1 = 0.0;
    double value = 0.0;
    double reference = 0.0;
    long samples = 0;
    long skipped = 0;

    void observe(double err, double a0, double a1, double v, double ref) {
        ++samples;
        if (err > error) {
            error = err;
            arg0 = a0;
            arg1 = a1;
            value = v;
            reference = ref;
        }
    }
    void skip() { ++skipped; }

    [[nodiscard]] std::string describe(const char* n0 = "arg0",
                                       const char* n1 = "arg1") const {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "worst %.4e over %ld samples (%ld skipped) at %s=%.10g %s=%.10g "
                      "[value %.17g, reference %.17g]",
                      error, samples, skipped, n0, arg0, n1, arg1, value, reference);
        return std::string(buf);
    }
};

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

/// `count` points geometrically spaced over [lo, hi].  Both must be > 0.
inline std::vector<double> log_space(double lo, double hi, int count) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(count));
    const double a = std::log(lo);
    const double b = std::log(hi);
    for (int i = 0; i < count; ++i) {
        const double u = (count == 1) ? 0.0 : static_cast<double>(i) / (count - 1);
        out.push_back(std::exp(a + u * (b - a)));
    }
    return out;
}

inline std::vector<double> lin_space(double lo, double hi, int count) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const double u = (count == 1) ? 0.0 : static_cast<double>(i) / (count - 1);
        out.push_back(lo + u * (hi - lo));
    }
    return out;
}

/// The normalised Black domain, as used by the pricing and implied-vol tests.
///
/// The ranges are deliberately far wider than anything a market produces:
/// |x| up to 1e2 is a strike ratio of e^100, and s up to 1e2 is a total
/// volatility of 10000%.  An engine asked only about plausible inputs is an
/// engine whose edge-case behaviour is unknown.
struct NormalisedDomain {
    std::vector<double> abs_x;  ///< |log(F/K)|
    std::vector<double> s;      ///< sigma*sqrt(T)
};

inline NormalisedDomain normalised_domain(int nx = 90, int ns = 90) {
    return {log_space(1e-8, 1e2, nx), log_space(1e-8, 1e2, ns)};
}

/// Market-plausible inputs, for tests about behaviour rather than limits.
struct MarketDomain {
    std::vector<double> strike_ratio;  ///< K/F
    std::vector<double> vol;
    std::vector<double> years;
};

inline MarketDomain market_domain() {
    return {log_space(0.1, 10.0, 25), log_space(0.01, 3.0, 12),
            log_space(1.0 / 8760.0, 10.0, 12)};
}

}  // namespace vl::test
