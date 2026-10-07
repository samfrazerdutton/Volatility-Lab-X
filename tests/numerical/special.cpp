// SPDX-License-Identifier: MIT
/// Validates math/special.hpp against the double-double reference over the
/// whole domain, and pins the measured accuracy figures that the headers and
/// docs quote.  If one of those numbers changes, this test fails and the claim
/// gets updated -- which is the point: a documented accuracy figure that
/// nothing checks is a documented accuracy figure that drifts.

#include "vl_test_support.hpp"

#include "volatility_lab/math/reference_special.hpp"
#include "volatility_lab/math/special.hpp"

using namespace vl;
using namespace vl::math;
using vl::test::WorstCase;

namespace {
double rel_error_local(double a, double b) { return vl::math::rel_error(a, b); }
}  // namespace

namespace {

/// The error figures published in math/special.hpp.  A test that merely
/// asserts "better than 1e-9" would pass while the implementation silently
/// got 1000x worse, so each bound is set just above the measured value.
struct PublishedUlpBounds {
    static constexpr double kNormCdf = 2000.0;   ///< fast path, tail-degrading
    static constexpr double kNormCdfHp = 12.0;   ///< measured 9
    static constexpr double kErfcx = 12.0;       ///< measured 9
    static constexpr double kNormInv = 8.0;      ///< measured 4
    static constexpr double kExpSq = 4.0;        ///< measured 2
};

}  // namespace

// ===========================================================================
// exp with an exact argument
// ===========================================================================

TEST(Special, ExpSqBeatsNaiveByTwoOrdersOfMagnitude) {
    // The header claims 498 ulps -> 2 ulps for exp(x*x) over x in [0.5, 26].
    // Both halves of that claim are checked: the naive form really is that
    // bad, and the exact-argument form really is that good.  Checking only the
    // second half would leave the justification for the extra fma unverified.
    WorstCase naive;
    WorstCase exact;
    for (double x = 0.5; x < 26.0; x += 0.0137) {
        const double ref = exp_dd(DDouble(x) * DDouble(x)).to_double();
        naive.observe(error_in_ulps(std::exp(x * x), ref), x, 0, std::exp(x * x), ref);
        exact.observe(error_in_ulps(exp_sq(x), ref), x, 0, exp_sq(x), ref);
    }
    EXPECT_GT(naive.error, 100.0) << "naive exp(x*x) is expected to be bad; if it is not, "
                                     "the platform changed and exp_sq may be unnecessary. "
                                  << naive.describe("x");
    EXPECT_LT(exact.error, PublishedUlpBounds::kExpSq) << exact.describe("x");
}

TEST(Special, ExpNegHalfSqMatchesReference) {
    WorstCase w;
    for (double x = -40.0; x <= 40.0; x += 0.01) {
        const DDouble ref_dd = exp_dd(DDouble(-0.5) * DDouble(x) * DDouble(x));
        const double ref = ref_dd.to_double();
        if (!(ref > 1e-300)) {
            w.skip();
            continue;
        }
        w.observe(error_in_ulps(exp_neg_half_sq(x), ref), x, 0, exp_neg_half_sq(x), ref);
    }
    EXPECT_LT(w.error, 4.0) << w.describe("x");
}

TEST(Special, ExpNegHalfSumSqKeepsTheWholeExponentExact) {
    // The argument reaches -568 in the deep-OTM corner of the Black vega, and
    // exp converts an absolute argument error into a relative result error, so
    // a single rounding of a^2 + b^2 costs ~a^2*eps/2 relative.  Checked
    // against the naive form so the extra work stays justified.
    WorstCase naive;
    WorstCase exact;
    for (double a : {0.5, 3.0, 10.0, 25.0, 33.7}) {
        for (double b : {0.001, 0.0444, 0.5, 2.0}) {
            const DDouble ref_dd =
                exp_dd(DDouble(-0.5) * (DDouble(a) * DDouble(a) + DDouble(b) * DDouble(b)));
            const double ref = ref_dd.to_double();
            if (!(ref > 1e-300)) {
                naive.skip();
                exact.skip();
                continue;
            }
            const double n = std::exp(-0.5 * std::fma(a, a, b * b));
            naive.observe(error_in_ulps(n, ref), a, b, n, ref);
            const double g = exp_neg_half_sum_sq(a, b);
            exact.observe(error_in_ulps(g, ref), a, b, g, ref);
        }
    }
    EXPECT_GT(naive.error, 20.0)
        << "the fma-only form is expected to lose accuracy at large arguments; "
           "if it no longer does, exp_neg_half_sum_sq is unnecessary. "
        << naive.describe("a", "b");
    EXPECT_LT(exact.error, 4.0) << exact.describe("a", "b");
}

// ===========================================================================
// Normal CDF
// ===========================================================================

TEST(Special, NormCdfKnownValues) {
    // Values independently computed with mpmath at 60 digits.
    EXPECT_CLOSE_TOL(norm_cdf(0.0), 0.5, tol::kNormCdf);
    EXPECT_CLOSE_TOL(norm_cdf(1.0), 0.84134474606854294859, tol::kNormCdf);
    EXPECT_CLOSE_TOL(norm_cdf(-1.0), 0.15865525393145705141, tol::kNormCdf);
    // The tail values are asserted against `norm_cdf_hp`, not `norm_cdf`.
    // That is not a convenience: the fast path is *documented* as degrading
    // like x^2*eps out there (19 ulps at x = -10), and asserting a few-eps
    // bound on it would contradict the design rather than test it.  The
    // degradation itself is pinned by
    // NormCdfFastPathDegradesInTheTailAsDocumented below.
    EXPECT_CLOSE_TOL(norm_cdf_hp(-10.0), 7.619853024160526066e-24, tol::kNormCdf);
    EXPECT_CLOSE_TOL(norm_cdf_hp(-30.0), 4.9067139271481870595e-198, tol::kNormCdf);
    EXPECT_LT(rel_error_local(norm_cdf(-10.0), 7.619853024160526066e-24), 1e-14);
}

TEST(Special, NormCdfSymmetry) {
    // Phi(x) + Phi(-x) = 1 exactly in exact arithmetic.  In floating point the
    // claim is that the two are computed consistently, which this checks
    // without needing a reference at all.
    WorstCase w;
    for (double x = 0.0; x <= 8.0; x += 0.001) {
        const double sum = norm_cdf(x) + norm_cdf(-x);
        w.observe(std::abs(sum - 1.0), x, 0, sum, 1.0);
    }
    EXPECT_LT(w.error, 4.0 * kEps) << w.describe("x");
}

TEST(Special, NormCdfFastPathDegradesInTheTailAsDocumented) {
    // This test exists to keep the documentation honest.  norm_cdf is *not*
    // uniformly accurate, the header says so and says why, and the figure
    // below is the measured worst case.  Asserting a tight bound here would
    // require changing the implementation; asserting no bound would let the
    // documented table rot.
    WorstCase w;
    for (double x = -40.0; x <= 40.0; x += 0.01) {
        const double ref = reference::norm_cdf_ref(x);
        if (!(ref > 1e-300)) {
            w.skip();
            continue;
        }
        w.observe(error_in_ulps(norm_cdf(x), ref), x, 0, norm_cdf(x), ref);
    }
    EXPECT_LT(w.error, PublishedUlpBounds::kNormCdf) << w.describe("x");
    EXPECT_GT(w.error, 100.0)
        << "norm_cdf is documented as degrading in the tail; it no longer does, so "
           "either the platform libm improved or norm_cdf_hp is now redundant. "
        << w.describe("x");
}

TEST(Special, NormCdfHpIsUniformlyAccurate) {
    // The claim that matters: the self-compensating form is accurate to a few
    // ulps *independent of x*, which is what makes it usable for densities and
    // arbitrage diagnostics in the far wings.
    WorstCase w;
    for (double x = -40.0; x <= 40.0; x += 0.005) {
        const double ref = reference::norm_cdf_ref(x);
        if (!(ref > 1e-300)) {
            w.skip();
            continue;
        }
        w.observe(error_in_ulps(norm_cdf_hp(x), ref), x, 0, norm_cdf_hp(x), ref);
    }
    EXPECT_LT(w.error, PublishedUlpBounds::kNormCdfHp) << w.describe("x");
}

TEST(Special, NormCdfPairAgreesWithScalarCalls) {
    for (double x = -12.0; x <= 12.0; x += 0.013) {
        const auto [cdf, pdf] = norm_cdf_pair(x);
        EXPECT_BITWISE_EQ(cdf, norm_cdf(x));
        EXPECT_BITWISE_EQ(pdf, norm_pdf(x));
    }
}

// ===========================================================================
// erfcx
// ===========================================================================

TEST(Special, ErfcxKnownValues) {
    // mpmath, 60 digits.  The first three are in the direct branch, the rest
    // in the continued fraction, so both are pinned by literal values that do
    // not come from this codebase.
    struct Row { double x, truth; };
    const Row rows[] = {
        {0.0, 1.0},
        {3.0, 0.17900115118138995042},
        {5.0, 0.11070463773306862637},
        {8.0, 0.069985166200880927723},
        {9.0, 0.062307724037774684147},
        {26.0, 0.021683584850562906616},
        {30.0, 0.018795888861416751497},
        {100.0, 0.0056416137829894329036},
    };
    for (const auto& r : rows) {
        EXPECT_LT(error_in_ulps(erfcx(r.x), r.truth), PublishedUlpBounds::kErfcx)
            << "erfcx(" << r.x << ") = " << erfcx(r.x) << ", expected " << r.truth;
    }
}

TEST(Special, ErfcxContinuedFractionBranchIsNotReciprocated) {
    // Regression test for a real bug.  Modified Lentz with b0 = 1 returns the
    // continued fraction K itself, not 1/K, and the first implementation
    // forgot the reciprocal.  The resulting error is a factor of
    // (1 + 1/(2x^2)) -- 1.5% at x = 8, 1e-4 at x = 100 -- which is small
    // enough to look like a tolerance problem rather than a sign that the
    // formula is wrong.  Checking the asymptotic limit catches it immediately:
    // x*sqrt(pi)*erfcx(x) -> 1 from *below*, never above.
    constexpr double kSqrtPi = 1.77245385090551602730;
    for (double x : {8.0, 12.0, 20.0, 50.0, 100.0, 1e4}) {
        const double scaled = x * kSqrtPi * erfcx(x);
        // The correction is 1 - 1/(2x^2) + O(x^-4), so the product is strictly
        // below 1 wherever that correction is still representable.  By x = 1e8
        // it is 5e-17 and rounds away, hence the upper limit of the sweep.
        EXPECT_LT(scaled, 1.0) << "x = " << x
                               << ": x*sqrt(pi)*erfcx(x) must approach 1 from below";
        EXPECT_GT(scaled, 1.0 - 1.0 / (x * x)) << "x = " << x;
    }
    EXPECT_LE(1e8 * kSqrtPi * erfcx(1e8), 1.0);
}

TEST(Special, ErfcxMatchesReferenceOverTheDomain) {
    WorstCase w;
    for (double x = -25.0; x <= 25.0; x += 0.01) {
        const double ref = reference::erfcx_ref(x);
        if (!std::isfinite(ref) || ref == 0.0) {
            w.skip();
            continue;
        }
        w.observe(error_in_ulps(erfcx(x), ref), x, 0, erfcx(x), ref);
    }
    for (double x = 8.0; x < 1e7; x *= 1.05) {
        const double ref = reference::erfcx_ref(x);
        w.observe(error_in_ulps(erfcx(x), ref), x, 0, erfcx(x), ref);
    }
    EXPECT_LT(w.error, PublishedUlpBounds::kErfcx) << w.describe("x");
}

TEST(Special, ErfcxIsFiniteWhereExpSqOverflows) {
    // exp(x^2) overflows above x = 26.64, so any implementation that forms the
    // product directly returns inf there.  erfcx must not.
    for (double x : {26.0, 26.7, 30.0, 40.0, 100.0, 1e5, 1e100}) {
        EXPECT_TRUE(std::isfinite(erfcx(x))) << "erfcx(" << x << ") = " << erfcx(x);
        EXPECT_GT(erfcx(x), 0.0) << "x = " << x;
    }
    EXPECT_EQ(erfcx(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_TRUE(std::isnan(erfcx(std::numeric_limits<double>::quiet_NaN())));
}

TEST(Special, ErfcxIsMonotoneDecreasing) {
    double prev = erfcx(-20.0);
    for (double x = -20.0 + 0.01; x <= 30.0; x += 0.01) {
        const double cur = erfcx(x);
        ASSERT_LE(cur, prev) << "erfcx is not monotone at x = " << x;
        prev = cur;
    }
}

// ===========================================================================
// Inverse normal CDF
// ===========================================================================

TEST(Special, NormInvKnownValues) {
    EXPECT_BITWISE_EQ(norm_inv(0.5), 0.0);
    // z_{0.975}, exact value 1.959963984540054235525 (mpmath).
    EXPECT_LT(error_in_ulps(norm_inv(0.975), 1.9599639845400543), 8.0);
    EXPECT_LT(error_in_ulps(norm_inv(0.025), -1.9599639845400543), 8.0);
}

TEST(Special, NormInvRoundTripsAgainstNormCdfInTheLowerTail) {
    // Only the lower tail, and that restriction is the point.
    //
    // For x > 0, Phi(x) approaches 1 and the representable doubles near 1 are
    // spaced 1.1e-16 apart, while dPhi/dx = phi(x).  So the round trip can
    // only recover x to about ulp(1)/phi(x), which at x = 8 is
    // 1.1e-16/5.1e-15 = 0.02 -- and indeed an earlier version of this test
    // measured 9.6e-3 and was "failing" on a representation limit rather than
    // on an implementation defect.  There is no implementation that does
    // better from a double p; the API answer is `norm_inv_upper`, exercised in
    // NormInvUpperRecoversWhatNormInvCannot.
    WorstCase w;
    for (double x = -8.0; x <= 0.0; x += 0.001) {
        const double p = norm_cdf_hp(x);
        if (p <= 0.0 || p >= 1.0) {
            w.skip();
            continue;
        }
        const double back = norm_inv(p);
        w.observe(std::abs(back - x), x, 0, back, x);
    }
    // Absolute, not relative: near x = 0 the relative error is unbounded by
    // construction, and absolute accuracy is what a caller of Phi^{-1} relies
    // on.
    EXPECT_LT(w.error, 1e-13) << w.describe("x");
}

TEST(Special, NormInvUpperRoundTripsInTheUpperTail) {
    // The mirror of the test above, done the way the API intends: pass the
    // upper-tail probability directly and the round trip is as accurate on the
    // right as on the left.
    WorstCase w;
    for (double x = 0.0; x <= 8.0; x += 0.001) {
        const double q = norm_cdf_hp(-x);  // = 1 - Phi(x), computed as a tail
        if (q <= 0.0 || q >= 1.0) {
            w.skip();
            continue;
        }
        const double back = norm_inv_upper(q);
        w.observe(std::abs(back - x), x, 0, back, x);
    }
    EXPECT_LT(w.error, 1e-13) << w.describe("x");
}

TEST(Special, NormInvIsAccurateNearOneHalf) {
    // Regression test: the first implementation computed the Halley residual
    // as Phi(x) - p, which cancels catastrophically when both are near 0.5.
    // It was 1450 ulps wrong just off the centre.
    WorstCase w;
    for (double p = 0.4; p <= 0.6; p += 1e-5) {
        const double ref = reference::norm_inv_ref(p);
        w.observe(error_in_ulps(norm_inv(p), ref), p, 0, norm_inv(p), ref);
    }
    EXPECT_LT(w.error, PublishedUlpBounds::kNormInv) << w.describe("p");
}

TEST(Special, NormInvIsAccurateInBothTails) {
    WorstCase w;
    for (double p = 1e-300; p < 1e-3; p *= 1.2) {
        const double ref = reference::norm_inv_ref(p);
        w.observe(error_in_ulps(norm_inv(p), ref), p, 0, norm_inv(p), ref);
    }
    EXPECT_LT(w.error, PublishedUlpBounds::kNormInv) << w.describe("p");
}

TEST(Special, NormInvCentralRegionMatchesReference) {
    WorstCase w;
    for (int i = 1; i < 100000; ++i) {
        const double p = i / 100000.0;
        const double ref = reference::norm_inv_ref(p);
        w.observe(error_in_ulps(norm_inv(p), ref), p, 0, norm_inv(p), ref);
    }
    EXPECT_LT(w.error, PublishedUlpBounds::kNormInv) << w.describe("p");
}

TEST(Special, NormInvEdgeCases) {
    EXPECT_EQ(norm_inv(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(norm_inv(1.0), std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(norm_inv(-0.1)));
    EXPECT_TRUE(std::isnan(norm_inv(1.1)));
    EXPECT_TRUE(std::isnan(norm_inv(std::numeric_limits<double>::quiet_NaN())));
}

TEST(Special, NormInvUpperRecoversWhatNormInvCannot) {
    // The representation argument for having a separate upper-tail entry
    // point: 1 - 1e-17 is not representable, so norm_inv of it is +inf, while
    // the upper-tail formulation answers correctly.
    EXPECT_TRUE(std::isinf(norm_inv(1.0 - 1e-17)));
    const double z = norm_inv_upper(1e-17);
    EXPECT_TRUE(std::isfinite(z));
    EXPECT_NEAR(z, 8.4937932241095968, 1e-12);

    // And it agrees with norm_inv by symmetry wherever both are meaningful.
    for (double q = 1e-12; q < 0.4; q *= 1.7) {
        EXPECT_LT(error_in_ulps(norm_inv_upper(q), -norm_inv(q)), 2.0) << "q = " << q;
    }
}

TEST(Special, NormInvIsMonotone) {
    double prev = -std::numeric_limits<double>::infinity();
    for (int i = 1; i < 20000; ++i) {
        const double cur = norm_inv(i / 20000.0);
        ASSERT_GE(cur, prev) << "norm_inv is not monotone at i = " << i;
        prev = cur;
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

TEST(Special, ClampPropagatesNaN) {
    // std::clamp would return the bound here.  Silently clamping a NaN hides
    // exactly the input the library is supposed to complain about.
    EXPECT_TRUE(std::isnan(clamp_nan_aware(std::numeric_limits<double>::quiet_NaN(), 0.0, 1.0)));
    EXPECT_EQ(clamp_nan_aware(-1.0, 0.0, 1.0), 0.0);
    EXPECT_EQ(clamp_nan_aware(2.0, 0.0, 1.0), 1.0);
    EXPECT_EQ(clamp_nan_aware(0.5, 0.0, 1.0), 0.5);
}

TEST(Special, IsPositiveFinite) {
    EXPECT_TRUE(is_positive_finite(1e-300));
    EXPECT_FALSE(is_positive_finite(0.0));
    EXPECT_FALSE(is_positive_finite(-1.0));
    EXPECT_FALSE(is_positive_finite(std::numeric_limits<double>::infinity()));
    EXPECT_FALSE(is_positive_finite(std::numeric_limits<double>::quiet_NaN()));
}
