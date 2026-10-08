// SPDX-License-Identifier: MIT
/// Validates the AVX2 erfcx kernel: against the double-double reference
/// directly (inheriting the scalar formula's own accuracy claim), against
/// the scalar polynomial kernel (the SIMD-equivalence claim, with its own
/// measured -- not guessed -- tolerance), and the batch-size edge cases
/// (tails not divisible by 4, the overflow guard, NaN propagation) a
/// hand-written SIMD kernel is most likely to get wrong.

#include "vl_test_support.hpp"

#include "simd/erfcx_avx2.hpp"

#include "scalar/erfcx_poly.hpp"
#include "volatility_lab/math/reference_special.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace vl;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

std::vector<double> batch(std::span<const double> x) {
    std::vector<double> out(x.size());
    kernels::simd::erfcx_avx2_batch(x, out);
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Against the double-double reference -- inherits the scalar kernel's claim
// ---------------------------------------------------------------------------

TEST(ErfcxAvx2, MatchesTheDoubleDoubleReferenceWithinTheSameBudgetAsTheScalarKernel) {
    const std::vector<double> xs = test::lin_space(-10.0, 10.0, 20001);
    const std::vector<double> out = batch(xs);

    WorstCase worst;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double ref = math::reference::erfcx_ref(xs[i]);
        if (!std::isfinite(ref) || !std::isfinite(out[i])) {
            worst.skip();
            continue;
        }
        worst.observe(rel_error(out[i], ref), xs[i], 0.0, out[i], ref);
    }
    EXPECT_GT(worst.samples, 0);
    EXPECT_CLOSE_TOL(worst.value, worst.reference, math::tol::kErfcxPoly)
        << worst.describe("x", "(unused)");
}

// ---------------------------------------------------------------------------
// SIMD vs scalar equivalence -- the measured, corrected budget
// ---------------------------------------------------------------------------

TEST(ErfcxAvx2, MatchesTheScalarKernelWithinTheMeasuredSimdEquivalenceBudget) {
    // x in [-26, 100]: covers the nonneg branch's whole practical range, the
    // reflection branch up to just short of the overflow guard (kExpSqMax ~=
    // 26.6416), and the region (worst case measured near x=-25.7) where the
    // two independent exp approximations disagree most.
    std::vector<double> xs;
    for (double x = -26.0; x <= 100.0; x += 0.0005) xs.push_back(x);
    const std::vector<double> out = batch(xs);

    double max_rel = 0.0;
    std::size_t n_checked = 0, n_skipped = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double scal = kernels::scalar::erfcx_poly(xs[i]);
        if (!std::isfinite(scal) || !std::isfinite(out[i])) {
            ++n_skipped;
            continue;
        }
        max_rel = std::max(max_rel, rel_error(out[i], scal));
        ++n_checked;
    }
    EXPECT_GT(n_checked, 0u);
    EXPECT_LE(max_rel, math::tol::kSimdEquivalence.rtol)
        << "max relative error " << max_rel << " over " << n_checked << " points ("
        << n_skipped << " skipped as non-finite)";
}

TEST(ErfcxAvx2, BothBranchesAgreeExactlyWithTheirOwnPairingWhenFinite) {
    // Every finite AVX2/scalar pair must agree -- not "most of them".
    const std::vector<double> xs = test::lin_space(-20.0, 20.0, 8001);
    const std::vector<double> out = batch(xs);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double scal = kernels::scalar::erfcx_poly(xs[i]);
        ASSERT_EQ(std::isfinite(out[i]), std::isfinite(scal)) << "x=" << xs[i];
        if (std::isfinite(scal)) {
            EXPECT_CLOSE_TOL(out[i], scal, math::tol::kSimdEquivalence) << "x=" << xs[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Batch-size edge cases: the part a hand-written SIMD kernel most often
// gets wrong, so tested explicitly rather than only via a sweep that
// happens to be a multiple of 4.
// ---------------------------------------------------------------------------

TEST(ErfcxAvx2, ExactMultipleOfFourUsesNoScalarTail) {
    const std::vector<double> xs = {0.1, 0.5, 1.0, 2.0};  // exactly 4
    const std::vector<double> out = batch(xs);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_CLOSE_TOL(out[i], kernels::scalar::erfcx_poly(xs[i]), math::tol::kSimdEquivalence);
    }
}

TEST(ErfcxAvx2, PureTailOfOneThroughThreeElementsIsHandledCorrectly) {
    for (std::size_t n : {1u, 2u, 3u}) {
        std::vector<double> xs;
        for (std::size_t i = 0; i < n; ++i) xs.push_back(0.3 + 0.7 * static_cast<double>(i));
        const std::vector<double> out = batch(xs);
        ASSERT_EQ(out.size(), n);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_CLOSE_TOL(out[i], kernels::scalar::erfcx_poly(xs[i]),
                            math::tol::kSimdEquivalence)
                << "n=" << n << " i=" << i;
        }
    }
}

TEST(ErfcxAvx2, FourFullLanesPlusATailOfThreeMatchesElementwise) {
    std::vector<double> xs;
    for (int i = 0; i < 7; ++i) xs.push_back(-3.0 + 0.9 * static_cast<double>(i));  // n=7
    const std::vector<double> out = batch(xs);
    ASSERT_EQ(out.size(), 7u);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_CLOSE_TOL(out[i], kernels::scalar::erfcx_poly(xs[i]), math::tol::kSimdEquivalence)
            << "i=" << i;
    }
}

TEST(ErfcxAvx2, EmptyBatchDoesNothingAndDoesNotCrash) {
    std::vector<double> empty;
    std::vector<double> out;
    kernels::simd::erfcx_avx2_batch(empty, out);
    EXPECT_TRUE(out.empty());
}

// ---------------------------------------------------------------------------
// Overflow guard and NaN -- same contract as the scalar kernel
// ---------------------------------------------------------------------------

TEST(ErfcxAvx2, NegativeXPastTheOverflowGuardReturnsPositiveInfinity) {
    const std::vector<double> xs = {-30.0, -27.0, -26.6416};
    const std::vector<double> out = batch(xs);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        EXPECT_TRUE(std::isinf(out[i]) && out[i] > 0.0) << "x=" << xs[i] << " got " << out[i];
    }
}

TEST(ErfcxAvx2, NegativeXJustInsideTheGuardIsFiniteAndMatchesScalar) {
    const std::vector<double> xs = {-26.0, -25.0, -20.0};
    const std::vector<double> out = batch(xs);
    for (std::size_t i = 0; i < xs.size(); ++i) {
        ASSERT_TRUE(std::isfinite(out[i])) << "x=" << xs[i];
        EXPECT_CLOSE_TOL(out[i], kernels::scalar::erfcx_poly(xs[i]), math::tol::kSimdEquivalence)
            << "x=" << xs[i];
    }
}

TEST(ErfcxAvx2, NanInputPropagatesNanInEveryLane) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> xs = {1.0, nan, -1.0, 2.0};
    const std::vector<double> out = batch(xs);
    EXPECT_FALSE(std::isnan(out[0]));
    EXPECT_TRUE(std::isnan(out[1]));
    EXPECT_FALSE(std::isnan(out[2]));
    EXPECT_FALSE(std::isnan(out[3]));
}
