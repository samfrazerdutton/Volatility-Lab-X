// SPDX-License-Identifier: MIT
/// Validates the double-double layer.  This is the base of the whole
/// validation chain: if `DDouble` is wrong then every accuracy claim in the
/// project is measured against a broken ruler, so these tests lean on
/// identities and on constants computed independently (mpmath, 60 digits)
/// rather than on anything this codebase produces.

#include "vl_test_support.hpp"

#include "volatility_lab/math/dd_real.hpp"
#include "volatility_lab/math/reference_special.hpp"

using namespace vl::math;

namespace {

/// Relative error of a double-double value against an exactly-known decimal,
/// measured at double-double precision rather than double.
///
/// The hi+lo pair carries ~106 bits, so `to_double()` would throw away exactly
/// the precision under test.  Instead the known value is split into its own
/// hi/lo pair the same way and the two are compared component-wise.
double dd_rel_error(DDouble got, double truth_hi, double truth_lo) {
    const DDouble truth{truth_hi, truth_lo};
    const DDouble diff = got - truth;
    const double d = std::abs(diff.hi() + diff.lo());
    const double t = std::abs(truth_hi);
    return (t > 0.0) ? d / t : d;
}

constexpr double kDdEps = 1e-31;  // ~106 bits

}  // namespace

// ===========================================================================
// Error-free transformations
// ===========================================================================

TEST(DdReal, TwoSumProducesTheCorrectlyRoundedSumAndItsResidual) {
    // Knuth two-sum guarantees a + b == s + e *exactly*, where s is the
    // correctly-rounded sum.  Two things can be checked without appealing to
    // higher precision (which would mean testing the type with itself):
    //
    //   1. s is exactly the rounded sum;
    //   2. the pair is already normalised, i.e. re-running two_sum on (s, e)
    //      is a fixed point.  That is equivalent to |e| <= ulp(s)/2, which is
    //      the non-overlap invariant the whole type depends on.
    //
    // Note what is *not* asserted: `s + e == s`.  That fails legitimately when
    // |e| is exactly half an ulp and round-half-to-even goes up, and asserting
    // it was simply wrong.
    const double cases[] = {1.0, 1e300, 1e-300, 0.1, 3.0, -7.5, 1.0 / 3.0, 12345.6789};
    for (double a : cases) {
        for (double b : cases) {
            double s = 0.0, e = 0.0;
            two_sum(a, b, s, e);
            EXPECT_EQ(s, a + b) << "a=" << a << " b=" << b;
            ASSERT_TRUE(std::isfinite(s)) << "a=" << a << " b=" << b;

            double s2 = 0.0, e2 = 0.0;
            two_sum(s, e, s2, e2);
            EXPECT_EQ(s2, s) << "pair is not normalised: a=" << a << " b=" << b;
            EXPECT_EQ(e2, e) << "pair is not normalised: a=" << a << " b=" << b;

            // Non-overlap: e is at most half an ulp of s.
            if (s != 0.0) {
                const double ulp = std::abs(std::nextafter(s, std::numeric_limits<double>::infinity()) - s);
                EXPECT_LE(std::abs(e), ulp) << "a=" << a << " b=" << b;
            }
        }
    }
}

TEST(DdReal, TwoSumRecoversTheResidualOfAnInexactAddition) {
    // The case the type exists for: adding a tiny number to a large one.  The
    // double sum discards the tiny addend entirely, and `e` must be exactly
    // what was discarded.
    double s = 0.0, e = 0.0;
    two_sum(1.0, 1e-20, s, e);
    EXPECT_EQ(s, 1.0);
    EXPECT_EQ(e, 1e-20);

    two_sum(1e16, 1.0, s, e);
    EXPECT_EQ(s, 1e16);
    EXPECT_EQ(e, 1.0);
}

TEST(DdReal, QuickTwoSumRequiresOrderedMagnitudes) {
    double s, e;
    quick_two_sum(1.0, 1e-20, s, e);
    EXPECT_EQ(s, 1.0);
    EXPECT_EQ(e, 1e-20);
}

TEST(DdReal, TwoProductIsExact) {
    const double cases[] = {1.0, 3.0, 0.1, 1.0 / 7.0, 1e100, 1e-100, 12345.6789};
    for (double a : cases) {
        for (double b : cases) {
            double p, e;
            two_product(a, b, p, e);
            EXPECT_EQ(p, a * b);
            // p + e must reproduce the product to double-double accuracy; the
            // strongest cheap check is that recomputing the residual with fma
            // agrees.
            EXPECT_EQ(e, std::fma(a, b, -p)) << "a=" << a << " b=" << b;
        }
    }
}

// ===========================================================================
// Arithmetic
// ===========================================================================

TEST(DdReal, DivisionInvertsMultiplication) {
    // (a/b)*b == a to double-double precision.  A plain-double version of this
    // test would pass trivially; it only has teeth because the intermediate
    // carries 106 bits.
    vl::test::WorstCase w;
    for (double a : {1.0, 3.0, 7.0, 1e5, 1e-5, 0.1}) {
        for (double b : {1.0, 3.0, 7.0, 11.0, 1e3, 1e-3}) {
            const DDouble q = DDouble(a) / DDouble(b);
            const DDouble back = q * DDouble(b);
            const DDouble diff = back - DDouble(a);
            w.observe(std::abs(diff.hi() + diff.lo()) / a, a, b, back.hi(), a);
        }
    }
    EXPECT_LT(w.error, kDdEps) << w.describe("a", "b");
}

TEST(DdReal, ThirdTimesThreeIsOne) {
    const DDouble third = DDouble(1.0) / DDouble(3.0);
    const DDouble back = third * DDouble(3.0);
    const DDouble diff = back - DDouble(1.0);
    EXPECT_LT(std::abs(diff.hi() + diff.lo()), kDdEps);
    // And 1/3 really does carry more than 53 bits: the lo limb must be nonzero
    // and of the right magnitude.
    EXPECT_NE(third.lo(), 0.0);
    EXPECT_LT(std::abs(third.lo()), std::abs(third.hi()) * 1e-15);
}

TEST(DdReal, AdditionIsAssociativeToDdPrecision) {
    // Not exactly associative -- nothing in floating point is -- but the
    // discrepancy must be at the dd epsilon, not the double epsilon.
    const DDouble a(1.0, 1e-20);
    const DDouble b(1e-10, 1e-30);
    const DDouble c(3.0, -1e-18);
    const DDouble left = (a + b) + c;
    const DDouble right = a + (b + c);
    const DDouble diff = left - right;
    EXPECT_LT(std::abs(diff.hi() + diff.lo()), kDdEps * 4.0);
}

TEST(DdReal, LoLimbActuallyCarriesPrecision) {
    // The point of the type.  In plain double, 1 + 1e-20 == 1.  Here it must
    // not be, or the whole reference layer is a no-op.
    const DDouble one(1.0);
    const DDouble nudged = one + 1e-20;
    EXPECT_NE(nudged.lo(), 0.0);
    EXPECT_EQ(nudged.hi(), 1.0);
    const DDouble back = nudged - one;
    EXPECT_NEAR(back.hi() + back.lo(), 1e-20, 1e-35);
}

// ===========================================================================
// Elementary functions
// ===========================================================================

TEST(DdReal, SqrtIsAccurateToDdPrecision) {
    vl::test::WorstCase w;
    for (double a : {1e-10, 0.1, 1.0, 2.0, 3.0, 10.0, 1e5, 1e10}) {
        const DDouble r = sqrt_dd(DDouble(a));
        const DDouble back = r * r;
        const DDouble diff = back - DDouble(a);
        w.observe(std::abs(diff.hi() + diff.lo()) / a, a, 0, back.hi(), a);
    }
    EXPECT_LT(w.error, kDdEps) << w.describe("a");
    EXPECT_EQ(sqrt_dd(DDouble(0.0)).to_double(), 0.0);
}

TEST(DdReal, Sqrt2MatchesIndependentConstant) {
    // sqrt(2) = 1.41421356237309504880168872420969807857 (mpmath).
    EXPECT_LT(dd_rel_error(sqrt_dd(DDouble(2.0)), 1.41421356237309515e+00,
                           -9.66729331345291345e-17),
              kDdEps);
}

TEST(DdReal, ExpAndLogAreInverse) {
    vl::test::WorstCase w;
    for (double a = -600.0; a <= 600.0; a += 7.3) {
        const DDouble e = exp_dd(DDouble(a));
        if (!(e.hi() > 1e-300) || !std::isfinite(e.hi())) {
            w.skip();
            continue;
        }
        const DDouble back = log_dd(e);
        const DDouble diff = back - DDouble(a);
        const double scale = std::max(std::abs(a), 1.0);
        w.observe(std::abs(diff.hi() + diff.lo()) / scale, a, 0, back.hi(), a);
    }
    EXPECT_LT(w.error, 1e-29) << w.describe("a");
}

TEST(DdReal, ExpOfOneMatchesE) {
    // e = 2.71828182845904523536028747135266249776 (mpmath).
    EXPECT_LT(dd_rel_error(exp_dd(DDouble(1.0)), 2.71828182845904509e+00,
                           1.44564689172925016e-16),
              kDdEps);
}

TEST(DdReal, ExpHandlesOverflowAndUnderflow) {
    EXPECT_EQ(exp_dd(DDouble(-1000.0)).to_double(), 0.0);
    EXPECT_TRUE(std::isinf(exp_dd(DDouble(1000.0)).to_double()));
}

TEST(DdReal, LogOfNonPositiveIsNegativeInfinity) {
    EXPECT_TRUE(std::isinf(log_dd(DDouble(0.0)).to_double()));
    EXPECT_LT(log_dd(DDouble(0.0)).to_double(), 0.0);
    EXPECT_LT(log_dd(DDouble(-1.0)).to_double(), 0.0);
}

// ===========================================================================
// Constants
// ===========================================================================

TEST(DdReal, ConstantsMatchIndependentlyComputedValues) {
    // Every value here was produced by mpmath at 60 digits and split into
    // hi/lo by exact rational arithmetic, outside this codebase.  Two of these
    // constants were initially wrong in the source by ~1e-16 in the lo limb,
    // and this test is what would have caught it.
    EXPECT_LT(dd_rel_error(dd_const::kPi, 3.14159265358979312e+00,
                           1.22464679914735321e-16), kDdEps);
    EXPECT_LT(dd_rel_error(dd_const::kSqrt2, 1.41421356237309515e+00,
                           -9.66729331345291345e-17), kDdEps);
    EXPECT_LT(dd_rel_error(dd_const::kSqrtHalf, 7.07106781186547573e-01,
                           -4.83364665672645673e-17), kDdEps);
    EXPECT_LT(dd_rel_error(dd_const::kSqrt2Pi, 2.50662827463100069e+00,
                           -1.83285799804591668e-16), kDdEps);
    EXPECT_LT(dd_rel_error(dd_const::kInvSqrt2Pi, 3.98942280401432703e-01,
                           -2.49232720227773004e-17), kDdEps);
    EXPECT_LT(dd_rel_error(dd_const::kTwoOverSqrtPi, 1.12837916709551256e+00,
                           1.53354596131658812e-17), kDdEps);
}

TEST(DdReal, ConstantsSatisfyTheirDefiningIdentities) {
    // Independent of the literals above: sqrt(2pi)^2 == 2pi, and
    // 1/sqrt(2pi) * sqrt(2pi) == 1.
    const DDouble two_pi = dd_const::kPi * 2.0;
    const DDouble sq = dd_const::kSqrt2Pi * dd_const::kSqrt2Pi;
    const DDouble d1 = sq - two_pi;
    EXPECT_LT(std::abs(d1.hi() + d1.lo()) / two_pi.hi(), kDdEps);

    const DDouble unity = dd_const::kInvSqrt2Pi * dd_const::kSqrt2Pi;
    const DDouble d2 = unity - DDouble(1.0);
    EXPECT_LT(std::abs(d2.hi() + d2.lo()), kDdEps);

    const DDouble half = dd_const::kSqrtHalf * dd_const::kSqrtHalf;
    const DDouble d3 = half - DDouble(0.5);
    EXPECT_LT(std::abs(d3.hi() + d3.lo()), kDdEps);
}

// ===========================================================================
// The reference special functions built on top
// ===========================================================================

TEST(DdReal, ReferenceErfcMatchesIndependentValues) {
    struct Row { double z, truth; };
    const Row rows[] = {
        {3.0, 2.2090496998585441373e-5},
        {6.35, 2.7013718824626958331e-19},
        {8.98, 5.9404004347274542514e-37},
        {26.0, 5.6631924088561428465e-296},
    };
    for (const auto& r : rows) {
        const double got = reference::erfc_dd(DDouble(r.z)).to_double();
        // The reference must be good to well under a double ulp, since
        // everything else is validated against it.
        EXPECT_LT(vl::math::rel_error(got, r.truth), 1e-15)
            << "erfc_dd(" << r.z << ") = " << got << ", expected " << r.truth;
    }
}

TEST(DdReal, ReferenceNormCdfMatchesIndependentValues) {
    struct Row { double x, truth; };
    const Row rows[] = {
        {-36.83, 3.0566987395559564295e-297},
        {-30.0, 4.9067139271481870595e-198},
        {-10.0, 7.619853024160526066e-24},
        {-1.0, 0.15865525393145705141},
        {0.0, 0.5},
        {1.0, 0.84134474606854294859},
    };
    for (const auto& r : rows) {
        EXPECT_LT(vl::math::rel_error(reference::norm_cdf_ref(r.x), r.truth), 1e-15)
            << "x = " << r.x;
    }
}

TEST(DdReal, ReferenceErfcxIsFiniteBeyondExpSqOverflow) {
    // The reference must be at least as robust as the implementation it
    // validates.  An earlier version computed exp(x^2)*erfc(x) and returned
    // inf*0 = NaN for x >= 26.65, which would have silently excluded the
    // entire far tail from validation.
    for (double x : {26.0, 27.0, 30.0, 50.0, 200.0}) {
        const double v = reference::erfcx_ref(x);
        EXPECT_TRUE(std::isfinite(v)) << "erfcx_ref(" << x << ") = " << v;
        EXPECT_GT(v, 0.0) << "x = " << x;
    }
    EXPECT_LT(vl::math::rel_error(reference::erfcx_ref(30.0), 0.018795888861416751497),
              1e-15);
    EXPECT_LT(vl::math::rel_error(reference::erfcx_ref(100.0), 0.0056416137829894329036),
              1e-15);
}
