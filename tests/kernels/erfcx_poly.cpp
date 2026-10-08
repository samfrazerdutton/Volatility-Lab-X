// SPDX-License-Identifier: MIT
/// Validates the scalar-optimised erfcx kernel (`kernels/scalar/erfcx_poly.hpp`)
/// against the double-double reference, per the directive's numerical
/// contract (section 9): max absolute error, max relative error, max ULP,
/// and explicit difficult-case coverage, not just friendly inputs. This is
/// the REFERENCE -> SCALAR step of the validation ladder -- the AVX2
/// kernel (once it exists) is checked against *this*, in addition to its
/// own direct reference comparison.

#include "vl_test_support.hpp"

#include "scalar/erfcx_poly.hpp"

#include "volatility_lab/math/reference_special.hpp"
#include "volatility_lab/math/special.hpp"

#include <cmath>
#include <limits>

using namespace vl;
using vl::math::error_in_ulps;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

/// The full accuracy sweep, reused by every test below so the error-report
/// numbers printed in failure messages are always computed the same way.
WorstCase sweep_against_reference() {
    WorstCase worst;
    for (double x : test::lin_space(-10.0, 10.0, 20001)) {
        const double ref = math::reference::erfcx_ref(x);
        const double approx = kernels::scalar::erfcx_poly(x);
        if (!std::isfinite(ref) || !std::isfinite(approx)) {
            worst.skip();
            continue;
        }
        worst.observe(rel_error(approx, ref), x, 0.0, approx, ref);
    }
    return worst;
}

}  // namespace

TEST(ErfcxPoly, MatchesTheDoubleDoubleReferenceWithinItsPublishedBudget) {
    const WorstCase worst = sweep_against_reference();
    EXPECT_GT(worst.samples, 0);
    EXPECT_CLOSE_TOL(worst.value, worst.reference, math::tol::kErfcxPoly)
        << worst.describe("x", "(unused)");
}

TEST(ErfcxPoly, ErrorReport) {
    // Not an assertion -- a printed report, exactly what the directive's
    // section 9 asks for (max abs, max relative, ULP, worst case), so a
    // human reviewing this kernel's accuracy does not have to re-derive it
    // from the pass/fail result alone.
    double max_abs = 0.0, max_rel = 0.0, max_ulp_err = 0.0;
    double worst_rel_x = 0.0, worst_ulp_x = 0.0;
    long n = 0, skipped = 0;
    for (double x : test::lin_space(-10.0, 10.0, 20001)) {
        const double ref = math::reference::erfcx_ref(x);
        const double approx = kernels::scalar::erfcx_poly(x);
        if (!std::isfinite(ref) || !std::isfinite(approx)) {
            ++skipped;
            continue;
        }
        ++n;
        const double a = std::abs(approx - ref);
        const double r = rel_error(approx, ref);
        const double u = error_in_ulps(approx, ref);
        if (a > max_abs) max_abs = a;
        if (r > max_rel) { max_rel = r; worst_rel_x = x; }
        if (u > max_ulp_err) { max_ulp_err = u; worst_ulp_x = x; }
    }
    std::printf(
        "erfcx_poly vs double-double reference, x in [-10, 10], n=%ld (%ld skipped):\n"
        "  max abs error:      %.6e\n"
        "  max relative error: %.6e  (at x=%.6f)\n"
        "  max error in ulps:  %.6e  (at x=%.6f)\n",
        n, skipped, max_abs, max_rel, worst_rel_x, max_ulp_err, worst_ulp_x);
    EXPECT_GT(n, 0);
    // Not re-asserted here (the budget test above already does) -- this
    // test's purpose is the printed report.
    SUCCEED();
}

// ---------------------------------------------------------------------------
// Difficult cases, named explicitly rather than left to the sweep alone
// ---------------------------------------------------------------------------

TEST(ErfcxPoly, AtZeroMatchesTheKnownValueOfOne) {
    EXPECT_CLOSE_TOL(kernels::scalar::erfcx_poly(0.0), 1.0, math::tol::kErfcxPoly);
}

TEST(ErfcxPoly, NearZeroOnEitherSideIsContinuous) {
    const double left = kernels::scalar::erfcx_poly(-1e-8);
    const double right = kernels::scalar::erfcx_poly(1e-8);
    EXPECT_NEAR(left, right, 1e-6);
}

TEST(ErfcxPoly, LargePositiveXMatchesTheAsymptoticDecay) {
    // erfcx(x) ~ 1/(x*sqrt(pi)) for large x -- the CF switch region in the
    // production erfcx, exercised here with no domain switch at all (see
    // the header for why none is needed).
    for (double x : {15.0, 26.0, 50.0, 100.0, 1000.0}) {
        const double ref = math::reference::erfcx_ref(x);
        const double approx = kernels::scalar::erfcx_poly(x);
        ASSERT_TRUE(std::isfinite(ref)) << "x=" << x;
        EXPECT_CLOSE_TOL(approx, ref, math::tol::kErfcxPoly) << "x=" << x;
    }
}

TEST(ErfcxPoly, NegativeXPastTheProductionOverflowGuardReturnsInfinity) {
    // math::exp_sq's own guard, kExpSqMax ~= 26.6416 -- beyond that,
    // erfcx(x) for negative x is not finite-representable via the
    // reflection formula either (it needs exp(x^2)), and the kernel must
    // say so explicitly rather than return a silently wrong finite number.
    EXPECT_TRUE(std::isinf(kernels::scalar::erfcx_poly(-30.0)));
    EXPECT_GT(kernels::scalar::erfcx_poly(-30.0), 0.0);
}

TEST(ErfcxPoly, NegativeXWellWithinTheGuardMatchesTheReference) {
    for (double x : {-1.0, -5.0, -10.0, -20.0, -26.0}) {
        const double ref = math::reference::erfcx_ref(x);
        const double approx = kernels::scalar::erfcx_poly(x);
        ASSERT_TRUE(std::isfinite(ref)) << "x=" << x;
        EXPECT_CLOSE_TOL(approx, ref, math::tol::kErfcxPoly) << "x=" << x;
    }
}

TEST(ErfcxPoly, NanInputPropagatesNanRatherThanBeingSwallowed) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(kernels::scalar::erfcx_poly(nan)));
}

// ---------------------------------------------------------------------------
// Cross-check against the existing production kernel
// ---------------------------------------------------------------------------

TEST(ErfcxPoly, AgreesWithProductionMathErfcxToWithinItsOwnBudget) {
    // math::erfcx is already validated to near machine precision against
    // the same reference -- so this is really the same budget as the
    // reference comparison above, just against a cheaper-to-call oracle,
    // confirming the two independent validations (dd reference, production
    // erfcx) do not disagree about what this kernel should return.
    WorstCase worst;
    for (double x : test::lin_space(-10.0, 10.0, 20001)) {
        const double prod = math::erfcx(x);
        const double approx = kernels::scalar::erfcx_poly(x);
        if (!std::isfinite(prod) || !std::isfinite(approx)) {
            worst.skip();
            continue;
        }
        worst.observe(rel_error(approx, prod), x, 0.0, approx, prod);
    }
    EXPECT_GT(worst.samples, 0);
    EXPECT_CLOSE_TOL(worst.value, worst.reference, math::tol::kErfcxPoly)
        << worst.describe("x", "(unused)");
}
