// SPDX-License-Identifier: MIT
/// Validates the SVI and SSVI slice models.
///
/// The centrepiece is `SsviGatheralJacquierConditionsImplyPositiveDensity`:
/// SSVI's entire reason for existing is that two closed-form inequalities on
/// three global parameters are supposed to *guarantee* a non-negative implied
/// density at every strike and maturity.  That is a theorem, and a theorem the
/// library leans on heavily -- so it is checked directly, by fuzzing the
/// parameter space and confirming that admissible parameters never produce a
/// negative Durrleman function and that inadmissible ones sometimes do.  If
/// the conditions were transcribed wrongly, that test fails and nothing else
/// about the surface can be trusted.

#include "vl_test_support.hpp"

#include "volatility_lab/volatility/slice.hpp"
#include "volatility_lab/volatility/ssvi.hpp"
#include "volatility_lab/volatility/svi.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace vl;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

/// A realistic equity-index smile: downward skew, moderate curvature.
SviParams reference_svi() {
    SviParams p;
    p.a = 0.012;
    p.b = 0.085;
    p.rho = -0.42;
    p.m = 0.018;
    p.sigma = 0.14;
    p.years = 0.5;
    return p;
}

}  // namespace

// ===========================================================================
// SVI: evaluation and derivatives
// ===========================================================================

TEST(Svi, DerivativesMatchFiniteDifferences) {
    const auto p = reference_svi();
    WorstCase w1;
    WorstCase w2;
    const double h = 1e-6;
    for (double k = -1.5; k <= 1.5; k += 0.003) {
        const SliceJet j = svi_jet(p, k);
        const double fd1 =
            (svi_total_variance(p, k + h) - svi_total_variance(p, k - h)) / (2.0 * h);
        const double fd2 = (svi_total_variance(p, k + h) - 2.0 * svi_total_variance(p, k) +
                            svi_total_variance(p, k - h)) /
                           (h * h);
        w1.observe(std::abs(j.dw - fd1), k, 0, j.dw, fd1);
        w2.observe(std::abs(j.d2w - fd2), k, 0, j.d2w, fd2);
    }
    EXPECT_LT(w1.error, 1e-9) << w1.describe("k");
    // Second differences lose half the digits to the h^2 divisor; the bound is
    // the truncation limit of the difference, not of the analytic value.
    EXPECT_LT(w2.error, 1e-3) << w2.describe("k");
}

TEST(Svi, JetAgreesWithTheScalarEvaluator) {
    const auto p = reference_svi();
    for (double k = -2.0; k <= 2.0; k += 0.013) {
        EXPECT_BITWISE_EQ(svi_jet(p, k).w, svi_total_variance(p, k));
    }
}

TEST(Svi, BatchAgreesWithScalarExactly) {
    const auto p = reference_svi();
    std::vector<double> ks;
    for (double k = -2.0; k <= 2.0; k += 0.001) ks.push_back(k);
    std::vector<double> ws(ks.size());
    svi_total_variance_batch(p, ks, ws);
    for (std::size_t i = 0; i < ks.size(); ++i) {
        EXPECT_BITWISE_EQ(ws[i], svi_total_variance(p, ks[i])) << "i = " << i;
    }
}

TEST(Svi, BatchHandlesMismatchedSpanSizes) {
    const auto p = reference_svi();
    std::vector<double> ks{0.0, 0.1, 0.2};
    std::vector<double> ws(2, -1.0);
    svi_total_variance_batch(p, ks, ws);  // must write only 2
    EXPECT_NE(ws[0], -1.0);
    EXPECT_NE(ws[1], -1.0);
    std::vector<double> ks2{0.0};
    std::vector<double> ws2(3, -1.0);
    svi_total_variance_batch(p, ks2, ws2);
    EXPECT_NE(ws2[0], -1.0);
    EXPECT_EQ(ws2[1], -1.0);
}

// ===========================================================================
// SVI: structural properties
// ===========================================================================

TEST(Svi, IsAlwaysConvexInLogMoneyness) {
    // w'' = b*sigma^2/r^3 > 0 whenever b > 0 and sigma > 0.  Convexity is
    // necessary for the absence of butterfly arbitrage, and it is the property
    // that a polynomial fit cannot be given.
    std::mt19937_64 rng(5551212u);
    std::uniform_real_distribution<double> ua(-0.05, 0.2);
    std::uniform_real_distribution<double> ub(1e-4, 0.6);
    std::uniform_real_distribution<double> ur(-0.95, 0.95);
    std::uniform_real_distribution<double> um(-0.5, 0.5);
    std::uniform_real_distribution<double> us(1e-3, 1.0);
    for (int t = 0; t < 400; ++t) {
        SviParams p{ua(rng), ub(rng), ur(rng), um(rng), us(rng), 1.0};
        for (double k = -3.0; k <= 3.0; k += 0.05) {
            ASSERT_GT(svi_jet(p, k).d2w, 0.0) << "trial " << t << ", k = " << k;
        }
    }
}

TEST(Svi, MinimumVarianceFormulaIsExact) {
    // The constraint a + b*sigma*sqrt(1-rho^2) >= 0 is enforced as a hard
    // bound, so it had better be the true minimum rather than a conservative
    // proxy.  Checked against a dense search.
    std::mt19937_64 rng(777u);
    std::uniform_real_distribution<double> ua(-0.02, 0.2);
    std::uniform_real_distribution<double> ub(1e-3, 0.5);
    std::uniform_real_distribution<double> ur(-0.95, 0.95);
    std::uniform_real_distribution<double> um(-0.4, 0.4);
    std::uniform_real_distribution<double> us(1e-2, 0.8);

    for (int t = 0; t < 200; ++t) {
        const SviParams p{ua(rng), ub(rng), ur(rng), um(rng), us(rng), 1.0};
        const double claimed = svi_min_variance(p);
        const double argmin = svi_argmin(p);

        // The claimed minimiser must actually attain the claimed minimum.
        EXPECT_NEAR(svi_total_variance(p, argmin), claimed,
                    1e-12 * std::max(1.0, std::abs(claimed)))
            << "trial " << t;

        // And nothing on a dense sweep may beat it.
        double searched = std::numeric_limits<double>::infinity();
        for (double k = argmin - 5.0; k <= argmin + 5.0; k += 0.002) {
            searched = std::min(searched, svi_total_variance(p, k));
        }
        EXPECT_GE(searched, claimed - 1e-10) << "trial " << t << ": found " << searched
                                             << " below the claimed minimum " << claimed;
    }
}

TEST(Svi, WingSlopesAreTheAsymptoticDerivatives) {
    const auto p = reference_svi();
    const SviWings wings = svi_wings(p);
    EXPECT_NEAR(svi_jet(p, -1e6).dw, wings.left, 1e-9);
    EXPECT_NEAR(svi_jet(p, 1e6).dw, wings.right, 1e-9);
    // Wings open outward: left slope negative, right positive.
    EXPECT_LT(wings.left, 0.0);
    EXPECT_GT(wings.right, 0.0);
}

TEST(Svi, TotalVarianceGrowsAtMostLinearlyInTheWings) {
    // Lee's moment formula: w must be O(|k|), never O(k^2).  This is the
    // property that makes SVI safe to extrapolate and a polynomial fit unsafe.
    const auto p = reference_svi();
    for (double k : {10.0, 100.0, 1000.0, 1e6}) {
        const double ratio = svi_total_variance(p, k) / k;
        EXPECT_LT(ratio, 1.0) << "k = " << k;
        EXPECT_GT(ratio, 0.0) << "k = " << k;
    }
}

TEST(Svi, AdmissibilityAndProjection) {
    EXPECT_TRUE(svi_parameters_admissible(reference_svi()));

    // Each structural violation must be caught.
    SviParams p = reference_svi();
    p.b = -0.1;
    EXPECT_FALSE(svi_parameters_admissible(p));
    p = reference_svi();
    p.sigma = 0.0;
    EXPECT_FALSE(svi_parameters_admissible(p));
    p = reference_svi();
    p.rho = 1.0;
    EXPECT_FALSE(svi_parameters_admissible(p));
    p = reference_svi();
    p.a = -10.0;  // drives the minimum negative
    EXPECT_FALSE(svi_parameters_admissible(p));
    p = reference_svi();
    p.m = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(svi_parameters_admissible(p));

    // Projection must produce something admissible from anything.
    std::mt19937_64 rng(31337u);
    std::uniform_real_distribution<double> wild(-5.0, 5.0);
    for (int t = 0; t < 500; ++t) {
        SviParams bad{wild(rng), wild(rng), wild(rng), wild(rng), wild(rng), 1.0};
        if (t % 17 == 0) bad.a = std::numeric_limits<double>::quiet_NaN();
        if (t % 23 == 0) bad.sigma = std::numeric_limits<double>::infinity();
        const SviParams fixed = svi_project_to_admissible(bad);
        ASSERT_TRUE(svi_parameters_admissible(fixed))
            << "trial " << t << ": projection produced an inadmissible slice";
        // And w >= 0 everywhere, which is what the projection is for.
        for (double k = -5.0; k <= 5.0; k += 0.1) {
            ASSERT_GE(svi_total_variance(fixed, k), -1e-12) << "trial " << t;
        }
    }
}

TEST(Svi, ProjectionRaisesTheLevelRatherThanLoweringTheWings) {
    // The documented choice: `a` is the weakly identified parameter, so moving
    // it does least damage to the fit, and raising it cannot introduce a new
    // violation since dw_min/da = +1.
    SviParams p = reference_svi();
    const double b_before = p.b;
    const double sigma_before = p.sigma;
    p.a = -1.0;  // violates positivity
    ASSERT_FALSE(svi_parameters_admissible(p));
    const SviParams fixed = svi_project_to_admissible(p);
    EXPECT_EQ(fixed.b, b_before) << "b must not be touched";
    EXPECT_EQ(fixed.sigma, sigma_before) << "sigma must not be touched";
    EXPECT_GT(fixed.a, p.a);
    EXPECT_NEAR(svi_min_variance(fixed), 0.0, 1e-15);
}

// ===========================================================================
// SVI: the Durrleman condition
// ===========================================================================

TEST(Svi, ButterflyCheckFindsAKnownBadSlice) {
    // A slice with a steep wing and too little curvature has negative density.
    // Constructed by pushing b up until the check complains, then asserting
    // both that it complains and that the density really is negative there --
    // the check must agree with the quantity it claims to be detecting.
    SviParams bad = reference_svi();
    bad.b = 1.2;
    bad.sigma = 0.02;
    bad.rho = -0.9;
    const auto chk = svi_butterfly_check(bad, -1.0, 1.0);
    ASSERT_FALSE(chk.arbitrage_free) << "expected this slice to be arbitrageable";
    EXPECT_LT(chk.worst_g, 0.0);
    EXPECT_LT(chk.worst_density, 0.0) << "a negative g must mean a negative density";
    // The reported location must be the location.
    const SliceJet j = svi_jet(bad, chk.worst_k);
    EXPECT_NEAR(durrleman_g(chk.worst_k, j), chk.worst_g, 1e-12);
}

TEST(Svi, ButterflyCheckPassesAWellBehavedSlice) {
    const auto chk = svi_butterfly_check(reference_svi(), -1.5, 1.5);
    EXPECT_TRUE(chk.arbitrage_free) << "worst g = " << chk.worst_g << " at k = "
                                    << chk.worst_k;
    EXPECT_GT(chk.worst_g, 0.0);
    EXPECT_GT(chk.worst_density, 0.0);
}

TEST(Svi, ButterflyCheckGridIsDenseEnough) {
    // The default grid is a judgement call, so it is checked against a 100x
    // denser one: the coarse grid must not declare a slice clean that the fine
    // grid rejects.
    std::mt19937_64 rng(90210u);
    std::uniform_real_distribution<double> ua(-0.01, 0.1);
    std::uniform_real_distribution<double> ub(0.01, 1.5);
    std::uniform_real_distribution<double> ur(-0.95, 0.95);
    std::uniform_real_distribution<double> um(-0.3, 0.3);
    std::uniform_real_distribution<double> us(0.01, 0.5);

    int disagreements = 0;
    int bad_slices = 0;
    for (int t = 0; t < 400; ++t) {
        SviParams p = svi_project_to_admissible(
            SviParams{ua(rng), ub(rng), ur(rng), um(rng), us(rng), 1.0});
        const auto coarse = svi_butterfly_check(p, -1.0, 1.0, kDefaultGridPoints);
        const auto fine = svi_butterfly_check(p, -1.0, 1.0, kDefaultGridPoints * 100);
        if (!fine.arbitrage_free) ++bad_slices;
        if (coarse.arbitrage_free && !fine.arbitrage_free) ++disagreements;
    }
    EXPECT_GT(bad_slices, 10) << "the fuzz range must actually produce bad slices, or "
                                 "this test proves nothing";
    EXPECT_EQ(disagreements, 0)
        << disagreements << " slices passed the default grid but failed a 100x finer one";
}

TEST(Svi, DurrlemanIsMinusInfinityAtNonPositiveVariance) {
    SliceJet j;
    j.w = 0.0;
    EXPECT_TRUE(std::isinf(durrleman_g(0.0, j)));
    EXPECT_LT(durrleman_g(0.0, j), 0.0);
    EXPECT_EQ(implied_density(0.0, j), 0.0);
}

TEST(Svi, SliceInspectionReportsEveryDefectItFinds) {
    // The bitmask must accumulate rather than report only the first problem: a
    // badly fitted wing typically violates several conditions at once.
    SliceJet nonfinite;
    nonfinite.w = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(has(inspect_slice_point(0.0, nonfinite), SliceDefect::NonFinite));

    SliceJet negative;
    negative.w = -1.0;
    EXPECT_TRUE(has(inspect_slice_point(0.0, negative), SliceDefect::NonPositiveVariance));

    // Steep wing beyond the Lee bound, far enough out for the bound to apply.
    SliceJet steep;
    steep.w = 0.1;
    steep.dw = 3.0;
    steep.d2w = 0.0;
    const auto defects = inspect_slice_point(1.0, steep);
    EXPECT_TRUE(has(defects, SliceDefect::SlopeOutOfBounds));
    // The same slope near the money must not trip it: the bound is asymptotic.
    EXPECT_FALSE(has(inspect_slice_point(0.01, steep), SliceDefect::SlopeOutOfBounds));

    SliceJet good;
    good.w = 0.04;
    good.dw = 0.0;
    good.d2w = 0.1;
    EXPECT_EQ(inspect_slice_point(0.0, good), SliceDefect::None);
}

// ===========================================================================
// SVI: the quasi-explicit reduction
// ===========================================================================

TEST(Svi, ReducedCoordinatesRoundTrip) {
    // The reduction is the foundation of the calibrator, so the mapping has to
    // be exactly invertible or the inner linear solve optimises the wrong
    // thing.
    std::mt19937_64 rng(246810u);
    std::uniform_real_distribution<double> ua(0.0, 0.2);
    std::uniform_real_distribution<double> ub(1e-3, 0.6);
    std::uniform_real_distribution<double> ur(-0.95, 0.95);
    std::uniform_real_distribution<double> um(-0.4, 0.4);
    std::uniform_real_distribution<double> us(1e-2, 0.8);

    for (int t = 0; t < 500; ++t) {
        const SviParams p{ua(rng), ub(rng), ur(rng), um(rng), us(rng), 0.75};
        const SviReduced r = svi_to_reduced(p);
        const SviParams back = svi_from_reduced(r, p.m, p.sigma, p.years);
        EXPECT_NEAR(back.a, p.a, 1e-14) << "trial " << t;
        EXPECT_NEAR(back.b, p.b, 1e-13 * std::max(1.0, p.b)) << "trial " << t;
        EXPECT_NEAR(back.rho, p.rho, 1e-13) << "trial " << t;
        EXPECT_EQ(back.m, p.m);
        EXPECT_EQ(back.sigma, p.sigma);
        EXPECT_EQ(back.years, p.years);
    }
}

TEST(Svi, ReducedFormEvaluatesToTheSameVariance) {
    // The whole reduction rests on w being linear in (adash, d, c) at fixed
    // (m, sigma).  Verified by evaluating both forms at the same point.
    const auto p = reference_svi();
    const SviReduced r = svi_to_reduced(p);
    WorstCase w;
    for (double k = -2.0; k <= 2.0; k += 0.001) {
        const double z = (k - p.m) / p.sigma;
        const double direct = svi_total_variance(p, k);
        const double reduced = svi_reduced_variance(r, z);
        w.observe(rel_error(reduced, direct), k, 0, reduced, direct);
    }
    EXPECT_LT(w.error, 1e-14) << w.describe("k");
}

TEST(Svi, ReducedFormIsLinearInItsCoefficients) {
    // The property the inner solve exploits: doubling the coefficients doubles
    // the variance, and a sum of coefficient vectors gives the sum of
    // variances.  If this failed, a linear least squares would be the wrong
    // inner problem.
    const SviReduced r1{0.01, -0.02, 0.05};
    const SviReduced r2{0.03, 0.01, 0.02};
    const SviReduced sum{r1.adash + r2.adash, r1.d + r2.d, r1.c + r2.c};
    for (double z = -5.0; z <= 5.0; z += 0.01) {
        EXPECT_NEAR(svi_reduced_variance(sum, z),
                    svi_reduced_variance(r1, z) + svi_reduced_variance(r2, z), 1e-14)
            << "z = " << z;
    }
}

// ===========================================================================
// SSVI
// ===========================================================================

TEST(Ssvi, SliceParamsMappingIsExact) {
    // SSVI at a fixed theta *is* a raw SVI slice.  That identity is what lets
    // every downstream consumer -- pricer, batch kernels, Greeks, Durrleman
    // check -- be shared between the two models, so it has to hold to machine
    // precision rather than approximately.
    std::mt19937_64 rng(13579u);
    std::uniform_real_distribution<double> urho(-0.9, 0.9);
    std::uniform_real_distribution<double> ueta(0.05, 3.0);
    std::uniform_real_distribution<double> ugam(0.05, 0.95);
    std::uniform_real_distribution<double> utheta(1e-4, 1.5);

    WorstCase w;
    for (int t = 0; t < 300; ++t) {
        SsviParams g;
        g.rho = urho(rng);
        g.eta = ueta(rng);
        g.gamma = ugam(rng);
        g.phi_kind = (t % 2 == 0) ? SsviPhiKind::PowerLaw : SsviPhiKind::Heston;
        g.lambda = ueta(rng);
        const double theta = utheta(rng);
        const SviParams sp = ssvi_slice_params(g, theta, 1.0);

        for (double k = -1.5; k <= 1.5; k += 0.05) {
            const double direct = ssvi_total_variance(g, theta, k);
            const double via_svi = svi_total_variance(sp, k);
            w.observe(rel_error(via_svi, direct), theta, k, via_svi, direct);
        }
    }
    EXPECT_LT(w.error, 1e-13) << w.describe("theta", "k");
}

TEST(Ssvi, SliceJetMatchesTheSviJetOfTheMappedSlice) {
    SsviParams g;
    g.rho = -0.5;
    g.eta = 1.2;
    g.gamma = 0.4;
    const double theta = 0.05;
    const SviParams sp = ssvi_slice_params(g, theta, 1.0);
    WorstCase w;
    for (double k = -1.2; k <= 1.2; k += 0.003) {
        const SliceJet a = ssvi_jet(g, theta, k);
        const SliceJet b = svi_jet(sp, k);
        w.observe(rel_error(a.w, b.w), k, 0, a.w, b.w);
        w.observe(rel_error(a.dw, b.dw), k, 1, a.dw, b.dw);
        w.observe(rel_error(a.d2w, b.d2w), k, 2, a.d2w, b.d2w);
    }
    EXPECT_LT(w.error, 1e-12) << w.describe("k", "which");
}

TEST(Ssvi, AtmVarianceIsExactlyTheta) {
    // theta is defined as w(0, T), so the model must reproduce it at k = 0 or
    // the term structure the calibrator reads off the ATM quotes is not the
    // parameter the model uses.
    SsviParams g;
    g.rho = -0.35;
    g.eta = 0.9;
    g.gamma = 0.45;
    for (double theta : {1e-4, 0.001, 0.01, 0.04, 0.3, 1.2}) {
        EXPECT_LT(rel_error(ssvi_total_variance(g, theta, 0.0), theta), 1e-14)
            << "theta = " << theta;
    }
}

TEST(Ssvi, PhiDerivativeMatchesFiniteDifference) {
    // The calendar condition is stated in terms of d(theta*phi)/d(theta), so a
    // wrong dphi/dtheta would silently mis-classify calendar arbitrage.
    for (auto kind : {SsviPhiKind::PowerLaw, SsviPhiKind::Heston}) {
        SsviParams g;
        g.phi_kind = kind;
        g.eta = 1.3;
        g.gamma = 0.37;
        g.lambda = 2.1;
        WorstCase w;
        for (double theta = 1e-3; theta < 2.0; theta *= 1.05) {
            // The step is a relative 1e-5, not 1e-6.  phi varies on the scale
            // of theta itself, so the central-difference round-off term is
            // eps*|phi|/(h*|dphi|); at h = 1e-6*theta that is ~3e-7 even with a
            // perfect phi, and it amplifies any error in phi by ~1e9.  An
            // earlier version used 1e-6 and was measuring the difference rather
            // than the derivative.
            const double h = theta * 1e-5;
            const double fd =
                (ssvi_phi(g, theta + h).phi - ssvi_phi(g, theta - h).phi) / (2.0 * h);
            const double an = ssvi_phi(g, theta).dphi_dtheta;
            w.observe(rel_error(an, fd), theta, 0, an, fd);
        }
        EXPECT_LT(w.error, 1e-6) << to_string(kind) << ": " << w.describe("theta");
    }
}

TEST(Ssvi, HestonPhiIsAccurateForSmallTheta) {
    // The Heston phi contains 1 - (1-exp(-u))/u, which cancels to u/2 for
    // small u.  Short-dated slices are exactly where theta is small, so the
    // series expansion is not optional.  Compared against the limit phi -> 1/2
    // as theta -> 0.
    SsviParams g;
    g.phi_kind = SsviPhiKind::Heston;
    g.lambda = 1.0;
    for (double theta : {1e-10, 1e-8, 1e-6, 1e-4}) {
        const double phi = ssvi_phi(g, theta).phi;
        EXPECT_TRUE(std::isfinite(phi)) << "theta = " << theta;
        EXPECT_NEAR(phi, 0.5, 1e-3) << "theta = " << theta
                                    << ": the small-u expansion is wrong";
    }
    // And it is continuous across the switch at u = 1e-3.
    const double below = ssvi_phi(g, 0.999e-3).phi;
    const double above = ssvi_phi(g, 1.001e-3).phi;
    EXPECT_LT(rel_error(below, above), 1e-6);
}

// ---------------------------------------------------------------------------
// The theorem
// ---------------------------------------------------------------------------

TEST(Ssvi, GatheralJacquierConditionsImplyPositiveDensity) {
    // **The test that earns SSVI its place in the library.**
    //
    // The claim is a theorem: if
    //     theta*phi*(1+|rho|) < 4   and   theta*phi^2*(1+|rho|) <= 4
    // then the slice has no butterfly arbitrage -- that is, the Durrleman
    // function g(k) is non-negative at *every* k, not merely on the quoted
    // range.  The whole value proposition of the model is that this can be
    // checked in closed form instead of searched for.
    //
    // So: fuzz the parameter space, and for every parameter set that the
    // conditions accept, verify directly on a dense grid that g >= 0.  A
    // transcription error in either inequality shows up here immediately, and
    // nothing else about the surface would be trustworthy if it did not.
    std::mt19937_64 rng(1140712u);
    std::uniform_real_distribution<double> urho(-0.97, 0.97);
    std::uniform_real_distribution<double> ueta(0.01, 6.0);
    std::uniform_real_distribution<double> ugam(0.0, 1.0);
    std::uniform_real_distribution<double> ulam(0.05, 8.0);
    std::uniform_real_distribution<double> utheta(1e-5, 2.0);

    long admissible = 0;
    long inadmissible = 0;
    long violations_among_admissible = 0;
    WorstCase worst_admissible_g;
    worst_admissible_g.error = -1e308;  // we track the *minimum* g, see below

    double min_g_admissible = std::numeric_limits<double>::infinity();
    double min_g_at_rho = 0.0;
    double min_g_at_theta = 0.0;

    for (int t = 0; t < 4000; ++t) {
        SsviParams g;
        g.rho = urho(rng);
        g.eta = ueta(rng);
        g.gamma = ugam(rng);
        g.lambda = ulam(rng);
        g.phi_kind = (t % 3 == 0) ? SsviPhiKind::Heston : SsviPhiKind::PowerLaw;
        const double theta = utheta(rng);

        const auto chk = ssvi_arbitrage_check(g, theta);
        if (!chk.butterfly_free) {
            ++inadmissible;
            continue;
        }
        ++admissible;

        // Dense grid, deliberately far wider than any quoted strike range:
        // the theorem is a statement about all k, so a test restricted to
        // [-1, 1] would not test it.
        for (double k = -8.0; k <= 8.0; k += 0.01) {
            const SliceJet j = ssvi_jet(g, theta, k);
            if (!(j.w > 0.0)) continue;
            const double gv = durrleman_g(k, j);
            if (gv < min_g_admissible) {
                min_g_admissible = gv;
                min_g_at_rho = g.rho;
                min_g_at_theta = theta;
            }
            if (gv < -1e-10) {
                ++violations_among_admissible;
                ADD_FAILURE() << "GJ conditions accepted parameters with negative "
                                 "density: rho=" << g.rho << " eta=" << g.eta
                              << " gamma=" << g.gamma << " lambda=" << g.lambda
                              << " kind=" << to_string(g.phi_kind) << " theta=" << theta
                              << " k=" << k << " g=" << gv;
                break;
            }
        }
        if (violations_among_admissible > 3) break;  // enough evidence
    }

    EXPECT_GT(admissible, 500) << "the fuzz range must actually produce admissible "
                                  "parameter sets";
    EXPECT_GT(inadmissible, 100) << "the fuzz range must also exercise the rejection "
                                    "path, or the conditions are not binding";
    EXPECT_EQ(violations_among_admissible, 0);
    EXPECT_GE(min_g_admissible, -1e-10)
        << "worst g over all admissible parameters was " << min_g_admissible
        << " at rho=" << min_g_at_rho << ", theta=" << min_g_at_theta;
}

TEST(Ssvi, ConditionsActuallyRejectArbitrageableParameters) {
    // The converse direction, which stops the previous test from being
    // satisfied vacuously by a check that accepts nothing: parameters chosen
    // to be beyond the butterfly bound must be rejected *and* must really
    // produce a negative density.
    SsviParams g;
    g.rho = -0.9;
    g.eta = 5.0;
    g.gamma = 0.1;
    g.phi_kind = SsviPhiKind::PowerLaw;
    const double theta = 1.0;

    const auto chk = ssvi_arbitrage_check(g, theta);
    ASSERT_FALSE(chk.butterfly_free)
        << "slack1 = " << chk.butterfly_slack_1 << ", slack2 = " << chk.butterfly_slack_2;
    EXPECT_GT(chk.violation(), 0.0) << "a violated check must report a positive penalty";

    bool found_negative = false;
    for (double k = -5.0; k <= 5.0; k += 0.002) {
        const SliceJet j = ssvi_jet(g, theta, k);
        if (j.w > 0.0 && durrleman_g(k, j) < 0.0) {
            found_negative = true;
            break;
        }
    }
    EXPECT_TRUE(found_negative)
        << "the conditions rejected parameters that are in fact arbitrage-free, which "
           "means they are more conservative than documented";
}

TEST(Ssvi, ViolationIsZeroWhenAdmissibleAndPositiveOtherwise) {
    // The penalty must be exactly zero inside the admissible set, or a
    // constrained calibration is biased away from the boundary even when
    // nothing is wrong.
    SsviParams g;
    g.rho = -0.3;
    g.eta = 0.5;
    g.gamma = 0.5;
    const auto ok = ssvi_arbitrage_check(g, 0.04);
    ASSERT_TRUE(ok.admissible());
    EXPECT_EQ(ok.violation(), 0.0);

    SsviParams bad = g;
    bad.eta = 20.0;
    const auto nope = ssvi_arbitrage_check(bad, 1.0);
    EXPECT_GT(nope.violation(), 0.0);
}

TEST(Ssvi, TermCheckCatchesNonMonotoneTheta) {
    // The first calendar condition, d(theta)/dT >= 0, is a property of the
    // data rather than of the parameters, and a caller handing over a
    // non-monotone term structure must be told.
    SsviParams g;
    g.rho = -0.3;
    g.eta = 0.5;
    g.gamma = 0.5;
    const std::vector<double> good{0.01, 0.02, 0.05, 0.1};
    const std::vector<double> bad{0.01, 0.05, 0.03, 0.1};
    EXPECT_TRUE(ssvi_arbitrage_check_term(g, good).calendar_free);
    EXPECT_FALSE(ssvi_arbitrage_check_term(g, bad).calendar_free);
}

TEST(Ssvi, AdmissibilityAndProjection) {
    EXPECT_TRUE(ssvi_parameters_admissible(SsviParams{}));
    SsviParams p;
    p.rho = 1.5;
    EXPECT_FALSE(ssvi_parameters_admissible(p));
    p = SsviParams{};
    p.eta = -1.0;
    EXPECT_FALSE(ssvi_parameters_admissible(p));
    p = SsviParams{};
    p.gamma = 2.0;
    EXPECT_FALSE(ssvi_parameters_admissible(p));
    p = SsviParams{};
    p.phi_kind = SsviPhiKind::Heston;
    p.lambda = 0.0;
    EXPECT_FALSE(ssvi_parameters_admissible(p));

    std::mt19937_64 rng(8642u);
    std::uniform_real_distribution<double> wild(-10.0, 10.0);
    for (int t = 0; t < 400; ++t) {
        SsviParams bad;
        bad.rho = wild(rng);
        bad.eta = wild(rng);
        bad.gamma = wild(rng);
        bad.lambda = wild(rng);
        bad.phi_kind = (t % 2 == 0) ? SsviPhiKind::PowerLaw : SsviPhiKind::Heston;
        ASSERT_TRUE(ssvi_parameters_admissible(ssvi_project_to_admissible(bad)))
            << "trial " << t;
    }
}

TEST(Ssvi, DegenerateThetaIsHandled) {
    SsviParams g;
    EXPECT_EQ(ssvi_total_variance(g, 0.0, 0.5), 0.0);
    EXPECT_EQ(ssvi_total_variance(g, -1.0, 0.5), 0.0);
    EXPECT_EQ(ssvi_jet(g, 0.0, 0.5).w, 0.0);
    EXPECT_EQ(ssvi_phi(g, 0.0).phi, 0.0);
    // The mapped slice must still be admissible, not a division by zero.
    const SviParams sp = ssvi_slice_params(g, 0.0, 1.0);
    EXPECT_TRUE(std::isfinite(sp.sigma));
    EXPECT_GT(sp.sigma, 0.0);
}

TEST(SliceKindNames, AreAllDistinct) {
    EXPECT_STREQ(to_string(SliceKind::Flat), "flat");
    EXPECT_STREQ(to_string(SliceKind::Svi), "svi");
    EXPECT_STREQ(to_string(SliceKind::Ssvi), "ssvi");
    EXPECT_STREQ(to_string(SliceKind::Grid), "grid");
    EXPECT_STREQ(to_string(SsviPhiKind::PowerLaw), "power-law");
    EXPECT_STREQ(to_string(SsviPhiKind::Heston), "heston");
}
