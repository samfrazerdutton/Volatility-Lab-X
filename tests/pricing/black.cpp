// SPDX-License-Identifier: MIT
/// Validates the Black engine against the double-double reference, and pins
/// the structural properties the rest of the library depends on: exact
/// put-call parity, monotonicity, the single inflection point, and overflow
/// safety at extreme total volatility.

#include "vl_test_support.hpp"

#include "volatility_lab/pricing/black.hpp"
#include "volatility_lab/pricing/reference.hpp"

using namespace vl;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

/// Worst relative error of `normalised_black` against the double-double
/// reference, measured over the full log domain.  3e-12 is the measured
/// figure; see docs/numerical-validation.md for the per-branch breakdown.
constexpr double kNormalisedBlackRtol = 1.0e-11;

/// Below this the reference itself underflows and there is nothing to compare.
constexpr double kReferenceFloor = 1e-280;

}  // namespace

// ===========================================================================
// Known values
// ===========================================================================

TEST(Black, AtmCallMatchesClosedForm) {
    // F = K = 100, sigma = 20%, T = 1.  Undiscounted ATM value is
    // 100 * erf(sigma*sqrt(T) / (2 sqrt2)) = 100 * erf(0.0707106781186...).
    const double got = black_undiscounted(100.0, 100.0, 0.2, 1.0, OptionType::Call);
    const double closed = 100.0 * std::erf(0.2 / (2.0 * std::sqrt(2.0)));
    EXPECT_CLOSE_TOL(got, closed, math::tol::kBlackPrice);
    EXPECT_CLOSE_TOL(got, reference::black_undiscounted_ref(100.0, 100.0, 0.2, 1.0,
                                                            OptionType::Call),
                     math::tol::kBlackPrice);
}

TEST(Black, AtmBranchIsExactlyTheErfForm) {
    for (double s : vl::test::log_space(1e-6, 10.0, 60)) {
        const auto nb = normalised_black(0.0, s);
        EXPECT_EQ(static_cast<int>(nb.branch), static_cast<int>(BlackBranch::Atm));
        EXPECT_BITWISE_EQ(nb.value, std::erf(s * 0.5 * math::kSqrtHalf));
    }
}

TEST(Black, SpotParameterisationAgreesWithForwardParameterisation) {
    const double S = 100.0, K = 95.0, v = 0.25, T = 1.5, r = 0.03, q = 0.01;
    const double F = S * std::exp((r - q) * T);
    const double df = std::exp(-r * T);
    for (auto type : {OptionType::Call, OptionType::Put}) {
        EXPECT_CLOSE_TOL(black_scholes_price(S, K, v, T, r, q, type),
                         df * black_undiscounted(F, K, v, T, type), math::tol::kBlackPrice);
    }
}

// ===========================================================================
// Put-call parity -- the structural claim
// ===========================================================================

TEST(Black, PutCallParityHoldsToTheLastBit) {
    // The library prices the OTM side and adds the exact intrinsic, so both
    // legs share one normalised value and parity is satisfied by construction
    // rather than to within a tolerance.  The bound below is therefore on the
    // *representation* of F - K, not on the pricing: 4 eps relative to
    // max(F, K) is what a single rounding of the intrinsic costs.
    WorstCase w;
    const auto dom = vl::test::market_domain();
    for (double ratio : dom.strike_ratio) {
        for (double v : dom.vol) {
            for (double T : dom.years) {
                const double F = 100.0;
                const double K = F * ratio;
                const double c = black_undiscounted(F, K, v, T, OptionType::Call);
                const double p = black_undiscounted(F, K, v, T, OptionType::Put);
                const double resid = std::abs((c - p) - (F - K)) / std::max(F, K);
                w.observe(resid, K, v, c - p, F - K);
            }
        }
    }
    EXPECT_LT(w.error, 4.0 * math::kEps) << w.describe("strike", "vol");
}

TEST(Black, DeepItmKeepsRelativeAccuracyOnBothLegs) {
    // A deep-ITM call is intrinsic + epsilon where epsilon is ~1e-13 of the
    // intrinsic.  Computing it as F*Phi(d1) - K*Phi(d2) returns noise for
    // epsilon; pricing the OTM put and adding the intrinsic does not.  Both
    // legs are checked against the reference in *relative* terms, which is the
    // demanding direction.
    for (double K : {1.0, 10.0, 25.0, 50.0, 90.0}) {
        const double c = black_undiscounted(100.0, K, 0.2, 1.0, OptionType::Call);
        const double p = black_undiscounted(100.0, K, 0.2, 1.0, OptionType::Put);
        const double cr = reference::black_undiscounted_ref(100.0, K, 0.2, 1.0,
                                                            OptionType::Call);
        const double pr = reference::black_undiscounted_ref(100.0, K, 0.2, 1.0,
                                                            OptionType::Put);
        EXPECT_LT(rel_error(c, cr), 1e-14) << "call, K = " << K;
        EXPECT_LT(rel_error(p, pr), 1e-12) << "put (the OTM leg), K = " << K;
    }
}

// ===========================================================================
// Against the reference, over the whole domain
// ===========================================================================

TEST(Black, NormalisedBlackMatchesReferenceOverTheFullDomain) {
    WorstCase w;
    const auto dom = vl::test::normalised_domain(110, 110);
    for (double ax : dom.abs_x) {
        for (double s : dom.s) {
            const double ref = reference::normalised_black_ref(-ax, s);
            if (!(ref > kReferenceFloor)) {
                w.skip();
                continue;
            }
            const double got = normalised_black_value(-ax, s);
            w.observe(rel_error(got, ref), ax, s, got, ref);
        }
    }
    EXPECT_GT(w.samples, 5000) << "the sweep must actually cover the domain";
    EXPECT_LT(w.error, kNormalisedBlackRtol) << w.describe("|x|", "s");
}

TEST(Black, EveryBranchIsExercisedAndAccurate) {
    // Branch coverage with teeth: each branch must be reached by the sweep
    // *and* be accurate where it is reached.  A branch that is never taken is
    // a branch whose correctness is unknown, and silently unreachable code is
    // how a fallback path rots.
    constexpr int kBranches = 6;
    WorstCase per_branch[kBranches];
    const auto dom = vl::test::normalised_domain(110, 110);
    for (double ax : dom.abs_x) {
        for (double s : dom.s) {
            const double ref = reference::normalised_black_ref(-ax, s);
            if (!(ref > kReferenceFloor)) continue;
            const auto nb = normalised_black(-ax, s);
            const int b = static_cast<int>(nb.branch);
            ASSERT_GE(b, 0);
            ASSERT_LT(b, kBranches);
            per_branch[b].observe(rel_error(nb.value, ref), ax, s, nb.value, ref);
        }
    }
    const char* names[kBranches] = {"Atm",          "Series",    "ErfcxDirect",
                                    "ZeroVariance", "Saturated", "HighVariance"};
    // Series, ErfcxDirect, Saturated and HighVariance must all be reached.
    for (int b : {static_cast<int>(BlackBranch::Series),
                  static_cast<int>(BlackBranch::ErfcxDirect),
                  static_cast<int>(BlackBranch::Saturated),
                  static_cast<int>(BlackBranch::HighVariance)}) {
        EXPECT_GT(per_branch[b].samples, 0) << "branch " << names[b] << " was never taken";
        EXPECT_LT(per_branch[b].error, kNormalisedBlackRtol)
            << "branch " << names[b] << ": " << per_branch[b].describe("|x|", "s");
    }
}

TEST(Black, TheTextbookFormIsMeasurablyWorse) {
    // Justification for form (2) existing at all.  The naive
    // exp(x/2)Phi(h+t) - exp(-x/2)Phi(h-t) is not catastrophic at these
    // points -- it is about 1e-12 -- but it is three orders of magnitude
    // worse than the erfcx form, and that gap is what the extra machinery
    // buys.  Stating it as a measured ratio keeps the header honest.
    double worst_naive = 0.0;
    double worst_good = 0.0;
    for (auto xs : std::vector<std::pair<double, double>>{
             {-0.05, 0.01}, {-0.2, 0.02}, {-0.5, 0.05}, {-1.0, 0.1}, {-2.0, 0.2}}) {
        const double x = xs.first, s = xs.second;
        const double h = x / s, t = 0.5 * s;
        const double naive = std::exp(0.5 * x) * math::norm_cdf(h + t) -
                             std::exp(-0.5 * x) * math::norm_cdf(h - t);
        const double ref = reference::normalised_black_ref(x, s);
        worst_naive = std::max(worst_naive, rel_error(naive, ref));
        worst_good = std::max(worst_good, rel_error(normalised_black_value(x, s), ref));
    }
    EXPECT_GT(worst_naive, 1e-13) << "the textbook form is documented as worse; it is not";
    EXPECT_LT(worst_good, 5e-14);
    EXPECT_GT(worst_naive / std::max(worst_good, 1e-17), 50.0)
        << "naive " << worst_naive << " vs erfcx form " << worst_good;
}

// ===========================================================================
// Structural properties
// ===========================================================================

TEST(Black, NormalisedBlackIsStrictlyIncreasingInTotalVolatility) {
    // Fact 1 of the implied-vol convergence proof.  If this fails the
    // inversion has no unique root and the solver's guarantees are void.
    for (double ax : {0.0, 1e-4, 0.01, 0.5, 2.0, 8.0}) {
        double prev = -1.0;
        for (double s : vl::test::log_space(1e-4, 50.0, 500)) {
            const double b = normalised_black_value(-ax, s);
            ASSERT_GE(b, prev) << "not monotone at |x| = " << ax << ", s = " << s;
            prev = b;
        }
    }
}

TEST(Black, NormalisedBlackIsBoundedByItsLimits) {
    for (double ax : {0.0, 1e-3, 0.5, 2.0, 10.0}) {
        const double cap = std::exp(-0.5 * ax);
        for (double s : vl::test::log_space(1e-6, 1e3, 200)) {
            const double b = normalised_black_value(-ax, s);
            EXPECT_GE(b, 0.0) << "|x| = " << ax << ", s = " << s;
            EXPECT_LE(b, cap) << "|x| = " << ax << ", s = " << s;
        }
    }
}

TEST(Black, VegaIsPositiveAndMatchesTheReferenceDerivative) {
    // Compared against a *double-double* central difference, not a double one.
    // A double difference cannot settle this: b has a log-derivative of 4e5 at
    // |x| = 3, s = 0.089, so any step large enough to survive the subtraction
    // of two nearly equal doubles is a big excursion in the exponent, and the
    // truncation error swamps the quantity under test.  See
    // reference::normalised_black_vega_ref.
    WorstCase w;
    for (double ax : {1e-3, 0.01, 0.2, 1.0, 3.0}) {
        for (double s : vl::test::log_space(0.05, 5.0, 25)) {
            const double an = normalised_black_vega(-ax, s);
            // Vega is proportional to exp(-(h^2+t^2)/2) and legitimately
            // underflows to zero once |x|/s exceeds ~38 -- at |x| = 3,
            // s = 0.05 the exact value is exp(-1800).  Zero is the correct
            // answer there, not a defect.
            if (an == 0.0) {
                w.skip();
                continue;
            }
            ASSERT_GT(an, 0.0) << "|x| = " << ax << ", s = " << s;
            const double ref = reference::normalised_black_vega_ref(-ax, s);
            if (!(ref > 0.0)) {
                w.skip();
                continue;
            }
            w.observe(rel_error(an, ref), ax, s, an, ref);
        }
    }
    EXPECT_LT(w.error, 1e-13) << w.describe("|x|", "s");
}

TEST(Black, VegaAgreesWithTheValueReturnedAlongsideThePrice) {
    for (double ax : vl::test::log_space(1e-6, 10.0, 40)) {
        for (double s : vl::test::log_space(1e-4, 20.0, 40)) {
            EXPECT_BITWISE_EQ(normalised_black(-ax, s).dv_ds,
                              normalised_black_vega(-ax, s));
        }
    }
}

TEST(Black, SecondDerivativeChangesSignExactlyOnceAtSqrtTwoAbsX) {
    // Fact 2 of the convergence proof: convex below s_c, concave above, one
    // crossing.  Verified by counting sign changes over a fine sweep, not by
    // trusting the formula.
    for (double ax : {1e-3, 0.01, 0.2, 1.0, 5.0}) {
        const auto infl = normalised_black_inflection(-ax);
        EXPECT_NEAR(infl.s_c, std::sqrt(2.0 * ax), 1e-15 * infl.s_c);

        int sign_changes = 0;
        double prev_sign = 0.0;
        const auto grid = vl::test::log_space(infl.s_c * 1e-3, infl.s_c * 1e3, 4000);
        for (double s : grid) {
            const double d2 = normalised_black_d2(-ax, s);
            if (d2 == 0.0) continue;
            const double sg = (d2 > 0.0) ? 1.0 : -1.0;
            if (prev_sign != 0.0 && sg != prev_sign) ++sign_changes;
            prev_sign = sg;
        }
        EXPECT_LE(sign_changes, 1) << "|x| = " << ax << ": more than one inflection";

        // Convex strictly below, concave strictly above.
        EXPECT_GT(normalised_black_d2(-ax, infl.s_c * 0.5), 0.0) << "|x| = " << ax;
        EXPECT_LT(normalised_black_d2(-ax, infl.s_c * 2.0), 0.0) << "|x| = " << ax;
    }
}

TEST(Black, InflectionValueMatchesAPriceAtThatPoint) {
    for (double ax : vl::test::log_space(1e-5, 20.0, 40)) {
        const auto infl = normalised_black_inflection(-ax);
        EXPECT_BITWISE_EQ(infl.b_c, normalised_black_value(-ax, infl.s_c));
    }
}

// ===========================================================================
// Extremes and malformed input
// ===========================================================================

TEST(Black, HighVarianceDoesNotOverflow) {
    // erfcx(y) ~ 2 exp(y^2) overflows for y < -26.6, i.e. s > 75, so the
    // direct erfcx-difference form computes 0 * inf there.  The reflected form
    // must keep it finite and must saturate exactly at exp(x/2).
    for (double ax : {0.0, 0.5, 2.0, 10.0}) {
        const double cap = std::exp(-0.5 * ax);
        for (double s : {50.0, 75.0, 100.0, 1e3, 1e6}) {
            const double b = normalised_black_value(-ax, s);
            ASSERT_TRUE(std::isfinite(b)) << "|x| = " << ax << ", s = " << s;
            EXPECT_LE(b, cap);
            EXPECT_GT(b, 0.9 * cap) << "should have saturated near its bound";
        }
        EXPECT_BITWISE_EQ(normalised_black_value(-ax, 1e8), cap);
    }
}

TEST(Black, ZeroVarianceGivesIntrinsic) {
    for (double ax : {0.0, 0.5, 3.0}) {
        const auto nb = normalised_black(-ax, 0.0);
        EXPECT_EQ(nb.value, 0.0);
        EXPECT_EQ(static_cast<int>(nb.branch), static_cast<int>(BlackBranch::ZeroVariance));
    }
    EXPECT_EQ(black_undiscounted(100.0, 90.0, 0.2, 0.0, OptionType::Call), 10.0);
    EXPECT_EQ(black_undiscounted(100.0, 90.0, 0.0, 1.0, OptionType::Call), 10.0);
    EXPECT_EQ(black_undiscounted(100.0, 110.0, 0.0, 1.0, OptionType::Call), 0.0);
    EXPECT_EQ(black_undiscounted(100.0, 110.0, 0.0, 1.0, OptionType::Put), 10.0);
}

TEST(Black, MalformedInputReturnsNaNRatherThanAPlausibleNumber) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isnan(black_undiscounted(0.0, 100.0, 0.2, 1.0, OptionType::Call)));
    EXPECT_TRUE(std::isnan(black_undiscounted(-100.0, 100.0, 0.2, 1.0, OptionType::Call)));
    EXPECT_TRUE(std::isnan(black_undiscounted(100.0, 0.0, 0.2, 1.0, OptionType::Call)));
    EXPECT_TRUE(std::isnan(black_undiscounted(100.0, 100.0, -0.2, 1.0, OptionType::Call)));
    EXPECT_TRUE(std::isnan(black_undiscounted(100.0, 100.0, 0.2, -1.0, OptionType::Call)));
    EXPECT_TRUE(std::isnan(black_undiscounted(100.0, 100.0, nan, 1.0, OptionType::Call)));
    EXPECT_TRUE(std::isnan(black_undiscounted(100.0, 100.0, inf, 1.0, OptionType::Call)));
    EXPECT_TRUE(std::isnan(normalised_black(nan, 1.0).value));
    EXPECT_TRUE(std::isnan(normalised_black(-1.0, nan).value));
    EXPECT_TRUE(std::isnan(normalised_black(-1.0, -1.0).value));
}

TEST(Black, XSignIsFoldedSoCallersCannotGetTheSkewBackwards) {
    // b(x, s) is documented as requiring x <= 0 and folding defensively.  The
    // fold must be exact, because a sign error here silently inverts every
    // skew in the library.
    for (double ax : vl::test::log_space(1e-6, 10.0, 40)) {
        for (double s : vl::test::log_space(1e-3, 10.0, 40)) {
            EXPECT_BITWISE_EQ(normalised_black_value(ax, s), normalised_black_value(-ax, s));
        }
    }
}

TEST(Black, PriceBoundsAreTheCorrectLimits) {
    const double F = 100.0, K = 120.0;
    const auto cb = forward_price_bounds(F, K, OptionType::Call);
    EXPECT_EQ(cb.lower, 0.0);
    EXPECT_EQ(cb.upper, F);
    const auto pb = forward_price_bounds(F, K, OptionType::Put);
    EXPECT_EQ(pb.lower, K - F);
    EXPECT_EQ(pb.upper, K);

    // And the limits are attained: tiny vol reaches the lower bound, huge vol
    // approaches the upper.
    EXPECT_NEAR(black_undiscounted(F, K, 1e-8, 1.0, OptionType::Call), cb.lower, 1e-12);
    EXPECT_GT(black_undiscounted(F, K, 50.0, 1.0, OptionType::Call), 0.99 * cb.upper);
    EXPECT_LE(black_undiscounted(F, K, 1e6, 1.0, OptionType::Call), cb.upper);
}

// ===========================================================================
// Reductions
// ===========================================================================

TEST(Black, LogMoneynessAndTotalVarianceRoundTrip) {
    EXPECT_NEAR(log_moneyness(100.0, 100.0), 0.0, 1e-16);
    EXPECT_NEAR(log_moneyness(100.0, 110.0), std::log(1.1), 1e-15);
    // Orientation: k = log(K/F) increases with strike.
    EXPECT_GT(log_moneyness(100.0, 110.0), 0.0);
    EXPECT_LT(log_moneyness(100.0, 90.0), 0.0);

    EXPECT_NEAR(total_variance(0.2, 2.0), 0.08, 1e-16);
    EXPECT_NEAR(vol_from_total_variance(0.08, 2.0), 0.2, 1e-15);
    EXPECT_EQ(vol_from_total_variance(0.0, 1.0), 0.0);
    EXPECT_EQ(vol_from_total_variance(0.1, 0.0), 0.0);
}

TEST(Black, PayoffSignIsTheBranchFreeMultiplier) {
    EXPECT_EQ(payoff_sign(OptionType::Call), 1.0);
    EXPECT_EQ(payoff_sign(OptionType::Put), -1.0);
    EXPECT_EQ(forward_intrinsic(100.0, 90.0, OptionType::Call), 10.0);
    EXPECT_EQ(forward_intrinsic(100.0, 110.0, OptionType::Call), 0.0);
    EXPECT_EQ(forward_intrinsic(100.0, 110.0, OptionType::Put), 10.0);
    EXPECT_EQ(forward_intrinsic(100.0, 90.0, OptionType::Put), 0.0);
}
