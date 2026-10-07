// SPDX-License-Identifier: MIT
/// Validates the surface assembly: variant dispatch, the term-structure
/// interpolation, and the two no-arbitrage properties the interpolation choice
/// is supposed to guarantee.
///
/// The load-bearing test here is
/// `TermInterpolationCannotCreateCalendarArbitrage`: the surface interpolates
/// *total variance linearly in T*, and the claim is that doing so cannot turn
/// two admissible slices into an arbitrageable surface.  That claim is what
/// justifies the choice over the more obvious "interpolate volatility", and it
/// is checked by construction rather than asserted.

#include "vl_test_support.hpp"

#include "volatility_lab/pricing/black.hpp"
#include "volatility_lab/pricing/implied_vol.hpp"
#include "volatility_lab/volatility/surface.hpp"

#include <cmath>
#include <random>
#include <vector>

using namespace vl;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

SviParams make_svi(double years, double atm_var, double skew = -0.4) {
    SviParams p;
    // Chosen so that w(0) is close to atm_var and the slice is admissible.
    p.b = 0.09 * std::sqrt(years + 0.1);
    p.rho = skew;
    p.m = 0.01;
    p.sigma = 0.15;
    p.years = years;
    p.a = atm_var - p.b * (p.rho * (0.0 - p.m) +
                           std::sqrt(p.m * p.m + p.sigma * p.sigma));
    return svi_project_to_admissible(p);
}

/// A four-expiry SVI surface with an increasing ATM variance term structure.
VolSurface make_surface() {
    std::vector<SliceVariant> slices;
    slices.emplace_back(make_svi(0.08, 0.0032));
    slices.emplace_back(make_svi(0.25, 0.0100));
    slices.emplace_back(make_svi(1.00, 0.0400));
    slices.emplace_back(make_svi(2.00, 0.0820));
    return VolSurface(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
}

}  // namespace

// ===========================================================================
// Construction and validation
// ===========================================================================

TEST(Surface, ValidSurfaceReportsNoProblems) {
    const auto s = make_surface();
    EXPECT_TRUE(s.valid()) << s.diagnostics().summary();
    EXPECT_EQ(s.num_slices(), 4u);
    EXPECT_EQ(s.diagnostics().count(Severity::Fatal), 0u);
    const auto exp = s.expiries();
    EXPECT_NEAR(exp[0], 0.08, 1e-15);
    EXPECT_NEAR(exp[3], 2.00, 1e-15);
}

TEST(Surface, EmptySurfaceIsFatalNotACrash) {
    const VolSurface s({}, TermCurve::flat(100.0), TermCurve::flat(1.0));
    EXPECT_FALSE(s.valid());
    EXPECT_GT(s.diagnostics().count(Severity::Fatal), 0u);
    // And it must still answer queries without dereferencing anything.
    EXPECT_EQ(s.total_variance(0.0, 1.0), 0.0);
    EXPECT_EQ(s.vol(0.0, 1.0), 0.0);
    EXPECT_EQ(s.jet(0.0, 1.0).w, 0.0);
    EXPECT_EQ(s.num_slices(), 0u);
}

TEST(Surface, OutOfOrderOrDuplicateExpiriesAreRejected) {
    std::vector<SliceVariant> slices;
    slices.emplace_back(make_svi(1.0, 0.04));
    slices.emplace_back(make_svi(0.5, 0.02));  // out of order
    const VolSurface s(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
    EXPECT_FALSE(s.valid());
    EXPECT_GT(s.diagnostics().count(DiagCode::DuplicateQuote), 0u);

    std::vector<SliceVariant> dup;
    dup.emplace_back(make_svi(1.0, 0.04));
    dup.emplace_back(make_svi(1.0, 0.05));
    const VolSurface s2(std::move(dup), TermCurve::flat(100.0), TermCurve::flat(1.0));
    EXPECT_FALSE(s2.valid());
}

TEST(Surface, NonPositiveExpiryIsRejected) {
    std::vector<SliceVariant> slices;
    slices.emplace_back(make_svi(0.0, 0.04));
    const VolSurface s(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
    EXPECT_FALSE(s.valid());
    EXPECT_GT(s.diagnostics().count(DiagCode::NonPositiveExpiry), 0u);
}

// ===========================================================================
// Variant dispatch
// ===========================================================================

TEST(Surface, SliceDispatchCoversEveryVariantAlternative) {
    const SliceVariant flat = FlatParams{0.04, 1.0};
    const SliceVariant svi = make_svi(1.0, 0.04);
    SsviSlice ss;
    ss.global.rho = -0.4;
    ss.global.eta = 1.0;
    ss.global.gamma = 0.5;
    ss.theta = 0.04;
    ss.years = 1.0;
    const SliceVariant ssvi = ss;
    const SliceVariant grid =
        GridSlice({-0.3, -0.1, 0.0, 0.1, 0.3}, {0.05, 0.042, 0.04, 0.041, 0.048}, 1.0);

    for (const auto* s : {&flat, &svi, &ssvi, &grid}) {
        EXPECT_EQ(slice_years(*s), 1.0);
        EXPECT_GT(slice_total_variance(*s, 0.0), 0.0);
        EXPECT_GT(slice_jet(*s, 0.0).w, 0.0);
        // The jet and the scalar evaluator must agree, or a Greek computed
        // from the jet disagrees with the price computed from the scalar.
        EXPECT_NEAR(slice_jet(*s, 0.05).w, slice_total_variance(*s, 0.05), 1e-15);
    }
    EXPECT_EQ(slice_kind(flat), SliceKind::Flat);
    EXPECT_EQ(slice_kind(svi), SliceKind::Svi);
    EXPECT_EQ(slice_kind(ssvi), SliceKind::Ssvi);
    EXPECT_EQ(slice_kind(grid), SliceKind::Grid);
}

TEST(Surface, SliceBatchAgreesWithScalarForEveryVariant) {
    // The batch path is where the performance lives, so it must be bitwise
    // identical to the scalar path rather than merely close: a batch/scalar
    // discrepancy would show up as a risk number that changes depending on how
    // the caller asked for it.
    const SliceVariant flat = FlatParams{0.04, 1.0};
    const SliceVariant svi = make_svi(1.0, 0.04);
    SsviSlice ss;
    ss.global.rho = -0.4;
    ss.theta = 0.04;
    ss.years = 1.0;
    const SliceVariant ssvi = ss;
    const SliceVariant grid =
        GridSlice({-0.3, -0.1, 0.0, 0.1, 0.3}, {0.05, 0.042, 0.04, 0.041, 0.048}, 1.0);

    std::vector<double> ks;
    for (double k = -0.6; k <= 0.6; k += 0.001) ks.push_back(k);
    std::vector<double> ws(ks.size());

    for (const auto* s : {&flat, &svi, &grid}) {
        slice_total_variance_batch(*s, ks, ws);
        for (std::size_t i = 0; i < ks.size(); ++i) {
            ASSERT_EQ(ws[i], slice_total_variance(*s, ks[i]))
                << "kind " << to_string(slice_kind(*s)) << ", i = " << i;
        }
    }
    // SSVI goes through the exact SVI mapping in the batch path, so the
    // agreement is to a few ulps rather than bitwise -- and that is worth
    // pinning, because the mapping is the thing being relied on.
    slice_total_variance_batch(ssvi, ks, ws);
    WorstCase w;
    for (std::size_t i = 0; i < ks.size(); ++i) {
        const double direct = slice_total_variance(ssvi, ks[i]);
        w.observe(rel_error(ws[i], direct), ks[i], 0, ws[i], direct);
    }
    EXPECT_LT(w.error, 1e-14) << w.describe("k");
}

TEST(Surface, GridSliceHasNoParametricFormAndSaysSo) {
    // Callers that need a parametric slice (the scenario engine, the SIMD
    // kernels) must be told rather than handed an approximation.
    const SliceVariant grid =
        GridSlice({-0.2, 0.0, 0.2}, {0.045, 0.04, 0.046}, 1.0);
    EXPECT_FALSE(slice_as_svi(grid).has_value());

    EXPECT_TRUE(slice_as_svi(SliceVariant{make_svi(1.0, 0.04)}).has_value());
    EXPECT_TRUE(slice_as_svi(SliceVariant{FlatParams{0.04, 1.0}}).has_value());
    SsviSlice ss;
    ss.theta = 0.04;
    ss.years = 1.0;
    EXPECT_TRUE(slice_as_svi(SliceVariant{ss}).has_value());
}

TEST(Surface, GridSliceClampsNegativeVarianceButKeepsDerivatives) {
    // A cubic through noisy quotes can dip below zero between knots.  The
    // value is clamped because no consumer can use a negative variance, but
    // the derivatives are left alone so the arbitrage engine can still see the
    // shape that caused it.
    const GridSlice g({-0.2, 0.0, 0.2}, {0.01, -0.02, 0.01}, 1.0);
    EXPECT_GE(g.total_variance(0.0), 0.0);
    EXPECT_EQ(g.total_variance(0.0), 0.0);
    EXPECT_TRUE(std::isfinite(g.jet(0.0).d2w));
    EXPECT_EQ(g.k_min(), -0.2);
    EXPECT_EQ(g.k_max(), 0.2);
}

TEST(Surface, EmptyGridSliceIsHarmless) {
    const GridSlice g;
    EXPECT_TRUE(g.empty());
    EXPECT_EQ(g.total_variance(0.0), 0.0);
    EXPECT_EQ(g.k_min(), 0.0);
    std::vector<double> k{0.0, 0.1};
    std::vector<double> w(2, -1.0);
    g.total_variance_batch(k, w);
    EXPECT_EQ(w[0], 0.0);
}

// ===========================================================================
// Term structure
// ===========================================================================

TEST(Surface, QueryAtASliceExpiryReturnsThatSlice) {
    const auto s = make_surface();
    for (std::size_t i = 0; i < s.num_slices(); ++i) {
        const double t = s.expiries()[i];
        for (double k = -0.4; k <= 0.4; k += 0.05) {
            EXPECT_NEAR(s.total_variance(k, t), slice_total_variance(s.slice(i), k), 1e-14)
                << "slice " << i << ", k = " << k;
        }
    }
}

TEST(Surface, TermInterpolationIsLinearInTotalVariance) {
    // Not in volatility.  Checked by confirming that the interpolated total
    // variance sits exactly on the straight line between the two bracketing
    // slices -- which also means the *volatility* does not, and that is the
    // intended behaviour.
    const auto s = make_surface();
    const double t0 = s.expiries()[1];
    const double t1 = s.expiries()[2];
    const double k = -0.1;
    const double w0 = slice_total_variance(s.slice(1), k);
    const double w1 = slice_total_variance(s.slice(2), k);

    for (double u : {0.0, 0.17, 0.5, 0.83, 1.0}) {
        const double t = t0 + u * (t1 - t0);
        const double expected = w0 + u * (w1 - w0);
        EXPECT_NEAR(s.total_variance(k, t), expected, 1e-13) << "u = " << u;
    }
}

TEST(Surface, TermInterpolationCannotCreateCalendarArbitrage) {
    // The claim that justifies interpolating total variance linearly in T.
    //
    // Calendar arbitrage is total variance *decreasing* in T at some fixed
    // moneyness.  If the two bracketing slices satisfy w_lo(k) <= w_hi(k),
    // then a convex combination is monotone in the interpolation weight, so no
    // intermediate maturity can dip below either -- the interpolator cannot
    // manufacture arbitrage out of admissible data.  Interpolating *volatility*
    // linearly has no such property.
    //
    // Verified across the whole surface on a dense (k, T) grid, including
    // inside every interval and across every knot.
    const auto s = make_surface();
    WorstCase worst_decrease;
    const double t_lo = s.expiries().front();
    const double t_hi = s.expiries().back();

    for (double k = -0.8; k <= 0.8; k += 0.01) {
        double prev = s.total_variance(k, t_lo);
        for (double t = t_lo; t <= t_hi; t += 0.004) {
            const double w = s.total_variance(k, t);
            const double decrease = prev - w;
            worst_decrease.observe(decrease, k, t, w, prev);
            prev = w;
        }
    }
    // Zero tolerance beyond rounding: this is a structural property, not a
    // numerical one.
    EXPECT_LT(worst_decrease.error, 1e-15)
        << "total variance decreased with maturity: " << worst_decrease.describe("k", "T");
}

TEST(Surface, LinearInVolWouldCreateCalendarArbitrageOnThisSurface) {
    // The counterexample that makes the previous test meaningful.  If
    // interpolating volatility linearly were equally safe, there would be no
    // reason to prefer total variance, so the alternative is constructed
    // explicitly and shown to fail.
    //
    // sigma(t) linear between (t0, sigma0) and (t1, sigma1) gives
    // w(t) = t * sigma(t)^2, which is not monotone in t when sigma falls
    // steeply enough -- and a falling term structure of volatility with a
    // rising term structure of variance is entirely normal.
    const double t0 = 0.08, t1 = 1.0;
    const double w0 = 0.0032, w1 = 0.0400;   // both increasing: admissible data
    const double sig0 = std::sqrt(w0 / t0);  // ~20%
    const double sig1 = std::sqrt(w1 / t1);  // ~20%
    ASSERT_LT(w0, w1);

    // Make the short end sharply higher vol, as a stressed front month is.
    const double sig0_stressed = 3.0 * sig0;
    const double w0_stressed = sig0_stressed * sig0_stressed * t0;
    ASSERT_LT(w0_stressed, w1) << "the data must still be calendar-admissible";

    bool vol_linear_decreases = false;
    double prev = w0_stressed;
    for (double t = t0; t <= t1; t += 0.001) {
        const double u = (t - t0) / (t1 - t0);
        const double sig = sig0_stressed + u * (sig1 - sig0_stressed);
        const double w = sig * sig * t;
        if (w < prev - 1e-15) vol_linear_decreases = true;
        prev = w;
    }
    EXPECT_TRUE(vol_linear_decreases)
        << "linear-in-vol interpolation is expected to produce calendar arbitrage on "
           "this admissible data; if it does not, the justification for "
           "linear-in-variance needs revisiting";
}

TEST(Surface, ExtrapolationBeyondTheLastExpiryHoldsVolatilityNotVariance) {
    // Holding total variance flat beyond the last quote would assert that
    // nothing can happen after the longest listed expiry -- which also makes a
    // longer-dated option worth less than a shorter one at the same strike, so
    // it is arbitrageable, not merely wrong.  The surface holds *volatility*
    // flat instead, i.e. variance grows linearly in T.
    const auto s = make_surface();
    const double t_last = s.expiries().back();
    const double k = 0.05;
    const double w_last = s.total_variance(k, t_last);
    const double vol_last = std::sqrt(w_last / t_last);

    for (double mult : {1.5, 2.0, 5.0}) {
        const double t = t_last * mult;
        const double w = s.total_variance(k, t);
        EXPECT_NEAR(w, w_last * mult, 1e-12 * w_last * mult) << "mult = " << mult;
        EXPECT_NEAR(std::sqrt(w / t), vol_last, 1e-12) << "volatility must stay flat";
        EXPECT_GT(w, w_last) << "longer maturity must not be worth less";
    }
}

TEST(Surface, ExtrapolationBeforeTheFirstExpiryShrinksVarianceToZero) {
    const auto s = make_surface();
    const double t_first = s.expiries().front();
    const double k = 0.0;
    const double w_first = s.total_variance(k, t_first);
    for (double mult : {0.5, 0.1, 0.01}) {
        const double w = s.total_variance(k, t_first * mult);
        EXPECT_NEAR(w, w_first * mult, 1e-12 * w_first) << "mult = " << mult;
        EXPECT_LT(w, w_first);
        EXPECT_GT(w, 0.0);
    }
}

TEST(Surface, JetIsConsistentWithTheScalarQuery) {
    const auto s = make_surface();
    WorstCase w;
    for (double t : {0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0}) {
        for (double k = -0.5; k <= 0.5; k += 0.01) {
            const auto j = s.jet(k, t);
            const double direct = s.total_variance(k, t);
            w.observe(rel_error(j.w, direct), k, t, j.w, direct);
        }
    }
    EXPECT_LT(w.error, 1e-14) << w.describe("k", "T");
}

TEST(Surface, JetDerivativesMatchFiniteDifferencesAcrossInterpolatedExpiries) {
    // The interpolated slice must have correct k-derivatives, not just a
    // correct level: gamma and the density come from them.
    const auto s = make_surface();
    WorstCase w1;
    const double h = 1e-6;
    for (double t : {0.15, 0.6, 1.4}) {
        for (double k = -0.4; k <= 0.4; k += 0.01) {
            const auto j = s.jet(k, t);
            const double fd =
                (s.total_variance(k + h, t) - s.total_variance(k - h, t)) / (2.0 * h);
            w1.observe(std::abs(j.dw - fd), k, t, j.dw, fd);
        }
    }
    EXPECT_LT(w1.error, 1e-8) << w1.describe("k", "T");
}

// ===========================================================================
// Batch queries
// ===========================================================================

TEST(Surface, SingleExpiryBatchAgreesWithScalar) {
    const auto s = make_surface();
    std::vector<double> ks;
    for (double k = -0.7; k <= 0.7; k += 0.0007) ks.push_back(k);
    std::vector<double> ws(ks.size());

    // At a knot (single slice), between knots (two slices blended), and
    // extrapolated -- the three code paths.
    for (double t : {0.25, 0.6, 4.0, 0.01}) {
        s.total_variance_batch_single_expiry(t, ks, ws);
        WorstCase w;
        for (std::size_t i = 0; i < ks.size(); ++i) {
            const double direct = s.total_variance(ks[i], t);
            w.observe(rel_error(ws[i], direct), ks[i], t, ws[i], direct);
        }
        EXPECT_LT(w.error, 1e-14) << "T = " << t << ": " << w.describe("k", "T");
    }
}

TEST(Surface, MixedExpiryBatchAgreesWithScalar) {
    const auto s = make_surface();
    std::mt19937_64 rng(4242u);
    std::uniform_real_distribution<double> uk(-0.8, 0.8);
    std::uniform_real_distribution<double> ut(0.01, 4.0);
    const std::size_t n = 5000;
    std::vector<double> ks(n), ts(n), ws(n);
    for (std::size_t i = 0; i < n; ++i) {
        ks[i] = uk(rng);
        ts[i] = ut(rng);
    }
    s.total_variance_batch(ks, ts, ws);
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_EQ(ws[i], s.total_variance(ks[i], ts[i])) << "i = " << i;
    }
}

TEST(Surface, BatchHandlesEmptyAndMismatchedSpans) {
    const auto s = make_surface();
    std::vector<double> none;
    std::vector<double> out;
    s.total_variance_batch(none, none, out);  // must not crash
    s.total_variance_batch_single_expiry(1.0, none, out);

    std::vector<double> ks{0.0, 0.1, 0.2};
    std::vector<double> ws(2, -1.0);
    s.total_variance_batch_single_expiry(1.0, ks, ws);
    EXPECT_NE(ws[0], -1.0);
    EXPECT_NE(ws[1], -1.0);
}

// ===========================================================================
// Curves
// ===========================================================================

TEST(Surface, TermCurveIsLogLinearSoForwardRatesStayPositive) {
    // Log-linear interpolation of discount factors is piecewise-constant
    // forward rates.  Linear interpolation of discount factors can imply a
    // negative forward rate between nodes, which is the failure this avoids.
    const TermCurve df({0.5, 1.0, 2.0}, {0.98, 0.96, 0.91});
    // Geometric mean at the midpoint in T, not arithmetic.
    EXPECT_NEAR(df(0.75), std::sqrt(0.98 * 0.96), 1e-12);
    // Monotone decreasing, so every implied forward rate is positive.
    double prev = df(0.5);
    for (double t = 0.5; t <= 2.0; t += 0.01) {
        const double v = df(t);
        ASSERT_LE(v, prev + 1e-15) << "t = " << t;
        prev = v;
    }
    EXPECT_GT(df(0.0), 0.0);
    EXPECT_GT(df(10.0), 0.0) << "extrapolated discount factors must stay positive";
}

TEST(Surface, FlatCurveIsConstant) {
    const auto c = TermCurve::flat(100.0);
    EXPECT_EQ(c(0.0), 100.0);
    EXPECT_EQ(c(1.0), 100.0);
    EXPECT_EQ(c(1e6), 100.0);
    const TermCurve empty;
    EXPECT_GT(empty(1.0), 0.0) << "a default curve must be usable, not zero";
}

TEST(Surface, AtStrikeResolvesTheForwardAndReportsExtrapolation) {
    std::vector<SliceVariant> slices;
    slices.emplace_back(
        GridSlice({-0.2, -0.1, 0.0, 0.1, 0.2}, {0.05, 0.043, 0.04, 0.042, 0.049}, 1.0));
    const VolSurface s(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(0.97));

    const auto atm = s.at_strike(100.0, 1.0);
    EXPECT_NEAR(atm.forward, 100.0, 1e-12);
    EXPECT_NEAR(atm.discount, 0.97, 1e-12);
    EXPECT_NEAR(atm.log_moneyness, 0.0, 1e-15);
    EXPECT_NEAR(atm.total_variance, 0.04, 1e-12);
    EXPECT_NEAR(atm.vol, 0.2, 1e-12);
    EXPECT_FALSE(atm.extrapolated_in_strike);
    EXPECT_FALSE(atm.extrapolated_in_time);

    // Well outside the quoted strike range: flagged.
    const auto far = s.at_strike(300.0, 1.0);
    EXPECT_TRUE(far.extrapolated_in_strike);
    // Outside the quoted expiry range: also flagged.
    EXPECT_TRUE(s.at_strike(100.0, 5.0).extrapolated_in_time);

    // Degenerate inputs must not produce garbage.
    EXPECT_EQ(s.at_strike(0.0, 1.0).total_variance, 0.0);
    EXPECT_EQ(s.at_strike(-1.0, 1.0).total_variance, 0.0);
}

// ===========================================================================
// Immutability
// ===========================================================================

TEST(Surface, WithSliceProducesANewSurfaceAndLeavesTheOriginalAlone) {
    // The surface is immutable so that the parallel backend can query one from
    // any number of threads without synchronisation, and so that the
    // incremental update path can replace one slice without rebuilding the
    // rest.  Both depend on this.
    const auto original = make_surface();
    const double before = original.total_variance(0.0, 1.0);

    const auto updated = original.with_slice(2, FlatParams{0.09, 1.0});
    EXPECT_EQ(original.total_variance(0.0, 1.0), before) << "the original was mutated";
    EXPECT_NEAR(updated.total_variance(0.0, 1.0), 0.09, 1e-12);
    EXPECT_EQ(updated.num_slices(), original.num_slices());
    EXPECT_EQ(slice_kind(updated.slice(2)), SliceKind::Flat);
    EXPECT_EQ(slice_kind(original.slice(2)), SliceKind::Svi);

    // Out-of-range index is a no-op rather than undefined behaviour.
    const auto noop = original.with_slice(99, FlatParams{0.5, 1.0});
    EXPECT_EQ(noop.total_variance(0.0, 1.0), before);
}

TEST(Surface, AtmTermStructureIsTheThetaSequenceSsviNeeds) {
    const auto s = make_surface();
    const auto theta = s.atm_total_variance_term();
    ASSERT_EQ(theta.size(), s.num_slices());
    for (std::size_t i = 0; i < theta.size(); ++i) {
        EXPECT_NEAR(theta[i], slice_total_variance(s.slice(i), 0.0), 1e-15);
    }
    // And for a sensible surface it is increasing -- the first calendar
    // condition.
    EXPECT_TRUE(math::is_non_decreasing(theta));
}

// ===========================================================================
// End to end: the surface prices options
// ===========================================================================

TEST(Surface, SurfacePricesConsistentlyWithTheBlackEngine) {
    // The join between phases 2 and 3: a surface query feeds the pricer, and
    // the implied volatility recovered from the resulting price must be the
    // volatility the surface reported.  This is the round trip that catches a
    // total-variance/volatility confusion or a k-sign error, either of which
    // would otherwise produce plausible-looking but wrong prices.
    const auto s = make_surface();
    WorstCase w;
    for (double t : {0.1, 0.25, 0.5, 1.0, 2.0}) {
        for (double strike : {60.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0}) {
            const auto pt = s.at_strike(strike, t);
            ASSERT_GT(pt.vol, 0.0);
            const double px =
                black_undiscounted(pt.forward, strike, pt.vol, t, OptionType::Call);
            const auto iv =
                implied_volatility_undiscounted(px, pt.forward, strike, t, OptionType::Call);
            ASSERT_TRUE(iv.ok()) << "strike " << strike << ", T " << t << ": "
                                 << to_string(iv.status);
            w.observe(rel_error(iv.volatility, pt.vol), strike, t, iv.volatility, pt.vol);
        }
    }
    EXPECT_LT(w.error, 1e-11) << w.describe("strike", "T");
}

TEST(Surface, SkewHasTheRightSign) {
    // The standing trap: the surface uses k = log(K/F) and the Black
    // normalisation uses x = log(F/K).  Getting them confused inverts the
    // skew, and the result still looks like a smile.  With rho < 0 the put
    // wing must be the expensive one, i.e. implied volatility must *fall* as
    // strike rises near the money.
    const auto s = make_surface();
    const double t = 1.0;
    const double vol_low = s.at_strike(85.0, t).vol;
    const double vol_atm = s.at_strike(100.0, t).vol;
    const double vol_high = s.at_strike(115.0, t).vol;
    EXPECT_GT(vol_low, vol_atm) << "downside volatility must exceed ATM for rho < 0";
    EXPECT_GT(vol_low, vol_high) << "the skew is inverted";
}
