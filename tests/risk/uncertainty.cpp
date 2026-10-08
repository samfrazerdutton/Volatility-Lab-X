// SPDX-License-Identifier: MIT
/// Validates the uncertainty engine: that it correctly ranks ATM-vs-wing and
/// in-range-vs-extrapolated confidence, that the degenerate "no coverage"
/// case is flagged rather than guessed at, and the surprising-but-correct
/// invariance of this module's *relative* confidence measure to a uniformly
/// stale snapshot (see the header comment for why that is not a bug).

#include "vl_test_support.hpp"

#include "volatility_lab/risk/uncertainty.hpp"

#include "volatility_lab/calibration/weights.hpp"
#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/options/normalize.hpp"

#include <cmath>
#include <limits>
#include <vector>

using namespace vl;

namespace {

/// A realistic, normalised-and-weighted quote book plus the ground-truth
/// surface it was generated from. Uncertainty estimation only needs *a*
/// surface and *a* quote book -- using the generator's ground truth here
/// (rather than first calibrating one) keeps this suite independent of the
/// calibrator, which has its own test suite.
struct Fixture {
    VolSurface surface;
    std::vector<OptionQuote> quotes;
    double min_k = 0.0, max_k = 0.0, min_t = 0.0, max_t = 0.0;
};

Fixture make_fixture(MarketRegime regime = MarketRegime::Normal, std::uint64_t seed = 20260207u) {
    Fixture f;
    auto market = generate_market(regime, seed);
    auto norm = normalize(market.snapshot);
    (void)assign_weights_by_slice(norm.quotes);
    f.surface = market.true_surface;
    f.quotes = std::move(norm.quotes);

    f.min_k = f.min_t = std::numeric_limits<double>::infinity();
    f.max_k = f.max_t = -std::numeric_limits<double>::infinity();
    for (const auto& q : f.quotes) {
        if (q.status == QuoteStatus::Rejected) continue;
        f.min_k = std::min(f.min_k, q.log_moneyness);
        f.max_k = std::max(f.max_k, q.log_moneyness);
        f.min_t = std::min(f.min_t, q.years);
        f.max_t = std::max(f.max_t, q.years);
    }
    return f;
}

}  // namespace

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

TEST(Uncertainty, EmptyBookReportsInfiniteUncertaintyRatherThanAGuess) {
    const auto f = make_fixture();
    const auto u =
        estimate_point_uncertainty(f.surface, std::span<const OptionQuote>{}, 0.0, 0.5);
    EXPECT_TRUE(u.no_local_coverage);
    EXPECT_TRUE(std::isinf(u.vol_std_error));
    EXPECT_EQ(u.effective_n, 0.0);
    // The point estimate itself is still reported -- the surface is defined
    // everywhere, it is only the *confidence* in it that is missing.
    EXPECT_EQ(u.vol_estimate, f.surface.vol(0.0, 0.5));
}

TEST(Uncertainty, AllQuotesRejectedIsEquivalentToAnEmptyBook) {
    auto f = make_fixture();
    for (auto& q : f.quotes) q.status = QuoteStatus::Rejected;
    const auto u = estimate_point_uncertainty(f.surface, f.quotes, 0.0, 0.5);
    EXPECT_TRUE(u.no_local_coverage);
    EXPECT_TRUE(std::isinf(u.vol_std_error));
}

TEST(Uncertainty, AllQuotesWithZeroCalibrationWeightIsEquivalentToAnEmptyBook) {
    // Not rejected (still counts toward the observed range), but excluded
    // from the fit entirely -- the uncertainty engine must treat "the
    // calibration ignored this" the same as "this isn't here".
    auto f = make_fixture();
    for (auto& q : f.quotes) q.weight = 0.0;
    const auto u = estimate_point_uncertainty(f.surface, f.quotes, 0.0, 0.5);
    EXPECT_TRUE(u.no_local_coverage);
    EXPECT_TRUE(std::isinf(u.vol_std_error));
}

// ---------------------------------------------------------------------------
// Ranking: ATM vs wing, in-range vs extrapolated
// ---------------------------------------------------------------------------

TEST(Uncertainty, DeepWingIsLessCertainThanAtTheMoney) {
    const auto f = make_fixture();
    const auto atm = estimate_point_uncertainty(f.surface, f.quotes, 0.0, 0.5);
    const auto wing = estimate_point_uncertainty(f.surface, f.quotes, 0.45, 0.5);
    ASSERT_FALSE(atm.no_local_coverage);
    ASSERT_FALSE(wing.no_local_coverage);
    EXPECT_GT(wing.vol_std_error, atm.vol_std_error);
    EXPECT_LT(wing.effective_n, atm.effective_n);
}

TEST(Uncertainty, StrikeExtrapolationIsFlaggedAndSharplyInflatesUncertainty) {
    const auto f = make_fixture();
    const auto in_range = estimate_point_uncertainty(f.surface, f.quotes, 0.0, 0.5);
    const auto beyond =
        estimate_point_uncertainty(f.surface, f.quotes, f.max_k + 0.5, 0.5);
    EXPECT_FALSE(in_range.extrapolated_in_strike);
    EXPECT_TRUE(beyond.extrapolated_in_strike);
    EXPECT_GT(beyond.vol_std_error, 5.0 * in_range.vol_std_error);
}

TEST(Uncertainty, TenorExtrapolationIsFlaggedAndInflatesUncertainty) {
    const auto f = make_fixture();
    const auto in_range = estimate_point_uncertainty(f.surface, f.quotes, 0.0, 0.5);
    const auto beyond = estimate_point_uncertainty(f.surface, f.quotes, 0.0, f.max_t + 5.0);
    EXPECT_FALSE(in_range.extrapolated_in_time);
    EXPECT_TRUE(beyond.extrapolated_in_time);
    EXPECT_GT(beyond.vol_std_error, in_range.vol_std_error);
}

TEST(Uncertainty, PointsStrictlyInsideTheObservedRangeAreNeverFlaggedAsExtrapolated) {
    const auto f = make_fixture();
    const double mid_k = 0.5 * (f.min_k + f.max_k);
    const double mid_t = 0.5 * (f.min_t + f.max_t);
    const auto u = estimate_point_uncertainty(f.surface, f.quotes, mid_k, mid_t);
    EXPECT_FALSE(u.extrapolated_in_strike);
    EXPECT_FALSE(u.extrapolated_in_time);
}

TEST(Uncertainty, DisablingTheExtrapolationPenaltyIsolatesTheInterpolationBehaviour) {
    const auto f = make_fixture();
    UncertaintyConfig cfg;
    cfg.extrapolation_penalty_per_bandwidth = 0.0;
    const auto in_range = estimate_point_uncertainty(f.surface, f.quotes, 0.0, 0.5, cfg);
    const auto beyond =
        estimate_point_uncertainty(f.surface, f.quotes, f.max_k + 0.5, 0.5, cfg);
    EXPECT_TRUE(beyond.extrapolated_in_strike);  // still flagged
    // But with the penalty off, the only remaining effect is the kernel
    // naturally having fewer nearby quotes -- no multiplicative blow-up.
    EXPECT_LT(beyond.vol_std_error, 5.0 * in_range.vol_std_error);
}

// ---------------------------------------------------------------------------
// Staleness: local vs global
// ---------------------------------------------------------------------------

TEST(Uncertainty, AgingOnlyTheLocallyRelevantQuotesChangesThisPointsUncertainty) {
    auto f = make_fixture();
    auto fresh = f.quotes;
    (void)assign_weights_by_slice(fresh);
    const auto u_fresh = estimate_point_uncertainty(f.surface, fresh, 0.0, 0.5);

    auto aged = f.quotes;
    for (auto& q : aged) {
        if (std::abs(q.log_moneyness) < 0.1 && std::abs(q.years - 0.5) < 0.1) {
            q.age_seconds += 3600.0;
        }
    }
    (void)assign_weights_by_slice(aged);
    const auto u_aged = estimate_point_uncertainty(f.surface, aged, 0.0, 0.5);

    EXPECT_GT(u_aged.vol_std_error, u_fresh.vol_std_error);
}

TEST(Uncertainty, UniformlyAgingTheWholeBookLeavesRelativeUncertaintyUnchanged) {
    // Documented in the header: Kish effective-N and the weighted mean are
    // both exactly invariant under a uniform rescaling of every weight, so
    // a book that is uniformly an hour old reports the *same* per-point
    // uncertainty as one that is uniformly fresh. This is correct -- nothing
    // about which points are relatively better or worse supported has
    // changed -- and is tested directly so a future change does not
    // "fix" it by bolting on an ad hoc global adjustment.
    auto f = make_fixture();
    const auto u_fresh = estimate_point_uncertainty(f.surface, f.quotes, 0.0, 0.5);

    auto aged = f.quotes;
    for (auto& q : aged) {
        if (q.age_seconds >= 0.0) q.age_seconds += 3600.0;
    }
    (void)assign_weights_by_slice(aged);
    const auto u_aged = estimate_point_uncertainty(f.surface, aged, 0.0, 0.5);

    EXPECT_NEAR(u_aged.vol_std_error, u_fresh.vol_std_error, 1e-6);
    EXPECT_NEAR(u_aged.effective_n, u_fresh.effective_n, 1.0);
    // The representative age itself, by contrast, must move -- that field is
    // reporting a raw observable, not a weight-invariant ratio.
    EXPECT_GT(u_aged.representative_age_seconds, u_fresh.representative_age_seconds + 3000.0);
}

// ---------------------------------------------------------------------------
// Batch convenience
// ---------------------------------------------------------------------------

TEST(Uncertainty, GridBatchMatchesPointwiseCallsExactly) {
    const auto f = make_fixture();
    const std::vector<double> ks = {-0.3, 0.0, 0.2, 0.4};
    const std::vector<double> ts = {0.25, 0.5, 0.5, 1.0};
    const auto batch = estimate_grid_uncertainty(f.surface, f.quotes, ks, ts);
    ASSERT_EQ(batch.size(), ks.size());
    for (std::size_t i = 0; i < ks.size(); ++i) {
        const auto single = estimate_point_uncertainty(f.surface, f.quotes, ks[i], ts[i]);
        EXPECT_EQ(batch[i].vol_std_error, single.vol_std_error);
        EXPECT_EQ(batch[i].effective_n, single.effective_n);
    }
}

TEST(Uncertainty, GridBatchTruncatesToTheShorterSpanRatherThanReadingOutOfBounds) {
    const auto f = make_fixture();
    const std::vector<double> ks = {0.0, 0.1, 0.2};
    const std::vector<double> ts = {0.5, 1.0};  // shorter
    const auto batch = estimate_grid_uncertainty(f.surface, f.quotes, ks, ts);
    EXPECT_EQ(batch.size(), 2u);
}

// ---------------------------------------------------------------------------
// Cross-regime sanity: a thin/illiquid book should read less certain overall
// ---------------------------------------------------------------------------

TEST(Uncertainty, IlliquidRegimeReportsHigherTypicalUncertaintyThanNormal) {
    const auto normal = make_fixture(MarketRegime::Normal);
    const auto illiquid = make_fixture(MarketRegime::Illiquid);

    const auto u_normal = estimate_point_uncertainty(normal.surface, normal.quotes, 0.0, 0.5);
    const auto u_illiquid =
        estimate_point_uncertainty(illiquid.surface, illiquid.quotes, 0.0, 0.5);
    ASSERT_FALSE(u_normal.no_local_coverage);
    ASSERT_FALSE(u_illiquid.no_local_coverage);
    EXPECT_GT(u_illiquid.vol_std_error, u_normal.vol_std_error);
}
