// SPDX-License-Identifier: MIT
/// Validates incremental (online) calibration: that refitting one slice
/// leaves every other slice bit-identical, that the refit result is
/// identical to what a standalone fit of that slice alone would produce
/// (no hidden interaction with the rest of the surface), that a brand-new
/// expiry is inserted at the correct sorted position, and the honest
/// (not rounded to 1/N) accounting of how much work is actually saved.

#include "vl_test_support.hpp"

#include "volatility_lab/calibration/incremental.hpp"

#include "volatility_lab/calibration/weights.hpp"
#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/options/normalize.hpp"

#include <map>
#include <vector>

using namespace vl;

namespace {

struct Fixture {
    VolSurface surface;
    std::map<double, std::vector<OptionQuote>> by_years;
    long total_residual_evaluations = 0;
};

Fixture make_fixture(MarketRegime regime = MarketRegime::Normal, std::uint64_t seed = 20260207u) {
    Fixture f;
    auto market = generate_market(regime, seed);
    auto norm = normalize(market.snapshot);
    (void)assign_weights_by_slice(norm.quotes);
    for (const auto& q : norm.quotes) f.by_years[q.years].push_back(q);

    std::vector<SliceVariant> slices;
    for (auto& [years, qs] : f.by_years) {
        const auto fit = calibrate_svi_slice(qs);
        slices.emplace_back(fit.params);
        f.total_residual_evaluations += fit.residual_evaluations;
    }
    f.surface = VolSurface(std::move(slices), TermCurve::flat(market.snapshot.spot),
                           TermCurve::flat(1.0));
    return f;
}

}  // namespace

// ---------------------------------------------------------------------------
// The core promise: only the touched slice changes
// ---------------------------------------------------------------------------

TEST(Incremental, EveryOtherSliceIsBitIdenticalAfterARefit) {
    const auto f = make_fixture();
    auto it = std::next(f.by_years.begin());  // some slice that isn't the first
    const auto updated = apply_incremental_update(f.surface, it->first, it->second);

    ASSERT_EQ(updated.surface.num_slices(), f.surface.num_slices());
    for (std::size_t i = 0; i < f.surface.num_slices(); ++i) {
        if (i == updated.slice_index) continue;
        const auto& a = std::get<SviParams>(f.surface.slice(i));
        const auto& b = std::get<SviParams>(updated.surface.slice(i));
        EXPECT_EQ(a.a, b.a) << "slice " << i;
        EXPECT_EQ(a.b, b.b) << "slice " << i;
        EXPECT_EQ(a.rho, b.rho) << "slice " << i;
        EXPECT_EQ(a.m, b.m) << "slice " << i;
        EXPECT_EQ(a.sigma, b.sigma) << "slice " << i;
    }
}

TEST(Incremental, RefitResultMatchesAStandaloneFitOfThatSliceExactly) {
    // The whole point: refitting via this path must not interact with the
    // rest of the surface at all -- same quotes in, bit-identical result to
    // calling calibrate_svi_slice directly.
    const auto f = make_fixture();
    auto it = f.by_years.begin();
    const auto standalone = calibrate_svi_slice(it->second);
    const auto updated = apply_incremental_update(f.surface, it->first, it->second);

    EXPECT_EQ(updated.fit.params.a, standalone.params.a);
    EXPECT_EQ(updated.fit.params.b, standalone.params.b);
    EXPECT_EQ(updated.fit.params.rho, standalone.params.rho);
    EXPECT_EQ(updated.fit.params.m, standalone.params.m);
    EXPECT_EQ(updated.fit.params.sigma, standalone.params.sigma);
    EXPECT_EQ(updated.fit.residual_evaluations, standalone.residual_evaluations);
    EXPECT_EQ(updated.fit.status, standalone.status);
}

TEST(Incremental, ReplacesByValueNotJustByIndexWhenYearsMatches) {
    const auto f = make_fixture();
    auto it = f.by_years.begin();
    const auto updated = apply_incremental_update(f.surface, it->first, it->second);
    EXPECT_FALSE(updated.inserted_new_slice);
    EXPECT_EQ(updated.surface.expiries()[updated.slice_index], it->first);
}

// ---------------------------------------------------------------------------
// Insertion of a brand-new expiry
// ---------------------------------------------------------------------------

TEST(Incremental, NewExpiryIsInsertedAtTheCorrectSortedPosition) {
    const auto f = make_fixture();
    auto it = f.by_years.begin();
    const double new_years = 0.37;  // between the generator's 0.25 and 0.5 slices
    std::vector<OptionQuote> fake_slice = it->second;
    for (auto& q : fake_slice) q.years = new_years;

    const auto inserted = apply_incremental_update(f.surface, new_years, fake_slice);
    ASSERT_TRUE(inserted.inserted_new_slice);
    ASSERT_EQ(inserted.surface.num_slices(), f.surface.num_slices() + 1);

    const auto expiries = inserted.surface.expiries();
    EXPECT_EQ(expiries[inserted.slice_index], new_years);
    for (std::size_t i = 1; i < expiries.size(); ++i) {
        EXPECT_GT(expiries[i], expiries[i - 1]) << "surface must stay strictly sorted";
    }
}

TEST(Incremental, InsertingDoesNotDisturbAnyExistingSlice) {
    const auto f = make_fixture();
    auto it = f.by_years.begin();
    const double new_years = 0.37;
    std::vector<OptionQuote> fake_slice = it->second;
    for (auto& q : fake_slice) q.years = new_years;
    const auto inserted = apply_incremental_update(f.surface, new_years, fake_slice);

    for (std::size_t i = 0; i < f.surface.num_slices(); ++i) {
        const double years = f.surface.expiries()[i];
        // Find the same expiry in the new (larger) surface and compare.
        const auto new_expiries = inserted.surface.expiries();
        const auto pos = std::find(new_expiries.begin(), new_expiries.end(), years);
        ASSERT_NE(pos, new_expiries.end());
        const std::size_t j = static_cast<std::size_t>(pos - new_expiries.begin());
        const auto& a = std::get<SviParams>(f.surface.slice(i));
        const auto& b = std::get<SviParams>(inserted.surface.slice(j));
        EXPECT_EQ(a.a, b.a) << "original slice at years=" << years;
    }
}

// ---------------------------------------------------------------------------
// Honest cost accounting -- see the header for why this is not "~= 1/N"
// ---------------------------------------------------------------------------

TEST(Incremental, UntouchedSlicesContributeExactlyZeroWorkToTheRefit) {
    // The always-true guarantee: an incremental update's cost is exactly the
    // refit slice's own standalone cost, regardless of how many OTHER
    // slices the surface has or how expensive they individually are.
    const auto f = make_fixture();
    ASSERT_GE(f.by_years.size(), 3u);

    for (auto& [years, qs] : f.by_years) {
        const auto standalone = calibrate_svi_slice(qs);
        const auto updated = apply_incremental_update(f.surface, years, qs);
        EXPECT_EQ(updated.fit.residual_evaluations, standalone.residual_evaluations)
            << "slice T=" << years;
    }
}

TEST(Incremental, TotalWorkForOneRefitIsStrictlyLessThanRefittingTheWholeSurface) {
    // Structurally always true (not a measurement of a specific ratio): one
    // slice's cost is one positive addend of the full refit's total, and
    // every other addend is strictly positive, so the sum strictly exceeds
    // any single term -- this holds no matter which slice is chosen or how
    // unevenly difficulty is distributed across slices.
    const auto f = make_fixture();
    ASSERT_GE(f.by_years.size(), 2u);
    for (auto& [years, qs] : f.by_years) {
        const auto updated = apply_incremental_update(f.surface, years, qs);
        EXPECT_LT(updated.fit.residual_evaluations, f.total_residual_evaluations)
            << "slice T=" << years;
    }
}

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

TEST(Incremental, TooFewQuotesDegradesTheSameWayAStandaloneFitWould) {
    const auto f = make_fixture();
    auto it = f.by_years.begin();
    const std::size_t n = std::min<std::size_t>(2, it->second.size());
    std::vector<OptionQuote> thin(it->second.begin(),
                                  it->second.begin() + static_cast<std::ptrdiff_t>(n));
    const auto standalone = calibrate_svi_slice(thin);
    const auto updated = apply_incremental_update(f.surface, it->first, thin);
    EXPECT_EQ(updated.fit.status, standalone.status);
}
