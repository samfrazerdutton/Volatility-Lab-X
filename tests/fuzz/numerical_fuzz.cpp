// SPDX-License-Identifier: MIT
/// \file numerical_fuzz.cpp
/// \brief Randomised, reproducible fuzzing of the numerical boundary
///        (directive section 26): the goal is not "does it crash" alone,
///        but "does it fail predictably, preserve its own invariants, and
///        report a useful diagnostic rather than a silently plausible
///        wrong answer."
///
/// Seeded with a fixed `std::mt19937_64` seed, not a time-based one: a
/// fuzz failure that cannot be reproduced on the next run is far less
/// useful than one that can, and this project's whole numerical-validation
/// discipline is built on reproducibility.

#include "vl_test_support.hpp"

#include "volatility_lab/calibration/svi_calibrator.hpp"
#include "volatility_lab/greeks/greeks.hpp"
#include "volatility_lab/pricing/implied_vol.hpp"
#include "volatility_lab/volatility/svi.hpp"

#include <cmath>
#include <limits>
#include <random>

using namespace vl;

namespace {

constexpr std::uint64_t kFuzzSeed = 0x564f4c585f465a5aULL;  // "VOLX_FZZ" in hex, arbitrary but fixed
constexpr int kFuzzIterations = 20000;

/// Draws from a mix of "plausible but extreme" and "outright pathological"
/// ranges -- the fuzzer's job is to cross the boundary between the two
/// without the caller choosing exactly where that boundary is.
double draw_extreme(std::mt19937_64& rng, double plausible_lo, double plausible_hi) {
    std::uniform_real_distribution<double> mode(0.0, 1.0);
    const double m = mode(rng);
    if (m < 0.70) {
        std::uniform_real_distribution<double> d(plausible_lo, plausible_hi);
        return d(rng);
    }
    if (m < 0.80) return 0.0;
    if (m < 0.85) return -1.0 * plausible_hi;
    if (m < 0.90) return std::numeric_limits<double>::infinity();
    if (m < 0.95) return -std::numeric_limits<double>::infinity();
    if (m < 0.98) return std::numeric_limits<double>::quiet_NaN();
    std::uniform_real_distribution<double> huge(1e10, 1e300);
    return huge(rng);
}

}  // namespace

// ---------------------------------------------------------------------------
// black_scholes_greeks: never crashes, NaN in -> NaN out, finite in -> finite
// or a well-defined degenerate result, never a silent garbage finite value
// ---------------------------------------------------------------------------

TEST(Fuzz, BlackScholesGreeksNeverCrashesAndPropagatesNanPredictably) {
    std::mt19937_64 rng(kFuzzSeed);
    long nan_in_cases = 0, finite_in_cases = 0;
    for (int i = 0; i < kFuzzIterations; ++i) {
        const double spot = draw_extreme(rng, 1.0, 1000.0);
        const double strike = draw_extreme(rng, 1.0, 1000.0);
        const double vol = draw_extreme(rng, 0.01, 3.0);
        const double years = draw_extreme(rng, 1.0 / 365.0, 10.0);
        const double rate = draw_extreme(rng, -0.1, 0.2);
        const double carry = draw_extreme(rng, -0.05, 0.1);
        const OptionType type = (i % 2 == 0) ? OptionType::Call : OptionType::Put;

        // The call itself must not crash -- if it does, the test binary
        // aborts and this assertion is never reached, which is itself the
        // failure signal.
        const auto g = black_scholes_greeks(spot, strike, vol, years, rate, carry, type);

        const bool any_nan_in = std::isnan(spot) || std::isnan(strike) || std::isnan(vol) ||
                                std::isnan(years) || std::isnan(rate) || std::isnan(carry);
        if (any_nan_in) {
            ++nan_in_cases;
            // At least the price must say "I don't know" rather than a
            // plausible-looking number manufactured from a NaN.
            EXPECT_TRUE(std::isnan(g.price))
                << "NaN input produced a non-NaN price=" << g.price << " spot=" << spot
                << " strike=" << strike << " vol=" << vol << " years=" << years
                << " rate=" << rate << " carry=" << carry;
        } else if (spot > 0.0 && spot < 1e6 && strike > 0.0 && strike < 1e6 && vol > 0.0 &&
                  vol < 10.0 && years > 0.0 && years < 100.0 && std::abs(rate) < 1.0 &&
                  std::abs(carry) < 1.0) {
            // "Well-posed" means realistically bounded, not merely
            // IEEE-finite: an earlier version of this test used
            // std::isfinite() alone and failed on inputs like
            // vol=5e299/years=9e299 -- technically finite doubles, but
            // values no real market or model input is ever close to, and
            // ordinary (correct) floating-point arithmetic legitimately
            // overflows to inf/NaN when fed numbers in that range (e.g.
            // exp(rate*years) with years~1e299 overflows on its own,
            // independent of anything this library does). That failure
            // was this test's own wrong assumption, not a production bug
            // -- fixed by bounding "well-posed" to the realistic range
            // draw_extreme's own "plausible" branch actually draws from,
            // with headroom, rather than accepting its "huge" branch's
            // output as something a sane price is still owed for.
            ++finite_in_cases;
            EXPECT_TRUE(std::isfinite(g.price))
                << "well-posed input produced a non-finite price: spot=" << spot
                << " strike=" << strike << " vol=" << vol << " years=" << years
                << " rate=" << rate << " carry=" << carry;
            EXPECT_GE(g.price, 0.0) << "a vanilla option's price must never be negative";
        }
        // Infinite or zero/negative inputs are deliberately NOT asserted on
        // here beyond "did not crash" -- e.g. spot=inf or vol=0 are
        // legitimate degenerate cases this project's own tests
        // (Black.HighVarianceDoesNotOverflow, Black.ZeroVarianceGivesIntrinsic)
        // already cover with specific expected behaviour; this fuzz test's
        // job is breadth across the *boundary*, not re-asserting what is
        // already pinned precisely elsewhere.
    }
    EXPECT_GT(nan_in_cases, 0) << "the fuzzer never drew a NaN input in "
                              << kFuzzIterations << " iterations -- check draw_extreme";
    EXPECT_GT(finite_in_cases, 0) << "the fuzzer never drew an all-well-posed input";
}

// ---------------------------------------------------------------------------
// implied_volatility: never crashes; a rejected input carries a real status,
// not a NaN masquerading as success
// ---------------------------------------------------------------------------

TEST(Fuzz, ImpliedVolatilityNeverCrashesAndRejectsRatherThanGuesses) {
    std::mt19937_64 rng(kFuzzSeed ^ 0x1234567890ABCDEFULL);
    for (int i = 0; i < kFuzzIterations; ++i) {
        const double price = draw_extreme(rng, 0.01, 500.0);
        const double forward = draw_extreme(rng, 1.0, 1000.0);
        const double strike = draw_extreme(rng, 1.0, 1000.0);
        const double years = draw_extreme(rng, 1.0 / 365.0, 10.0);
        const double discount = draw_extreme(rng, 0.5, 1.0);
        const OptionType type = (i % 2 == 0) ? OptionType::Call : OptionType::Put;

        const auto result = implied_volatility(price, forward, strike, years, discount, type);

        // Whatever the status, the result must be internally consistent:
        // a result claiming success must have a finite, non-negative
        // volatility; anything else must say so via its status rather than
        // leaving a stale or fabricated volatility value.
        if (result.status == IvStatus::Ok) {
            EXPECT_TRUE(std::isfinite(result.total_volatility))
                << "status Ok but non-finite volatility";
            EXPECT_GE(result.total_volatility, 0.0) << "status Ok but negative volatility";
        }
    }
}

// ---------------------------------------------------------------------------
// calibrate_svi_slice: never crashes on a randomly-assembled (and often
// nonsensical) quote set; a reported Ok status implies admissible params
// ---------------------------------------------------------------------------

TEST(Fuzz, SviCalibratorNeverCrashesAndOkStatusImpliesAdmissibleParams) {
    std::mt19937_64 rng(kFuzzSeed ^ 0xFEDCBA0987654321ULL);
    std::uniform_int_distribution<int> count_dist(0, 15);

    for (int trial = 0; trial < 500; ++trial) {
        const int n = count_dist(rng);
        std::vector<OptionQuote> quotes;
        quotes.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            OptionQuote q;
            q.log_moneyness = draw_extreme(rng, -1.0, 1.0);
            q.years = draw_extreme(rng, 0.01, 3.0);
            q.forward = draw_extreme(rng, 1.0, 1000.0);
            q.strike = q.forward * std::exp(q.log_moneyness);
            q.discount = 1.0;
            q.total_variance = draw_extreme(rng, 0.0001, 2.0);
            q.implied_vol =
                (q.total_variance > 0.0 && q.years > 0.0 && std::isfinite(q.total_variance))
                    ? std::sqrt(q.total_variance / q.years)
                    : 0.0;
            q.weight = draw_extreme(rng, 0.0, 1.0);
            q.status = (i % 4 == 0) ? QuoteStatus::Rejected : QuoteStatus::Ok;
            quotes.push_back(q);
        }

        // Must not crash regardless of how nonsensical `quotes` is.
        const auto fit = calibrate_svi_slice(quotes);

        if (fit.status == SviFitStatus::Ok) {
            EXPECT_TRUE(svi_parameters_admissible(fit.params))
                << "status Ok but the fitted parameters are not admissible";
            EXPECT_TRUE(std::isfinite(fit.objective)) << "status Ok but non-finite objective";
        }
    }
    // Not asserting how many trials reported Ok: most random draws here
    // are nonsensical by construction (that is the point), so
    // TooFewQuotes/DegradedToFlat/NotAdmissible dominating is expected and
    // healthy, not a test weakness.
}
