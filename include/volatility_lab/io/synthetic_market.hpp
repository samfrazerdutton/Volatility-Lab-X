// SPDX-License-Identifier: MIT
#pragma once
/// \file synthetic_market.hpp
/// \brief Deterministic synthetic market generator.
///
/// ## Why this exists and what it must guarantee
///
/// No proprietary market data may be used, so every test, benchmark and demo
/// in this project runs on generated data.  That places two hard requirements
/// on the generator, and the second is the one usually missed:
///
///  1. **Reproducible from a seed.**  A benchmark whose input changes between
///     runs is not a benchmark, and a calibration regression that cannot be
///     replayed cannot be fixed.  Every datum here is a pure function of the
///     config and the seed -- including the quote *ordering*, which is why the
///     generator fills strikes in a fixed nested loop rather than from a set.
///
///  2. **Honest about being synthetic.**  The generator produces data from a
///     *known* surface, which makes it tempting to use for validating the
///     calibrator -- "we recover the parameters we put in".  That is a
///     legitimate test and the suite uses it, but it is a weak one: it says
///     the calibrator can invert its own model, not that it can fit a market.
///     So the generator deliberately includes effects the slice models cannot
///     represent exactly (a Student-t quote noise, discrete tick rounding, a
///     bid/ask structure that widens non-linearly in the wings), and the
///     calibration tests assert fit *quality* rather than exact parameter
///     recovery wherever those are switched on.
///
/// ## Regimes
///
/// Six named regimes, each a different stress on the pipeline rather than
/// merely a different number:
///
///  * `Normal`    -- liquid index surface, moderate skew.  The baseline.
///  * `HighVol`   -- level shifted up; tests that nothing assumes sigma < 1.
///  * `Crash`     -- steep put skew, wide spreads, thin wings.  The case where
///                   SVI's wing slopes approach the Lee bound.
///  * `VolCrush`  -- very low vol, very short dates.  The small-total-variance
///                   corner of the pricer, and where the IV inversion is most
///                   ill-conditioned.
///  * `Earnings`  -- a single expiry with a variance bump, breaking the smooth
///                   term structure.  SSVI cannot fit this and should not
///                   pretend to; the quality report must show it.
///  * `Illiquid`  -- sparse strikes, zero bids, stale quotes, crossed markets.
///                   Exercises the normalisation layer rather than the maths.

#include <cstdint>
#include <string>
#include <vector>

#include "volatility_lab/options/quote.hpp"
#include "volatility_lab/volatility/surface.hpp"

namespace vl {

enum class MarketRegime : std::uint8_t {
    Normal = 0,
    HighVol = 1,
    Crash = 2,
    VolCrush = 3,
    Earnings = 4,
    Illiquid = 5
};

[[nodiscard]] constexpr const char* to_string(MarketRegime r) noexcept {
    switch (r) {
        case MarketRegime::Normal:   return "normal";
        case MarketRegime::HighVol:  return "high-vol";
        case MarketRegime::Crash:    return "crash";
        case MarketRegime::VolCrush: return "vol-crush";
        case MarketRegime::Earnings: return "earnings";
        case MarketRegime::Illiquid: return "illiquid";
    }
    return "?";
}

[[nodiscard]] bool parse_regime(std::string_view name, MarketRegime& out) noexcept;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct SyntheticMarketConfig {
    MarketRegime regime = MarketRegime::Normal;
    std::uint64_t seed = 20260207u;

    std::string underlying = "SYN";
    double spot = 100.0;
    double rate = 0.04;
    double dividend = 0.015;

    /// Expiries in years.  Empty means "use the regime's default ladder".
    std::vector<double> expiries;

    /// Strikes are placed at fixed *standard deviations* from the forward, not
    /// at fixed percentages.  That is what a listed market approximates and it
    /// matters for the fit: a fixed-percentage ladder puts almost all its
    /// strikes inside one standard deviation at long maturities and outside
    /// five at short ones, so a calibrator tested on it never sees a realistic
    /// strike distribution.
    double strike_span_sd = 3.0;
    int strikes_per_expiry = 21;

    /// Round strikes to this increment, as a listed market does.  Zero means
    /// no rounding.  Switched on by default because strike rounding is what
    /// makes the ladder non-uniform in log-moneyness, which is a real
    /// complication for interpolation.
    double strike_increment = 2.5;

    /// Quote noise, in volatility points, applied to the *implied volatility*
    /// before pricing.  Drawn from a Student-t with 4 degrees of freedom
    /// rather than a Gaussian: real quote errors have heavy tails, and a
    /// calibrator tuned against Gaussian noise is over-confident about
    /// outliers.
    double vol_noise = 0.0025;
    double noise_degrees_of_freedom = 4.0;

    /// Bid/ask half-spread in volatility points, at the money.  The spread
    /// widens with |k| and shrinks with maturity; see the implementation.
    double atm_half_spread_vol = 0.004;

    /// Round prices to this tick.  Non-zero by default: tick rounding is a
    /// real and significant source of implied-volatility noise in the wings,
    /// where a one-tick move is a large relative change.
    double price_tick = 0.01;

    /// Probability that a quote is damaged, in the Illiquid regime.  Zero in
    /// the others.
    double corrupt_fraction = 0.0;

    /// Emit implied volatilities directly instead of prices, bypassing the
    /// inversion.  Used by tests that want to isolate the calibrator from the
    /// solver.
    bool emit_vols_not_prices = false;
};

/// The config a named regime implies.  Exposed so the CLI can print it and a
/// test can override one field without rebuilding the rest.
[[nodiscard]] SyntheticMarketConfig regime_defaults(MarketRegime r,
                                                    std::uint64_t seed = 20260207u);

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/// The snapshot, plus the surface it was generated from.
///
/// Returning the true surface is what makes the generator useful for
/// validation: a calibration test can compare the fitted surface against the
/// one the data actually came from, rather than only against the quotes.  It
/// is also what makes it easy to cheat, so the tests that use it say so.
struct SyntheticMarket {
    MarketSnapshot snapshot;
    VolSurface true_surface;
    SyntheticMarketConfig config;

    /// The exact SSVI parameters used, when the regime is generated from SSVI.
    /// `Earnings` perturbs one slice away from the SSVI surface, so the
    /// parameters no longer describe it exactly -- flagged by
    /// `surface_is_exactly_ssvi`.
    SsviParams ssvi;
    bool surface_is_exactly_ssvi = true;
};

/// Generate a snapshot.  Deterministic in `config` alone.
[[nodiscard]] SyntheticMarket generate_market(const SyntheticMarketConfig& cfg);

/// Convenience: generate from a named regime and seed.
[[nodiscard]] SyntheticMarket generate_market(MarketRegime regime,
                                              std::uint64_t seed = 20260207u);

// ---------------------------------------------------------------------------
// The random engine
// ---------------------------------------------------------------------------

/// A small, explicit PRNG.
///
/// `std::mt19937_64` would do, but the *distributions* in `<random>` are not
/// specified to produce identical output across standard-library
/// implementations -- `std::normal_distribution` in particular differs between
/// libstdc++, libc++ and the MSVC STL.  Since this project's determinism tests
/// assert reproducibility across the three toolchains it builds on, the
/// distributions have to be ours.  The generator is splitmix64, which is
/// three lines, has no state beyond a single word, and passes the usual
/// statistical batteries at this scale.
class DeterministicRng {
  public:
    explicit DeterministicRng(std::uint64_t seed) noexcept : state_(seed) {}

    [[nodiscard]] std::uint64_t next_u64() noexcept;

    /// Uniform in [0, 1).  53-bit resolution, never exactly 1.
    [[nodiscard]] double uniform() noexcept;
    [[nodiscard]] double uniform(double lo, double hi) noexcept;

    /// Standard normal, by the Box-Muller transform.  Written out rather than
    /// taken from `<random>` so the output is identical on every platform.
    [[nodiscard]] double normal() noexcept;

    /// Student-t with `nu` degrees of freedom, as a normal scaled by an
    /// inverse-chi distribution.  Used for quote noise because real quote
    /// errors have heavy tails.
    [[nodiscard]] double student_t(double nu) noexcept;

    [[nodiscard]] bool bernoulli(double p) noexcept { return uniform() < p; }

  private:
    std::uint64_t state_;
    double spare_normal_ = 0.0;
    bool has_spare_ = false;
};

}  // namespace vl
