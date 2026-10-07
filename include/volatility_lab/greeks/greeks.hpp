// SPDX-License-Identifier: MIT
#pragma once
/// \file greeks.hpp
/// \brief Closed-form spot-measure Greeks, single-instrument and batch, plus
///        the Taylor-vs-exact repricing diagnostic (brief section 9).
///
/// ## Why closed form, and why direct in d1/d2 rather than through the
/// normalised-Black folding
///
/// `pricing/black.hpp` prices through a folded, always-OTM normalised form
/// `b(x,s)` specifically because *price* has a cancellation catastrophe at
/// small total variance that the fold removes.  Greeks do not have that
/// problem: `Phi(d1)`, `phi(d1)` and the other pieces below are each a single
/// well-scaled quantity, not a difference of two nearly-equal ones, so there
/// is nothing to fold around.  Differentiating the folded form symbolically
/// would also require accounting for the kink the fold and the intrinsic term
/// have at F == K (which cancel in the price but have to be tracked through
/// every derivative), for no numerical benefit.  So this file works directly
/// from the standard
///
///     d1 = (ln(F/K) + s^2/2) / s,   d2 = d1 - s,   s = sigma*sqrt(T)
///
/// using the already-validated `norm_cdf`/`norm_pdf`, and is checked against
/// an *independent* double-double reference built the same way
/// (`reference::black_scholes_greeks_ref`) rather than against anything that
/// shares this derivation.
///
/// ## Sign and measure conventions
///
///  * `w = payoff_sign(type)` (+1 call, -1 put) is used throughout exactly as
///    in `core/types.hpp`, so every formula below is one branch-free
///    expression instead of a call/put switch.
///  * Rates and carry are continuously compounded; `F = S*exp((r-q)T)`.
///  * **Theta is the calendar-decay convention**, `d/dt = -d/dT`: a long
///    option's theta is (usually) negative, matching what a trading desk
///    reads off a risk screen.  `pricing/reference.hpp`'s *other* Greeks
///    struct, `ReferenceGreeks`, instead reports `dU/dT` directly on the
///    undiscounted forward price with the opposite sign convention -- the two
///    structs answer different questions and are not meant to be compared
///    field-for-field.
///  * Vega and volga are "per unit volatility" (d/dsigma with sigma in
///    decimal, e.g. 0.20 for 20%), not "per vol point"; divide by 100 at the
///    call site if that convention is wanted. Rho is per unit rate, same
///    reasoning.
///
/// ## What is new relative to first-order Greeks
///
/// `vanna`, `volga` (a.k.a. vomma), `charm` and `speed` are the second-order
/// and mixed terms the brief (section 19) calls volatility risk and
/// extrapolation risk.  They matter for exactly the reason section 9 of the
/// brief states: a first-order Greek book is a Taylor expansion, and the
/// Taylor expansion's own error -- not just its first two terms -- is
/// something a risk system has to be able to quantify.  See
/// `taylor_vs_exact_reprice` below.

#include <cstdint>
#include <span>

#include "volatility_lab/core/types.hpp"

namespace vl {

/// Spot-measure Greeks for one European option.  Field-for-field compatible
/// with `reference::SpotGreeks`, which is what validates this struct.
struct OptionGreeks {
    double price = 0.0;
    double delta = 0.0;  ///< dPrice/dS
    double gamma = 0.0;  ///< d2Price/dS2
    double vega = 0.0;   ///< dPrice/dsigma
    double theta = 0.0;  ///< dPrice/dt = -dPrice/dT (calendar decay)
    double rho = 0.0;    ///< dPrice/dr
    double vanna = 0.0;  ///< d2Price/(dS dsigma)
    double volga = 0.0;  ///< d2Price/dsigma2  (a.k.a. vomma)
    double charm = 0.0;  ///< dDelta/dt = -d2Price/(dS dT)
    double speed = 0.0;  ///< d3Price/dS3
};

/// All ten Greeks in one evaluation, sharing the d1/d2/common-factor
/// computation.  This is the one real implementation; every other overload in
/// this file funnels into it.
[[nodiscard]] OptionGreeks black_scholes_greeks(double spot, double strike, double vol,
                                                double years, double rate, double carry,
                                                OptionType type) noexcept;

/// Convenience overload taking a forward directly (carry folded in), for
/// callers who already have `F` from a surface query and a discount factor
/// rather than a raw spot and a rate/carry pair.  `rate` is still needed
/// because rho and theta's discounting term depend on it even when the
/// forward itself is supplied.
[[nodiscard]] OptionGreeks black_scholes_greeks_forward(double forward, double strike,
                                                        double vol, double years,
                                                        double rate, double discount,
                                                        OptionType type) noexcept;

// ---------------------------------------------------------------------------
// Batch
// ---------------------------------------------------------------------------

/// Greeks for every option in a parallel-column batch, in one call.  Plain
/// loop over the scalar formula -- each option's Greeks are independent, so
/// there is no batch-specific algorithm to speak of -- but having one entry
/// point here is what the SIMD phase can replace without touching any caller.
void black_scholes_greeks_batch(std::span<const double> spot, std::span<const double> strike,
                                std::span<const double> vol, std::span<const double> years,
                                std::span<const double> rate, std::span<const double> carry,
                                std::span<const std::int8_t> type_sign,
                                std::span<OptionGreeks> out) noexcept;

// ---------------------------------------------------------------------------
// Portfolio aggregation
// ---------------------------------------------------------------------------

/// Dollar Greeks summed over a book: `quantity[i] * multiplier[i] * greek[i]`.
///
/// This is the aggregation primitive both the portfolio module (position
/// sizing, PnL attribution) and the scenario engine build on.  It is kept
/// here, next to `OptionGreeks`, rather than in `portfolio/`, because it has
/// no concept of a "position" -- just three parallel arrays -- so it has no
/// reason to depend on anything portfolio-shaped.
struct PortfolioGreeks {
    double value = 0.0;  ///< sum(quantity * multiplier * price)
    double delta = 0.0;
    double gamma = 0.0;
    double vega = 0.0;
    double theta = 0.0;
    double rho = 0.0;
    double vanna = 0.0;
    double volga = 0.0;
    double charm = 0.0;
    double speed = 0.0;
};

[[nodiscard]] PortfolioGreeks aggregate_greeks(std::span<const OptionGreeks> greeks,
                                               std::span<const double> quantity,
                                               std::span<const double> multiplier) noexcept;

// ---------------------------------------------------------------------------
// Taylor approximation vs exact repricing (brief section 9)
// ---------------------------------------------------------------------------

/// One scenario's bump to the four axes a Greek book is sensitive to.  Zero in
/// every field is "no change"; this is intentionally the same shape as
/// `scenarios/scenario.hpp`'s per-axis shifts (added in a later phase) so a
/// scenario can be fed to both the Taylor estimator and the exact repricer
/// without translation.
struct GreekBump {
    double d_spot = 0.0;   ///< absolute shift in S
    double d_vol = 0.0;    ///< absolute shift in sigma
    double d_rate = 0.0;   ///< absolute shift in r
    double d_years = 0.0;  ///< absolute shift in T (negative = time passing)
};

/// Comparison of a Taylor-series PnL estimate against the exact repriced PnL
/// for one instrument under one bump, at increasing approximation order.
///
/// `order1`, `order2` are cumulative (order2 includes the order-1 terms), so
/// `order2 - order1` is exactly "how much the second-order terms added", and
/// `exact - order2` is exactly "how much repricing found that no finite
/// Taylor order captured" -- the two numbers the brief's "risk model error"
/// concept (section 9) is built from.
struct TaylorVsExact {
    double base_price = 0.0;
    double exact_price = 0.0;
    double exact_pnl = 0.0;

    double order1_pnl = 0.0;  ///< delta*dS + vega*dvol + theta*dT + rho*dr
    double order2_pnl = 0.0;  ///< order1 + 1/2 gamma dS^2 + 1/2 volga dvol^2
                              ///<        + vanna dS dvol + charm dS dT

    [[nodiscard]] double order1_error() const noexcept { return exact_pnl - order1_pnl; }
    [[nodiscard]] double order2_error() const noexcept { return exact_pnl - order2_pnl; }

    /// Error as a fraction of the exact move, guarding a near-zero move so
    /// the ratio does not explode on a trivial bump.
    [[nodiscard]] double order1_relative_error() const noexcept;
    [[nodiscard]] double order2_relative_error() const noexcept;
};

/// Compare the Taylor estimate (built from `greeks`, evaluated at the base
/// point) against the exact repriced value under `bump`, for one European
/// option.  `spot, strike, vol, years, rate, carry, type` describe the base
/// point `greeks` was evaluated at; passing a `greeks` from a different point
/// than the other arguments is a caller error (not checked here -- this is a
/// diagnostic, not a public safety boundary).
[[nodiscard]] TaylorVsExact taylor_vs_exact_reprice(const OptionGreeks& greeks, double spot,
                                                    double strike, double vol, double years,
                                                    double rate, double carry,
                                                    OptionType type,
                                                    const GreekBump& bump) noexcept;

}  // namespace vl
