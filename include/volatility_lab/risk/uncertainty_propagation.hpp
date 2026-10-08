// SPDX-License-Identifier: MIT
#pragma once
/// \file uncertainty_propagation.hpp
/// \brief Propagating quote uncertainty through Greeks and the portfolio
///        (directive Phase 8, section 16) -- not merely attaching a
///        confidence number to a quote and stopping there.
///
/// ## The method: delta (first-order Taylor) propagation
///
/// `risk/uncertainty.hpp` already estimates `vol_std_error`, the 1-sigma
/// uncertainty in the surface's vol at one (k, T) point. Propagating that
/// into price/delta/vega uncertainty needs nothing new: it is exactly what
/// `greeks::OptionGreeks`'s own second-order fields already measure --
/// `d(price)/d(vol) = vega`, `d(delta)/d(vol) = vanna`,
/// `d(vega)/d(vol) = volga` -- so `price_std_error ~= |vega| * vol_std_error`,
/// `delta_std_error ~= |vanna| * vol_std_error`, and so on. This is the
/// standard first-order ("delta method") propagation of uncertainty through
/// a smooth function, reusing Greeks this project has already independently
/// validated against the double-double reference, not a new sensitivity
/// computed from scratch.
///
/// ## Why "large move" and "large move, low uncertainty" are different claims
///
/// The directive's own framing (section 16): a large PnL move with a tight
/// `vol_std_error` behind it is a real signal; the same move with a wide
/// `vol_std_error` -- sparse coverage, extrapolation, wide quote spreads --
/// is much less trustworthy, even though the two look identical from the
/// PnL number alone. `PositionUncertainty`/`PortfolioUncertainty` exist
/// specifically to carry that second number alongside the first, not to
/// replace it.
///
/// ## The one real modelling choice: correlation across positions
///
/// Combining per-leg uncertainties into one portfolio number requires a
/// choice about correlation, and a wrong one is actively misleading (both
/// "add them up" and "root-sum-square them" are wrong on their own: the
/// first overstates risk from genuinely independent sources, the second
/// understates it for legs that share the same underlying noise). The
/// choice made here, stated explicitly rather than buried in the formula:
/// positions sharing the **same expiry** are treated as fully correlated
/// (their vol uncertainty comes substantially from the same slice
/// calibration), so their dollar-vega-weighted contributions are *summed*
/// within an expiry bucket; different expiries are treated as independent
/// and combined in *quadrature* across buckets. This is the same
/// "per-bucket vega risk" approximation a trading desk already uses for
/// ordinary vega risk aggregation -- not a new, bespoke assumption
/// invented for this module.

#include <span>
#include <utility>
#include <vector>

#include "volatility_lab/greeks/greeks.hpp"
#include "volatility_lab/options/quote.hpp"
#include "volatility_lab/portfolio/portfolio.hpp"
#include "volatility_lab/risk/uncertainty.hpp"

namespace vl {

/// One position's Greeks, re-expressed as 1-sigma uncertainty via the delta
/// method against its own `vol_std_error`.
struct PositionGreekUncertainty {
    double vol_std_error = 0.0;    ///< the input this was propagated from
    double price_std_error = 0.0;  ///< |vega| * vol_std_error
    double delta_std_error = 0.0;  ///< |vanna| * vol_std_error
    double vega_std_error = 0.0;   ///< |volga| * vol_std_error
};

[[nodiscard]] PositionGreekUncertainty propagate_greek_uncertainty(const OptionGreeks& greeks,
                                                                   double vol_std_error) noexcept;

/// Portfolio-level uncertainty: dollar price/delta uncertainty, combined
/// across positions using the same-expiry-correlated /
/// different-expiry-independent rule described in the file comment.
struct PortfolioUncertainty {
    double price_std_error = 0.0;
    double delta_std_error = 0.0;

    /// Per-expiry breakdown (years, dollar-vega-weighted price std error
    /// contribution from that expiry alone, before cross-expiry
    /// combination) -- the "why" behind the headline number, not just the
    /// number itself.
    std::vector<std::pair<double, double>> price_contribution_by_expiry;

    /// True if every position's uncertainty estimate had local coverage
    /// (`PointUncertainty::no_local_coverage == false`); false means at
    /// least one leg's uncertainty is formally infinite and the headline
    /// numbers above are not meaningful -- reported rather than silently
    /// producing an infinite or NaN portfolio number.
    bool all_positions_had_coverage = true;
};

/// Propagates quote uncertainty through every position's Greeks and
/// combines them into a portfolio-level number.
///
/// `valuations` must be `value_portfolio`'s own output for `positions`
/// against `surface` -- the Greeks used here must be the ones actually
/// priced from, not independently recomputed, so attribution and
/// uncertainty never disagree about which Greeks they are describing.
[[nodiscard]] PortfolioUncertainty propagate_portfolio_uncertainty(
    std::span<const Position> positions, std::span<const PositionValuation> valuations,
    const VolSurface& surface, std::span<const OptionQuote> quotes,
    const UncertaintyConfig& cfg = {});

}  // namespace vl
