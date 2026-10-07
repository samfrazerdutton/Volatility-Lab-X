// SPDX-License-Identifier: MIT
#pragma once
/// \file portfolio.hpp
/// \brief Positions, portfolio-level Greeks, and PnL attribution that
///        reconciles exactly (brief section 10).
///
/// ## Scope
///
/// A `Position` is a European option leg: the static contract terms
/// (strike, expiry, type, quantity, contract multiplier) plus a label.  It
/// deliberately does not carry a volatility of its own -- every leg's
/// implied vol comes from a `VolSurface` query at valuation time, which is
/// what makes "reprice this whole book under a different surface" (the
/// scenario engine's job, and this module's PnL-attribution job) a matter of
/// supplying a different surface rather than editing every position.
///
/// This phase assumes a single underlier per portfolio (one spot, one rate,
/// one carry, shared across legs) -- cross-underlier books are Phase 15 of
/// the roadmap and build on this rather than replacing it.
///
/// ## PnL attribution: the exact-reconciliation requirement
///
/// Section 10 of the brief requires
///
///     Total PnL  ==  Σ attribution_components + residual     (exactly)
///
/// and explicitly forbids discarding the residual.  The way this is made
/// exact rather than approximate: the residual is *defined* as whatever the
/// second-order Taylor attribution did not capture, i.e.
///
///     residual := exact_pnl - (order1_components + order2_components)
///
/// so the reconciliation identity holds by construction, to floating-point
/// precision, for every portfolio and every market transition -- there is no
/// approximation left unaccounted for, only a number that says how large the
/// approximation's error was and, per leg, exactly where it came from (via
/// `PnLAttribution::leg_residual`).
///
/// The individual components (spot, vol, rate, theta, and the cross terms
/// vanna/volga/charm) are exactly the quantities `greeks::taylor_vs_exact_reprice`
/// already computes for one instrument; this module is the aggregation of
/// that across a book, weighted by quantity and multiplier, with each leg's
/// own vol bump read off two different surface snapshots instead of a single
/// shared `GreekBump`.

#include <span>
#include <string>
#include <vector>

#include "volatility_lab/core/types.hpp"
#include "volatility_lab/greeks/greeks.hpp"
#include "volatility_lab/volatility/surface.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

struct Position {
    std::string label;
    double quantity = 0.0;    ///< signed number of contracts
    double multiplier = 1.0;  ///< contract size, e.g. 100 for listed equity options
    double strike = 0.0;
    double years = 0.0;  ///< time to expiry as of the valuation date
    OptionType type = OptionType::Call;

    [[nodiscard]] double signed_notional() const noexcept { return quantity * multiplier; }
};

/// A fully resolved market point for a single-underlier book: everything a
/// position needs besides its own contract terms and the surface's implied
/// vol at its own moneyness.
struct MarketPoint {
    double spot = 0.0;
    double rate = 0.0;
    double carry = 0.0;  ///< dividend / borrow yield
};

// ---------------------------------------------------------------------------
// Valuation
// ---------------------------------------------------------------------------

/// Greeks for one position, read off `surface` at the position's own
/// (log-moneyness, years).  `vol_used` is reported alongside the Greeks
/// because it is the one input that came from the surface rather than from
/// the position or the market point, and attribution needs to know it to
/// form each leg's volatility bump between two snapshots.
struct PositionValuation {
    OptionGreeks greeks;
    double vol_used = 0.0;
    double forward = 0.0;
    double log_moneyness = 0.0;
};

[[nodiscard]] PositionValuation value_position(const Position& p, const VolSurface& surface,
                                               const MarketPoint& market) noexcept;

/// All positions in one call.  A plain loop -- valuation is independent
/// per-leg -- kept here so the parallel/incremental phases have one entry
/// point to replace without touching any caller.
void value_portfolio(std::span<const Position> positions, const VolSurface& surface,
                     const MarketPoint& market, std::span<PositionValuation> out) noexcept;

/// Dollar Greeks for the whole book at one market point/surface.
[[nodiscard]] PortfolioGreeks portfolio_greeks(std::span<const Position> positions,
                                               const VolSurface& surface,
                                               const MarketPoint& market) noexcept;

// ---------------------------------------------------------------------------
// PnL attribution
// ---------------------------------------------------------------------------

/// Portfolio-level decomposition of the PnL between two (surface, market)
/// snapshots.  Every *_pnl field here is a dollar amount (already multiplied
/// by quantity and contract multiplier, summed across the book).
///
/// `residual` is never dropped and is exactly
/// `total_exact_pnl - (spot_pnl + vol_pnl + rate_pnl + theta_pnl + gamma_pnl +
///  vanna_pnl + volga_pnl + charm_pnl)` by construction -- see the file
/// comment.
struct PnLAttribution {
    double total_exact_pnl = 0.0;  ///< Σ qty*mult*(exact reprice - base price)
    double base_value = 0.0;
    double new_value = 0.0;

    // First order.
    double spot_pnl = 0.0;   ///< Σ qty*mult*delta*dS
    double vol_pnl = 0.0;    ///< Σ qty*mult*vega*dvol   (per-leg dvol)
    double rate_pnl = 0.0;   ///< Σ qty*mult*rho*dr
    double theta_pnl = 0.0;  ///< Σ qty*mult*theta*dt

    // Second order / cross terms.
    double gamma_pnl = 0.0;  ///< Σ qty*mult*0.5*gamma*dS^2
    double vanna_pnl = 0.0;  ///< Σ qty*mult*vanna*dS*dvol
    double volga_pnl = 0.0;  ///< Σ qty*mult*0.5*volga*dvol^2
    double charm_pnl = 0.0;  ///< Σ qty*mult*charm*dS*dt

    /// Whatever the second-order Taylor expansion did not capture.  Defined
    /// so the reconciliation identity below holds exactly; never zeroed out
    /// or dropped to make a report look cleaner.
    double residual = 0.0;

    /// Per-leg exact PnL and residual, in the same order as the positions
    /// passed to `compute_pnl_attribution` -- the "where did the unexplained
    /// PnL come from" answer, not just the "how much" one.
    std::vector<double> leg_exact_pnl;
    std::vector<double> leg_residual;

    [[nodiscard]] double explained_pnl() const noexcept { return total_exact_pnl - residual; }

    /// Fraction of |total PnL| that the attribution explains.  This is the
    /// "risk explanation coverage" metric (brief section 49): 0.95 means 95%
    /// of the move is explained by the attributed components, 5% is residual.
    /// Guards a near-zero total so the ratio does not explode on a
    /// economically trivial move.
    [[nodiscard]] double explained_fraction() const noexcept;
};

/// Compute the attribution between two full snapshots of the same book.
///
/// `positions` is the same book at both ends -- a position that is closed
/// out between snapshots should have quantity 0 in one of the two framings
/// supplied by the caller (not handled specially here: a vanished position
/// contributes 0 to every component on the side it is absent from).
/// `new_years`, if empty, reuses each position's own `years` (same-day
/// comparison, dt = 0); supplying it lets a caller model time passing
/// between snapshots (dt = new_years[i] - positions[i].years, typically
/// negative).
[[nodiscard]] PnLAttribution compute_pnl_attribution(std::span<const Position> positions,
                                                      const VolSurface& base_surface,
                                                      const MarketPoint& base_market,
                                                      const VolSurface& new_surface,
                                                      const MarketPoint& new_market,
                                                      std::span<const double> new_years = {});

// ---------------------------------------------------------------------------
// Research-query style helpers
// ---------------------------------------------------------------------------

/// The `n` positions with the largest |exact PnL| from a `PnLAttribution`,
/// as indices into the original `positions` span -- the direct answer to
/// "which positions drove this move" (brief section 26).
[[nodiscard]] std::vector<std::size_t> largest_pnl_contributors(const PnLAttribution& attr,
                                                                std::size_t n);

}  // namespace vl
