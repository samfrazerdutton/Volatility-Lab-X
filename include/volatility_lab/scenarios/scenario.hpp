// SPDX-License-Identifier: MIT
#pragma once
/// \file scenario.hpp
/// \brief The scenario graph: named market shocks, applied consistently, and
///        reconciled against the portfolio and surface-differential engines.
///
/// ## What ties together here
///
/// Two phases already exist independently: the surface differential engine
/// (`diagnostics/surface_differential.hpp`) *measures* a move between two
/// surfaces as {level, skew, curvature, term, forward, event} shifts in
/// annualised-vol space; the portfolio module (`portfolio/portfolio.hpp`)
/// *reprices* a book exactly between two (surface, market) snapshots and
/// reconciles that exact PnL against a Taylor approximation. A scenario is
/// the missing third piece: a named shock that *specifies* target
/// {level, skew, curvature, term, spot} values, constructs the (surface,
/// market) pair that realises them, and feeds that pair to both of the
/// existing engines.
///
/// That the shock is specified in exactly the same four-coefficient
/// vol-space basis the differential engine regresses in is deliberate, not
/// incidental: it means a scenario's own effect can be cross-checked by
/// measuring it back with `compute_surface_differential` and comparing
/// against what was asked for -- see
/// `tests/scenarios/scenario.cpp`'s `RealizedShockMatchesTheRequestedShock...`
/// tests, which do exactly that rather than trusting construction blindly.
///
/// ## How the shocked surface is actually built
///
/// `VolSurface` is parametric (SVI/SSVI) slice by slice, not a raw grid, so
/// there is no single "add this number to the surface" operation on the
/// slices themselves. Instead, for each of the base surface's own expiries,
/// the shock is evaluated as an additive change in *annualised volatility*
/// -- `level + skew*k + curvature*k^2 + term*T` -- on top of that slice's
/// own `vol(k, T)` at a dense moneyness grid, and the result becomes a new
/// `GridSlice` (one of `SliceVariant`'s own alternatives, see
/// `volatility/surface.hpp`) at that same expiry. This is exact by
/// construction at every grid point; `compute_surface_differential` later
/// recovering it is what confirms the construction and the regression agree
/// on what "level" etc. mean, rather than two independent, driftable
/// conventions.
///
/// ## What a shock does and does not model
///
/// A `ShockSpec` moves: the vol surface's shape (level/skew/curvature/term,
/// in vol space), the forward curve and market spot together (by the same
/// multiplicative factor, so the scenario stays internally consistent --
/// a spot move without a matching forward move is the "sticky moneyness"
/// case already covered by the portfolio module's own tests, not a
/// scenario), and calendar time (shrinking every position's own years).
///
/// Deliberately **not** modelled: a rate shock (the discount and rate-carry
/// curves are left exactly as given) and an event/residual shape (the
/// differential engine's `event_shift` is a *measured* diagnostic, not
/// something a shock is specified in terms of -- synthesising "an event" as
/// an input would mean inventing a market move with no stated numerical
/// basis, which is exactly what the brief forbids). A caller who wants a
/// rate shock or an event-shaped move can still build one directly as a
/// `VolSurface`/`MarketPoint` pair and call `compute_pnl_attribution`
/// itself -- this module adds a convenient, consistent *subset* of shocks on
/// top of that existing, more general path, not a replacement for it.

#include <span>
#include <string>
#include <vector>

#include "volatility_lab/diagnostics/surface_differential.hpp"
#include "volatility_lab/portfolio/portfolio.hpp"
#include "volatility_lab/volatility/surface.hpp"

namespace vl {

struct ShockSpec {
    std::string label;

    // --- vol-space shape shock, same basis as SurfaceDifferential ----------
    double level_shift = 0.0;      ///< additive change in annualised vol
    double skew_shift = 0.0;       ///< coefficient on k
    double curvature_shift = 0.0;  ///< coefficient on k^2
    double term_shift = 0.0;       ///< coefficient on T

    // --- spot / forward ------------------------------------------------------
    /// Fractional spot move, e.g. -0.10 for a 10% decline. Applied to both
    /// `MarketPoint::spot` and the surface's own forward curve (scaled by
    /// the same `1 + spot_pct` at every tenor), so the scenario is
    /// internally consistent rather than a sticky-moneyness spot move.
    double spot_pct = 0.0;

    // --- time ----------------------------------------------------------------
    /// Calendar days advanced; each position's `years` shrinks by
    /// `time_decay_days / 365`.
    double time_decay_days = 0.0;
};

/// Build the shocked surface: see the file comment for how the vol-space
/// shock is realised as per-expiry `GridSlice`s, and the forward curve
/// scaling.
///
/// `k_grid` is the moneyness grid each new `GridSlice` is built on; empty
/// uses a default (|k| up to 1.0, step 0.05) wide and dense enough that
/// `default_differential_grid()`'s range sits well clear of both the held-
/// slope extrapolation boundary and the natural-spline edge effects --
/// see `apply_shock_to_surface`'s own comment in the .cpp for the measured
/// numbers behind that choice.
[[nodiscard]] VolSurface apply_shock_to_surface(const VolSurface& base, const ShockSpec& shock,
                                                 std::span<const double> k_grid = {});

/// Build the shocked market point: `spot *= 1 + spot_pct`, rate and carry
/// unchanged (no rate shock modelled -- see file comment).
[[nodiscard]] MarketPoint apply_shock_to_market(const MarketPoint& base,
                                                const ShockSpec& shock) noexcept;

struct ScenarioResult {
    std::string label;

    /// Exact PnL and the full Taylor-vs-exact decomposition, reconciling
    /// exactly (`portfolio/portfolio.hpp`).
    PnLAttribution attribution;

    /// What the shock actually did to the surface, as independently
    /// *measured* by the differential engine -- compare against the
    /// `ShockSpec`'s own level/skew/curvature/term/spot_pct to see how
    /// exactly the construction realised what was asked for.
    SurfaceDifferential realized_shock;
};

/// Evaluate one scenario end to end against a base book.
[[nodiscard]] ScenarioResult evaluate_scenario(std::span<const Position> positions,
                                               const VolSurface& base_surface,
                                               const MarketPoint& base_market,
                                               const ShockSpec& shock);

/// Evaluate many scenarios against the same base book -- the scenario
/// *graph*: each scenario is independent of the others (a plain loop, no
/// shared mutable state), which is what lets a caller run a whole stress
/// report, or eventually parallelise it, without this function changing.
[[nodiscard]] std::vector<ScenarioResult> evaluate_scenarios(std::span<const Position> positions,
                                                             const VolSurface& base_surface,
                                                             const MarketPoint& base_market,
                                                             std::span<const ShockSpec> shocks);

}  // namespace vl
