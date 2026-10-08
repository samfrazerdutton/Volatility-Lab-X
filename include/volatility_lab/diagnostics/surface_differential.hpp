// SPDX-License-Identifier: MIT
#pragma once
/// \file surface_differential.hpp
/// \brief The surface differential engine -- VOLATILITY LAB X's first
///        flagship feature (brief section 2).
///
/// ## What this answers
///
/// Not "did the surface change" but "*what kind* of change was it": a level
/// shift, a skew move, a change in curvature, a term-structure twist, a move
/// in the forward, an isolated event bump, or something none of the above
/// explain.
///
/// ## The decomposition is a regression, not a lookup table
///
/// The brief is explicit that this must be numerical and testable, not
/// hardcoded percentages.  So it is built as a small weighted least-squares
/// fit of the *observed* annualised-volatility change, over a canonical grid
/// of (log-moneyness, years) points, onto four interpretable basis
/// functions:
///
///     dSigma(k, T)  ~=  c_level   * 1
///                     + c_skew    * k
///                     + c_curv    * k^2
///                     + c_term    * T
///                     + residual(k, T)
///
/// reusing exactly the same `math::NormalEquations` + `cholesky_solve`
/// machinery the SVI calibrator uses -- this is a four-parameter linear
/// regression, which is the simplest model that can separate "the whole
/// smile moved up" (c_level) from "the smile tilted" (c_skew) from "the
/// smile got more convex" (c_curv) from "short and long expiries moved
/// differently" (c_term).  Being linear, the fit's own residual is defined
/// as observed-minus-fitted at every point, so
///
///     dSigma(point) == c_level + c_skew*k + c_curv*k^2 + c_term*T + residual(point)
///
/// holds exactly, by construction of ordinary least squares, for every
/// point on the grid -- not approximately.
///
/// The regression is deliberately done in *volatility* space, not total
/// variance, even though total variance (W = sigma^2 * T) is the more
/// "natural" quantity elsewhere in this codebase.  A uniform annualised-vol
/// shift of, say, +4 points at every tenor is the textbook definition of a
/// pure level move -- it is what "did overall implied volatility move"
/// (the brief's own phrasing) means.  In total-variance terms that same move
/// is dW(T) = (sigma_new^2 - sigma_old^2)*T: exactly linear in T, i.e.
/// indistinguishable from a term-structure twist under a variance-space fit.
/// Regressing dSigma rather than dW is what makes c_level actually mean
/// "the level moved" rather than "something T-independent happened to
/// variance, which for a vol-level move it structurally never is."
///
/// ## Why forward is reported separately, and what that implies
///
/// This surface is parameterised in log-moneyness k = log(K/F): the whole
/// point of that choice (see `volatility/surface.hpp`) is that the smile's
/// *shape* is independent of where the forward happens to sit.  One direct
/// consequence, worth stating plainly: **a pure forward/spot move leaves
/// the (k, T) variance grid completely unchanged**, so level, skew,
/// curvature and term structure are all exactly zero for it.  The forward
/// move is real and has to be reported, but it is reported from the
/// surface's own forward curve directly (`VolSurface::forwards()`), not
/// smuggled into the shape regression where it would not show up anyway.
/// This orthogonality is checked directly in the test suite rather than
/// merely asserted here.
///
/// ## Event detection
///
/// An "event" (brief section 2's `DeltaEvent`) is modelled as one tenor's
/// move being poorly explained by the smooth trend the *rest* of the
/// surface's tenors establish -- which is the shape a bump to a single
/// expiry's volatility produces (see the synthetic generator's `Earnings`
/// regime, which does precisely this).
///
/// This is deliberately **not** "check the global fit's residual at the
/// shortest tenor": an earlier version did exactly that, and failed on its
/// own test suite.  A single tenor with a large, genuinely isolated bump
/// pulls the *global* level/skew/curvature/term fit toward itself, which
/// spreads comparably-sized residual (of the opposite sign) across every
/// OTHER tenor too -- so an ordinary skew change, present at every tenor by
/// construction and not an event in any sense, produced a *larger* naive
/// front-tenor residual than a genuinely isolated bump at a single tenor.
///
/// Instead, event detection uses leave-one-tenor-out: for each distinct
/// tenor, refit level/skew/curvature/term using every *other* tenor's
/// points only, then measure how badly that fit -- which never saw this
/// tenor's data -- predicts this tenor's own observed move, relative to how
/// well it predicts the tenors it was actually fitted on.  A tenor that is
/// well-explained by the rest of the surface's smooth trend scores low; a
/// genuinely isolated bump, which the rest of the surface carries no
/// information about, scores high.  `event_shift` is the largest such
/// excess across all candidate tenors, floored at zero.
///
/// ## What is explicitly out of scope here
///
/// Liquidity and cross-underlier correlation are named in the same list in
/// the brief (section 2) but need information this engine does not have --
/// quote-level bid/ask and staleness, and a second underlier's surface,
/// respectively.  Liquidity-aware uncertainty is the next phase
/// (`risk/uncertainty.hpp`); cross-underlier relative value is a much later
/// one.  Reporting them here would mean inventing numbers from nothing,
/// which is exactly what section 50 of the brief forbids.

#include <span>
#include <vector>

#include "volatility_lab/volatility/surface.hpp"

namespace vl {

/// One point of the canonical grid the differential is measured on.
struct DifferentialGridPoint {
    double k = 0.0;      ///< log-moneyness, log(K/F)
    double years = 0.0;
};

/// Standard tenors (1m, 3m, 6m, 1y, 2y) crossed with five moneyness points
/// spanning a realistic quoted range (|k| up to ~0.4, roughly a 1.5x strike
/// ratio).  Deliberately independent of either surface's own slice
/// placement -- see the file comment for why that matters for comparing two
/// surfaces that may not share the same expiries.
[[nodiscard]] std::vector<DifferentialGridPoint> default_differential_grid();

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

struct SurfaceDifferential {
    // --- the regression coefficients: shape-of-smile changes ---------------
    double level_shift = 0.0;      ///< c_level: average total-variance shift
    double skew_shift = 0.0;       ///< c_skew: coefficient on k
    double curvature_shift = 0.0;  ///< c_curv: coefficient on k^2
    double term_shift = 0.0;       ///< c_term: coefficient on T

    // --- measured, not regressed -------------------------------------------
    /// Average log change in the forward curve across the grid's distinct
    /// tenors, log(F_new(T)/F_old(T)).  Exactly zero whenever the forward
    /// curve is unchanged, regardless of anything else that moved.
    double forward_shift = 0.0;

    /// Excess RMS residual concentrated in the grid's shortest tenor,
    /// floored at zero.  See the file comment.
    double event_shift = 0.0;

    // --- fit quality --------------------------------------------------------
    double residual_rms = 0.0;      ///< RMS of dW(point) - fitted(point)
    double total_delta_rms = 0.0;   ///< RMS of the raw observed dW(point)
    double condition_estimate = 1.0;  ///< from the regression's normal matrix

    /// Per-point diagnostics, in the same order as the grid passed in.
    std::vector<double> grid_k;
    std::vector<double> grid_years;
    std::vector<double> grid_delta_w;      ///< observed W_new - W_old
    std::vector<double> grid_fitted_w;     ///< c_level + c_skew*k + c_curv*k^2 + c_term*T
    std::vector<double> grid_residual_w;   ///< grid_delta_w - grid_fitted_w, exactly

    /// 1 - residual_rms/total_delta_rms, guarded against a near-zero total
    /// move.  The headline number: what fraction of the observed surface
    /// change is explained by level + skew + curvature + term alone.
    [[nodiscard]] double explained_fraction() const noexcept;
};

/// Compute the differential between two surfaces over `grid`.
///
/// Both surfaces are queried at every grid point via their own
/// interpolation/extrapolation (`VolSurface::total_variance`), so this works
/// even when the two surfaces have different expiries, strike ranges, or
/// underlying slice models entirely -- the grid, not either surface's
/// internal structure, defines what is compared.
[[nodiscard]] SurfaceDifferential compute_surface_differential(
    const VolSurface& old_surface, const VolSurface& new_surface,
    std::span<const DifferentialGridPoint> grid = {});

}  // namespace vl
