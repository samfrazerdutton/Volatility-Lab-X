// SPDX-License-Identifier: MIT
#pragma once
/// \file uncertainty.hpp
/// \brief Uncertainty-aware volatility: how much to trust the surface's own
///        point estimate at a given (log-moneyness, tenor).
///
/// ## What this answers
///
/// `VolSurface::vol(k, T)` always returns a number -- that is what a
/// parametric model is for, it is defined on the whole plane. But "defined"
/// is not "known equally well everywhere": a point sitting on top of a dozen
/// tight, liquid, fresh quotes and a point sitting in a thin wing with one
/// stale, wide-spread print both get a confident-looking number out of the
/// same `.vol()` call. This module estimates, from the *quotes themselves*,
/// how much uncertainty actually surrounds that number.
///
/// ## Built from three independently-measurable observables, not invented
///
/// 1. **Each quote's own bid/ask spread**, converted to volatility points via
///    `half_spread_in_vol` (`calibration/weights.hpp`) -- the same
///    vega-divided conversion the calibration weighting pipeline already
///    uses, reused rather than re-derived so there is exactly one
///    implementation of "how wide is this quote, in vol terms" in the
///    codebase.
/// 2. **How many quotes are actually near this point**, combined with their
///    own calibration weight (so a quote the fit already discounted --
///    illiquid, stale, degraded -- discounts its vote here too) and a local
///    relevance kernel (Gaussian in log-moneyness scaled by the surface's own
///    ATM total variance at this tenor -- exactly the scaling convention
///    `calibration/weights.hpp`'s moneyness kernel already uses -- times a
///    Gaussian in tenor). The standard-error-of-the-mean shrinkage,
///    `spread / sqrt(effective_n)`, is where "a dozen tight quotes agreeing"
///    becomes a tighter estimate than "one tight quote alone" -- Kish's
///    effective sample size (`sum(w)^2 / sum(w^2)`) is the standard way to
///    turn an *unequally weighted* count of contributors into the
///    equivalent-sized unweighted sample.
/// 3. **Whether the point is outside the quotes' own observed range at all**
///    -- extrapolation, not interpolation. This is measured directly from
///    the quote book's own `log_moneyness`/`years` range rather than from
///    `VolSurface`'s own `extrapolated_in_strike` flag, because that flag is
///    only meaningful for `GridSlice` (see `volatility/surface.hpp`'s
///    `at_strike`) -- a parametric SVI/SSVI slice is defined on the whole
///    line and that flag is always false for one, which would make every
///    query past the wings look exactly as well-supported as the ATM point.
///
/// ## What this is not
///
/// Not a rigorous Bayesian posterior over the fitted SVI/SSVI parameters.
/// The quasi-explicit calibrator's inner solve is a *constrained* (boxed)
/// least squares (see `calibration/svi_calibrator.hpp`): when a constraint is
/// active the standard unconstrained covariance formula
/// `(J^T W J)^-1 sigma^2` does not apply, and deriving the constrained
/// analogue is a materially larger undertaking than this module's scope.
/// What is built instead is a local, transparent aggregation of real,
/// observable quote-level noise -- defensible and testable, but a
/// description of "how much raw disagreement/thinness is nearby", not a
/// formal confidence interval on the fitted parameters.
///
/// A sharper, non-obvious consequence of that: this reports *relative*
/// confidence across one snapshot, not the snapshot's own absolute
/// freshness. If every quote in the book ages by the same amount, every
/// weight scales by (approximately) the same factor, and both the Kish
/// effective sample size (`sum(w)^2 / sum(w^2)`) and the weighted mean
/// (`sum(w*x) / sum(w)`) are exactly invariant under a uniform rescaling of
/// every weight -- so a book that is uniformly an hour old reports *the same*
/// per-point uncertainty as one that is uniformly fresh. That is correct for
/// what this function measures (nothing has changed about *which* points are
/// relatively better or worse supported than others) and would be wrong to
/// "fix" by blending in an ad hoc global staleness adjustment; it does mean a
/// caller who also needs to know "is this entire snapshot too old to trust"
/// has to ask that question separately, from the snapshot's own age, not
/// from this module.
///
/// ## What is explicitly out of scope here
///
/// Propagating this point-level uncertainty into Greeks, PnL attribution, or
/// the dislocation engine is later work (the brief's confidence-aware
/// dislocation engine). This module answers "how much do I trust this one
/// number", not yet "and therefore how much should a downstream consumer of
/// it be worried."

#include <span>
#include <vector>

#include "volatility_lab/calibration/weights.hpp"
#include "volatility_lab/options/quote.hpp"
#include "volatility_lab/volatility/surface.hpp"

namespace vl {

struct UncertaintyConfig {
    /// Width of the local moneyness relevance kernel, in standard deviations
    /// of the surface's own ATM total variance at the query's tenor -- same
    /// convention as `WeightConfig::moneyness_kernel_sd`.
    double moneyness_kernel_sd = 1.5;

    /// Width of the local tenor relevance kernel, as a fraction of the
    /// query's own years (so a 1-week query and a 2-year query get
    /// proportionately scaled windows, not the same absolute width).
    double time_bandwidth_frac = 0.35;

    /// Floor on the tenor bandwidth, so a very short-dated query does not get
    /// an unreasonably tight window.
    double min_time_bandwidth_years = 7.0 / 365.0;

    /// How strongly to inflate vol_std_error per local-bandwidth-unit of
    /// distance outside the quotes' observed range. 0 disables the
    /// extrapolation penalty entirely (useful for isolating the interpolation
    /// behaviour in tests).
    double extrapolation_penalty_per_bandwidth = 1.0;

    /// Floor on the reported uncertainty, so a point with an accidentally
    /// tiny aggregate (e.g. a single contributing quote with a locked
    /// market) is never reported as having literally zero uncertainty.
    double min_vol_std_error = 1.0e-4;

    /// Passed through to `half_spread_in_vol` for its own fallback/floor
    /// behaviour on quotes with no two-sided market or underflowed vega.
    WeightConfig spread_weights{};
};

struct PointUncertainty {
    double k = 0.0;
    double years = 0.0;

    /// The surface's own point estimate, carried for convenience.
    double vol_estimate = 0.0;

    /// The headline number: estimated 1-sigma uncertainty in `vol_estimate`,
    /// in volatility points. `std::numeric_limits<double>::infinity()` when
    /// `no_local_coverage` is true -- an explicit, unmistakable "this number
    /// means nothing" rather than a finite-looking but arbitrary guess.
    double vol_std_error = 0.0;

    /// Kish effective sample size of the quotes that actually contributed
    /// (weight * relevance kernel), not a raw count -- a dozen quotes far out
    /// in the kernel's tail contribute far less than one quote at its centre.
    double effective_n = 0.0;

    /// Weighted-average age of the contributing quotes, or -1 if none of them
    /// reported an age.
    double representative_age_seconds = -1.0;

    /// Outside the quotes' own observed log-moneyness / years range,
    /// measured directly from the quotes (see file comment for why this is
    /// not `VolSurface`'s own extrapolation flag).
    bool extrapolated_in_strike = false;
    bool extrapolated_in_time = false;

    /// True when no quote contributed any weight at all at this point (every
    /// quote was rejected, excluded by calibration, or infinitely far in the
    /// relevance kernel). `vol_std_error` is +infinity in this case.
    bool no_local_coverage = false;
};

/// Estimate the uncertainty in the surface's vol at one (k, years) point,
/// from the quotes that informed (or should have informed) that region.
///
/// `quotes` should be the same, already-normalised-and-weighted book the
/// surface was calibrated from (`OptionQuote::weight` and `::log_moneyness`
/// populated) -- this does not recompute weights, it reuses them.
[[nodiscard]] PointUncertainty estimate_point_uncertainty(
    const VolSurface& surface, std::span<const OptionQuote> quotes, double k, double years,
    const UncertaintyConfig& cfg = {});

/// Batch convenience over parallel (k, years) arrays, matching
/// `VolSurface::total_variance_batch`'s calling convention.
[[nodiscard]] std::vector<PointUncertainty> estimate_grid_uncertainty(
    const VolSurface& surface, std::span<const OptionQuote> quotes, std::span<const double> k,
    std::span<const double> years, const UncertaintyConfig& cfg = {});

}  // namespace vl
