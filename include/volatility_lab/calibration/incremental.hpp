// SPDX-License-Identifier: MIT
#pragma once
/// \file incremental.hpp
/// \brief Incremental (online) calibration: refit one expiry's slice without
///        rebuilding the whole surface.
///
/// ## The case this exists for
///
/// A market feed ticks one expiry at a time -- a new print, an updated
/// bid/ask, one more quote filling in -- and a surface held by anything
/// downstream (Greeks, PnL attribution, the differential engine) needs to
/// reflect that quickly. Refitting every slice because one of them has new
/// data is wasted work scaling with the number of expiries on the surface,
/// for a change that is local to one of them.
///
/// `VolSurface` is already built for this: it is immutable, and `with_slice`
/// (`volatility/surface.hpp`) replaces exactly one slice and leaves every
/// other one bit-identical, including the `TermCurve`s. What this module
/// adds is the calibration side of that: given new quotes for one expiry,
/// refit *that slice alone* with the existing quasi-explicit SVI calibrator
/// (`svi_calibrator.hpp`) and hand back a new surface built with
/// `with_slice`.
///
/// ## How much this actually saves, measured not claimed
///
/// The quasi-explicit calibrator already counts its own work per slice
/// (`SviFitResult::inner_solves`, `::outer_iterations`,
/// `::residual_evaluations`) for exactly this kind of comparison.
///
/// The precise, always-true guarantee -- checked directly in
/// `tests/calibration/incremental.cpp` -- is **not** "this costs 1/N of a
/// full refit": one slice's own difficulty (how many Levenberg-Marquardt
/// iterations its (m, sigma) search needs) varies a lot slice to slice, so a
/// single refit slice can cost well above the per-slice average -- measured
/// directly, one slice in the synthetic `Normal` regime costs 46% of the
/// *entire* ten-slice surface's total work, because that slice alone needed
/// 58 outer iterations against 7-12 for the rest. What *is* always exactly
/// true: the incremental call's cost equals that one slice's own
/// standalone `calibrate_svi_slice` cost -- no more, no less -- and every
/// other slice does *zero* work, which a naive full refit cannot say. The
/// aggregate saving across many updates is real and large; a single
/// update's saving depends on which slice changed and is reported
/// factually rather than rounded to a tidy 1/N.
///
/// ## What decides "refit vs insert"
///
/// If `years` matches one of `base_surface`'s existing expiries (within
/// `years_match_tolerance`, which exists only to absorb floating-point
/// round-trip noise -- expiries are not expected to be *deliberately*
/// close together without being equal), that slice is replaced in place via
/// `with_slice`. Otherwise this is a new expiry the surface did not have
/// before, and it is inserted at the correct sorted position via
/// `with_slices` -- `VolSurface` requires strictly increasing expiries, so
/// insertion, not append, is what keeps that invariant for an expiry that
/// lands in the middle of the existing term structure.
///
/// ## What is explicitly out of scope here
///
/// Deciding *whether* a slice needs refitting at all -- i.e. "is this new
/// quote different enough from what is already fitted to be worth the
/// work" -- is the dirty-region/dependency-graph phase's job, not this
/// module's. This module answers "refit this one slice cheaply", not "should
/// I". A caller who wants that decision made automatically should consult
/// that engine first and call this one only for the slices it flags.

#include <span>

#include "volatility_lab/calibration/svi_calibrator.hpp"
#include "volatility_lab/options/quote.hpp"
#include "volatility_lab/volatility/surface.hpp"

namespace vl {

struct IncrementalUpdateResult {
    VolSurface surface;
    SviFitResult fit;

    /// Index of the updated slice in `surface`'s own slice list (not
    /// `base_surface`'s -- insertion shifts every later index up by one).
    std::size_t slice_index = 0;

    /// True if `years` was not one of `base_surface`'s existing expiries, so
    /// a new slice was inserted rather than an existing one replaced.
    bool inserted_new_slice = false;
};

/// Refit exactly one expiry's slice and return a new surface with only that
/// slice different.
///
/// `quotes_for_slice` must all share `years` (the same contract
/// `calibrate_svi_slice` already has) and must already be normalised and
/// weighted -- this does not call `normalize` or `assign_weights_*` itself,
/// consistent with `calibrate_svi_slice`'s own contract, so that a caller
/// who has already normalised the whole book once does not pay for it
/// again per slice.
[[nodiscard]] IncrementalUpdateResult apply_incremental_update(
    const VolSurface& base_surface, double years, std::span<const OptionQuote> quotes_for_slice,
    const SviCalibratorConfig& cfg = {}, double years_match_tolerance = 1.0e-9);

}  // namespace vl
