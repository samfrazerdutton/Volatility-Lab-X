// SPDX-License-Identifier: MIT
#pragma once
/// \file full_rebuild.hpp
/// \brief A deliberately independent, non-incremental reference pipeline
///        (Phase 2 section 8).
///
/// `IncrementalEngine`'s own constructor already runs every node once
/// (every node starts dirty), which is *a* full rebuild, but it gets there
/// through the same `DependencyGraph`/`NodeKind` machinery the incremental
/// path uses -- a bug in that machinery could in principle affect both
/// paths identically and hide from a comparison between them. `full_rebuild`
/// does not touch `DependencyGraph` at all: it calls the same low-level,
/// separately-tested pipeline functions
/// (`calibrate_svi_slice`/`VolSurface`/`compute_surface_differential`/
/// `estimate_point_uncertainty`/`value_position`/`aggregate_greeks`/
/// `compute_pnl_attribution`) directly, in a straight sequence, grouping
/// quotes by expiry with a plain scan rather than any index. This is
/// intentional: the point of this function is to be the simplest possible
/// correct implementation, not a fast one, so that a disagreement between
/// it and `IncrementalEngine` is actually informative.

#include "volatility_lab/runtime/incremental_engine.hpp"

namespace vl {

struct FullRebuildResult {
    VolSurface surface;
    SurfaceDifferential differential;
    PointUncertainty uncertainty;
    std::vector<PositionValuation> valuations;
    PortfolioGreeks portfolio;
    PnLAttribution pnl;
};

/// Recomputes everything from `quotes` and `config` with no dependency
/// graph, no dirty tracking, and no indexed quote lookup -- reusing
/// `IncrementalEngine::Config` only for its (already-validated) field
/// set, not any of its runtime behaviour. Uses `config.baseline_market`
/// as the market point, matching what a freshly-constructed
/// `IncrementalEngine` starts from before any `update_market_point` call.
[[nodiscard]] FullRebuildResult full_rebuild(std::span<const OptionQuote> quotes,
                                             const IncrementalEngine::Config& config);

}  // namespace vl
