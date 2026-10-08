// SPDX-License-Identifier: MIT
#pragma once
/// \file incremental_engine.hpp
/// \brief The runtime dependency graph, wired to the real pipeline
///        (directive Phase 2, section 6): quote -> expiry slice -> surface
///        -> surface differential / uncertainty -> Greeks -> portfolio ->
///        PnL.
///
/// ## What this is, and is not
///
/// `core/dependency_graph.hpp` is deliberately a bookkeeping-only DAG: it
/// tracks labels and dirty flags and calls nothing. This module is the
/// *runtime* that sits on top of it -- it owns the real pipeline state
/// (the quotes, the calibrated slices, the assembled `VolSurface`, the
/// portfolio) and, for each dirty node the graph reports, calls the one
/// already-tested function that recomputes exactly that piece:
/// `calibrate_svi_slice` for an expiry slice, `VolSurface`'s own
/// constructor for the assembled surface, `compute_surface_differential`
/// and `estimate_point_uncertainty` for the diagnostic layer,
/// `value_position`/`portfolio_greeks` for risk, `compute_pnl_attribution`
/// for PnL. Nothing here reimplements any of that math.
///
/// ## Why a market-point change does not recalibrate anything
///
/// This project has already found, independently, in three different
/// modules (`portfolio/portfolio.hpp`'s sticky-moneyness tests,
/// `diagnostics/surface_differential.hpp`'s forward-orthogonality
/// guarantee, `scenarios/scenario.hpp`'s spot-only shock test) that a pure
/// spot/forward move leaves a log-moneyness-parameterised surface's *shape*
/// exactly unchanged -- only where the forward sits. `update_market_point`
/// applies that same, by-now well-established finding here: it marks the
/// Surface node (and everything downstream of it) dirty directly, and the
/// Surface node's own recompute only rebuilds the forward/discount
/// `TermCurve`s and reassembles the *existing* calibrated slices -- it does
/// not touch a single `ExpirySlice` node, and so does not recalibrate
/// anything. A quote tick, by contrast, dirties exactly the one
/// `ExpirySlice` node whose expiry the quote belongs to, and nothing else.
///
/// ## A deliberate v1 scope limit: the set of expiries is fixed at construction
///
/// `core/dependency_graph.hpp`'s `add_node` can add a new node with
/// dependencies on existing nodes, but there is no operation to add a new
/// *dependency* to a node that already exists -- so a brand-new expiry
/// arriving after construction cannot be slotted in as one more dependency
/// of the already-built Surface node. `apply_event` reports this
/// explicitly (`RuntimeError::UnknownExpiry`) rather than silently dropping
/// the event or rebuilding the whole graph behind the caller's back.
/// Supporting a graph whose shape can grow at runtime is a real extension,
/// not attempted here speculatively.

#include <chrono>
#include <map>
#include <unordered_map>
#include <vector>

#include "volatility_lab/calibration/svi_calibrator.hpp"
#include "volatility_lab/calibration/weights.hpp"
#include "volatility_lab/core/dependency_graph.hpp"
#include "volatility_lab/diagnostics/surface_differential.hpp"
#include "volatility_lab/options/normalize.hpp"
#include "volatility_lab/portfolio/portfolio.hpp"
#include "volatility_lab/risk/uncertainty.hpp"
#include "volatility_lab/runtime/market_event.hpp"

namespace vl {

enum class RuntimeError : std::uint8_t {
    /// `event.expiry` does not match (within tolerance) any expiry this
    /// runtime was constructed with. See the file comment.
    UnknownExpiry = 0,
};

[[nodiscard]] constexpr const char* to_string(RuntimeError e) noexcept {
    switch (e) {
        case RuntimeError::UnknownExpiry: return "unknown-expiry";
    }
    return "?";
}

/// What one `recompute()` pass actually did, in real, measured terms --
/// never fabricated. `total_nodes`/`recomputed_nodes` come directly from
/// the graph's own bookkeeping; `latency` from `std::chrono::steady_clock`
/// around the actual recompute work.
struct RecomputeReport {
    std::size_t total_nodes = 0;
    std::size_t recomputed_nodes = 0;
    std::size_t reused_nodes = 0;
    std::chrono::nanoseconds latency{0};

    [[nodiscard]] double fraction_avoided() const noexcept {
        return (total_nodes > 0) ? static_cast<double>(reused_nodes) /
                                       static_cast<double>(total_nodes)
                                 : 0.0;
    }
};

class IncrementalEngine {
  public:
    struct Config {
        VolSurface baseline_surface;    ///< fixed reference for the differential/PnL baseline
        MarketPoint baseline_market;
        std::vector<Position> positions;
        SviCalibratorConfig calibrator{};
        WeightConfig weights{};
        NormalizationConfig normalize{};
        UncertaintyConfig uncertainty{};
        /// (k, years) the Uncertainty node reports on -- a representative
        /// point, not full per-point propagation (that is the directive's
        /// later, separate "uncertainty must propagate" phase).
        double uncertainty_k = 0.0;
        double uncertainty_years = 0.25;
    };

    /// `initial_quotes` fixes the set of expiries this engine will ever
    /// know about -- see the file comment. Quotes need not be normalised
    /// or weighted yet; this constructor does that itself, once per
    /// expiry, exactly as a quote-tick recompute would.
    IncrementalEngine(std::vector<OptionQuote> initial_quotes, Config config);

    /// Updates the stored quote for `event.instrument` and marks its
    /// expiry's node (and everything downstream) dirty. Does not
    /// recompute -- call `recompute()` to do that, so a batch of events
    /// can be applied before paying for one recompute pass.
    [[nodiscard]] Expected<std::monostate, RuntimeError> apply_event(const MarketEvent& event);

    /// Updates the market point and marks the Surface node (not any
    /// ExpirySlice node) dirty -- see the file comment for why a pure
    /// market-point change never recalibrates anything.
    void update_market_point(MarketPoint market) noexcept;

    /// Recomputes every currently-dirty node, in the graph's own safe
    /// order, calling the real function for each node kind. Returns a
    /// report of exactly how much work that was, and how much was
    /// avoided.
    RecomputeReport recompute();

    [[nodiscard]] const VolSurface& surface() const noexcept { return surface_; }
    [[nodiscard]] const MarketPoint& market() const noexcept { return market_; }
    [[nodiscard]] const SurfaceDifferential& differential() const noexcept { return differential_; }
    [[nodiscard]] const PointUncertainty& uncertainty() const noexcept { return uncertainty_; }
    [[nodiscard]] std::span<const PositionValuation> position_valuations() const noexcept {
        return valuations_;
    }
    [[nodiscard]] const PortfolioGreeks& portfolio() const noexcept { return portfolio_greeks_; }
    [[nodiscard]] const PnLAttribution& pnl() const noexcept { return pnl_; }

    [[nodiscard]] std::size_t node_count() const noexcept { return graph_.size(); }
    [[nodiscard]] bool all_clean() const noexcept { return graph_.all_clean(); }

  private:
    enum class NodeKind : std::uint8_t {
        ExpirySlice,
        Surface,
        SurfaceDifferential,
        Uncertainty,
        PositionGreeks,
        Portfolio,
        Pnl,
    };

    void recompute_node(NodeId id);
    void rebuild_surface_from_slices();

    DependencyGraph graph_;
    Config config_;

    // Node bookkeeping: what kind each node is, and which expiry/position
    // index an ExpirySlice/PositionGreeks node owns.
    std::vector<NodeKind> node_kind_;
    std::vector<double> node_expiry_years_;    // valid for ExpirySlice nodes
    std::vector<std::size_t> node_position_;   // valid for PositionGreeks nodes
    std::map<double, NodeId> expiry_node_by_years_;
    std::vector<NodeId> position_node_ids_;
    NodeId surface_node_ = kInvalidNodeId;
    NodeId differential_node_ = kInvalidNodeId;
    NodeId uncertainty_node_ = kInvalidNodeId;
    NodeId portfolio_node_ = kInvalidNodeId;
    NodeId pnl_node_ = kInvalidNodeId;

    // Pipeline state, owned here and mutated only by recompute_node.
    std::vector<OptionQuote> quotes_;
    std::unordered_map<InstrumentKey, std::size_t> quote_index_by_instrument_;
    std::map<double, SliceVariant> slices_by_expiry_;
    VolSurface surface_;
    MarketPoint market_;
    SurfaceDifferential differential_;
    PointUncertainty uncertainty_;
    std::vector<PositionValuation> valuations_;
    PortfolioGreeks portfolio_greeks_;
    PnLAttribution pnl_;

    static constexpr double kYearsMatchTolerance = 1.0e-9;
};

}  // namespace vl
