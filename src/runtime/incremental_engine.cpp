// SPDX-License-Identifier: MIT
#include "volatility_lab/runtime/incremental_engine.hpp"

#include <algorithm>
#include <cmath>

namespace vl {

namespace {

/// Standard forward/discount derivation (the same convention documented on
/// `OptionQuote::forward`/`::discount`: "derived if not supplied:
/// S*exp((r-q)T)"), evaluated at each of `years` to build the TermCurve a
/// VolSurface needs -- not re-deriving the formula, applying the one the
/// rest of this codebase already uses.
TermCurve build_forward_curve(const MarketPoint& market, std::span<const double> years) {
    if (years.empty()) return TermCurve::flat(market.spot);
    std::vector<double> ys(years.begin(), years.end());
    std::vector<double> fwds;
    fwds.reserve(ys.size());
    for (double t : ys) fwds.push_back(market.spot * std::exp((market.rate - market.carry) * t));
    return TermCurve(std::move(ys), std::move(fwds));
}

TermCurve build_discount_curve(const MarketPoint& market, std::span<const double> years) {
    if (years.empty()) return TermCurve::flat(1.0);
    std::vector<double> ys(years.begin(), years.end());
    std::vector<double> dfs;
    dfs.reserve(ys.size());
    for (double t : ys) dfs.push_back(std::exp(-market.rate * t));
    return TermCurve(std::move(ys), std::move(dfs));
}

}  // namespace

IncrementalEngine::IncrementalEngine(std::vector<OptionQuote> initial_quotes, Config config)
    : config_(std::move(config)),
      surface_(config_.baseline_surface),
      market_(config_.baseline_market) {
    quotes_ = std::move(initial_quotes);
    // Key is the quote's own structural identity -- (years, strike, type)
    // -- via the typed, allocation-free InstrumentKey (core/types.hpp).
    // OptionQuote has no instrument-symbol field (deliberately: it is a
    // single-underlier quote, not a feed record), so this is the natural
    // identity rather than a stand-in for a missing field.
    for (std::size_t i = 0; i < quotes_.size(); ++i) {
        const auto& q = quotes_[i];
        quote_index_by_instrument_[InstrumentKey{Years{q.years}, Strike{q.strike}, q.type}] = i;
    }

    // One ExpirySlice node per distinct expiry present in the initial book.
    std::vector<double> distinct_years;
    for (const auto& q : quotes_) {
        if (std::find_if(distinct_years.begin(), distinct_years.end(), [&](double t) {
                return std::abs(t - q.years) <= kYearsMatchTolerance;
            }) == distinct_years.end()) {
            distinct_years.push_back(q.years);
        }
    }
    std::sort(distinct_years.begin(), distinct_years.end());

    node_kind_.reserve(distinct_years.size() + 5 + config_.positions.size());
    node_expiry_years_.reserve(distinct_years.size());
    std::vector<NodeId> expiry_ids;
    for (double years : distinct_years) {
        const NodeId id = graph_.add_node("expiry_slice[" + std::to_string(years) + "]");
        expiry_node_by_years_[years] = id;
        expiry_ids.push_back(id);
        node_kind_.push_back(NodeKind::ExpirySlice);
        node_expiry_years_.push_back(years);
        node_position_.push_back(0);
    }

    // Build the per-expiry quote index once, up front: O(quotes * log
    // expiries) via `find_expiry_node`'s binary search, rather than the
    // O(quotes * expiries) a linear per-quote-per-expiry scan would cost.
    // `quote_indices_by_node_` only has entries for `ExpirySlice` node ids,
    // which are exactly `0..distinct_years.size()-1` since they are the
    // first nodes created above; sized to the eventual total node count so
    // it can be indexed directly by NodeId with no further bookkeeping.
    quote_indices_by_node_.resize(distinct_years.size() + 5 + config_.positions.size());
    for (std::size_t i = 0; i < quotes_.size(); ++i) {
        const NodeId node = find_expiry_node(quotes_[i].years);
        if (node != kInvalidNodeId) quote_indices_by_node_[node].push_back(i);
    }

    surface_node_ = graph_.add_node("surface", expiry_ids);
    node_kind_.push_back(NodeKind::Surface);
    node_expiry_years_.push_back(0.0);
    node_position_.push_back(0);

    differential_node_ = graph_.add_node("surface_differential", std::array{surface_node_});
    node_kind_.push_back(NodeKind::SurfaceDifferential);
    node_expiry_years_.push_back(0.0);
    node_position_.push_back(0);

    uncertainty_node_ = graph_.add_node("uncertainty", std::array{surface_node_});
    node_kind_.push_back(NodeKind::Uncertainty);
    node_expiry_years_.push_back(0.0);
    node_position_.push_back(0);

    position_node_ids_.reserve(config_.positions.size());
    for (std::size_t i = 0; i < config_.positions.size(); ++i) {
        const NodeId id =
            graph_.add_node("position_greeks[" + std::to_string(i) + "]", std::array{surface_node_});
        position_node_ids_.push_back(id);
        node_kind_.push_back(NodeKind::PositionGreeks);
        node_expiry_years_.push_back(0.0);
        node_position_.push_back(i);
    }
    valuations_.resize(config_.positions.size());

    portfolio_node_ = graph_.add_node("portfolio", position_node_ids_);
    node_kind_.push_back(NodeKind::Portfolio);
    node_expiry_years_.push_back(0.0);
    node_position_.push_back(0);

    pnl_node_ = graph_.add_node("pnl", std::array{portfolio_node_});
    node_kind_.push_back(NodeKind::Pnl);
    node_expiry_years_.push_back(0.0);
    node_position_.push_back(0);

    // Every node starts dirty (DependencyGraph's own invariant) -- run one
    // full recompute now so the engine is immediately queryable, exactly
    // the "full rebuild" baseline the directive's benchmark matrix compares
    // incremental recomputation against.
    recompute();
}

Expected<std::monostate, RuntimeError> IncrementalEngine::apply_event(const MarketEvent& event) {
    const double years = event.expiry.value();
    const NodeId node = find_expiry_node(years);
    if (node == kInvalidNodeId) {
        return make_unexpected(RuntimeError::UnknownExpiry);
    }

    const InstrumentKey key{event.expiry, event.strike, event.option_type};
    const auto idx_it = quote_index_by_instrument_.find(key);

    // An update to a *known* instrument must start from its existing quote,
    // not a blank one: `OptionQuote` carries fields a `MarketEvent` has no
    // opinion on at all (volume, open_interest, age_seconds, last, style,
    // source_index, slice_index), and `assign_weights_by_slice`'s liquidity
    // factor reads volume/open_interest directly. Building a fresh,
    // default-constructed `OptionQuote` here silently zeroed those out,
    // changing that one quote's calibration weight relative to a full
    // rebuild from the same final quotes -- found because the engineering
    // demo's incremental-vs-full-rebuild PnL differed by ~$20, far outside
    // the 1e-9 bitwise-match guarantee `IncrementalUpdateMatchesAFreshFullRebuildExactly`
    // already protects.
    OptionQuote q = (idx_it != quote_index_by_instrument_.end()) ? quotes_[idx_it->second]
                                                                  : OptionQuote{};
    q.strike = event.strike.value();
    q.years = years;
    q.type = event.option_type;
    q.spot = market_.spot;
    q.bid = event.bid.value();
    q.ask = event.ask.value();
    q.mid = event.mid.value();
    q.rate = market_.rate;
    q.dividend = market_.carry;
    q.status = QuoteStatus::Unvalidated;

    if (idx_it != quote_index_by_instrument_.end()) {
        // Existing instrument: same slot, same expiry bucket -- no index
        // maintenance needed.
        quotes_[idx_it->second] = q;
    } else {
        // New instrument: record its slot and add it to its expiry's
        // bucket so a future ExpirySlice recompute finds it without
        // rescanning the whole book.
        const std::size_t new_index = quotes_.size();
        quote_index_by_instrument_[key] = new_index;
        quotes_.push_back(q);
        quote_indices_by_node_[node].push_back(new_index);
    }

    graph_.mark_dirty(node);
    return std::monostate{};
}

void IncrementalEngine::update_market_point(MarketPoint market) noexcept {
    market_ = market;
    graph_.mark_dirty(surface_node_);
}

NodeId IncrementalEngine::find_expiry_node(double years) const noexcept {
    // See the header's comment: construction only ever inserts one entry
    // per expiry *more than* kYearsMatchTolerance apart from every other,
    // so lower_bound can find at most one candidate within tolerance.
    auto it = expiry_node_by_years_.lower_bound(years - kYearsMatchTolerance);
    if (it != expiry_node_by_years_.end() && std::abs(it->first - years) <= kYearsMatchTolerance) {
        return it->second;
    }
    return kInvalidNodeId;
}

void IncrementalEngine::rebuild_surface_from_slices() {
    std::vector<double> years;
    std::vector<SliceVariant> slices;
    years.reserve(slices_by_expiry_.size());
    slices.reserve(slices_by_expiry_.size());
    for (const auto& [y, slice] : slices_by_expiry_) {
        years.push_back(y);
        slices.push_back(slice);
    }
    surface_ = VolSurface(std::move(slices), build_forward_curve(market_, years),
                          build_discount_curve(market_, years));
}

void IncrementalEngine::recompute_node(NodeId id, RecomputeReport& report) {
    const NodeKind kind = node_kind_[id];
    switch (kind) {
        case NodeKind::ExpirySlice: {
            const double years = node_expiry_years_[id];
            // Indexed lookup, not a scan of `quotes_`: `quote_indices_by_node_[id]`
            // holds exactly this expiry's own quote slots, built once at
            // construction and kept current by `apply_event`. Recomputing
            // one expiry out of however many the book has now costs
            // O(that expiry's own quote count), not O(total quotes) --
            // see docs/INCREMENTAL_RUNTIME.md for the measured before/after.
            const auto& indices = quote_indices_by_node_[id];
            std::vector<OptionQuote> slice_quotes;
            slice_quotes.reserve(indices.size());
            for (std::size_t idx : indices) slice_quotes.push_back(quotes_[idx]);
            report.quotes_examined += slice_quotes.size();
            report.calibrations_run += 1;

            MarketSnapshot snapshot;
            snapshot.underlying = "engine";
            snapshot.spot = market_.spot;
            snapshot.quotes = slice_quotes;
            auto normalized = normalize(snapshot, config_.normalize);
            (void)assign_weights_by_slice(normalized.quotes, config_.weights);
            const auto fit = calibrate_svi_slice(normalized.quotes, config_.calibrator);
            slices_by_expiry_[years] = SliceVariant(fit.params);
            break;
        }
        case NodeKind::Surface: {
            rebuild_surface_from_slices();
            break;
        }
        case NodeKind::SurfaceDifferential: {
            differential_ = compute_surface_differential(config_.baseline_surface, surface_);
            break;
        }
        case NodeKind::Uncertainty: {
            uncertainty_ = estimate_point_uncertainty(surface_, quotes_, config_.uncertainty_k,
                                                      config_.uncertainty_years,
                                                      config_.uncertainty);
            break;
        }
        case NodeKind::PositionGreeks: {
            const std::size_t i = node_position_[id];
            valuations_[i] = value_position(config_.positions[i], surface_, market_);
            break;
        }
        case NodeKind::Portfolio: {
            std::vector<OptionGreeks> greeks(valuations_.size());
            std::vector<double> qty(config_.positions.size()), mult(config_.positions.size());
            for (std::size_t i = 0; i < config_.positions.size(); ++i) {
                greeks[i] = valuations_[i].greeks;
                qty[i] = config_.positions[i].quantity;
                mult[i] = config_.positions[i].multiplier;
            }
            portfolio_greeks_ = aggregate_greeks(greeks, qty, mult);
            break;
        }
        case NodeKind::Pnl: {
            pnl_ = compute_pnl_attribution(config_.positions, config_.baseline_surface,
                                           config_.baseline_market, surface_, market_);
            break;
        }
    }
}

RecomputeReport IncrementalEngine::recompute() {
    RecomputeReport report;
    report.total_nodes = graph_.size();
    report.quotes_total = quotes_.size();
    const auto dirty = graph_.dirty_nodes_in_order();
    report.recomputed_nodes = dirty.size();
    report.reused_nodes = report.total_nodes - report.recomputed_nodes;

    const auto start = std::chrono::steady_clock::now();
    for (NodeId id : dirty) {
        recompute_node(id, report);
        graph_.mark_clean(id);
    }
    report.latency = std::chrono::steady_clock::now() - start;
    return report;
}

}  // namespace vl
