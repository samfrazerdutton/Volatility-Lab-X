// SPDX-License-Identifier: MIT
#pragma once
/// \file replay.hpp
/// \brief Deterministic replay: fold a market event stream into a sequence
///        of fingerprinted states (directive Phase 1, section 5).
///
/// ## What "state" means at this point in the pipeline
///
/// The full pipeline the directive describes
/// (events -> normalized state -> surface -> factors -> uncertainty ->
/// regime -> risk -> PnL -> cross-underlier -> decision state) does not
/// exist as a wired-together runtime yet -- that is Phase 2's job (the
/// dependency-graph runtime, built on `core/dependency_graph.hpp`), which
/// this intentionally comes *before*, per the directive's own implementation
/// order. What replay can fingerprint *today*, honestly, is the one state
/// that is fully determined by the event stream alone: `MarketState`, the
/// most-recent observation for every instrument seen so far. Once Phase 2
/// wires events into calibration/surface/Greeks/portfolio, those derived
/// states get their own fingerprints via the exact same `StateHasher`
/// primitive (`state_hash.hpp`) -- a `surface_hash` or `portfolio_hash` is
/// not a different mechanism, just the same one applied one stage further
/// down a pipeline that does not exist yet. Claiming those fingerprints now
/// would be fabricating a capability nothing produces.
///
/// ## Why `MarketState` is a `std::map`, not a `std::unordered_map`
///
/// Fingerprinting requires iterating every instrument's state in some
/// order. A `std::map`'s iteration order is the key's total order --
/// identical on every run, every platform, every standard library.  A
/// `std::unordered_map`'s iteration order depends on the hash function and
/// bucket layout, which is implementation-defined and not guaranteed stable
/// even between two runs of the *same* binary with different bucket
/// growth history. Hashing an unordered_map "in iteration order" would
/// quietly make the fingerprint depend on the standard library's internal
/// hashing, not on the market state -- exactly the class of hidden
/// nondeterminism this module exists to rule out.
///
/// ## What `apply` does, deliberately simply
///
/// Every event -- `Quote` or `Trade` -- unconditionally overwrites the
/// instrument's stored bid/ask/mid/size with the event's own fields. This
/// is "the state is literally the most recent observation", nothing more.
/// A `Trade` print that does not carry a meaningful bid/ask is not
/// specially fused with the prior quote here; that kind of "keep the last
/// known two-sided market separate from the last trade print" refinement
/// belongs to the *normalized* market state the real pipeline will build on
/// top of this, not to the raw replay state.

#include <cstddef>
#include <map>
#include <vector>

#include "volatility_lab/runtime/market_event.hpp"
#include "volatility_lab/runtime/state_hash.hpp"

namespace vl {

struct InstrumentState {
    std::string underlier;
    Years expiry{};
    Strike strike{};
    OptionType option_type = OptionType::Call;
    Money bid{};
    Money ask{};
    Money mid{};
    double size = 0.0;
    MarketEventType last_event_type = MarketEventType::Quote;
    SequenceNumber last_sequence{};
    Timestamp last_timestamp{};
};

class MarketState {
  public:
    /// Overwrites the stored state for `event.instrument` with the event's
    /// own fields. See the file comment for why this is deliberately not
    /// smarter than that.
    void apply(const MarketEvent& event);

    [[nodiscard]] std::size_t size() const noexcept { return instruments_.size(); }
    [[nodiscard]] bool empty() const noexcept { return instruments_.empty(); }
    [[nodiscard]] const InstrumentState* find(const std::string& instrument) const noexcept;
    [[nodiscard]] const std::map<std::string, InstrumentState>& instruments() const noexcept {
        return instruments_;
    }

    /// Fold every instrument's state, in key (sorted instrument-id) order,
    /// into a single fingerprint. Field combine order is fixed:
    /// instrument id, underlier, expiry, strike, option_type, bid, ask,
    /// mid, size, last_event_type, last_sequence, last_timestamp.
    [[nodiscard]] StateHash fingerprint() const noexcept;

  private:
    std::map<std::string, InstrumentState> instruments_;
};

/// One event's effect on the fingerprint: which event produced it, and the
/// state hash immediately after applying it.
struct ReplayStep {
    SequenceNumber sequence;
    StateHash state_hash;
};

struct ReplayResult {
    MarketState final_state;
    std::vector<ReplayStep> steps;  ///< one per event, in stream order
};

/// Replays `stream` from empty state, applying every event in order and
/// recording the state fingerprint after each one. Pure: does not mutate
/// `stream`, touches no global or static state, and reads no wall clock --
/// calling this twice on the same stream is guaranteed, not merely
/// expected, to produce identical `ReplayResult`s.
///
/// Performance note, stated honestly rather than left to be discovered:
/// `fingerprint()` folds *every* instrument's full state on *every* call,
/// so this is O(events * instruments), not O(events). That is the right
/// trade for a correctness-first implementation (the directive's own
/// `REFERENCE -> SCALAR -> SIMD -> PARALLEL -> INCREMENTAL` ladder puts
/// optimisation after correctness, not before it) and is fine at the
/// scale this is tested at. An incremental fingerprint that updates only
/// the changed instrument's contribution is the natural next step once a
/// benchmark actually shows this costing something at million-event scale
/// -- not done here pre-emptively.
[[nodiscard]] ReplayResult replay(const MarketEventStream& stream);

}  // namespace vl
