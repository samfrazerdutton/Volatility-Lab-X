// SPDX-License-Identifier: MIT
#pragma once
/// \file market_event.hpp
/// \brief The deterministic market event: the atomic unit the runtime
///        (replay, dependency-graph execution, state hashing) is built on.
///
/// ## What this is for
///
/// Everything from here on -- replay (`runtime/replay.hpp`), incremental
/// recomputation, state hashing -- is only as deterministic as the event
/// model underneath it. A `MarketEvent` is therefore immutable once
/// constructed (enforced by `const` members, not just documented: there is
/// no setter and no assignment operator to misuse), and a
/// `MarketEventStream` only ever appends, in strictly increasing sequence
/// order, checked at the point of append rather than assumed.
///
/// ## Why both `underlier` and `instrument`
///
/// `underlier` is the thing the option is written on ("SPX"); `instrument`
/// is the feed's own identifier for this specific contract (a ticker/OCC
/// symbol or similar). They are kept distinct because a dependency graph
/// built from an event stream (the whole point of Phase 2 of the runtime
/// directive) dirties state keyed by *underlier and expiry*, not by the
/// feed's raw symbol -- two different instrument symbols can resolve to the
/// same underlier/expiry/strike/type (a reissued series, a corporate-action
/// adjusted symbol), and collapsing that distinction at the event level
/// would silently hide that case rather than let a later normalisation
/// stage decide what to do with it.
///
/// ## Why `expiry` is `Years`, not a calendar date
///
/// Every other quantity in this codebase (`OptionQuote::years`,
/// `Position::years`) is a year-fraction as observed at a point in time,
/// not a calendar date -- introducing a `Date` type here, used nowhere
/// else, would be a second way to express the same thing. `expiry` is
/// therefore "years to expiry as observed at `timestamp`": a fixed,
/// per-event value (it does not drift with when the event is later
/// *processed*, only with when it was *observed* -- which is exactly the
/// property determinism needs), consistent with how the rest of the
/// codebase already represents time to expiry.

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "volatility_lab/core/expected.hpp"
#include "volatility_lab/core/types.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Strong integer identifiers
// ---------------------------------------------------------------------------

/// A stream-local, strictly-increasing event identifier. Ordering, not
/// arithmetic, is the only operation that is meaningful for a sequence
/// number, so (unlike `Scalar<Tag>`) no `+`/`-` is provided.
class SequenceNumber {
  public:
    SequenceNumber() = default;
    constexpr explicit SequenceNumber(std::uint64_t v) noexcept : v_(v) {}
    [[nodiscard]] constexpr std::uint64_t value() const noexcept { return v_; }

    friend constexpr auto operator<=>(SequenceNumber a, SequenceNumber b) noexcept {
        return a.v_ <=> b.v_;
    }
    friend constexpr bool operator==(SequenceNumber a, SequenceNumber b) noexcept {
        return a.v_ == b.v_;
    }

  private:
    std::uint64_t v_ = 0;
};

/// Nanoseconds since an arbitrary, stream-local epoch -- not necessarily
/// wall-clock UTC, that is the producer's choice. Integer, not floating
/// point, specifically so two timestamps compare and hash *exactly*: a
/// replay's determinism requirement would otherwise be at the mercy of
/// floating-point rounding for no reason, since nanosecond-resolution
/// integers already have more than enough range and precision
/// (`std::int64_t` nanoseconds covers roughly +-292 years).
class Timestamp {
  public:
    Timestamp() = default;
    constexpr explicit Timestamp(std::int64_t nanos_since_epoch) noexcept
        : ns_(nanos_since_epoch) {}
    [[nodiscard]] constexpr std::int64_t nanoseconds_since_epoch() const noexcept { return ns_; }

    friend constexpr auto operator<=>(Timestamp a, Timestamp b) noexcept { return a.ns_ <=> b.ns_; }
    friend constexpr bool operator==(Timestamp a, Timestamp b) noexcept { return a.ns_ == b.ns_; }
    /// Duration between two timestamps, in nanoseconds.
    friend constexpr std::int64_t operator-(Timestamp a, Timestamp b) noexcept {
        return a.ns_ - b.ns_;
    }

  private:
    std::int64_t ns_ = 0;
};

// ---------------------------------------------------------------------------
// Event kind
// ---------------------------------------------------------------------------

enum class MarketEventType : std::uint8_t {
    Quote = 0,  ///< a bid/ask (two-sided market) update
    Trade = 1,  ///< a print (last/size), bid/ask may be stale or absent
};

[[nodiscard]] constexpr const char* to_string(MarketEventType t) noexcept {
    switch (t) {
        case MarketEventType::Quote: return "quote";
        case MarketEventType::Trade: return "trade";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// MarketEvent
// ---------------------------------------------------------------------------

/// One immutable market observation. Every field is `const`: there is no
/// setter, and the compiler-generated copy/move *assignment* operators are
/// implicitly deleted because of it (copy/move *construction* -- what a
/// `std::vector<MarketEvent>` needs to grow -- remains available). This is
/// the immutability requirement enforced in the type itself rather than
/// left to a comment a future change could invalidate silently.
struct MarketEvent {
    const SequenceNumber sequence;
    const Timestamp timestamp;
    const std::string underlier;
    const std::string instrument;
    const Years expiry;  ///< years to expiry as observed at `timestamp`
    const Strike strike;
    const OptionType option_type;
    const MarketEventType event_type;
    const Money bid;
    const Money ask;
    const Money mid;  ///< the feed's own mid, not necessarily 0.5*(bid+ask)
    const double size = 0.0;  ///< contract size/quantity of this observation
};

static_assert(!std::is_copy_assignable_v<MarketEvent>,
             "MarketEvent must stay immutable -- if this fires, a field lost its const");
static_assert(std::is_move_constructible_v<MarketEvent>,
             "MarketEvent must stay constructible into a growing std::vector");

/// Convenience for constructing a `Quote` event when the feed gives a
/// simple arithmetic mid rather than its own. Returns `Money{0.5*(bid+ask)}`
/// -- not stored anywhere, so a caller who has a *real* feed-provided mid
/// should use that directly instead of this.
[[nodiscard]] constexpr Money natural_mid(Money bid, Money ask) noexcept {
    return Money{0.5 * (bid.value() + ask.value())};
}

// ---------------------------------------------------------------------------
// MarketEventStream: append-only, strictly ordered
// ---------------------------------------------------------------------------

enum class EventStreamError : std::uint8_t {
    /// `event.sequence` was not strictly greater than the stream's last
    /// appended sequence number. This is a producer bug (two events racing,
    /// a replay feeding events out of order, an accidental duplicate), not
    /// a data-quality issue -- it is reported as a value (not an assert)
    /// specifically so the determinism-checking replay harness
    /// (`runtime/replay.hpp`) can catch it in a release build too, not only
    /// in a debug build where an assert would fire.
    SequenceNotIncreasing = 0,
};

[[nodiscard]] constexpr const char* to_string(EventStreamError e) noexcept {
    switch (e) {
        case EventStreamError::SequenceNotIncreasing: return "sequence-not-increasing";
    }
    return "?";
}

/// An append-only, strictly-sequence-increasing log of `MarketEvent`s. No
/// erase, no non-const element access, no reordering -- the only mutating
/// operation is `append`, and it is the one place ordering is checked.
class MarketEventStream {
  public:
    MarketEventStream() = default;

    /// Appends `event` if its sequence number is strictly greater than the
    /// last appended one (or this is the first event). Returns the new
    /// event's index on success.
    [[nodiscard]] Expected<std::size_t, EventStreamError> append(MarketEvent event);

    [[nodiscard]] std::size_t size() const noexcept { return events_.size(); }
    [[nodiscard]] bool empty() const noexcept { return events_.empty(); }
    [[nodiscard]] const MarketEvent& operator[](std::size_t i) const noexcept {
        return events_[i];
    }
    [[nodiscard]] const MarketEvent& at(std::size_t i) const { return events_.at(i); }

    [[nodiscard]] auto begin() const noexcept { return events_.begin(); }
    [[nodiscard]] auto end() const noexcept { return events_.end(); }

  private:
    std::vector<MarketEvent> events_;
    SequenceNumber last_sequence_{};
    bool has_last_ = false;
};

}  // namespace vl
