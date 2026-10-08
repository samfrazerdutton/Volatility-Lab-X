// SPDX-License-Identifier: MIT
/// Validates the deterministic market event model: that MarketEvent is
/// genuinely immutable (checked at compile time, not just by convention),
/// that the event stream enforces strictly-increasing sequence numbers
/// rather than assuming a well-behaved caller, and that it is otherwise a
/// plain, faithful append-only log.

#include "vl_test_support.hpp"

#include "volatility_lab/runtime/market_event.hpp"

using namespace vl;

namespace {

MarketEvent make_event(std::uint64_t seq, OptionType type = OptionType::Call) {
    return MarketEvent{
        SequenceNumber{seq},
        Timestamp{static_cast<std::int64_t>(seq) * 1'000'000},
        "SPX",
        "SPXW_241220C05900",
        Years{0.25},
        Strike{5900.0},
        type,
        MarketEventType::Quote,
        Money{12.0},
        Money{12.4},
        natural_mid(Money{12.0}, Money{12.4}),
        10.0,
    };
}

}  // namespace

// ---------------------------------------------------------------------------
// Immutability
// ---------------------------------------------------------------------------

TEST(MarketEvent, IsNotCopyOrMoveAssignable) {
    // Compile-time, not runtime: the static_asserts in market_event.hpp
    // already prove this at header-inclusion time for every translation
    // unit that includes it. This test exists so a *passing build* is
    // itself evidence the property held, and documents the expectation
    // in the one place a test reader would look for it.
    static_assert(!std::is_copy_assignable_v<MarketEvent>);
    static_assert(!std::is_move_assignable_v<MarketEvent>);
    static_assert(std::is_move_constructible_v<MarketEvent>);
    SUCCEED();
}

TEST(MarketEvent, FieldsReadBackExactlyAsConstructed) {
    const auto e = make_event(1);
    EXPECT_EQ(e.sequence.value(), 1u);
    EXPECT_EQ(e.underlier, "SPX");
    EXPECT_EQ(e.strike.value(), 5900.0);
    EXPECT_EQ(e.option_type, OptionType::Call);
    EXPECT_EQ(e.event_type, MarketEventType::Quote);
    EXPECT_DOUBLE_EQ(e.mid.value(), 12.2);
}

// ---------------------------------------------------------------------------
// Strong-type ordering
// ---------------------------------------------------------------------------

TEST(SequenceNumber, OrdersByValue) {
    EXPECT_LT(SequenceNumber{1}, SequenceNumber{2});
    EXPECT_FALSE(SequenceNumber{5} < SequenceNumber{5});
    EXPECT_EQ(SequenceNumber{7}, SequenceNumber{7});
}

TEST(Timestamp, SubtractionGivesNanosecondDuration) {
    const Timestamp a{1'000};
    const Timestamp b{2'500};
    EXPECT_EQ(b - a, 1500);
    EXPECT_EQ(a - b, -1500);
}

// ---------------------------------------------------------------------------
// Event stream: append-only, strictly increasing
// ---------------------------------------------------------------------------

TEST(MarketEventStream, AppendsInIncreasingOrderSucceed) {
    MarketEventStream stream;
    for (std::uint64_t seq = 0; seq < 5; ++seq) {
        const auto result = stream.append(make_event(seq));
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result.value(), seq);
    }
    EXPECT_EQ(stream.size(), 5u);
    for (std::size_t i = 0; i < stream.size(); ++i) {
        EXPECT_EQ(stream[i].sequence.value(), i);
    }
}

TEST(MarketEventStream, OutOfOrderSequenceIsRejectedNotSilentlyAccepted) {
    MarketEventStream stream;
    ASSERT_TRUE(stream.append(make_event(5)).has_value());
    const auto result = stream.append(make_event(3));  // goes backward
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), EventStreamError::SequenceNotIncreasing);
    // The rejected event must not have been appended.
    EXPECT_EQ(stream.size(), 1u);
}

TEST(MarketEventStream, DuplicateSequenceIsRejected) {
    MarketEventStream stream;
    ASSERT_TRUE(stream.append(make_event(5)).has_value());
    const auto result = stream.append(make_event(5));  // same, not greater
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), EventStreamError::SequenceNotIncreasing);
    EXPECT_EQ(stream.size(), 1u);
}

TEST(MarketEventStream, RejectionDoesNotCorruptSubsequentAppends) {
    // A rejected append must not disturb the stream's own notion of "last
    // sequence" -- a later, correctly-ordered event must still succeed.
    MarketEventStream stream;
    ASSERT_TRUE(stream.append(make_event(5)).has_value());
    ASSERT_FALSE(stream.append(make_event(3)).has_value());
    const auto result = stream.append(make_event(6));
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(stream.size(), 2u);
}

TEST(MarketEventStream, EmptyStreamAcceptsAnyFirstSequenceNumber) {
    // There is nothing to be "out of order" relative to yet -- including a
    // sequence number of 0, and including a stream that starts somewhere
    // other than 0 (a replay resuming mid-stream, say).
    MarketEventStream stream;
    EXPECT_TRUE(stream.append(make_event(42)).has_value());
}

TEST(MarketEventStream, IterationVisitsEveryEventInAppendOrder) {
    MarketEventStream stream;
    for (std::uint64_t seq : {1u, 2u, 3u}) {
        ASSERT_TRUE(stream.append(make_event(seq)).has_value());
    }
    std::vector<std::uint64_t> seen;
    for (const auto& e : stream) seen.push_back(e.sequence.value());
    EXPECT_EQ(seen, (std::vector<std::uint64_t>{1, 2, 3}));
}
