// SPDX-License-Identifier: MIT
/// Validates deterministic replay (directive section 5): that two
/// independently-built streams carrying the same event data produce
/// identical state hashes at every step, that a single perturbed field
/// changes the fingerprint, and that MarketState::apply behaves exactly as
/// documented (unconditional overwrite, no special-casing by event type).

#include "vl_test_support.hpp"

#include "volatility_lab/runtime/replay.hpp"

using namespace vl;

namespace {

MarketEvent make_event(std::uint64_t seq, double bid, double ask, const char* instrument = "A",
                       MarketEventType type = MarketEventType::Quote, double size = 10.0) {
    return MarketEvent{
        SequenceNumber{seq},
        Timestamp{static_cast<std::int64_t>(seq) * 1000},
        "SPX",
        instrument,
        Years{0.25},
        Strike{5900.0},
        OptionType::Call,
        type,
        Money{bid},
        Money{ask},
        natural_mid(Money{bid}, Money{ask}),
        size,
    };
}

MarketEventStream build_three_instrument_stream() {
    MarketEventStream s;
    EXPECT_TRUE(s.append(make_event(0, 12.0, 12.4, "A")).has_value());
    EXPECT_TRUE(s.append(make_event(1, 5.0, 5.2, "B")).has_value());
    EXPECT_TRUE(s.append(make_event(2, 12.1, 12.5, "A")).has_value());
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// MarketState::apply
// ---------------------------------------------------------------------------

TEST(MarketState, ApplyCreatesANewInstrumentEntry) {
    MarketState state;
    state.apply(make_event(0, 12.0, 12.4, "A"));
    ASSERT_EQ(state.size(), 1u);
    const auto* s = state.find("A");
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->bid.value(), 12.0);
    EXPECT_EQ(s->ask.value(), 12.4);
    EXPECT_EQ(s->underlier, "SPX");
}

TEST(MarketState, SecondEventForSameInstrumentOverwritesEveryField) {
    MarketState state;
    state.apply(make_event(0, 12.0, 12.4, "A"));
    state.apply(make_event(1, 13.0, 13.4, "A"));
    ASSERT_EQ(state.size(), 1u);  // still one instrument, not two
    const auto* s = state.find("A");
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->bid.value(), 13.0);
    EXPECT_EQ(s->ask.value(), 13.4);
    EXPECT_EQ(s->last_sequence.value(), 1u);
}

TEST(MarketState, TradeEventOverwritesBidAskUnconditionally) {
    // Deliberately simple per the header comment: no special-casing of
    // Trade vs Quote at this layer. A Trade event's bid/ask, even if the
    // feed sent zeros, replaces whatever was there.
    MarketState state;
    state.apply(make_event(0, 12.0, 12.4, "A", MarketEventType::Quote));
    state.apply(make_event(1, 0.0, 0.0, "A", MarketEventType::Trade, 3.0));
    const auto* s = state.find("A");
    ASSERT_NE(s, nullptr);
    EXPECT_EQ(s->bid.value(), 0.0);
    EXPECT_EQ(s->ask.value(), 0.0);
    EXPECT_EQ(s->size, 3.0);
    EXPECT_EQ(s->last_event_type, MarketEventType::Trade);
}

TEST(MarketState, UnknownInstrumentFindReturnsNull) {
    MarketState state;
    state.apply(make_event(0, 12.0, 12.4, "A"));
    EXPECT_EQ(state.find("does-not-exist"), nullptr);
}

TEST(MarketState, DifferentInstrumentsAreTrackedSeparately) {
    MarketState state;
    state.apply(make_event(0, 12.0, 12.4, "A"));
    state.apply(make_event(1, 5.0, 5.2, "B"));
    EXPECT_EQ(state.size(), 2u);
    EXPECT_EQ(state.find("A")->bid.value(), 12.0);
    EXPECT_EQ(state.find("B")->bid.value(), 5.0);
}

// ---------------------------------------------------------------------------
// Determinism: the central claim
// ---------------------------------------------------------------------------

TEST(Replay, TwoIndependentlyBuiltStreamsWithTheSameDataFingerprintIdenticallyAtEveryStep) {
    const auto stream_a = build_three_instrument_stream();
    const auto stream_b = build_three_instrument_stream();

    const auto replay_a = replay(stream_a);
    const auto replay_b = replay(stream_b);

    ASSERT_EQ(replay_a.steps.size(), replay_b.steps.size());
    for (std::size_t i = 0; i < replay_a.steps.size(); ++i) {
        EXPECT_EQ(replay_a.steps[i].sequence.value(), replay_b.steps[i].sequence.value());
        EXPECT_TRUE(replay_a.steps[i].state_hash == replay_b.steps[i].state_hash)
            << "diverged at step " << i;
    }
    EXPECT_TRUE(replay_a.final_state.fingerprint() == replay_b.final_state.fingerprint());
}

TEST(Replay, CallingReplayTwiceOnTheSameStreamGivesTheSameResult) {
    const auto stream = build_three_instrument_stream();
    const auto first = replay(stream);
    const auto second = replay(stream);
    ASSERT_EQ(first.steps.size(), second.steps.size());
    for (std::size_t i = 0; i < first.steps.size(); ++i) {
        EXPECT_TRUE(first.steps[i].state_hash == second.steps[i].state_hash);
    }
}

TEST(Replay, APerturbedFieldInOneEventChangesTheFinalFingerprint) {
    // Sanity check on the hash itself: it must actually be sensitive to the
    // state, not accidentally constant or insensitive to a small change.
    auto base = build_three_instrument_stream();
    MarketEventStream perturbed;
    ASSERT_TRUE(perturbed.append(make_event(0, 12.0, 12.4, "A")).has_value());
    ASSERT_TRUE(perturbed.append(make_event(1, 5.0, 5.2, "B")).has_value());
    ASSERT_TRUE(perturbed.append(make_event(2, 12.1, 12.50001, "A")).has_value());  // tiny change

    const auto r_base = replay(base);
    const auto r_perturbed = replay(perturbed);
    EXPECT_FALSE(r_base.final_state.fingerprint() == r_perturbed.final_state.fingerprint());
}

TEST(Replay, StepFingerprintsFormAStrictSequenceMatchingTheEventSequenceNumbers) {
    const auto stream = build_three_instrument_stream();
    const auto result = replay(stream);
    ASSERT_EQ(result.steps.size(), 3u);
    EXPECT_EQ(result.steps[0].sequence.value(), 0u);
    EXPECT_EQ(result.steps[1].sequence.value(), 1u);
    EXPECT_EQ(result.steps[2].sequence.value(), 2u);
    // The fingerprint after the last step must equal the final state's own
    // fingerprint -- the running record and the end state must agree.
    EXPECT_TRUE(result.steps.back().state_hash == result.final_state.fingerprint());
}

TEST(Replay, EmptyStreamReplaysToAnEmptyStateWithAWellDefinedFingerprint) {
    const MarketEventStream empty;
    const auto result = replay(empty);
    EXPECT_TRUE(result.final_state.empty());
    EXPECT_TRUE(result.steps.empty());
    // Still well-defined and reproducible, not undefined/garbage.
    const auto result2 = replay(empty);
    EXPECT_TRUE(result.final_state.fingerprint() == result2.final_state.fingerprint());
}

// ---------------------------------------------------------------------------
// StateHash / StateHasher
// ---------------------------------------------------------------------------

TEST(StateHasher, SameSequenceOfCombinesProducesTheSameHash) {
    StateHasher h1, h2;
    h1.combine(std::uint64_t{42}).combine(3.14).combine(std::string_view("abc"));
    h2.combine(std::uint64_t{42}).combine(3.14).combine(std::string_view("abc"));
    EXPECT_TRUE(h1.finish() == h2.finish());
}

TEST(StateHasher, DifferentOrderOfTheSameValuesProducesADifferentHash) {
    StateHasher h1, h2;
    h1.combine(std::uint64_t{1}).combine(std::uint64_t{2});
    h2.combine(std::uint64_t{2}).combine(std::uint64_t{1});
    EXPECT_FALSE(h1.finish() == h2.finish());
}

TEST(StateHasher, AmbiguousStringConcatenationDoesNotCollide) {
    // The classic hash bug this guards against: combining "ab","c" must not
    // equal combining "a","bc" -- the length-prefixing in
    // StateHasher::combine(string_view) is what prevents it.
    StateHasher h1, h2;
    h1.combine(std::string_view("ab")).combine(std::string_view("c"));
    h2.combine(std::string_view("a")).combine(std::string_view("bc"));
    EXPECT_FALSE(h1.finish() == h2.finish());
}

TEST(StateHash, ToHexProducesSixteenLowercaseHexDigits) {
    const StateHash h{0x9d8e3f1a2b4c5d6eULL};
    EXPECT_EQ(h.to_hex(), "9d8e3f1a2b4c5d6e");
}

TEST(StateHash, ZeroHashesToSixteenZeroDigits) {
    const StateHash h{0};
    EXPECT_EQ(h.to_hex(), "0000000000000000");
}
