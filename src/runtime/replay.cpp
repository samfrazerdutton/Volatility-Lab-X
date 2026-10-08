// SPDX-License-Identifier: MIT
#include "volatility_lab/runtime/replay.hpp"

namespace vl {

void MarketState::apply(const MarketEvent& event) {
    InstrumentState& s = instruments_[event.instrument];
    s.underlier = event.underlier;
    s.expiry = event.expiry;
    s.strike = event.strike;
    s.option_type = event.option_type;
    s.bid = event.bid;
    s.ask = event.ask;
    s.mid = event.mid;
    s.size = event.size;
    s.last_event_type = event.event_type;
    s.last_sequence = event.sequence;
    s.last_timestamp = event.timestamp;
}

const InstrumentState* MarketState::find(const std::string& instrument) const noexcept {
    const auto it = instruments_.find(instrument);
    return (it != instruments_.end()) ? &it->second : nullptr;
}

StateHash MarketState::fingerprint() const noexcept {
    StateHasher h;
    // instruments_ is a std::map: iteration order is the key's total
    // order, identical on every run -- see the header comment for why that
    // is the whole point of using one here.
    for (const auto& [instrument, s] : instruments_) {
        h.combine(std::string_view(instrument));
        h.combine(std::string_view(s.underlier));
        h.combine(s.expiry.value());
        h.combine(s.strike.value());
        h.combine(static_cast<std::uint64_t>(static_cast<std::int8_t>(s.option_type)));
        h.combine(s.bid.value());
        h.combine(s.ask.value());
        h.combine(s.mid.value());
        h.combine(s.size);
        h.combine(static_cast<std::uint64_t>(static_cast<std::uint8_t>(s.last_event_type)));
        h.combine(s.last_sequence.value());
        h.combine(s.last_timestamp.nanoseconds_since_epoch());
    }
    return h.finish();
}

ReplayResult replay(const MarketEventStream& stream) {
    ReplayResult out;
    out.steps.reserve(stream.size());
    for (const auto& event : stream) {
        out.final_state.apply(event);
        out.steps.push_back(ReplayStep{event.sequence, out.final_state.fingerprint()});
    }
    return out;
}

}  // namespace vl
