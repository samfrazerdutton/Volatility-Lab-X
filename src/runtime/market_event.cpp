// SPDX-License-Identifier: MIT
#include "volatility_lab/runtime/market_event.hpp"

namespace vl {

Expected<std::size_t, EventStreamError> MarketEventStream::append(MarketEvent event) {
    if (has_last_ && !(event.sequence > last_sequence_)) {
        return make_unexpected(EventStreamError::SequenceNotIncreasing);
    }
    last_sequence_ = event.sequence;
    has_last_ = true;
    events_.push_back(std::move(event));
    return events_.size() - 1;
}

}  // namespace vl
