// SPDX-License-Identifier: MIT
#include "volatility_lab/options/quote.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <set>

namespace vl {

std::string quote_label(const OptionQuote& q) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4gy-%c%.6g", q.years,
                  q.type == OptionType::Call ? 'C' : 'P', q.strike);
    return std::string(buf);
}

// ===========================================================================
// QuoteBook
// ===========================================================================

void QuoteBook::resize(std::size_t n) {
    // Every column is padded to a whole number of vector registers so the SIMD
    // kernels need no masked epilogue.  The pad value matters: a zero weight
    // makes a pad lane contribute nothing to any reduction, and a strike of 1
    // (rather than 0) keeps log(K/F) finite if a pad lane is ever read by
    // accident.  Both are belt-and-braces -- the kernels only read pad lanes
    // for values they discard -- but a NaN in a pad lane would poison a
    // vectorised reduction and be very hard to trace.
    constexpr std::size_t lanes = kMaxVectorLanesF64;
    strike_.resize(n, lanes, 1.0);
    forward_.resize(n, lanes, 1.0);
    years_.resize(n, lanes, 1.0);
    implied_vol_.resize(n, lanes, 0.0);
    discount_.resize(n, lanes, 1.0);
    log_moneyness_.resize(n, lanes, 0.0);
    total_variance_.resize(n, lanes, 0.0);
    mid_.resize(n, lanes, 0.0);
    weight_.resize(n, lanes, 0.0);
    type_sign_.resize(n, lanes, static_cast<std::int8_t>(1));
    slice_index_.resize(n, lanes, 0u);
    size_ = n;
    slices_.clear();
}

QuoteBook QuoteBook::from_quotes(std::span<const OptionQuote> quotes) {
    QuoteBook b(quotes.size());
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        const OptionQuote& q = quotes[i];
        b.strike_[i] = q.strike;
        b.forward_[i] = q.forward;
        b.years_[i] = q.years;
        b.implied_vol_[i] = q.implied_vol;
        b.discount_[i] = q.discount;
        b.log_moneyness_[i] = q.log_moneyness;
        b.total_variance_[i] = q.total_variance;
        b.mid_[i] = q.mid;
        b.weight_[i] = q.weight;
        b.type_sign_[i] = static_cast<std::int8_t>(q.type);
        b.slice_index_[i] = q.slice_index;
    }
    b.sort_and_group();
    return b;
}

std::vector<OptionQuote> QuoteBook::to_quotes() const {
    std::vector<OptionQuote> out(size_);
    for (std::size_t i = 0; i < size_; ++i) {
        OptionQuote& q = out[i];
        q.strike = strike_[i];
        q.forward = forward_[i];
        q.years = years_[i];
        q.implied_vol = implied_vol_[i];
        q.discount = discount_[i];
        q.log_moneyness = log_moneyness_[i];
        q.total_variance = total_variance_[i];
        q.mid = mid_[i];
        q.weight = weight_[i];
        q.type = static_cast<OptionType>(type_sign_[i]);
        q.slice_index = slice_index_[i];
        q.source_index = static_cast<std::uint32_t>(i);
        q.status = (weight_[i] > 0.0) ? QuoteStatus::Ok : QuoteStatus::Rejected;
    }
    return out;
}

QuoteBookView QuoteBook::view() const noexcept {
    QuoteBookView v;
    v.strike = strike_.view();
    v.forward = forward_.view();
    v.years = years_.view();
    v.implied_vol = implied_vol_.view();
    v.discount = discount_.view();
    v.log_moneyness = log_moneyness_.view();
    v.total_variance = total_variance_.view();
    v.mid = mid_.view();
    v.weight = weight_.view();
    v.type_sign = type_sign_.view();
    v.slice_index = slice_index_.view();
    return v;
}

void QuoteBook::sort_and_group() {
    slices_.clear();
    if (size_ == 0) return;

    // Sort by (expiry, strike, type) via a permutation, then apply it to every
    // column.  Sorting a permutation rather than the rows themselves is the
    // point of SoA: there is no 152-byte row to swap, just eleven gathers.
    //
    // std::stable_sort, not std::sort: two quotes can share an expiry, strike
    // and type (a call and a put at the same point, or a duplicate from the
    // feed), and a stable order means the resulting book -- and therefore every
    // downstream fit and every diagnostic index -- is a function of the input
    // alone.  An unstable sort would make the output depend on the library's
    // pivot choices, which is exactly the kind of irreproducibility the
    // determinism tests exist to catch.
    std::vector<std::uint32_t> perm(size_);
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(), [&](std::uint32_t a, std::uint32_t b) {
        if (years_[a] != years_[b]) return years_[a] < years_[b];
        if (strike_[a] != strike_[b]) return strike_[a] < strike_[b];
        return type_sign_[a] < type_sign_[b];
    });

    const auto apply = [&](auto& column) {
        using T = typename std::decay_t<decltype(column)>::value_type;
        AlignedBuffer<T> tmp(size_, kMaxVectorLanesF64);
        for (std::size_t i = 0; i < size_; ++i) tmp[i] = column[perm[i]];
        column = std::move(tmp);
    };
    apply(strike_);
    apply(forward_);
    apply(years_);
    apply(implied_vol_);
    apply(discount_);
    apply(log_moneyness_);
    apply(total_variance_);
    apply(mid_);
    apply(weight_);
    apply(type_sign_);
    apply(slice_index_);

    // Group into contiguous expiry ranges.  Exact equality on the expiry is
    // intended: expiries come from a calendar, not from arithmetic, so two
    // quotes on the same expiry carry bit-identical year fractions.  Grouping
    // with a tolerance would merge genuinely distinct near-dated expiries --
    // weekly options a day apart differ by 0.0027 -- which is worse.
    std::size_t begin = 0;
    for (std::size_t i = 1; i <= size_; ++i) {
        if (i == size_ || years_[i] != years_[begin]) {
            slices_.push_back(SliceRange{years_[begin], begin, i});
            begin = i;
        }
    }
    for (std::size_t s = 0; s < slices_.size(); ++s) {
        for (std::size_t i = slices_[s].begin; i < slices_[s].end; ++i) {
            slice_index_[i] = static_cast<std::uint32_t>(s);
        }
    }
}

// ===========================================================================
// MarketSnapshot
// ===========================================================================

std::vector<double> MarketSnapshot::expiries() const {
    std::set<double> distinct;
    for (const auto& q : quotes) distinct.insert(q.years);
    return std::vector<double>(distinct.begin(), distinct.end());
}

std::size_t MarketSnapshot::count(QuoteStatus s) const noexcept {
    std::size_t n = 0;
    for (const auto& q : quotes) {
        if (q.status == s) ++n;
    }
    return n;
}

}  // namespace vl
