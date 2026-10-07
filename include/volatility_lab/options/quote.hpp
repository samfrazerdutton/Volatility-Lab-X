// SPDX-License-Identifier: MIT
#pragma once
/// \file quote.hpp
/// \brief The market snapshot: one option quote, and a batch of them.
///
/// ## Two layouts, and why both exist
///
/// `OptionQuote` is an array-of-structures record: one option, all its fields
/// adjacent.  `QuoteBook` is the structure-of-arrays form: one contiguous
/// column per field.  Both are provided, with explicit conversions, because
/// they are optimal for different access patterns and the difference is
/// measurable:
///
///  * **AoS is right at the boundary.**  A market feed delivers one quote at a
///    time; a CSV row is one quote; a diagnostic names one quote.  Code that
///    touches every field of one option wants them in one cache line.
///
///  * **SoA is right in the kernels.**  A batch pricer reads strike, forward,
///    expiry and vol and writes price -- five of the twenty-two fields.  In AoS
///    that is a 168-byte stride, so every option touches three cache lines and
///    brings in seventeen fields nobody asked for; the loop cannot vectorise
///    because the data it needs is not adjacent.  In SoA each column is dense,
///    the prefetcher sees five linear streams, and the loop vectorises.
///
/// The crossover is measured in benchmarks/pricing (`--layout`), not asserted
/// here; the expected shape is that AoS and SoA are indistinguishable while
/// the batch fits in L1 and diverge once it does not, because the difference
/// is bandwidth, not instruction count.
///
/// ## Field layout of OptionQuote
///
/// Fields are ordered largest-alignment-first so there is no interior padding,
/// and the hot five (strike, forward, years, vol, type) are placed together at
/// the front so that even the AoS path touches as few lines as possible.  The
/// struct is 168 bytes -- just under three cache lines -- and the
/// `static_assert` below pins that, so a field added in the wrong place fails
/// the build rather than silently costing a line in every batch pricer.
///
/// ## Ownership
///
/// `QuoteBook` owns its columns (over-aligned, see core/aligned_buffer.hpp).
/// `QuoteBookView` is a borrowed, non-owning set of spans over them -- that is
/// what kernels take, so a kernel cannot extend the lifetime of the data it
/// reads and cannot be handed a copy by accident.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "volatility_lab/core/aligned_buffer.hpp"
#include "volatility_lab/core/config.hpp"
#include "volatility_lab/core/types.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Quote status
// ---------------------------------------------------------------------------

/// What normalisation concluded about a quote.  A quote is never silently
/// dropped: it is marked, and the mark travels with it, so a downstream
/// report can say how many quotes were used out of how many received.
enum class QuoteStatus : std::uint8_t {
    Unvalidated = 0,  ///< straight off the feed, not yet checked
    Ok = 1,           ///< usable, full weight
    Degraded = 2,     ///< usable but down-weighted (wide, stale, illiquid)
    Rejected = 3      ///< unusable; excluded from the fit
};

[[nodiscard]] constexpr const char* to_string(QuoteStatus s) noexcept {
    switch (s) {
        case QuoteStatus::Unvalidated: return "unvalidated";
        case QuoteStatus::Ok:          return "ok";
        case QuoteStatus::Degraded:    return "degraded";
        case QuoteStatus::Rejected:    return "rejected";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// OptionQuote
// ---------------------------------------------------------------------------

/// One option quote as received, plus the fields normalisation derives.
///
/// Both the raw inputs and the derived quantities live here on purpose.  The
/// alternative -- a separate "normalised quote" type -- means every diagnostic
/// has to carry a back-reference to find the raw values it is complaining
/// about, and every round trip through the pipeline risks the two drifting out
/// of sync.
struct OptionQuote {
    // -- the hot five, placed first ---------------------------------------
    double strike = 0.0;
    double forward = 0.0;     ///< derived if not supplied: S*exp((r-q)T)
    double years = 0.0;       ///< time to expiry, year fraction
    double implied_vol = 0.0; ///< derived by normalisation from the mid price
    OptionType type = OptionType::Call;
    ExerciseStyle style = ExerciseStyle::European;
    QuoteStatus status = QuoteStatus::Unvalidated;
    std::uint8_t pad0_ = 0;
    std::uint32_t pad1_ = 0;

    // -- market inputs -----------------------------------------------------
    double spot = 0.0;
    double bid = 0.0;
    double ask = 0.0;
    double last = 0.0;
    double rate = 0.0;      ///< continuously compounded
    double dividend = 0.0;  ///< continuous yield / borrow
    double volume = 0.0;
    double open_interest = 0.0;

    /// Seconds since the quote was last updated.  Used by the staleness check
    /// and the weighting model; a negative value means "unknown", which is
    /// treated as "not stale" rather than as an error.
    double age_seconds = -1.0;

    // -- derived by normalisation ------------------------------------------
    double mid = 0.0;              ///< the price actually fitted
    double discount = 1.0;         ///< DF to the payment date
    double log_moneyness = 0.0;    ///< k = log(K/F)
    double total_variance = 0.0;   ///< implied_vol^2 * years
    double vega = 0.0;             ///< dPrice/dVol at the implied vol
    double weight = 0.0;           ///< calibration weight; 0 means excluded

    /// Index of this quote in the book it came from, so a diagnostic can point
    /// back at it after the book has been filtered or reordered.
    std::uint32_t source_index = 0;
    std::uint32_t slice_index = 0;  ///< which expiry group it belongs to
};

// The layout claims above are load-bearing, so they are checked rather than
// described.  If a field is added in the wrong place the build fails here
// instead of the cost showing up as an unexplained benchmark regression.
static_assert(std::is_trivially_copyable_v<OptionQuote>);
static_assert(std::is_standard_layout_v<OptionQuote>);
static_assert(sizeof(OptionQuote) == 168,
              "OptionQuote size changed: re-check the field ordering, and update "
              "the AoS/SoA stride figures in docs/performance-report.md");
static_assert(offsetof(OptionQuote, strike) == 0, "the hot fields must stay first");

/// A human-readable identifier, built on demand rather than stored.
///
/// Storing a string per quote would make `OptionQuote` non-trivially-copyable
/// and put a heap allocation on the feed path, so the name is formatted only
/// when a diagnostic actually needs one.  Format: `<T>y-<C|P><strike>`, e.g.
/// `0.25y-C5000`.
[[nodiscard]] std::string quote_label(const OptionQuote& q);

// ---------------------------------------------------------------------------
// QuoteBook: the SoA form
// ---------------------------------------------------------------------------

/// Borrowed columns over a `QuoteBook`.  What kernels take.
///
/// Non-owning by design: a kernel handed a view cannot extend the lifetime of
/// the data, cannot copy it by accident, and cannot resize it.  Every span has
/// the same length, which is `size`.
struct QuoteBookView {
    std::span<const double> strike;
    std::span<const double> forward;
    std::span<const double> years;
    std::span<const double> implied_vol;
    std::span<const double> discount;
    std::span<const double> log_moneyness;
    std::span<const double> total_variance;
    std::span<const double> mid;
    std::span<const double> weight;
    std::span<const std::int8_t> type_sign;  ///< +1 call, -1 put: the branch-free multiplier
    std::span<const std::uint32_t> slice_index;

    [[nodiscard]] std::size_t size() const noexcept { return strike.size(); }
    [[nodiscard]] bool empty() const noexcept { return strike.empty(); }
};

/// Structure-of-arrays market snapshot.
///
/// Columns are 64-byte aligned with vector tail padding, so the SIMD kernels
/// can process ceil(n/W) registers with aligned loads and no masked epilogue.
/// See core/aligned_buffer.hpp for why that matters and what it costs.
class QuoteBook {
  public:
    QuoteBook() = default;
    explicit QuoteBook(std::size_t n) { resize(n); }

    void resize(std::size_t n);
    void clear() { resize(0); }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Build from AoS records.  This is the one place the conversion happens,
    /// so the transpose cost is paid once per snapshot rather than per kernel.
    static QuoteBook from_quotes(std::span<const OptionQuote> quotes);

    /// Copy back to AoS, for reporting and for the CSV writer.
    [[nodiscard]] std::vector<OptionQuote> to_quotes() const;

    [[nodiscard]] QuoteBookView view() const noexcept;

    // Mutable column access, for the normaliser and the synthetic generator.
    [[nodiscard]] std::span<double> strike() noexcept { return strike_.view(); }
    [[nodiscard]] std::span<double> forward() noexcept { return forward_.view(); }
    [[nodiscard]] std::span<double> years() noexcept { return years_.view(); }
    [[nodiscard]] std::span<double> implied_vol() noexcept { return implied_vol_.view(); }
    [[nodiscard]] std::span<double> discount() noexcept { return discount_.view(); }
    [[nodiscard]] std::span<double> log_moneyness() noexcept {
        return log_moneyness_.view();
    }
    [[nodiscard]] std::span<double> total_variance() noexcept {
        return total_variance_.view();
    }
    [[nodiscard]] std::span<double> mid() noexcept { return mid_.view(); }
    [[nodiscard]] std::span<double> weight() noexcept { return weight_.view(); }
    [[nodiscard]] std::span<std::int8_t> type_sign() noexcept { return type_sign_.view(); }
    [[nodiscard]] std::span<std::uint32_t> slice_index() noexcept {
        return slice_index_.view();
    }

    /// Number of distinct expiries, and the half-open row range of each.
    ///
    /// The book is sorted by (expiry, strike) during construction, so each
    /// slice is a contiguous row range.  That is what lets the calibrator and
    /// the batch pricer process a slice with one pass over dense memory
    /// instead of gathering scattered rows.
    struct SliceRange {
        double years = 0.0;
        std::size_t begin = 0;
        std::size_t end = 0;
        [[nodiscard]] std::size_t count() const noexcept { return end - begin; }
    };
    [[nodiscard]] std::span<const SliceRange> slice_ranges() const noexcept {
        return slices_;
    }

    /// Sort by (expiry, strike) and rebuild the slice ranges.  Idempotent.
    void sort_and_group();

  private:
    AlignedBuffer<double> strike_;
    AlignedBuffer<double> forward_;
    AlignedBuffer<double> years_;
    AlignedBuffer<double> implied_vol_;
    AlignedBuffer<double> discount_;
    AlignedBuffer<double> log_moneyness_;
    AlignedBuffer<double> total_variance_;
    AlignedBuffer<double> mid_;
    AlignedBuffer<double> weight_;
    AlignedBuffer<std::int8_t> type_sign_;
    AlignedBuffer<std::uint32_t> slice_index_;
    std::vector<SliceRange> slices_;
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// MarketSnapshot: the AoS form plus its context
// ---------------------------------------------------------------------------

/// A complete market observation: the quotes plus what they are quotes *of*.
struct MarketSnapshot {
    std::string underlying;
    double spot = 0.0;
    double observation_time = 0.0;  ///< seconds since an arbitrary epoch
    std::vector<OptionQuote> quotes;

    [[nodiscard]] std::size_t size() const noexcept { return quotes.size(); }
    [[nodiscard]] bool empty() const noexcept { return quotes.empty(); }

    /// The distinct expiries present, sorted ascending.
    [[nodiscard]] std::vector<double> expiries() const;

    /// Count by status, for the quality report.
    [[nodiscard]] std::size_t count(QuoteStatus s) const noexcept;
};

}  // namespace vl
