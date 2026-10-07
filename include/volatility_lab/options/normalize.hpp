// SPDX-License-Identifier: MIT
#pragma once
/// \file normalize.hpp
/// \brief Market-data normalisation: from a dirty feed to fittable data, with
///        every rejection accounted for.
///
/// ## The contract
///
/// Nothing here silently accepts bad input, and nothing here silently discards
/// it either.  Every quote comes out with a `QuoteStatus` and, if anything was
/// wrong with it, one or more `Diagnostic` records naming the quote, the field,
/// the observed value and the admissible range.  A snapshot of 50,000 quotes
/// with 4,000 problems produces 4,000 structured records and a grouped summary
/// -- not a log line, and not a count.
///
/// The reason is operational rather than aesthetic.  When a surface fit comes
/// out wrong at 09:31, the question is always "which quotes went into it", and
/// a normaliser that answers "about forty thousand" has not helped.
///
/// ## The pipeline
///
///     raw quotes
///       |- 1. shape         finite, positive, European, not duplicated
///       |- 2. quote quality crossed, locked, zero-bid, stale, wide
///       |- 3. forward       derive F and DF if not supplied
///       |- 4. price bounds  intrinsic <= price <= bound, else reject
///       |- 5. invert        implied volatility, OTM side only
///       |- 6. weight        see calibration/weights.hpp
///       `- 7. group         sort by (expiry, strike), build slice ranges
///
/// The order is not arbitrary.  Each stage depends on the previous one having
/// succeeded, and the price-bound check (4) must precede the inversion (5)
/// because the inversion of an out-of-bounds price has no solution -- feeding
/// it one and interpreting the failure is strictly worse than checking first.
///
/// ## Which price is fitted
///
/// The mid, by default -- but the *OTM* mid.  For a strike below the forward
/// the put is out of the money and the call is intrinsic plus a sliver, so the
/// call's mid carries almost no volatility information while its bid/ask
/// spread carries all the noise.  Normalisation picks the OTM side at each
/// strike and converts through put-call parity, which is the single largest
/// improvement available to the quality of a surface fit and costs nothing.
///
/// ## Bad data is not an exception
///
/// A crossed market is a Tuesday, not an error condition.  So normalisation
/// returns results rather than throwing, and the caller decides whether the
/// surviving quotes are enough to fit.  `NormalizationResult::usable()` is the
/// question most callers actually want answered.

#include <cstdint>
#include <span>
#include <vector>

#include "volatility_lab/core/diagnostics.hpp"
#include "volatility_lab/options/quote.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Thresholds for the quote-quality stage.
///
/// Every one of these is a *policy* choice rather than a mathematical fact, so
/// they are configuration rather than constants, and the defaults are stated
/// with the reasoning rather than left as magic numbers.
struct NormalizationConfig {
    /// Reject a quote whose relative bid/ask spread exceeds this.
    ///
    /// 2.0 means the ask is three times the bid.  Deliberately loose: deep
    /// wings legitimately trade 1 bid / 5 ask, and throwing those away
    /// truncates the smile exactly where its shape is most informative.  The
    /// weighting model (not this threshold) is what stops a wide quote from
    /// dominating the fit.
    double max_relative_spread = 2.0;

    /// Reject if the absolute spread exceeds this fraction of the forward.
    /// Catches the case the relative test misses: a 0.01/0.02 quote has a 100%
    /// relative spread but is perfectly good information.
    double max_spread_over_forward = 0.10;

    /// Quotes older than this are marked `Degraded` and down-weighted, not
    /// rejected -- a stale quote in the wings is still better than no quote.
    /// Negative `age_seconds` means unknown, which is treated as fresh.
    double max_age_seconds = 300.0;

    /// Reject expiries shorter than this.  At 1e-5 years (five minutes) the
    /// inversion is still well posed but the quote is dominated by
    /// microstructure; below that the OTM price underflows and carries no
    /// volatility information at all.
    double min_years = 1.0e-5;

    /// Reject expiries longer than this.  Ten years is past the liquid listed
    /// market for every underlying this library targets.
    double max_years = 10.0;

    /// Volatilities outside this band are rejected as implausible rather than
    /// returned.  Wide on purpose: 5% to 500% covers everything from a
    /// rates-like underlying to a single stock into earnings.
    double min_vol = 0.005;
    double max_vol = 5.0;

    /// Reject a quote whose |log(K/F)| exceeds this.  At 3.0 the strike is
    /// e^3 = 20x the forward; beyond that the option is worth less than a tick
    /// and the quote is a placeholder.
    double max_abs_log_moneyness = 3.0;

    /// A slice with fewer than this many usable quotes cannot identify five
    /// SVI parameters and is reported as `SliceTooFewQuotes`.  Five is the
    /// bare minimum; the calibrator falls back to fewer parameters below it
    /// rather than fitting noise.
    std::size_t min_quotes_per_slice = 5;

    /// Treat a zero bid as a rejection rather than a degradation.  Off by
    /// default: a zero bid with a positive ask is the normal state of a deep
    /// wing, and the *ask* still bounds the volatility from above.
    bool reject_zero_bid = false;

    /// Use the OTM side at each strike, converting through put-call parity.
    /// On by default; see the header comment for why.
    bool prefer_otm_side = true;

    /// Accept a quote that has an implied volatility supplied directly, rather
    /// than requiring a price to invert.  Used by the synthetic generator and
    /// by feeds that publish vols.
    bool accept_supplied_vol = true;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

struct NormalizationStats {
    std::size_t received = 0;
    std::size_t accepted = 0;
    std::size_t degraded = 0;
    std::size_t rejected = 0;
    std::size_t slices = 0;

    /// Rejections by cause, so a report can say *why* a snapshot was thin
    /// rather than only that it was.
    std::size_t rejected_shape = 0;
    std::size_t rejected_quality = 0;
    std::size_t rejected_bounds = 0;
    std::size_t rejected_inversion = 0;
    std::size_t rejected_vol_range = 0;

    [[nodiscard]] double acceptance_rate() const noexcept {
        return received > 0 ? static_cast<double>(accepted) / static_cast<double>(received)
                            : 0.0;
    }
};

struct NormalizationResult {
    /// The surviving quotes, with derived fields filled in, sorted by
    /// (expiry, strike).  Rejected quotes are *excluded* from here but counted
    /// in `stats` and described in `diagnostics`.
    std::vector<OptionQuote> quotes;

    /// Every quote, in input order, with its status set.  Kept separately so a
    /// caller can audit what happened to a specific input without re-running
    /// anything.
    std::vector<OptionQuote> audit;

    DiagnosticSink diagnostics;
    NormalizationStats stats;

    /// Distinct expiries that survived, with at least `min_quotes_per_slice`
    /// usable quotes each.
    std::vector<double> fittable_expiries;

    [[nodiscard]] bool usable() const noexcept {
        return !quotes.empty() && diagnostics.usable() && !fittable_expiries.empty();
    }

    /// The SoA form, built once.
    [[nodiscard]] QuoteBook to_book() const { return QuoteBook::from_quotes(quotes); }
};

// ---------------------------------------------------------------------------
// The entry point
// ---------------------------------------------------------------------------

/// Normalise a snapshot.  Never throws; all problems are reported.
[[nodiscard]] NormalizationResult normalize(const MarketSnapshot& snapshot,
                                            const NormalizationConfig& cfg = {});

// ---------------------------------------------------------------------------
// Individual stages, exposed for testing and for callers with partial data
// ---------------------------------------------------------------------------

/// Derive the forward and discount factor from spot, rate and dividend, if
/// they were not supplied.
///
/// A supplied forward always wins. Feeds that publish a forward have usually
/// derived it from the put-call parity of the listed options themselves, which
/// is strictly better information than S*exp((r-q)T) with a guessed dividend --
/// and using the better number is what makes the put and call at a given
/// strike imply the same volatility.
void derive_forward_and_discount(OptionQuote& q) noexcept;

/// Which side of the market to fit at this strike, and the resulting
/// undiscounted OTM price.
struct OtmSelection {
    OptionType side = OptionType::Call;
    double undiscounted_price = 0.0;
    bool converted_through_parity = false;
};

[[nodiscard]] OtmSelection select_otm_side(const OptionQuote& q,
                                           const NormalizationConfig& cfg) noexcept;

/// Check the shape and quality of one quote, appending diagnostics.  Returns
/// the status it concluded.
[[nodiscard]] QuoteStatus validate_quote(const OptionQuote& q,
                                         const NormalizationConfig& cfg,
                                         DiagnosticSink& diags,
                                         std::uint32_t index);

}  // namespace vl
