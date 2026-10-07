// SPDX-License-Identifier: MIT
#pragma once
/// \file diagnostics.hpp
/// \brief Structured, machine-readable diagnostics.
///
/// The library never signals a data problem by returning NaN, clamping
/// silently, or logging to stderr.  Every rejection carries enough structure
/// for a caller to locate the offending instrument and field, decide whether
/// the severity warrants abandoning the fit, and render a human-readable
/// explanation without string-matching.
///
/// `Diagnostic` is deliberately a flat struct with a small fixed string field
/// rather than a polymorphic hierarchy: diagnostics are produced in bulk (one
/// per bad quote over a 50k-quote snapshot), collected into a vector, and
/// never dispatched on.  A heap-allocated std::string per diagnostic was the
/// first implementation and dominated the cost of normalising a deliberately
/// dirty snapshot; the inline buffer removed that.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace vl {

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

enum class Severity : std::uint8_t {
    Info = 0,     ///< observation worth reporting, no action taken
    Warning = 1,  ///< quote was adjusted or down-weighted but retained
    Error = 2,    ///< quote was rejected; the rest of the snapshot is usable
    Fatal = 3     ///< the snapshot cannot be used at all
};

[[nodiscard]] constexpr const char* to_string(Severity s) noexcept {
    switch (s) {
        case Severity::Info:    return "info";
        case Severity::Warning: return "warning";
        case Severity::Error:   return "error";
        case Severity::Fatal:   return "fatal";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Diagnostic codes
//
// Grouped by decade so that a code range identifies the producing subsystem
// without a lookup table.  Stable values: these appear in regression
// baselines (tests/regression) and in the CLI's JSON output.
// ---------------------------------------------------------------------------

enum class DiagCode : std::uint16_t {
    Ok = 0,

    // --- 1000: market data shape -------------------------------------------
    MissingField = 1000,
    NonFiniteValue = 1001,
    NegativeStrike = 1002,
    NonPositiveSpot = 1003,
    NonPositiveForward = 1004,
    NonPositiveExpiry = 1005,
    ExpiryTooFar = 1006,
    AmericanExerciseUnsupported = 1007,
    DuplicateQuote = 1008,

    // --- 1100: quote quality -----------------------------------------------
    CrossedMarket = 1100,       ///< bid > ask
    LockedMarket = 1101,        ///< bid == ask, zero-width
    ZeroBid = 1102,
    NonPositivePrice = 1103,
    ExcessiveSpread = 1104,
    StaleQuote = 1105,
    ZeroVolumeAndOpenInterest = 1106,
    WideSpreadRelativeToPrice = 1107,

    // --- 1200: no-arbitrage bounds on the price itself ----------------------
    PriceBelowIntrinsic = 1200,
    PriceAboveForwardBound = 1201,
    PriceBelowZero = 1202,

    // --- 1300: implied volatility inversion ---------------------------------
    IvSolverNoBracket = 1300,
    IvSolverMaxIterations = 1301,
    IvBelowFloor = 1302,
    IvAboveCeiling = 1303,
    IvVegaTooSmall = 1304,

    // --- 1400: surface fitting ----------------------------------------------
    SliceTooFewQuotes = 1400,
    CalibrationDidNotConverge = 1401,
    ParameterAtBound = 1402,
    SurfaceExtrapolated = 1403,
    SliceDegenerate = 1404,

    // --- 1500: static arbitrage in the fitted surface ------------------------
    ButterflyArbitrage = 1500,
    CalendarArbitrage = 1501,
    CallPriceNotMonotone = 1502,
    CallPriceNotConvex = 1503,
    PutCallParityViolation = 1504,
    NegativeDensity = 1505,
    TotalVarianceNotIncreasing = 1506,
    DurrlemanConditionViolated = 1507,

    // --- 1600: pricing / risk ------------------------------------------------
    GreekNumericallyUnstable = 1600,
    ScenarioOutOfDomain = 1601,
    PortfolioEmpty = 1602,

    // --- 1700: engine / configuration ----------------------------------------
    BackendUnavailable = 1700,
    DeterminismContractBroken = 1701,
    InvalidConfiguration = 1702,
};

[[nodiscard]] const char* to_string(DiagCode c) noexcept;

/// The subsystem a code belongs to, derived from its numeric range.
[[nodiscard]] const char* diag_subsystem(DiagCode c) noexcept;

// ---------------------------------------------------------------------------
// Diagnostic
// ---------------------------------------------------------------------------

/// A small inline string.  64 bytes holds every message the library produces
/// (the longest is 58 characters); longer text is truncated with a trailing
/// ellipsis rather than allocating.
class ShortText {
  public:
    static constexpr std::size_t kCapacity = 64;

    ShortText() noexcept { buf_[0] = '\0'; }
    ShortText(std::string_view s) noexcept { assign(s); }  // NOLINT(google-explicit-constructor)

    void assign(std::string_view s) noexcept {
        const std::size_t n = std::min(s.size(), kCapacity - 1);
        std::copy_n(s.data(), n, buf_.data());
        buf_[n] = '\0';
        if (n < s.size() && n >= 3) {
            buf_[n - 1] = '.';
            buf_[n - 2] = '.';
            buf_[n - 3] = '.';
        }
    }

    [[nodiscard]] const char* c_str() const noexcept { return buf_.data(); }
    [[nodiscard]] std::string_view view() const noexcept { return {buf_.data()}; }
    [[nodiscard]] bool empty() const noexcept { return buf_[0] == '\0'; }

  private:
    std::array<char, kCapacity> buf_{};
};

/// One structured complaint about one piece of input or output.
///
/// `observed`, `expected_lo`, `expected_hi` form the "value vs admissible
/// range" triple required by the spec.  A code that has no natural range
/// leaves the bounds as NaN, and the formatter omits them.
struct Diagnostic {
    DiagCode code = DiagCode::Ok;
    Severity severity = Severity::Info;

    /// Index of the offending quote within the snapshot it came from, or
    /// `kNoIndex` when the diagnostic is about the snapshot as a whole.
    static constexpr std::uint32_t kNoIndex = 0xFFFFFFFFu;
    std::uint32_t quote_index = kNoIndex;

    ShortText instrument;  ///< e.g. "SPX 2026-12-18 C5000"
    ShortText field;       ///< e.g. "bid", "implied_vol", "theta_slice_3"

    double observed = 0.0;
    double expected_lo = qnan();
    double expected_hi = qnan();

    ShortText explanation;

    [[nodiscard]] static double qnan() noexcept;

    /// Single-line human-readable rendering, e.g.
    ///   error  [1100 CrossedMarket] SPX-2026-12-18-C5000.bid: observed 12.4,
    ///          expected <= 12.1 -- bid exceeds ask
    [[nodiscard]] std::string format() const;
};

/// A collection of diagnostics plus the aggregate verdict.
///
/// Thread-safety: `DiagnosticSink` is not internally synchronised.  Parallel
/// producers each own a sink and the results are merged in deterministic index
/// order by `merge_ordered`, so the final vector does not depend on thread
/// scheduling.  See docs/design-decisions.md (D-11).
class DiagnosticSink {
  public:
    void add(Diagnostic d) {
        worst_ = std::max(worst_, d.severity);
        ++counts_[static_cast<std::size_t>(d.severity)];
        items_.push_back(std::move(d));
    }

    void reserve(std::size_t n) { items_.reserve(n); }
    void clear() noexcept {
        items_.clear();
        counts_ = {};
        worst_ = Severity::Info;
    }

    /// Append `other` wholesale.  Used by the parallel normaliser: sinks are
    /// merged in ascending chunk index, so the output order is a function of
    /// the input only.
    void merge_ordered(const DiagnosticSink& other) {
        items_.insert(items_.end(), other.items_.begin(), other.items_.end());
        for (std::size_t i = 0; i < counts_.size(); ++i) counts_[i] += other.counts_[i];
        worst_ = std::max(worst_, other.worst_);
    }

    [[nodiscard]] const std::vector<Diagnostic>& items() const noexcept { return items_; }
    [[nodiscard]] std::size_t size() const noexcept { return items_.size(); }
    [[nodiscard]] bool empty() const noexcept { return items_.empty(); }
    [[nodiscard]] Severity worst() const noexcept { return worst_; }
    [[nodiscard]] std::size_t count(Severity s) const noexcept {
        return counts_[static_cast<std::size_t>(s)];
    }
    [[nodiscard]] bool usable() const noexcept { return worst_ < Severity::Fatal; }

    /// Number of distinct diagnostics with the given code.
    [[nodiscard]] std::size_t count(DiagCode c) const noexcept;

    /// Multi-line report grouped by code, with counts.  Used by the CLI.
    [[nodiscard]] std::string summary() const;

  private:
    std::vector<Diagnostic> items_;
    std::array<std::size_t, 4> counts_{};
    Severity worst_ = Severity::Info;
};

// ---------------------------------------------------------------------------
// Construction helpers.  These keep call sites to one line so that validation
// code reads as a list of rules rather than a list of struct initialisers.
// ---------------------------------------------------------------------------

[[nodiscard]] Diagnostic make_diag(DiagCode code, Severity sev, std::string_view instrument,
                                   std::string_view field, double observed,
                                   std::string_view explanation,
                                   std::uint32_t quote_index = Diagnostic::kNoIndex);

[[nodiscard]] Diagnostic make_range_diag(DiagCode code, Severity sev,
                                         std::string_view instrument, std::string_view field,
                                         double observed, double lo, double hi,
                                         std::string_view explanation,
                                         std::uint32_t quote_index = Diagnostic::kNoIndex);

}  // namespace vl
