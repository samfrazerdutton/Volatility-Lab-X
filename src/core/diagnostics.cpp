// SPDX-License-Identifier: MIT
#include "volatility_lab/core/diagnostics.hpp"

#include <cmath>
#include <limits>
#include <map>

namespace vl {

double Diagnostic::qnan() noexcept { return std::numeric_limits<double>::quiet_NaN(); }

const char* to_string(DiagCode c) noexcept {
    switch (c) {
        case DiagCode::Ok: return "Ok";

        case DiagCode::MissingField: return "MissingField";
        case DiagCode::NonFiniteValue: return "NonFiniteValue";
        case DiagCode::NegativeStrike: return "NegativeStrike";
        case DiagCode::NonPositiveSpot: return "NonPositiveSpot";
        case DiagCode::NonPositiveForward: return "NonPositiveForward";
        case DiagCode::NonPositiveExpiry: return "NonPositiveExpiry";
        case DiagCode::ExpiryTooFar: return "ExpiryTooFar";
        case DiagCode::AmericanExerciseUnsupported: return "AmericanExerciseUnsupported";
        case DiagCode::DuplicateQuote: return "DuplicateQuote";

        case DiagCode::CrossedMarket: return "CrossedMarket";
        case DiagCode::LockedMarket: return "LockedMarket";
        case DiagCode::ZeroBid: return "ZeroBid";
        case DiagCode::NonPositivePrice: return "NonPositivePrice";
        case DiagCode::ExcessiveSpread: return "ExcessiveSpread";
        case DiagCode::StaleQuote: return "StaleQuote";
        case DiagCode::ZeroVolumeAndOpenInterest: return "ZeroVolumeAndOpenInterest";
        case DiagCode::WideSpreadRelativeToPrice: return "WideSpreadRelativeToPrice";

        case DiagCode::PriceBelowIntrinsic: return "PriceBelowIntrinsic";
        case DiagCode::PriceAboveForwardBound: return "PriceAboveForwardBound";
        case DiagCode::PriceBelowZero: return "PriceBelowZero";

        case DiagCode::IvSolverNoBracket: return "IvSolverNoBracket";
        case DiagCode::IvSolverMaxIterations: return "IvSolverMaxIterations";
        case DiagCode::IvBelowFloor: return "IvBelowFloor";
        case DiagCode::IvAboveCeiling: return "IvAboveCeiling";
        case DiagCode::IvVegaTooSmall: return "IvVegaTooSmall";

        case DiagCode::SliceTooFewQuotes: return "SliceTooFewQuotes";
        case DiagCode::CalibrationDidNotConverge: return "CalibrationDidNotConverge";
        case DiagCode::ParameterAtBound: return "ParameterAtBound";
        case DiagCode::SurfaceExtrapolated: return "SurfaceExtrapolated";
        case DiagCode::SliceDegenerate: return "SliceDegenerate";

        case DiagCode::ButterflyArbitrage: return "ButterflyArbitrage";
        case DiagCode::CalendarArbitrage: return "CalendarArbitrage";
        case DiagCode::CallPriceNotMonotone: return "CallPriceNotMonotone";
        case DiagCode::CallPriceNotConvex: return "CallPriceNotConvex";
        case DiagCode::PutCallParityViolation: return "PutCallParityViolation";
        case DiagCode::NegativeDensity: return "NegativeDensity";
        case DiagCode::TotalVarianceNotIncreasing: return "TotalVarianceNotIncreasing";
        case DiagCode::DurrlemanConditionViolated: return "DurrlemanConditionViolated";

        case DiagCode::GreekNumericallyUnstable: return "GreekNumericallyUnstable";
        case DiagCode::ScenarioOutOfDomain: return "ScenarioOutOfDomain";
        case DiagCode::PortfolioEmpty: return "PortfolioEmpty";

        case DiagCode::BackendUnavailable: return "BackendUnavailable";
        case DiagCode::DeterminismContractBroken: return "DeterminismContractBroken";
        case DiagCode::InvalidConfiguration: return "InvalidConfiguration";
    }
    return "UnknownDiagCode";
}

const char* diag_subsystem(DiagCode c) noexcept {
    const auto v = static_cast<unsigned>(c);
    if (v == 0) return "ok";
    if (v < 1100) return "market-data";
    if (v < 1200) return "quote-quality";
    if (v < 1300) return "price-bounds";
    if (v < 1400) return "implied-vol";
    if (v < 1500) return "calibration";
    if (v < 1600) return "arbitrage";
    if (v < 1700) return "pricing-risk";
    return "engine";
}

std::string Diagnostic::format() const {
    char buf[320];
    const bool has_lo = std::isfinite(expected_lo);
    const bool has_hi = std::isfinite(expected_hi);

    char range[96] = {0};
    if (has_lo && has_hi) {
        std::snprintf(range, sizeof(range), ", expected in [%.10g, %.10g]", expected_lo,
                      expected_hi);
    } else if (has_lo) {
        std::snprintf(range, sizeof(range), ", expected >= %.10g", expected_lo);
    } else if (has_hi) {
        std::snprintf(range, sizeof(range), ", expected <= %.10g", expected_hi);
    }

    char idx[24] = {0};
    if (quote_index != kNoIndex) {
        std::snprintf(idx, sizeof(idx), " #%u", quote_index);
    }

    std::snprintf(buf, sizeof(buf), "%-7s [%u %s]%s %s%s%s: observed %.10g%s%s%s",
                  to_string(severity), static_cast<unsigned>(code), to_string(code), idx,
                  instrument.empty() ? "-" : instrument.c_str(),
                  field.empty() ? "" : ".", field.empty() ? "" : field.c_str(), observed,
                  range, explanation.empty() ? "" : " -- ",
                  explanation.empty() ? "" : explanation.c_str());
    return std::string(buf);
}

std::size_t DiagnosticSink::count(DiagCode c) const noexcept {
    std::size_t n = 0;
    for (const auto& d : items_) {
        if (d.code == c) ++n;
    }
    return n;
}

std::string DiagnosticSink::summary() const {
    std::string out;
    char head[192];
    std::snprintf(head, sizeof(head),
                  "diagnostics: %zu total (fatal %zu, error %zu, warning %zu, info %zu)\n",
                  items_.size(), count(Severity::Fatal), count(Severity::Error),
                  count(Severity::Warning), count(Severity::Info));
    out += head;
    if (items_.empty()) return out;

    // Grouped by code so that a snapshot with 4000 zero-bid quotes produces
    // one line, not 4000.  std::map keeps the order a function of the code
    // value only, which keeps CLI output byte-stable across runs.
    struct Agg {
        std::size_t n = 0;
        Severity sev = Severity::Info;
        double worst_observed = 0.0;
        const Diagnostic* exemplar = nullptr;
    };
    std::map<std::uint16_t, Agg> by_code;
    for (const auto& d : items_) {
        Agg& a = by_code[static_cast<std::uint16_t>(d.code)];
        if (a.n == 0) a.exemplar = &d;
        ++a.n;
        a.sev = std::max(a.sev, d.severity);
        if (std::abs(d.observed) > std::abs(a.worst_observed)) a.worst_observed = d.observed;
    }

    for (const auto& [code_value, a] : by_code) {
        const auto code = static_cast<DiagCode>(code_value);
        char line[320];
        std::snprintf(line, sizeof(line), "  %-7s %-5u %-28s %-14s x%-6zu  e.g. %s\n",
                      to_string(a.sev), code_value, to_string(code), diag_subsystem(code),
                      a.n, a.exemplar->explanation.empty() ? "-"
                                                           : a.exemplar->explanation.c_str());
        out += line;
    }
    return out;
}

Diagnostic make_diag(DiagCode code, Severity sev, std::string_view instrument,
                     std::string_view field, double observed, std::string_view explanation,
                     std::uint32_t quote_index) {
    Diagnostic d;
    d.code = code;
    d.severity = sev;
    d.quote_index = quote_index;
    d.instrument.assign(instrument);
    d.field.assign(field);
    d.observed = observed;
    d.explanation.assign(explanation);
    return d;
}

Diagnostic make_range_diag(DiagCode code, Severity sev, std::string_view instrument,
                           std::string_view field, double observed, double lo, double hi,
                           std::string_view explanation, std::uint32_t quote_index) {
    Diagnostic d = make_diag(code, sev, instrument, field, observed, explanation, quote_index);
    d.expected_lo = lo;
    d.expected_hi = hi;
    return d;
}

}  // namespace vl
