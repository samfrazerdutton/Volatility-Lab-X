// SPDX-License-Identifier: MIT
#include "vl_test_support.hpp"

#include "volatility_lab/core/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <string>

using namespace vl;

TEST(Diagnostics, ShortTextTruncatesRatherThanAllocating) {
    ShortText t("short");
    EXPECT_STREQ(t.c_str(), "short");
    EXPECT_FALSE(t.empty());

    const std::string long_text(200, 'x');
    ShortText u(long_text);
    EXPECT_EQ(std::string(u.c_str()).size(), ShortText::kCapacity - 1);
    // Truncation is marked, so a reader can tell the text was cut.
    EXPECT_EQ(std::string(u.c_str()).substr(ShortText::kCapacity - 4), "...");

    ShortText e;
    EXPECT_TRUE(e.empty());
}

TEST(Diagnostics, CodeNamesAndSubsystemsAreComplete) {
    // An unnamed code in a report is useless to whoever has to act on it.
    const DiagCode codes[] = {DiagCode::Ok,
                              DiagCode::MissingField,
                              DiagCode::CrossedMarket,
                              DiagCode::PriceBelowIntrinsic,
                              DiagCode::IvSolverNoBracket,
                              DiagCode::CalibrationDidNotConverge,
                              DiagCode::ButterflyArbitrage,
                              DiagCode::GreekNumericallyUnstable,
                              DiagCode::BackendUnavailable,
                              DiagCode::DurrlemanConditionViolated,
                              DiagCode::DeterminismContractBroken};
    for (auto c : codes) {
        EXPECT_STRNE(to_string(c), "UnknownDiagCode") << static_cast<unsigned>(c);
        EXPECT_STRNE(diag_subsystem(c), "") << static_cast<unsigned>(c);
    }
    // The decade encoding must actually classify.
    EXPECT_STREQ(diag_subsystem(DiagCode::Ok), "ok");
    EXPECT_STREQ(diag_subsystem(DiagCode::MissingField), "market-data");
    EXPECT_STREQ(diag_subsystem(DiagCode::CrossedMarket), "quote-quality");
    EXPECT_STREQ(diag_subsystem(DiagCode::PriceBelowIntrinsic), "price-bounds");
    EXPECT_STREQ(diag_subsystem(DiagCode::IvSolverNoBracket), "implied-vol");
    EXPECT_STREQ(diag_subsystem(DiagCode::SliceTooFewQuotes), "calibration");
    EXPECT_STREQ(diag_subsystem(DiagCode::ButterflyArbitrage), "arbitrage");
    EXPECT_STREQ(diag_subsystem(DiagCode::ScenarioOutOfDomain), "pricing-risk");
    EXPECT_STREQ(diag_subsystem(DiagCode::BackendUnavailable), "engine");
}

TEST(Diagnostics, SeverityNames) {
    EXPECT_STREQ(to_string(Severity::Info), "info");
    EXPECT_STREQ(to_string(Severity::Warning), "warning");
    EXPECT_STREQ(to_string(Severity::Error), "error");
    EXPECT_STREQ(to_string(Severity::Fatal), "fatal");
}

TEST(Diagnostics, FormattedLineCarriesTheWholeStructure) {
    const auto d =
        make_range_diag(DiagCode::CrossedMarket, Severity::Error, "SPX-2026-12-18-C5000",
                        "bid", 12.4, 0.0, 12.1, "bid exceeds ask", 17);
    const std::string s = d.format();
    EXPECT_NE(s.find("error"), std::string::npos);
    EXPECT_NE(s.find("1100"), std::string::npos);
    EXPECT_NE(s.find("CrossedMarket"), std::string::npos);
    EXPECT_NE(s.find("SPX-2026-12-18-C5000"), std::string::npos);
    EXPECT_NE(s.find("bid"), std::string::npos);
    EXPECT_NE(s.find("12.4"), std::string::npos);
    EXPECT_NE(s.find("12.1"), std::string::npos);
    EXPECT_NE(s.find("bid exceeds ask"), std::string::npos);
    EXPECT_NE(s.find("#17"), std::string::npos);
}

TEST(Diagnostics, FormatOmitsBoundsThatWereNotSupplied) {
    const auto d =
        make_diag(DiagCode::ZeroBid, Severity::Warning, "X", "bid", 0.0, "zero bid");
    const std::string s = d.format();
    EXPECT_EQ(s.find("expected"), std::string::npos);
    EXPECT_TRUE(std::isnan(d.expected_lo));
    EXPECT_TRUE(std::isnan(d.expected_hi));
}

TEST(Diagnostics, SinkTracksWorstSeverityAndCounts) {
    DiagnosticSink sink;
    EXPECT_TRUE(sink.empty());
    EXPECT_TRUE(sink.usable());
    EXPECT_EQ(sink.worst(), Severity::Info);

    sink.add(make_diag(DiagCode::ZeroBid, Severity::Warning, "A", "bid", 0.0, "w"));
    sink.add(make_diag(DiagCode::ZeroBid, Severity::Warning, "B", "bid", 0.0, "w"));
    sink.add(make_diag(DiagCode::CrossedMarket, Severity::Error, "C", "bid", 1.0, "e"));

    EXPECT_EQ(sink.size(), 3u);
    EXPECT_EQ(sink.worst(), Severity::Error);
    EXPECT_EQ(sink.count(Severity::Warning), 2u);
    EXPECT_EQ(sink.count(Severity::Error), 1u);
    EXPECT_EQ(sink.count(DiagCode::ZeroBid), 2u);
    EXPECT_TRUE(sink.usable()) << "an error is recoverable; only Fatal is not";

    sink.add(make_diag(DiagCode::NonPositiveSpot, Severity::Fatal, "D", "spot", -1.0, "f"));
    EXPECT_FALSE(sink.usable());

    sink.clear();
    EXPECT_TRUE(sink.empty());
    EXPECT_EQ(sink.worst(), Severity::Info);
    EXPECT_EQ(sink.count(Severity::Error), 0u);
}

TEST(Diagnostics, MergeIsOrderPreservingSoParallelOutputIsDeterministic) {
    // The parallel normaliser merges per-chunk sinks in ascending chunk index.
    // The resulting order must be a function of the input only, never of
    // thread scheduling -- which is why merge appends wholesale rather than
    // interleaving by timestamp.
    DiagnosticSink a;
    DiagnosticSink b;
    a.add(make_diag(DiagCode::ZeroBid, Severity::Warning, "a0", "bid", 0.0, "x"));
    a.add(make_diag(DiagCode::ZeroBid, Severity::Warning, "a1", "bid", 0.0, "x"));
    b.add(make_diag(DiagCode::CrossedMarket, Severity::Error, "b0", "bid", 0.0, "y"));

    DiagnosticSink merged;
    merged.merge_ordered(a);
    merged.merge_ordered(b);
    ASSERT_EQ(merged.size(), 3u);
    EXPECT_STREQ(merged.items()[0].instrument.c_str(), "a0");
    EXPECT_STREQ(merged.items()[1].instrument.c_str(), "a1");
    EXPECT_STREQ(merged.items()[2].instrument.c_str(), "b0");
    EXPECT_EQ(merged.worst(), Severity::Error);
    EXPECT_EQ(merged.count(Severity::Warning), 2u);
}

TEST(Diagnostics, SummaryGroupsByCodeRatherThanListingEveryItem) {
    // A snapshot with 4000 zero-bid quotes must produce one line, not 4000.
    DiagnosticSink sink;
    for (int i = 0; i < 4000; ++i) {
        sink.add(make_diag(DiagCode::ZeroBid, Severity::Warning, "X", "bid",
                           static_cast<double>(i), "zero bid"));
    }
    sink.add(
        make_diag(DiagCode::CrossedMarket, Severity::Error, "Y", "bid", 1.0, "crossed"));
    const std::string s = sink.summary();
    EXPECT_NE(s.find("4000"), std::string::npos);
    EXPECT_NE(s.find("ZeroBid"), std::string::npos);
    EXPECT_NE(s.find("CrossedMarket"), std::string::npos);
    // Header plus two grouped lines.
    const auto newlines = std::count(s.begin(), s.end(), '\n');
    EXPECT_LE(newlines, 4);
}

TEST(Diagnostics, EmptySummaryIsStillInformative) {
    DiagnosticSink sink;
    const std::string s = sink.summary();
    EXPECT_NE(s.find("0 total"), std::string::npos);
}

TEST(Diagnostics, NoIndexSentinelIsDistinguishable) {
    const auto d = make_diag(DiagCode::ZeroBid, Severity::Info, "X", "bid", 0.0, "x");
    EXPECT_EQ(d.quote_index, Diagnostic::kNoIndex);
    EXPECT_EQ(d.format().find("#"), std::string::npos);
}
