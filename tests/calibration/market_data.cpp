// SPDX-License-Identifier: MIT
/// Validates the market-data layer: the SoA/AoS round trip, the synthetic
/// generator's determinism contract, and the normaliser's promise that nothing
/// is silently accepted or silently discarded.

#include "vl_test_support.hpp"

#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/options/normalize.hpp"
#include "volatility_lab/pricing/black.hpp"

#include <cmath>
#include <map>
#include <set>
#include <vector>

using namespace vl;
using vl::math::rel_error;
using vl::test::WorstCase;

namespace {

constexpr MarketRegime kAllRegimes[] = {MarketRegime::Normal,   MarketRegime::HighVol,
                                        MarketRegime::Crash,    MarketRegime::VolCrush,
                                        MarketRegime::Earnings, MarketRegime::Illiquid};

}  // namespace

// ===========================================================================
// QuoteBook: the SoA layout
// ===========================================================================

TEST(QuoteBook, RoundTripsThroughAosAndBack) {
    const auto m = generate_market(MarketRegime::Normal);
    const auto n = normalize(m.snapshot);
    ASSERT_GT(n.quotes.size(), 50u);

    const QuoteBook book = QuoteBook::from_quotes(n.quotes);
    ASSERT_EQ(book.size(), n.quotes.size());
    const auto back = book.to_quotes();
    ASSERT_EQ(back.size(), n.quotes.size());

    // The book sorts, so compare as multisets keyed on (expiry, strike, type).
    std::map<std::tuple<double, double, int>, const OptionQuote*> original;
    for (const auto& q : n.quotes) {
        original[{q.years, q.strike, static_cast<int>(q.type)}] = &q;
    }
    for (const auto& q : back) {
        const auto it = original.find({q.years, q.strike, static_cast<int>(q.type)});
        ASSERT_NE(it, original.end()) << quote_label(q) << " vanished";
        const OptionQuote& o = *it->second;
        // Bitwise: a transpose that loses a bit would be a silent corruption of
        // every downstream price.
        EXPECT_EQ(q.strike, o.strike);
        EXPECT_EQ(q.forward, o.forward);
        EXPECT_EQ(q.years, o.years);
        EXPECT_EQ(q.implied_vol, o.implied_vol);
        EXPECT_EQ(q.total_variance, o.total_variance);
        EXPECT_EQ(q.log_moneyness, o.log_moneyness);
        EXPECT_EQ(q.discount, o.discount);
        EXPECT_EQ(q.weight, o.weight);
        EXPECT_EQ(static_cast<int>(q.type), static_cast<int>(o.type));
    }
}

TEST(QuoteBook, ColumnsAreOverAlignedAndTailPadded) {
    // The SIMD kernels assume aligned loads and a readable vector tail; both
    // are structural promises of the book rather than of the allocator.
    for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{64},
                          std::size_t{1001}}) {
        QuoteBook b(n);
        ASSERT_EQ(b.size(), n);
        const auto v = b.view();
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(v.strike.data()) % kSimdAlign, 0u);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(v.weight.data()) % kSimdAlign, 0u);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(v.type_sign.data()) % kSimdAlign, 0u);
        // All columns the same length, which every kernel relies on.
        EXPECT_EQ(v.forward.size(), n);
        EXPECT_EQ(v.years.size(), n);
        EXPECT_EQ(v.slice_index.size(), n);
    }
}

TEST(QuoteBook, SortsByExpiryThenStrikeAndGroupsIntoContiguousSlices) {
    const auto m = generate_market(MarketRegime::Normal);
    const auto n = normalize(m.snapshot);
    const QuoteBook book = QuoteBook::from_quotes(n.quotes);
    const auto v = book.view();

    for (std::size_t i = 1; i < book.size(); ++i) {
        const bool ordered = (v.years[i] > v.years[i - 1]) ||
                             (v.years[i] == v.years[i - 1] &&
                              v.strike[i] >= v.strike[i - 1]);
        ASSERT_TRUE(ordered) << "not sorted at " << i;
    }

    // Each slice range must be contiguous, non-empty, cover the book exactly,
    // and contain a single expiry.  That contiguity is what lets the
    // calibrator and the batch pricer process a slice in one pass over dense
    // memory rather than gathering scattered rows.
    const auto ranges = book.slice_ranges();
    ASSERT_GT(ranges.size(), 3u);
    std::size_t covered = 0;
    for (std::size_t s = 0; s < ranges.size(); ++s) {
        EXPECT_GT(ranges[s].count(), 0u);
        EXPECT_EQ(ranges[s].begin, covered);
        covered = ranges[s].end;
        for (std::size_t i = ranges[s].begin; i < ranges[s].end; ++i) {
            EXPECT_EQ(v.years[i], ranges[s].years) << "slice " << s << " row " << i;
            EXPECT_EQ(v.slice_index[i], s);
        }
        if (s > 0) EXPECT_GT(ranges[s].years, ranges[s - 1].years);
    }
    EXPECT_EQ(covered, book.size());
}

TEST(QuoteBook, SortIsStableSoTheResultDependsOnlyOnTheInput) {
    // Two quotes can share (expiry, strike, type) -- a duplicate from a feed.
    // An unstable sort would make the book, and therefore every downstream fit
    // and every diagnostic index, depend on the library's pivot choices.
    std::vector<OptionQuote> quotes;
    for (int i = 0; i < 8; ++i) {
        OptionQuote q;
        q.years = 1.0;
        q.strike = 100.0;
        q.type = OptionType::Call;
        q.forward = 100.0;
        q.implied_vol = 0.2 + 0.001 * i;  // distinguishable
        quotes.push_back(q);
    }
    const auto a = QuoteBook::from_quotes(quotes).to_quotes();
    const auto b = QuoteBook::from_quotes(quotes).to_quotes();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].implied_vol, b[i].implied_vol) << "i = " << i;
        // Stability: the original relative order is preserved.
        EXPECT_NEAR(a[i].implied_vol, 0.2 + 0.001 * static_cast<double>(i), 1e-15);
    }
}

TEST(QuoteBook, EmptyAndSingletonAreHarmless) {
    QuoteBook empty;
    EXPECT_EQ(empty.size(), 0u);
    EXPECT_TRUE(empty.view().empty());
    EXPECT_TRUE(empty.slice_ranges().empty());
    empty.sort_and_group();  // must not crash

    const QuoteBook one = QuoteBook::from_quotes(std::vector<OptionQuote>(1));
    EXPECT_EQ(one.size(), 1u);
    EXPECT_EQ(one.slice_ranges().size(), 1u);
}

TEST(QuoteLabel, IdentifiesTheQuoteWithoutAllocatingOnTheFeedPath) {
    // The label is built on demand precisely so that OptionQuote stays
    // trivially copyable and the feed path allocates nothing.
    OptionQuote q;
    q.years = 0.25;
    q.strike = 5000.0;
    q.type = OptionType::Call;
    const std::string s = quote_label(q);
    EXPECT_NE(s.find("0.25"), std::string::npos);
    EXPECT_NE(s.find("C"), std::string::npos);
    EXPECT_NE(s.find("5000"), std::string::npos);
    q.type = OptionType::Put;
    EXPECT_NE(quote_label(q).find("P"), std::string::npos);
}

// ===========================================================================
// The synthetic generator
// ===========================================================================

TEST(SyntheticMarket, IsBitwiseReproducibleFromASeed) {
    // The contract the whole benchmark and regression suite rests on.  A
    // generator whose output drifts makes a benchmark meaningless and a
    // calibration regression unfixable.
    for (auto regime : kAllRegimes) {
        const auto a = generate_market(regime, 4242u);
        const auto b = generate_market(regime, 4242u);
        ASSERT_EQ(a.snapshot.quotes.size(), b.snapshot.quotes.size())
            << to_string(regime);
        for (std::size_t i = 0; i < a.snapshot.quotes.size(); ++i) {
            const auto& x = a.snapshot.quotes[i];
            const auto& y = b.snapshot.quotes[i];
            // Bitwise, including the ordering: a downstream diagnostic refers
            // to quote indices, so a regression baseline would be useless if
            // the order could change.
            ASSERT_EQ(x.strike, y.strike) << to_string(regime) << " i=" << i;
            ASSERT_EQ(x.years, y.years) << to_string(regime) << " i=" << i;
            ASSERT_EQ(x.bid, y.bid) << to_string(regime) << " i=" << i;
            ASSERT_EQ(x.ask, y.ask) << to_string(regime) << " i=" << i;
            ASSERT_EQ(x.volume, y.volume) << to_string(regime) << " i=" << i;
            ASSERT_EQ(x.age_seconds, y.age_seconds) << to_string(regime) << " i=" << i;
        }
    }
}

TEST(SyntheticMarket, DifferentSeedsProduceDifferentData) {
    // The other half of the contract: if the seed did nothing, reproducibility
    // would be trivial and the generator would be a fixed fixture.
    const auto a = generate_market(MarketRegime::Normal, 1u);
    const auto b = generate_market(MarketRegime::Normal, 2u);
    ASSERT_EQ(a.snapshot.quotes.size(), b.snapshot.quotes.size());
    bool any_different = false;
    for (std::size_t i = 0; i < a.snapshot.quotes.size(); ++i) {
        if (a.snapshot.quotes[i].bid != b.snapshot.quotes[i].bid) any_different = true;
    }
    EXPECT_TRUE(any_different);
    // ...but the strike ladder is a function of the regime, not the seed.
    for (std::size_t i = 0; i < a.snapshot.quotes.size(); ++i) {
        EXPECT_EQ(a.snapshot.quotes[i].strike, b.snapshot.quotes[i].strike) << "i = " << i;
    }
}

TEST(SyntheticMarket, HasNoDuplicateStrikes) {
    // Regression test.  A fixed strike increment collapsed 25 target strikes
    // onto four listed ones at short maturities, producing 53 duplicate quotes
    // out of 250 -- which the normaliser then dutifully rejected, silently
    // discarding a fifth of the snapshot.  The increment now adapts to the
    // maturity, as real markets do.
    for (auto regime : kAllRegimes) {
        const auto m = generate_market(regime);
        std::set<std::tuple<double, double, int>> seen;
        for (const auto& q : m.snapshot.quotes) {
            const auto key = std::make_tuple(q.years, q.strike, static_cast<int>(q.type));
            ASSERT_TRUE(seen.insert(key).second)
                << to_string(regime) << ": duplicate " << quote_label(q);
        }
    }
}

TEST(SyntheticMarket, EverySliceHasEnoughStrikesToFit) {
    for (auto regime : kAllRegimes) {
        const auto m = generate_market(regime);
        std::map<double, std::size_t> per_expiry;
        for (const auto& q : m.snapshot.quotes) ++per_expiry[q.years];
        ASSERT_GE(per_expiry.size(), 4u) << to_string(regime);
        for (const auto& [years, count] : per_expiry) {
            EXPECT_GE(count, 7u) << to_string(regime) << " at T = " << years;
        }
    }
}

TEST(SyntheticMarket, QuotesStraddleTheForwardAndUseTheOtmSide) {
    // A listed market quotes the OTM side; so does the generator, and the
    // normaliser depends on it to exercise the parity conversion.
    const auto m = generate_market(MarketRegime::Normal);
    std::size_t calls_above = 0;
    std::size_t puts_below = 0;
    for (const auto& q : m.snapshot.quotes) {
        if (q.type == OptionType::Call) {
            EXPECT_GE(q.strike, q.forward) << quote_label(q);
            ++calls_above;
        } else {
            EXPECT_LT(q.strike, q.forward) << quote_label(q);
            ++puts_below;
        }
    }
    EXPECT_GT(calls_above, 20u);
    EXPECT_GT(puts_below, 20u);
}

TEST(SyntheticMarket, TrueSurfaceIsAdmissibleAndMatchesTheQuotes) {
    for (auto regime : kAllRegimes) {
        const auto m = generate_market(regime);
        ASSERT_TRUE(m.true_surface.valid())
            << to_string(regime) << ": " << m.true_surface.diagnostics().summary();
        // The ATM term structure must be non-decreasing, which is the first
        // calendar condition and a property of the regime ladders.
        const auto theta = m.true_surface.atm_total_variance_term();
        EXPECT_TRUE(math::is_non_decreasing(theta))
            << to_string(regime) << ": ATM total variance is not monotone";
        // And the surface must be SSVI-admissible where it claims to be.
        if (m.surface_is_exactly_ssvi) {
            const auto chk = ssvi_arbitrage_check_term(m.ssvi, theta);
            EXPECT_TRUE(chk.butterfly_free)
                << to_string(regime) << ": slack " << chk.butterfly_slack_1;
        }
    }
}

TEST(SyntheticMarket, OnlyTheEarningsRegimeIsNotExactlySsvi) {
    // The flag matters: a calibration test that compares a fit against
    // `true_surface` needs to know whether that surface is representable by
    // the model being fitted.
    for (auto regime : kAllRegimes) {
        const auto m = generate_market(regime);
        EXPECT_EQ(m.surface_is_exactly_ssvi, regime != MarketRegime::Earnings)
            << to_string(regime);
    }
}

TEST(SyntheticMarket, IlliquidRegimeActuallyProducesDamagedQuotes) {
    // The regime exists to exercise the normaliser, so it has to deliver the
    // damage it promises -- otherwise the normalisation tests pass vacuously.
    const auto m = generate_market(MarketRegime::Illiquid);
    std::size_t zero_bid = 0, crossed = 0, stale = 0, dead = 0, no_market = 0;
    for (const auto& q : m.snapshot.quotes) {
        if (q.bid <= 0.0 && q.ask > 0.0) ++zero_bid;
        if (q.bid > q.ask && q.ask > 0.0) ++crossed;
        if (q.age_seconds > 300.0) ++stale;
        if (q.volume <= 0.0 && q.open_interest <= 0.0) ++dead;
        if (q.bid <= 0.0 && q.ask <= 0.0) ++no_market;
    }
    EXPECT_GT(zero_bid + crossed + stale + dead + no_market, 3u)
        << "the illiquid regime produced no damaged quotes";
}

TEST(DeterministicRng, ProducesTheSameSequenceEveryTime) {
    DeterministicRng a(12345u);
    DeterministicRng b(12345u);
    for (int i = 0; i < 1000; ++i) {
        ASSERT_EQ(a.next_u64(), b.next_u64()) << "i = " << i;
    }
}

TEST(DeterministicRng, UniformIsInRangeAndNeverExactlyOne) {
    // `normal` takes a log of it, so a 1.0 would be a -inf and a 0.0 a +inf.
    DeterministicRng r(7u);
    double lo = 1.0, hi = 0.0;
    for (int i = 0; i < 200000; ++i) {
        const double u = r.uniform();
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
        lo = std::min(lo, u);
        hi = std::max(hi, u);
    }
    EXPECT_LT(lo, 0.001);
    EXPECT_GT(hi, 0.999);
}

TEST(DeterministicRng, NormalHasTheRightMomentsAndFiniteTails) {
    DeterministicRng r(99u);
    const int n = 500000;
    double sum = 0.0, sum_sq = 0.0, worst = 0.0;
    for (int i = 0; i < n; ++i) {
        const double z = r.normal();
        ASSERT_TRUE(std::isfinite(z)) << "i = " << i;
        sum += z;
        sum_sq += z * z;
        worst = std::max(worst, std::abs(z));
    }
    const double mean = sum / n;
    const double var = sum_sq / n - mean * mean;
    EXPECT_NEAR(mean, 0.0, 0.01);
    EXPECT_NEAR(var, 1.0, 0.02);
    EXPECT_GT(worst, 3.5) << "the tails are not being sampled at all";
    EXPECT_LT(worst, 8.0) << "implausibly heavy tail for a Gaussian";
}

TEST(DeterministicRng, StudentTIsHeavierTailedThanNormalButUnitVariance) {
    // The noise model is Student-t because real quote errors have heavy tails,
    // and it is scaled to unit variance so that `vol_noise` means a standard
    // deviation in both cases.  Both halves of that are checked.
    DeterministicRng rn(5u);
    DeterministicRng rt(5u);
    const int n = 400000;
    double t_sum = 0.0, t_sq = 0.0, t_worst = 0.0, n_worst = 0.0;
    for (int i = 0; i < n; ++i) {
        const double t = rt.student_t(4.0);
        ASSERT_TRUE(std::isfinite(t)) << "i = " << i;
        t_sum += t;
        t_sq += t * t;
        t_worst = std::max(t_worst, std::abs(t));
        n_worst = std::max(n_worst, std::abs(rn.normal()));
    }
    const double mean = t_sum / n;
    const double var = t_sq / n - mean * mean;
    EXPECT_NEAR(mean, 0.0, 0.02);
    EXPECT_NEAR(var, 1.0, 0.15) << "the unit-variance scaling is wrong";
    EXPECT_GT(t_worst, n_worst) << "the t distribution must have heavier tails";
}

// ===========================================================================
// Normalisation
// ===========================================================================

TEST(Normalize, AcceptsMostOfACleanSnapshotAndExplainsTheRest) {
    const auto m = generate_market(MarketRegime::Normal);
    const auto n = normalize(m.snapshot);
    EXPECT_TRUE(n.usable()) << n.diagnostics.summary();
    EXPECT_GT(n.stats.acceptance_rate(), 0.90)
        << "clean data should mostly survive: " << n.diagnostics.summary();

    // Every rejection is accounted for: the counts must add up exactly.
    EXPECT_EQ(n.stats.accepted + n.stats.rejected, n.stats.received);
    const std::size_t by_cause = n.stats.rejected_shape + n.stats.rejected_quality +
                                 n.stats.rejected_bounds + n.stats.rejected_inversion +
                                 n.stats.rejected_vol_range;
    EXPECT_EQ(by_cause, n.stats.rejected)
        << "a rejection was not attributed to a cause";
    // And every rejection produced a diagnostic.
    EXPECT_GE(n.diagnostics.count(Severity::Error), n.stats.rejected);
}

TEST(Normalize, AuditCoversEveryInputQuoteInInputOrder) {
    // The promise that nothing is silently discarded: a caller must be able to
    // ask what happened to a specific input without re-running anything.
    const auto m = generate_market(MarketRegime::Illiquid);
    const auto n = normalize(m.snapshot);
    ASSERT_EQ(n.audit.size(), m.snapshot.quotes.size());
    for (std::size_t i = 0; i < n.audit.size(); ++i) {
        EXPECT_EQ(n.audit[i].source_index, i);
        EXPECT_NE(static_cast<int>(n.audit[i].status),
                  static_cast<int>(QuoteStatus::Unvalidated))
            << "quote " << i << " was never classified";
        EXPECT_EQ(n.audit[i].strike, m.snapshot.quotes[i].strike);
    }
    // Accepted quotes are a subset of the audit, and all are Ok or Degraded.
    for (const auto& q : n.quotes) {
        EXPECT_NE(static_cast<int>(q.status), static_cast<int>(QuoteStatus::Rejected));
    }
}

TEST(Normalize, RecoversTheGeneratingVolatilityToWithinTheQuoteNoise) {
    // The end-to-end check of phases 2-4 together: price the true surface,
    // round the prices to a tick, add spread, then invert.  What comes back
    // must match the surface to within the noise that was injected -- if it
    // does not, something between the pricer, the parity conversion and the
    // inversion is wrong.
    for (auto regime : {MarketRegime::Normal, MarketRegime::HighVol,
                        MarketRegime::Earnings}) {
        const auto m = generate_market(regime);
        const auto n = normalize(m.snapshot);
        ASSERT_TRUE(n.usable()) << to_string(regime);

        WorstCase w;
        double sum_abs = 0.0;
        std::size_t count = 0;
        for (const auto& q : n.quotes) {
            const double truth = m.true_surface.vol(q.log_moneyness, q.years);
            const double d = std::abs(q.implied_vol - truth);
            w.observe(d, q.strike, q.years, q.implied_vol, truth);
            sum_abs += d;
            ++count;
        }
        const double mean_abs = sum_abs / static_cast<double>(count);
        // The mean error must be of the order of the injected noise, not
        // orders of magnitude larger.  The bound is generous because tick
        // rounding in the wings is a genuine additional error source.
        const double noise = m.config.vol_noise;
        EXPECT_LT(mean_abs, 10.0 * noise + 0.002)
            << to_string(regime) << ": " << w.describe("strike", "T");
    }
}

TEST(Normalize, ConvertsToTheOtmSideThroughParity) {
    // The single largest cheap improvement to fit quality, and the one most
    // easily got wrong by a sign.
    OptionQuote q;
    q.forward = 100.0;
    q.strike = 80.0;  // call is deep ITM here, so the put is the OTM side
    q.years = 1.0;
    q.discount = 1.0;
    q.type = OptionType::Call;
    const double true_vol = 0.25;
    const double call = black_undiscounted(100.0, 80.0, true_vol, 1.0, OptionType::Call);
    q.bid = call - 0.001;
    q.ask = call + 0.001;

    NormalizationConfig cfg;
    const auto sel = select_otm_side(q, cfg);
    EXPECT_EQ(static_cast<int>(sel.side), static_cast<int>(OptionType::Put));
    EXPECT_TRUE(sel.converted_through_parity);
    const double expected_put =
        black_undiscounted(100.0, 80.0, true_vol, 1.0, OptionType::Put);
    EXPECT_LT(rel_error(sel.undiscounted_price, expected_put), 1e-3);

    // With the conversion off, the ITM call is used as-is.
    cfg.prefer_otm_side = false;
    const auto raw = select_otm_side(q, cfg);
    EXPECT_EQ(static_cast<int>(raw.side), static_cast<int>(OptionType::Call));
    EXPECT_FALSE(raw.converted_through_parity);
    EXPECT_GT(raw.undiscounted_price, 19.0);
}

TEST(Normalize, DerivesForwardAndDiscountButPrefersSuppliedValues) {
    OptionQuote q;
    q.spot = 100.0;
    q.rate = 0.05;
    q.dividend = 0.02;
    q.years = 2.0;
    derive_forward_and_discount(q);
    EXPECT_NEAR(q.forward, 100.0 * std::exp(0.03 * 2.0), 1e-12);
    EXPECT_NEAR(q.discount, std::exp(-0.05 * 2.0), 1e-12);

    // A supplied forward wins: a feed that publishes one has usually implied
    // it from the parity of the listed options, which is better information
    // than spot times a guessed carry.
    OptionQuote s = q;
    s.forward = 123.0;
    s.discount = 0.9;
    derive_forward_and_discount(s);
    EXPECT_EQ(s.forward, 123.0);
    EXPECT_EQ(s.discount, 0.9);
}

TEST(Normalize, RejectsMalformedQuotesWithTheRightCode) {
    struct Case {
        const char* name;
        DiagCode code;
        std::function<void(OptionQuote&)> damage;
    };
    const Case cases[] = {
        {"negative strike", DiagCode::NegativeStrike, [](OptionQuote& q) { q.strike = -1.0; }},
        {"zero expiry", DiagCode::NonPositiveExpiry, [](OptionQuote& q) { q.years = 0.0; }},
        {"far expiry", DiagCode::ExpiryTooFar, [](OptionQuote& q) { q.years = 50.0; }},
        {"american", DiagCode::AmericanExerciseUnsupported,
         [](OptionQuote& q) { q.style = ExerciseStyle::American; }},
        {"crossed", DiagCode::CrossedMarket,
         [](OptionQuote& q) { std::swap(q.bid, q.ask); }},
        {"nan strike", DiagCode::NonFiniteValue,
         [](OptionQuote& q) { q.strike = std::numeric_limits<double>::quiet_NaN(); }},
    };

    for (const auto& c : cases) {
        OptionQuote q;
        q.strike = 110.0;
        q.forward = 100.0;
        q.years = 1.0;
        q.type = OptionType::Call;
        q.bid = 4.0;
        q.ask = 4.2;
        q.discount = 1.0;
        c.damage(q);

        DiagnosticSink diags;
        const QuoteStatus st = validate_quote(q, NormalizationConfig{}, diags, 0);
        EXPECT_EQ(static_cast<int>(st), static_cast<int>(QuoteStatus::Rejected)) << c.name;
        EXPECT_GT(diags.count(c.code), 0u)
            << c.name << " was rejected with the wrong code: " << diags.summary();
    }
}

TEST(Normalize, DegradesRatherThanRejectsRecoverableProblems) {
    // The distinction matters: a stale quote in the wings is still better than
    // no quote, so it is down-weighted, not discarded.
    struct Case {
        const char* name;
        DiagCode code;
        std::function<void(OptionQuote&)> damage;
    };
    const Case cases[] = {
        {"zero bid", DiagCode::ZeroBid, [](OptionQuote& q) { q.bid = 0.0; }},
        {"locked", DiagCode::LockedMarket, [](OptionQuote& q) { q.ask = q.bid; }},
        {"stale", DiagCode::StaleQuote, [](OptionQuote& q) { q.age_seconds = 3600.0; }},
        {"illiquid", DiagCode::ZeroVolumeAndOpenInterest, [](OptionQuote& q) {
             q.volume = 0.0;
             q.open_interest = 0.0;
         }},
    };
    for (const auto& c : cases) {
        OptionQuote q;
        q.strike = 110.0;
        q.forward = 100.0;
        q.years = 1.0;
        q.type = OptionType::Call;
        q.bid = 4.0;
        q.ask = 4.2;
        q.discount = 1.0;
        q.volume = 100.0;
        q.open_interest = 500.0;
        q.age_seconds = 1.0;
        c.damage(q);

        DiagnosticSink diags;
        const QuoteStatus st = validate_quote(q, NormalizationConfig{}, diags, 0);
        EXPECT_EQ(static_cast<int>(st), static_cast<int>(QuoteStatus::Degraded)) << c.name;
        EXPECT_GT(diags.count(c.code), 0u) << c.name;
    }
}

TEST(Normalize, RejectsPricesOutsideTheNoArbitrageBounds) {
    MarketSnapshot snap;
    snap.underlying = "T";
    snap.spot = 100.0;
    // Price below intrinsic: a 90-strike call must be worth at least 10.
    OptionQuote below;
    below.strike = 90.0;
    below.forward = 100.0;
    below.years = 1.0;
    below.type = OptionType::Call;
    below.discount = 1.0;
    below.bid = 2.0;
    below.ask = 3.0;
    snap.quotes.push_back(below);
    // Price above the forward bound.
    OptionQuote above = below;
    above.strike = 110.0;
    above.bid = 150.0;
    above.ask = 160.0;
    snap.quotes.push_back(above);

    const auto n = normalize(snap);
    EXPECT_EQ(n.stats.accepted, 0u);
    EXPECT_EQ(n.stats.rejected_bounds, 2u);
    EXPECT_GT(n.diagnostics.count(DiagCode::PriceBelowIntrinsic) +
                  n.diagnostics.count(DiagCode::PriceAboveForwardBound),
              1u);
}

TEST(Normalize, RejectsDuplicatesKeepingTheFirst) {
    MarketSnapshot snap;
    snap.spot = 100.0;
    for (int i = 0; i < 3; ++i) {
        OptionQuote q;
        q.strike = 110.0;
        q.forward = 100.0;
        q.years = 1.0;
        q.type = OptionType::Call;
        q.discount = 1.0;
        q.bid = 4.0 + i;
        q.ask = 4.2 + i;
        snap.quotes.push_back(q);
    }
    const auto n = normalize(snap);
    EXPECT_EQ(n.stats.accepted, 1u);
    EXPECT_EQ(n.diagnostics.count(DiagCode::DuplicateQuote), 2u);
    // The first wins, so the accepted quote is the one with bid 4.0.
    ASSERT_EQ(n.quotes.size(), 1u);
    EXPECT_EQ(n.quotes[0].bid, 4.0);
}

TEST(Normalize, EmptySnapshotIsFatalRatherThanACrash) {
    MarketSnapshot snap;
    const auto n = normalize(snap);
    EXPECT_FALSE(n.usable());
    EXPECT_GT(n.diagnostics.count(Severity::Fatal), 0u);
    EXPECT_TRUE(n.quotes.empty());
    EXPECT_TRUE(n.fittable_expiries.empty());
}

TEST(Normalize, SlicesWithTooFewQuotesAreReportedAndExcluded) {
    MarketSnapshot snap;
    snap.spot = 100.0;
    // One expiry with plenty of quotes, one with two.
    for (int i = 0; i < 9; ++i) {
        OptionQuote q;
        q.strike = 80.0 + 5.0 * i;
        q.forward = 100.0;
        q.years = 1.0;
        q.type = (q.strike >= 100.0) ? OptionType::Call : OptionType::Put;
        q.discount = 1.0;
        const double px = black_undiscounted(100.0, q.strike, 0.2, 1.0, q.type);
        q.bid = px * 0.98;
        q.ask = px * 1.02;
        snap.quotes.push_back(q);
    }
    for (int i = 0; i < 2; ++i) {
        OptionQuote q;
        q.strike = 95.0 + 10.0 * i;
        q.forward = 100.0;
        q.years = 2.0;
        q.type = (q.strike >= 100.0) ? OptionType::Call : OptionType::Put;
        q.discount = 1.0;
        const double px = black_undiscounted(100.0, q.strike, 0.2, 2.0, q.type);
        q.bid = px * 0.98;
        q.ask = px * 1.02;
        snap.quotes.push_back(q);
    }
    const auto n = normalize(snap);
    EXPECT_GT(n.diagnostics.count(DiagCode::SliceTooFewQuotes), 0u);
    ASSERT_EQ(n.fittable_expiries.size(), 1u);
    EXPECT_NEAR(n.fittable_expiries[0], 1.0, 1e-15);
    // The thin slice's quotes survive normalisation but are marked as
    // belonging to no fittable slice.
    for (const auto& q : n.quotes) {
        if (q.years == 2.0) {
            EXPECT_EQ(q.slice_index, std::numeric_limits<std::uint32_t>::max());
        }
    }
}

TEST(Normalize, AcceptsSuppliedVolatilitiesWhenThereIsNoPrice) {
    const auto cfg_market = [] {
        SyntheticMarketConfig c = regime_defaults(MarketRegime::Normal);
        c.emit_vols_not_prices = true;
        return c;
    }();
    const auto m = generate_market(cfg_market);
    const auto n = normalize(m.snapshot);
    EXPECT_TRUE(n.usable()) << n.diagnostics.summary();
    EXPECT_GT(n.stats.acceptance_rate(), 0.95);
    // The supplied vol must come through essentially untouched -- the only
    // difference is the generator's injected noise.
    WorstCase w;
    for (const auto& q : n.quotes) {
        const double truth = m.true_surface.vol(q.log_moneyness, q.years);
        w.observe(std::abs(q.implied_vol - truth), q.strike, q.years, q.implied_vol, truth);
    }
    EXPECT_LT(w.error, 0.03) << w.describe("strike", "T");
}

TEST(Normalize, EveryRegimeProducesAFittableSnapshot) {
    // The regimes exist to stress different parts of the pipeline, but all of
    // them must come out the other side usable -- otherwise the later phases
    // have nothing to test against.
    for (auto regime : kAllRegimes) {
        const auto m = generate_market(regime);
        const auto n = normalize(m.snapshot);
        EXPECT_TRUE(n.usable()) << to_string(regime) << ":\n" << n.diagnostics.summary();
        EXPECT_GE(n.fittable_expiries.size(), 3u) << to_string(regime);
        EXPECT_GT(n.stats.acceptance_rate(), 0.40)
            << to_string(regime) << " rejected too much:\n" << n.diagnostics.summary();
    }
}
