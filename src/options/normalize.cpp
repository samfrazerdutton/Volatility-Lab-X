// SPDX-License-Identifier: MIT
#include "volatility_lab/options/normalize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "volatility_lab/pricing/black.hpp"
#include "volatility_lab/pricing/implied_vol.hpp"

namespace vl {

// ===========================================================================
// Forward and discount
// ===========================================================================

void derive_forward_and_discount(OptionQuote& q) noexcept {
    // A supplied discount factor wins; otherwise derive from the rate.
    if (!(q.discount > 0.0) || !std::isfinite(q.discount)) {
        q.discount = std::exp(-q.rate * q.years);
    }
    // A supplied forward wins.  Feeds that publish one have usually implied it
    // from the put-call parity of the listed options, which is better
    // information than S*exp((r-q)T) with a guessed dividend -- and it is what
    // makes the put and the call at a given strike imply the same volatility.
    if (!(q.forward > 0.0) || !std::isfinite(q.forward)) {
        if (q.spot > 0.0) {
            q.forward = q.spot * std::exp((q.rate - q.dividend) * q.years);
        }
    }
}

// ===========================================================================
// OTM side selection
// ===========================================================================

OtmSelection select_otm_side(const OptionQuote& q, const NormalizationConfig& cfg) noexcept {
    OtmSelection sel;
    const double mid_raw = (q.bid > 0.0 && q.ask > 0.0)
                               ? 0.5 * (q.bid + q.ask)
                               : (q.ask > 0.0 ? q.ask : (q.bid > 0.0 ? q.bid : q.last));
    const double discount = (q.discount > 0.0) ? q.discount : 1.0;
    const double undiscounted = mid_raw / discount;

    sel.side = q.type;
    sel.undiscounted_price = undiscounted;

    if (!cfg.prefer_otm_side || !(q.forward > 0.0) || !(q.strike > 0.0)) return sel;

    const bool quote_is_call = (q.type == OptionType::Call);
    const bool call_is_otm = (q.strike >= q.forward);
    if (quote_is_call == call_is_otm) return sel;  // already the OTM side

    // Convert through put-call parity: C - P = F - K, undiscounted.
    //
    // This is the single largest cheap improvement to fit quality.  At a strike
    // below the forward the call is intrinsic plus a sliver: its mid carries
    // almost no volatility information while its spread carries all the noise.
    // The put at the same strike is the OTM one and carries the signal.
    //
    // The subtraction is exact in the direction that matters, because the
    // intrinsic F - K is the large part and is representable; what remains is
    // precisely the small OTM value.  (That the *input* mid may only pin that
    // value down to a few digits is a separate matter, reported by the
    // inversion as `attainable_rtol`.)
    const double intrinsic_call = q.forward - q.strike;
    sel.converted_through_parity = true;
    if (quote_is_call) {
        sel.side = OptionType::Put;
        sel.undiscounted_price = undiscounted - intrinsic_call;
    } else {
        sel.side = OptionType::Call;
        sel.undiscounted_price = undiscounted + intrinsic_call;
    }
    return sel;
}

// ===========================================================================
// Validation
// ===========================================================================

namespace {

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

bool all_finite(const OptionQuote& q) noexcept {
    return std::isfinite(q.strike) && std::isfinite(q.years) && std::isfinite(q.bid) &&
           std::isfinite(q.ask) && std::isfinite(q.rate) && std::isfinite(q.dividend);
}

}  // namespace

QuoteStatus validate_quote(const OptionQuote& q, const NormalizationConfig& cfg,
                           DiagnosticSink& diags, std::uint32_t index) {
    const std::string label = quote_label(q);
    QuoteStatus status = QuoteStatus::Ok;
    const auto reject = [&](DiagCode code, const char* field, double observed, double lo,
                            double hi, const char* why) {
        diags.add(make_range_diag(code, Severity::Error, label, field, observed, lo, hi, why,
                                  index));
        status = QuoteStatus::Rejected;
    };
    const auto degrade = [&](DiagCode code, const char* field, double observed,
                             const char* why) {
        diags.add(make_diag(code, Severity::Warning, label, field, observed, why, index));
        if (status == QuoteStatus::Ok) status = QuoteStatus::Degraded;
    };

    // --- 1. shape ---------------------------------------------------------
    if (!all_finite(q)) {
        reject(DiagCode::NonFiniteValue, "quote", kNan, -kInf, kInf,
               "a required field is NaN or infinite");
        return status;
    }
    if (!(q.strike > 0.0)) {
        reject(DiagCode::NegativeStrike, "strike", q.strike, 0.0, kInf,
               "strike must be strictly positive");
    }
    if (!(q.years > 0.0)) {
        reject(DiagCode::NonPositiveExpiry, "years", q.years, cfg.min_years, cfg.max_years,
               "expiry must be strictly positive");
    } else if (q.years < cfg.min_years) {
        reject(DiagCode::NonPositiveExpiry, "years", q.years, cfg.min_years, cfg.max_years,
               "expiry below the minimum; the option carries no vol information");
    } else if (q.years > cfg.max_years) {
        reject(DiagCode::ExpiryTooFar, "years", q.years, cfg.min_years, cfg.max_years,
               "expiry beyond the configured horizon");
    }
    if (q.style != ExerciseStyle::European) {
        // Rejected rather than priced with a European formula.  Silently
        // treating an American option as European is a real mispricing, not an
        // approximation, and the library has no American engine.
        reject(DiagCode::AmericanExerciseUnsupported, "style",
               static_cast<double>(q.style), 0.0, 0.0,
               "only European exercise is supported");
    }
    if (status == QuoteStatus::Rejected) return status;

    // --- 2. quote quality -------------------------------------------------
    const bool have_bid = q.bid > 0.0;
    const bool have_ask = q.ask > 0.0;

    if (have_bid && have_ask) {
        if (q.bid > q.ask) {
            reject(DiagCode::CrossedMarket, "bid", q.bid, 0.0, q.ask,
                   "bid exceeds ask");
            return status;
        }
        if (q.bid == q.ask) {
            degrade(DiagCode::LockedMarket, "bid", q.bid,
                    "zero-width market; no spread information");
        }
        const double mid = 0.5 * (q.bid + q.ask);
        const double spread = q.ask - q.bid;
        if (mid > 0.0 && spread / mid > cfg.max_relative_spread) {
            // Checked against *both* thresholds, and only rejected if both
            // bind.  A 0.01/0.02 quote has a 100% relative spread but is good
            // information; a 50/60 quote on a 100 forward is not, despite a
            // relative spread of only 18%.
            if (q.forward > 0.0 && spread / q.forward > cfg.max_spread_over_forward) {
                reject(DiagCode::ExcessiveSpread, "spread", spread / mid, 0.0,
                       cfg.max_relative_spread, "spread too wide both relatively and "
                                                "as a fraction of the forward");
                return status;
            }
            degrade(DiagCode::WideSpreadRelativeToPrice, "spread", spread / mid,
                    "wide relative spread; down-weighted rather than rejected");
        }
    } else if (!have_bid && !have_ask) {
        if (!(q.last > 0.0) && !(cfg.accept_supplied_vol && q.implied_vol > 0.0)) {
            reject(DiagCode::MissingField, "bid/ask", 0.0, 0.0, kInf,
                   "no bid, ask, last or supplied volatility");
            return status;
        }
        degrade(DiagCode::MissingField, "bid/ask", 0.0,
                "no two-sided market; using last or supplied vol");
    } else if (!have_bid) {
        if (cfg.reject_zero_bid) {
            reject(DiagCode::ZeroBid, "bid", q.bid, 0.0, kInf, "zero bid rejected by policy");
            return status;
        }
        // A zero bid with a positive ask is the normal state of a deep wing,
        // and the ask still bounds the volatility from above.
        degrade(DiagCode::ZeroBid, "bid", q.bid,
                "zero bid; the ask still bounds volatility from above");
    }

    if (q.age_seconds >= 0.0 && q.age_seconds > cfg.max_age_seconds) {
        degrade(DiagCode::StaleQuote, "age_seconds", q.age_seconds,
                "quote older than the staleness threshold");
    }
    if (q.volume <= 0.0 && q.open_interest <= 0.0) {
        degrade(DiagCode::ZeroVolumeAndOpenInterest, "volume", 0.0,
                "no volume and no open interest; illiquid");
    }
    return status;
}

// ===========================================================================
// The pipeline
// ===========================================================================

NormalizationResult normalize(const MarketSnapshot& snapshot,
                              const NormalizationConfig& cfg) {
    NormalizationResult out;
    out.stats.received = snapshot.quotes.size();
    out.audit.reserve(snapshot.quotes.size());
    out.quotes.reserve(snapshot.quotes.size());
    out.diagnostics.reserve(snapshot.quotes.size() / 8 + 8);

    if (snapshot.quotes.empty()) {
        out.diagnostics.add(make_diag(DiagCode::MissingField, Severity::Fatal,
                                      snapshot.underlying, "quotes", 0.0,
                                      "snapshot contains no quotes"));
        return out;
    }

    // Duplicate detection needs to see the whole snapshot, so it runs as a
    // pre-pass.  Keyed on (expiry, strike, type) exactly: these come from a
    // listing, not from arithmetic, so exact equality is the right test.
    std::map<std::tuple<double, double, std::int8_t>, std::uint32_t> seen;

    for (std::uint32_t i = 0; i < snapshot.quotes.size(); ++i) {
        OptionQuote q = snapshot.quotes[i];
        q.source_index = i;
        if (!(q.spot > 0.0) && snapshot.spot > 0.0) q.spot = snapshot.spot;

        // --- 1-2. shape and quality --------------------------------------
        QuoteStatus status = validate_quote(q, cfg, out.diagnostics, i);
        if (status == QuoteStatus::Rejected) {
            ++out.stats.rejected;
            ++out.stats.rejected_shape;
            q.status = status;
            q.weight = 0.0;
            out.audit.push_back(q);
            continue;
        }

        const auto key = std::make_tuple(q.years, q.strike, static_cast<std::int8_t>(q.type));
        if (const auto it = seen.find(key); it != seen.end()) {
            out.diagnostics.add(make_diag(DiagCode::DuplicateQuote, Severity::Warning,
                                          quote_label(q), "strike", q.strike,
                                          "duplicate of an earlier quote; the first wins",
                                          i));
            ++out.stats.rejected;
            ++out.stats.rejected_shape;
            q.status = QuoteStatus::Rejected;
            q.weight = 0.0;
            out.audit.push_back(q);
            continue;
        }
        seen.emplace(key, i);

        // --- 3. forward and discount -------------------------------------
        derive_forward_and_discount(q);
        if (!(q.forward > 0.0)) {
            out.diagnostics.add(make_range_diag(
                DiagCode::NonPositiveForward, Severity::Error, quote_label(q), "forward",
                q.forward, 0.0, kInf,
                "no forward supplied and none derivable (spot missing or non-positive)",
                i));
            ++out.stats.rejected;
            ++out.stats.rejected_shape;
            q.status = QuoteStatus::Rejected;
            q.weight = 0.0;
            out.audit.push_back(q);
            continue;
        }
        q.log_moneyness = log_moneyness(q.forward, q.strike);
        if (std::abs(q.log_moneyness) > cfg.max_abs_log_moneyness) {
            out.diagnostics.add(make_range_diag(
                DiagCode::ExcessiveSpread, Severity::Error, quote_label(q),
                "log_moneyness", q.log_moneyness, -cfg.max_abs_log_moneyness,
                cfg.max_abs_log_moneyness,
                "strike too far from the forward to carry information", i));
            ++out.stats.rejected;
            ++out.stats.rejected_shape;
            q.status = QuoteStatus::Rejected;
            q.weight = 0.0;
            out.audit.push_back(q);
            continue;
        }

        // --- 4-5. price bounds and inversion ------------------------------
        if (cfg.accept_supplied_vol && q.implied_vol > 0.0 && !(q.bid > 0.0) &&
            !(q.ask > 0.0)) {
            // A feed that publishes volatilities directly.  Price it so the
            // downstream weighting has a vega to work with.
            q.mid = black_price(q.forward, q.strike, q.implied_vol, q.years, q.discount,
                                q.type);
        } else {
            const OtmSelection sel = select_otm_side(q, cfg);
            const PriceBounds bounds = forward_price_bounds(q.forward, q.strike, sel.side);

            if (!(sel.undiscounted_price > bounds.lower)) {
                out.diagnostics.add(make_range_diag(
                    DiagCode::PriceBelowIntrinsic, Severity::Error, quote_label(q), "mid",
                    sel.undiscounted_price, bounds.lower, bounds.upper,
                    sel.converted_through_parity
                        ? "OTM value non-positive after parity conversion"
                        : "price at or below intrinsic",
                    i));
                ++out.stats.rejected;
                ++out.stats.rejected_bounds;
                q.status = QuoteStatus::Rejected;
                q.weight = 0.0;
                out.audit.push_back(q);
                continue;
            }
            if (sel.undiscounted_price >= bounds.upper) {
                out.diagnostics.add(make_range_diag(
                    DiagCode::PriceAboveForwardBound, Severity::Error, quote_label(q),
                    "mid", sel.undiscounted_price, bounds.lower, bounds.upper,
                    "price at or above the no-arbitrage upper bound", i));
                ++out.stats.rejected;
                ++out.stats.rejected_bounds;
                q.status = QuoteStatus::Rejected;
                q.weight = 0.0;
                out.audit.push_back(q);
                continue;
            }

            const auto iv = implied_volatility_undiscounted(
                sel.undiscounted_price, q.forward, q.strike, q.years, sel.side);
            if (!iv.ok()) {
                out.diagnostics.add(make_range_diag(
                    to_diag_code(iv.status), Severity::Error, quote_label(q), "mid",
                    sel.undiscounted_price, bounds.lower, bounds.upper,
                    to_string(iv.status), i));
                ++out.stats.rejected;
                ++out.stats.rejected_inversion;
                q.status = QuoteStatus::Rejected;
                q.weight = 0.0;
                out.audit.push_back(q);
                continue;
            }
            q.implied_vol = iv.volatility;
            q.mid = sel.undiscounted_price * q.discount;

            // A quote whose own precision does not pin the volatility down is
            // degraded, not rejected: it still constrains the fit, just less.
            // This is the ITM-conversion case, and reporting it is the reason
            // `attainable_rtol` exists.
            if (iv.attainable_rtol > 1e-6) {
                out.diagnostics.add(make_diag(
                    DiagCode::IvVegaTooSmall, Severity::Warning, quote_label(q),
                    "implied_vol", iv.attainable_rtol,
                    "input precision limits the implied vol to few digits", i));
                if (status == QuoteStatus::Ok) status = QuoteStatus::Degraded;
            }
        }

        // --- 6. volatility sanity ----------------------------------------
        if (q.implied_vol < cfg.min_vol || q.implied_vol > cfg.max_vol) {
            out.diagnostics.add(make_range_diag(
                q.implied_vol < cfg.min_vol ? DiagCode::IvBelowFloor
                                            : DiagCode::IvAboveCeiling,
                Severity::Error, quote_label(q), "implied_vol", q.implied_vol, cfg.min_vol,
                cfg.max_vol, "implied volatility outside the plausible band", i));
            ++out.stats.rejected;
            ++out.stats.rejected_vol_range;
            q.status = QuoteStatus::Rejected;
            q.weight = 0.0;
            out.audit.push_back(q);
            continue;
        }

        q.total_variance = total_variance(q.implied_vol, q.years);
        // Vega at the fitted volatility, in price-per-vol-point.  Needed by
        // the weighting model, so computed once here rather than re-derived.
        const double s = q.implied_vol * std::sqrt(q.years);
        const double x = -std::abs(q.log_moneyness);
        q.vega = q.discount * std::sqrt(q.forward) * std::sqrt(q.strike) *
                 normalised_black_vega(x, s) * std::sqrt(q.years);
        q.status = status;
        // The weight is assigned by calibration/weights.hpp; a non-zero
        // placeholder here keeps a quote that is never weighted from silently
        // vanishing.
        q.weight = 1.0;

        if (status == QuoteStatus::Degraded) ++out.stats.degraded;
        ++out.stats.accepted;
        out.audit.push_back(q);
        out.quotes.push_back(q);
    }

    // --- 7. group ---------------------------------------------------------
    std::stable_sort(out.quotes.begin(), out.quotes.end(),
                     [](const OptionQuote& a, const OptionQuote& b) {
                         if (a.years != b.years) return a.years < b.years;
                         if (a.strike != b.strike) return a.strike < b.strike;
                         return static_cast<std::int8_t>(a.type) <
                                static_cast<std::int8_t>(b.type);
                     });

    std::map<double, std::size_t> per_expiry;
    for (const auto& q : out.quotes) ++per_expiry[q.years];

    for (const auto& [years, count] : per_expiry) {
        if (count < cfg.min_quotes_per_slice) {
            out.diagnostics.add(make_range_diag(
                DiagCode::SliceTooFewQuotes, Severity::Warning, "slice", "count",
                static_cast<double>(count),
                static_cast<double>(cfg.min_quotes_per_slice), kInf,
                "too few usable quotes to identify a full slice"));
        } else {
            out.fittable_expiries.push_back(years);
        }
    }
    out.stats.slices = per_expiry.size();

    // Assign slice indices over the *fittable* expiries, which is what the
    // calibrator iterates.
    std::map<double, std::uint32_t> slice_of;
    for (std::uint32_t i = 0; i < out.fittable_expiries.size(); ++i) {
        slice_of[out.fittable_expiries[i]] = i;
    }
    for (auto& q : out.quotes) {
        const auto it = slice_of.find(q.years);
        q.slice_index = (it != slice_of.end()) ? it->second
                                               : std::numeric_limits<std::uint32_t>::max();
    }

    if (out.fittable_expiries.empty()) {
        out.diagnostics.add(make_diag(DiagCode::SliceTooFewQuotes, Severity::Fatal,
                                      snapshot.underlying, "slices", 0.0,
                                      "no expiry has enough usable quotes to fit"));
    }
    return out;
}

}  // namespace vl
