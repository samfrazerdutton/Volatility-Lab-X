// SPDX-License-Identifier: MIT
#include "volatility_lab/runtime/full_rebuild.hpp"

#include <algorithm>
#include <cmath>

namespace vl {

namespace {

// Deliberately re-derived here rather than shared with
// `incremental_engine.cpp`'s identical-looking helpers: the whole point of
// this file is to not depend on anything the optimized path depends on,
// so a shared bug in a shared helper cannot hide from the comparison
// between them.
TermCurve forward_curve(const MarketPoint& market, std::span<const double> years) {
    if (years.empty()) return TermCurve::flat(market.spot);
    std::vector<double> ys(years.begin(), years.end());
    std::vector<double> fwds;
    fwds.reserve(ys.size());
    for (double t : ys) fwds.push_back(market.spot * std::exp((market.rate - market.carry) * t));
    return TermCurve(std::move(ys), std::move(fwds));
}

TermCurve discount_curve(const MarketPoint& market, std::span<const double> years) {
    if (years.empty()) return TermCurve::flat(1.0);
    std::vector<double> ys(years.begin(), years.end());
    std::vector<double> dfs;
    dfs.reserve(ys.size());
    for (double t : ys) dfs.push_back(std::exp(-market.rate * t));
    return TermCurve(std::move(ys), std::move(dfs));
}

}  // namespace

FullRebuildResult full_rebuild(std::span<const OptionQuote> quotes,
                               const IncrementalEngine::Config& config) {
    constexpr double kYearsMatchTolerance = 1.0e-9;
    const MarketPoint market = config.baseline_market;

    // Distinct expiries: a plain scan, not an index -- see the header
    // comment for why that is the point, not an oversight.
    std::vector<double> distinct_years;
    for (const auto& q : quotes) {
        if (std::find_if(distinct_years.begin(), distinct_years.end(), [&](double y) {
                return std::abs(y - q.years) <= kYearsMatchTolerance;
            }) == distinct_years.end()) {
            distinct_years.push_back(q.years);
        }
    }
    std::sort(distinct_years.begin(), distinct_years.end());

    std::vector<SliceVariant> slices;
    slices.reserve(distinct_years.size());
    for (double years : distinct_years) {
        std::vector<OptionQuote> slice_quotes;
        for (const auto& q : quotes) {
            if (std::abs(q.years - years) <= kYearsMatchTolerance) slice_quotes.push_back(q);
        }
        MarketSnapshot snapshot;
        snapshot.underlying = "full_rebuild";
        snapshot.spot = market.spot;
        snapshot.quotes = slice_quotes;
        auto normalized = normalize(snapshot, config.normalize);
        (void)assign_weights_by_slice(normalized.quotes, config.weights);
        const auto fit = calibrate_svi_slice(normalized.quotes, config.calibrator);
        slices.emplace_back(fit.params);
    }

    FullRebuildResult out;
    out.surface = VolSurface(std::move(slices), forward_curve(market, distinct_years),
                             discount_curve(market, distinct_years));
    out.differential = compute_surface_differential(config.baseline_surface, out.surface);
    out.uncertainty = estimate_point_uncertainty(out.surface, quotes, config.uncertainty_k,
                                                  config.uncertainty_years, config.uncertainty);

    out.valuations.resize(config.positions.size());
    for (std::size_t i = 0; i < config.positions.size(); ++i) {
        out.valuations[i] = value_position(config.positions[i], out.surface, market);
    }

    std::vector<OptionGreeks> greeks(out.valuations.size());
    std::vector<double> qty(config.positions.size()), mult(config.positions.size());
    for (std::size_t i = 0; i < config.positions.size(); ++i) {
        greeks[i] = out.valuations[i].greeks;
        qty[i] = config.positions[i].quantity;
        mult[i] = config.positions[i].multiplier;
    }
    out.portfolio = aggregate_greeks(greeks, qty, mult);
    out.pnl = compute_pnl_attribution(config.positions, config.baseline_surface,
                                      config.baseline_market, out.surface, market);
    return out;
}

}  // namespace vl
