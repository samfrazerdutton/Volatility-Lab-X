// SPDX-License-Identifier: MIT
#include "volatility_lab/risk/uncertainty.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vl {

namespace {

/// Observed (log-moneyness, years) range across quotes that were not
/// outright rejected -- a degraded-but-used quote still counts as "the
/// market showed something here".
struct ObservedRange {
    double min_k = std::numeric_limits<double>::infinity();
    double max_k = -std::numeric_limits<double>::infinity();
    double min_t = std::numeric_limits<double>::infinity();
    double max_t = -std::numeric_limits<double>::infinity();
    bool any = false;
};

ObservedRange observed_range(std::span<const OptionQuote> quotes) {
    ObservedRange r;
    for (const auto& q : quotes) {
        if (q.status == QuoteStatus::Rejected) continue;
        r.any = true;
        r.min_k = std::min(r.min_k, q.log_moneyness);
        r.max_k = std::max(r.max_k, q.log_moneyness);
        r.min_t = std::min(r.min_t, q.years);
        r.max_t = std::max(r.max_t, q.years);
    }
    return r;
}

}  // namespace

PointUncertainty estimate_point_uncertainty(const VolSurface& surface,
                                             std::span<const OptionQuote> quotes, double k,
                                             double years, const UncertaintyConfig& cfg) {
    PointUncertainty out{};
    out.k = k;
    out.years = years;
    out.vol_estimate = surface.vol(k, years);

    const ObservedRange range = observed_range(quotes);
    if (range.any) {
        out.extrapolated_in_strike = (k < range.min_k || k > range.max_k);
        out.extrapolated_in_time = (years < range.min_t || years > range.max_t);
    }

    // Local relevance kernel scale: the surface's own ATM total variance at
    // this tenor, same convention as calibration/weights.hpp's moneyness
    // kernel (scale by standard deviation, not a fixed percentage of
    // moneyness, so a short-dated and a long-dated query are compared on a
    // footing where "how many standard deviations out of the money" means
    // the same thing).
    const double atm_w = surface.total_variance(0.0, years);
    const double sd = (atm_w > 0.0) ? std::sqrt(atm_w) : 0.0;

    // Capped at the book's own observed tenor span: without this, a query
    // far beyond the longest quoted tenor gets a bandwidth proportional to
    // its OWN years, which can grow wide enough to swallow the entire book
    // as if every short-dated quote were locally informative about it --
    // understating uncertainty exactly where it should be highest. The cap
    // is itself an observable (how wide the data actually is), not an
    // invented constant.
    const double span_cap = range.any ? std::max(cfg.min_time_bandwidth_years,
                                                  range.max_t - range.min_t)
                                      : std::numeric_limits<double>::infinity();
    const double t_bw = std::min(
        span_cap,
        std::max(cfg.min_time_bandwidth_years, cfg.time_bandwidth_frac * std::max(years, 0.0)));

    double sum_w = 0.0;
    double sum_w2 = 0.0;
    double sum_w_spread = 0.0;
    double sum_w_age = 0.0;
    double sum_w_age_valid = 0.0;

    for (const auto& q : quotes) {
        if (q.status == QuoteStatus::Rejected) continue;
        if (!(q.weight > 0.0)) continue;  // excluded from the fit entirely

        double moneyness_kernel = 1.0;
        if (sd > 0.0 && cfg.moneyness_kernel_sd > 0.0) {
            const double z = (q.log_moneyness - k) / (sd * cfg.moneyness_kernel_sd);
            moneyness_kernel = std::exp(-0.5 * z * z);
        }
        const double dt = (q.years - years) / t_bw;
        const double time_kernel = std::exp(-0.5 * dt * dt);

        const double w = q.weight * moneyness_kernel * time_kernel;
        if (!(w > 0.0) || !std::isfinite(w)) continue;

        sum_w += w;
        sum_w2 += w * w;
        sum_w_spread += w * half_spread_in_vol(q, cfg.spread_weights);

        if (q.age_seconds >= 0.0) {
            sum_w_age += w * q.age_seconds;
            sum_w_age_valid += w;
        }
    }

    if (!(sum_w > 0.0)) {
        out.no_local_coverage = true;
        out.vol_std_error = std::numeric_limits<double>::infinity();
        return out;
    }

    out.effective_n = (sum_w * sum_w) / sum_w2;
    out.representative_age_seconds =
        (sum_w_age_valid > 0.0) ? (sum_w_age / sum_w_age_valid) : -1.0;

    const double weighted_avg_spread = sum_w_spread / sum_w;
    double vol_std_error = weighted_avg_spread / std::sqrt(std::max(out.effective_n, 1.0));

    // Extrapolation inflation: distance beyond the observed range, in units
    // of the same local bandwidth the relevance kernel itself uses, so one
    // config knob controls both "how local is local" and "how much does
    // leaving that locality cost".
    if (out.extrapolated_in_strike && sd > 0.0) {
        const double dist = std::max(range.min_k - k, k - range.max_k);
        const double bw = sd * cfg.moneyness_kernel_sd;
        vol_std_error *= (1.0 + cfg.extrapolation_penalty_per_bandwidth * (dist / bw));
    }
    if (out.extrapolated_in_time) {
        const double dist = std::max(range.min_t - years, years - range.max_t);
        vol_std_error *= (1.0 + cfg.extrapolation_penalty_per_bandwidth * (dist / t_bw));
    }

    out.vol_std_error = std::max(vol_std_error, cfg.min_vol_std_error);
    return out;
}

std::vector<PointUncertainty> estimate_grid_uncertainty(const VolSurface& surface,
                                                         std::span<const OptionQuote> quotes,
                                                         std::span<const double> k,
                                                         std::span<const double> years,
                                                         const UncertaintyConfig& cfg) {
    std::vector<PointUncertainty> out;
    const std::size_t n = std::min(k.size(), years.size());
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(estimate_point_uncertainty(surface, quotes, k[i], years[i], cfg));
    }
    return out;
}

}  // namespace vl
