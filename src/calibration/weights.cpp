// SPDX-License-Identifier: MIT
#include "volatility_lab/calibration/weights.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "volatility_lab/pricing/black.hpp"

namespace vl {

/// Half-spread converted into volatility points.
///
/// The conversion is the whole point of the spread weight: a half-spread of
/// 0.05 on a 2.00 near-the-money option is a few basis points of volatility,
/// while the same 0.05 on a 0.10 wing option is several volatility points.
/// Dividing by vega is what makes the two comparable, and it is why `vega` is
/// computed during normalisation and carried on the quote.
double half_spread_in_vol(const OptionQuote& q, const WeightConfig& cfg) noexcept {
    if (!(q.ask > 0.0) || !(q.bid > 0.0) || q.ask < q.bid) {
        // No two-sided market.  Fall back to a wide default rather than a
        // tight one: the absence of a spread is not evidence of precision.
        return 10.0 * cfg.min_half_spread_vol;
    }
    const double half_price = 0.5 * (q.ask - q.bid);
    if (!(q.vega > 0.0)) {
        // Vega has underflowed, which happens in the deep wings.  The price
        // spread cannot be converted, and the quote carries essentially no
        // volatility information, so it gets the floor weight rather than a
        // division by zero.
        return 1.0 / cfg.min_half_spread_vol;  // deliberately huge -> tiny weight
    }
    return std::max(half_price / q.vega, cfg.min_half_spread_vol);
}

namespace {

double liquidity_factor(const OptionQuote& q, const WeightConfig& cfg) noexcept {
    // Square root rather than linear: the penalty should be gentle, because a
    // quote with no reported volume may still be a perfectly good two-sided
    // market and down-weighting it hard would discard the wings of every
    // surface.  Volume and open interest are combined by taking the better of
    // the two, since either one alone is sufficient evidence the strike is
    // real.
    const double v = (cfg.reference_volume > 0.0)
                         ? std::min(1.0, q.volume / cfg.reference_volume)
                         : 1.0;
    const double oi = (cfg.reference_open_interest > 0.0)
                          ? std::min(1.0, q.open_interest / cfg.reference_open_interest)
                          : 1.0;
    const double best = std::max(v, oi);
    // Floor at 0.1 so an illiquid strike still constrains the fit a little.
    return std::max(0.1, std::sqrt(best));
}

double moneyness_factor(const OptionQuote& q, double atm_total_variance,
                        const WeightConfig& cfg) noexcept {
    if (!(atm_total_variance > 0.0) || !(cfg.moneyness_kernel_sd > 0.0)) return 1.0;
    // Scaled by the slice's own standard deviation, not by a fixed
    // percentage.  A fixed-percentage kernel treats 10% out of the money as
    // equally far at one week and at two years, when in standard deviations it
    // is roughly four times further at the short end -- so the near-dated
    // wings would be weighted as though they were near the money.
    const double sd = std::sqrt(atm_total_variance);
    const double z = q.log_moneyness / (sd * cfg.moneyness_kernel_sd);
    return std::exp(-0.5 * z * z);
}

double staleness_factor(const OptionQuote& q, const WeightConfig& cfg) noexcept {
    if (q.age_seconds < 0.0 || !(cfg.stale_window_seconds > 0.0)) return 1.0;
    // Linear decay to a floor rather than a cliff at the threshold: a quote
    // does not become worthless one second after the window, and a
    // discontinuity in the weight makes the fit jump when a quote ages past
    // it.
    const double u = q.age_seconds / cfg.stale_window_seconds;
    return std::max(0.05, 1.0 - 0.95 * std::min(u, 1.0));
}

}  // namespace

// ===========================================================================
// assign_weights
// ===========================================================================

std::vector<WeightBreakdown> assign_weights(std::span<OptionQuote> quotes,
                                            double atm_total_variance,
                                            const WeightConfig& cfg) {
    std::vector<WeightBreakdown> out(quotes.size());
    if (quotes.empty()) return out;

    for (std::size_t i = 0; i < quotes.size(); ++i) {
        const OptionQuote& q = quotes[i];
        WeightBreakdown& b = out[i];

        if (q.status == QuoteStatus::Rejected) {
            b.combined = 0.0;
            b.normalised = 0.0;
            continue;
        }

        b.half_spread_vol = half_spread_in_vol(q, cfg);
        if (cfg.use_spread) {
            // Inverse variance.  Least squares is maximum likelihood under
            // Gaussian errors of equal variance; when the variances differ,
            // the correct estimator weights by their inverse, and the
            // half-spread in vol points is the market's own estimate of the
            // standard deviation of its quote.
            b.spread = 1.0 / (b.half_spread_vol * b.half_spread_vol);
        }
        if (cfg.use_liquidity) b.liquidity = liquidity_factor(q, cfg);
        if (cfg.use_moneyness) b.moneyness = moneyness_factor(q, atm_total_variance, cfg);
        if (cfg.use_staleness) b.staleness = staleness_factor(q, cfg);
        if (q.status == QuoteStatus::Degraded) b.status = cfg.degraded_multiplier;

        b.combined = b.spread * b.liquidity * b.moneyness * b.staleness * b.status;
        if (!std::isfinite(b.combined) || b.combined < 0.0) b.combined = 0.0;
    }

    // Cap relative to the median before normalising.
    //
    // This is the guard against the characteristic failure of
    // inverse-variance weighting: the estimator trusts whichever observation
    // *claims* the smallest error, so one suspiciously tight quote -- a locked
    // market, a stale print, a crossed quote that survived -- can carry more
    // weight than the rest of the slice combined.  The median is the right
    // reference because it is itself robust to exactly that outlier.
    std::vector<double> positives;
    positives.reserve(quotes.size());
    for (const auto& b : out) {
        if (b.combined > 0.0) positives.push_back(b.combined);
    }
    if (positives.empty()) return out;

    const std::size_t mid = positives.size() / 2;
    std::nth_element(positives.begin(), positives.begin() + static_cast<std::ptrdiff_t>(mid),
                     positives.end());
    const double median = positives[mid];
    const double median_cap = (cfg.max_weight_ratio > 0.0 && median > 0.0)
                                  ? median * cfg.max_weight_ratio
                                  : std::numeric_limits<double>::infinity();
    for (auto& b : out) {
        if (b.combined > 0.0) b.combined = std::min(b.combined, median_cap);
    }

    // Then the share cap, which is the constraint that actually expresses the
    // intent.  The median ratio alone does not imply it: with ten quotes and a
    // 20x median cap the outlier still carries 69% of the total.  Iterated
    // because capping changes the total it is a fraction of; three passes
    // reach the fixed point.
    double sum = 0.0;
    std::size_t used = 0;
    for (const auto& b : out) {
        if (b.combined > 0.0) {
            sum += b.combined;
            ++used;
        }
    }
    if (cfg.max_weight_fraction > 0.0 && cfg.max_weight_fraction < 1.0 && used > 1) {
        // The cap cannot be below 1/used: n weights summing to the total
        // cannot all be under (1/n) of it, so a tighter cap is infeasible and
        // the iteration drives every weight to exactly equal -- silently
        // discarding the entire weighting model.  That is what happened with
        // two quotes and a 25% cap, which is how this was found.  The floor of
        // 1.5/used leaves room for a genuine spread of weights in small
        // slices while still binding on the large ones the cap is for.
        const double fraction =
            std::max(cfg.max_weight_fraction, 1.5 / static_cast<double>(used));
        for (int pass = 0; pass < 3; ++pass) {
            const double share_cap = fraction * sum;
            double new_sum = 0.0;
            bool changed = false;
            for (auto& b : out) {
                if (b.combined <= 0.0) continue;
                if (b.combined > share_cap) {
                    b.combined = share_cap;
                    changed = true;
                }
                new_sum += b.combined;
            }
            sum = new_sum;
            if (!changed) break;
        }
    }

    // Normalise to a unit mean, so the reported objective and rms_residual are
    // comparable between slices with different quote counts and the LM
    // tolerances do not need re-tuning per slice.
    const double scale = (used > 0 && sum > 0.0)
                             ? static_cast<double>(used) / sum
                             : 0.0;
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        out[i].normalised = out[i].combined * scale;
        quotes[i].weight = out[i].normalised;
    }
    return out;
}

std::vector<WeightBreakdown> assign_weights_by_slice(std::span<OptionQuote> quotes,
                                                     const WeightConfig& cfg) {
    std::vector<WeightBreakdown> out(quotes.size());
    if (quotes.empty()) return out;

    // Group by expiry.  The book is sorted by (expiry, strike) upstream, but
    // this must not depend on that -- it is called on raw vectors too.
    std::map<double, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < quotes.size(); ++i) {
        groups[quotes[i].years].push_back(i);
    }

    for (const auto& [years, indices] : groups) {
        // The slice's ATM total variance, for the moneyness kernel: the
        // variance of the quote closest to the money.
        double atm_w = 0.0;
        double best_abs_k = std::numeric_limits<double>::infinity();
        for (std::size_t i : indices) {
            const double ak = std::abs(quotes[i].log_moneyness);
            if (ak < best_abs_k) {
                best_abs_k = ak;
                atm_w = quotes[i].total_variance;
            }
        }

        // Weighting is normalised *per slice*, not globally.  A global
        // normalisation would let a slice with 60 quotes outvote one with 12
        // simply by having more of them, when what is wanted is for each
        // expiry's shape to be fitted on its own terms.
        std::vector<OptionQuote> slice;
        slice.reserve(indices.size());
        for (std::size_t i : indices) slice.push_back(quotes[i]);

        const auto breakdowns = assign_weights(slice, atm_w, cfg);
        for (std::size_t j = 0; j < indices.size(); ++j) {
            quotes[indices[j]].weight = slice[j].weight;
            out[indices[j]] = breakdowns[j];
        }
    }
    return out;
}

// ===========================================================================
// Residuals
// ===========================================================================

double quote_residual(const OptionQuote& q, double model_total_variance,
                      ResidualKind kind) noexcept {
    switch (kind) {
        case ResidualKind::TotalVariance:
            return model_total_variance - q.total_variance;

        case ResidualKind::Volatility: {
            if (!(q.years > 0.0)) return 0.0;
            const double model_vol = (model_total_variance > 0.0)
                                         ? std::sqrt(model_total_variance / q.years)
                                         : 0.0;
            return model_vol - q.implied_vol;
        }

        case ResidualKind::Price: {
            if (!(q.years > 0.0) || !(q.forward > 0.0)) return 0.0;
            const double model_vol = (model_total_variance > 0.0)
                                         ? std::sqrt(model_total_variance / q.years)
                                         : 0.0;
            const double model_price =
                black_price(q.forward, q.strike, model_vol, q.years, q.discount, q.type);
            return model_price - q.mid;
        }

        case ResidualKind::VegaScaledPrice: {
            if (!(q.years > 0.0) || !(q.forward > 0.0)) return 0.0;
            const double model_vol = (model_total_variance > 0.0)
                                         ? std::sqrt(model_total_variance / q.years)
                                         : 0.0;
            const double model_price =
                black_price(q.forward, q.strike, model_vol, q.years, q.discount, q.type);
            // Degrades smoothly where vega is small instead of dividing by a
            // near-zero sensitivity: that is the entire reason this kind
            // exists alongside `Volatility`.
            const double denom = std::max(q.vega, 1e-10);
            return (model_price - q.mid) / denom;
        }
    }
    return 0.0;
}

double quote_residual_dw(const OptionQuote& q, double model_total_variance,
                         ResidualKind kind) noexcept {
    switch (kind) {
        case ResidualKind::TotalVariance:
            return 1.0;

        case ResidualKind::Volatility: {
            // sigma = sqrt(w/T), so dsigma/dw = 1/(2 sqrt(w T)).
            if (!(q.years > 0.0) || !(model_total_variance > 0.0)) return 0.0;
            return 1.0 / (2.0 * std::sqrt(model_total_variance * q.years));
        }

        case ResidualKind::Price:
        case ResidualKind::VegaScaledPrice: {
            if (!(q.years > 0.0) || !(model_total_variance > 0.0)) return 0.0;
            // dPrice/dw = vega * dsigma/dw, with vega at the *model* vol.
            const double model_vol = std::sqrt(model_total_variance / q.years);
            const double s = model_vol * std::sqrt(q.years);
            const double x = -std::abs(q.log_moneyness);
            const double vega = q.discount * std::sqrt(q.forward) * std::sqrt(q.strike) *
                                normalised_black_vega(x, s) * std::sqrt(q.years);
            const double dsigma_dw = 1.0 / (2.0 * std::sqrt(model_total_variance * q.years));
            const double d = vega * dsigma_dw;
            if (kind == ResidualKind::VegaScaledPrice) {
                return d / std::max(q.vega, 1e-10);
            }
            return d;
        }
    }
    return 0.0;
}

}  // namespace vl
