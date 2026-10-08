// SPDX-License-Identifier: MIT
#include "volatility_lab/risk/uncertainty_propagation.hpp"

#include <cmath>
#include <map>

namespace vl {

PositionGreekUncertainty propagate_greek_uncertainty(const OptionGreeks& greeks,
                                                      double vol_std_error) noexcept {
    PositionGreekUncertainty out;
    out.vol_std_error = vol_std_error;
    out.price_std_error = std::abs(greeks.vega) * vol_std_error;
    out.delta_std_error = std::abs(greeks.vanna) * vol_std_error;
    out.vega_std_error = std::abs(greeks.volga) * vol_std_error;
    return out;
}

PortfolioUncertainty propagate_portfolio_uncertainty(std::span<const Position> positions,
                                                      std::span<const PositionValuation> valuations,
                                                      const VolSurface& surface,
                                                      std::span<const OptionQuote> quotes,
                                                      const UncertaintyConfig& cfg) {
    PortfolioUncertainty out;
    if (positions.size() != valuations.size() || positions.empty()) return out;

    // Group dollar-vega and dollar-vanna by expiry, summing within a group
    // (same-expiry => treated as fully correlated) -- see the file comment
    // for why this is the rule rather than either extreme.
    std::map<double, double> dollar_vega_by_expiry;
    std::map<double, double> dollar_vanna_by_expiry;
    std::map<double, double> vol_std_error_by_expiry_sum;
    std::map<double, std::size_t> count_by_expiry;

    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto& p = positions[i];
        const auto& v = valuations[i];
        const double w = p.signed_notional();

        const PointUncertainty pu =
            estimate_point_uncertainty(surface, quotes, v.log_moneyness, p.years, cfg);
        if (pu.no_local_coverage) {
            out.all_positions_had_coverage = false;
            continue;
        }

        dollar_vega_by_expiry[p.years] += w * v.greeks.vega;
        dollar_vanna_by_expiry[p.years] += w * v.greeks.vanna;
        vol_std_error_by_expiry_sum[p.years] += pu.vol_std_error;
        count_by_expiry[p.years] += 1;
    }

    double price_var = 0.0;
    double delta_var = 0.0;
    out.price_contribution_by_expiry.reserve(dollar_vega_by_expiry.size());
    for (const auto& [years, dollar_vega] : dollar_vega_by_expiry) {
        const double n = static_cast<double>(count_by_expiry.at(years));
        const double avg_vol_std_error = vol_std_error_by_expiry_sum.at(years) / n;

        const double price_contribution = std::abs(dollar_vega) * avg_vol_std_error;
        const double delta_contribution = std::abs(dollar_vanna_by_expiry.at(years)) * avg_vol_std_error;

        out.price_contribution_by_expiry.emplace_back(years, price_contribution);
        // Different expiries combined in quadrature (independence
        // assumption across buckets); within a bucket the contribution
        // above already summed dollar-vega linearly before taking the
        // absolute value, which is the "fully correlated within the
        // bucket" treatment.
        price_var += price_contribution * price_contribution;
        delta_var += delta_contribution * delta_contribution;
    }

    out.price_std_error = std::sqrt(price_var);
    out.delta_std_error = std::sqrt(delta_var);
    return out;
}

}  // namespace vl
