// SPDX-License-Identifier: MIT
#include "volatility_lab/portfolio/portfolio.hpp"

#include <algorithm>
#include <cmath>

#include "volatility_lab/pricing/black.hpp"

namespace vl {

PositionValuation value_position(const Position& p, const VolSurface& surface,
                                 const MarketPoint& market) noexcept {
    PositionValuation out{};
    if (!(market.spot > 0.0) || !(p.strike > 0.0) || !(p.years >= 0.0)) {
        out.greeks.price = std::numeric_limits<double>::quiet_NaN();
        return out;
    }
    out.forward = market.spot * std::exp((market.rate - market.carry) * p.years);
    out.log_moneyness = log_moneyness(out.forward, p.strike);
    out.vol_used = surface.vol(out.log_moneyness, p.years);
    out.greeks = black_scholes_greeks(market.spot, p.strike, out.vol_used, p.years,
                                      market.rate, market.carry, p.type);
    return out;
}

void value_portfolio(std::span<const Position> positions, const VolSurface& surface,
                     const MarketPoint& market, std::span<PositionValuation> out) noexcept {
    const std::size_t n = std::min(positions.size(), out.size());
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = value_position(positions[i], surface, market);
    }
}

PortfolioGreeks portfolio_greeks(std::span<const Position> positions, const VolSurface& surface,
                                 const MarketPoint& market) noexcept {
    std::vector<PositionValuation> vals(positions.size());
    value_portfolio(positions, surface, market, vals);

    std::vector<OptionGreeks> greeks(positions.size());
    std::vector<double> quantity(positions.size());
    std::vector<double> multiplier(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        greeks[i] = vals[i].greeks;
        quantity[i] = positions[i].quantity;
        multiplier[i] = positions[i].multiplier;
    }
    return aggregate_greeks(greeks, quantity, multiplier);
}

double PnLAttribution::explained_fraction() const noexcept {
    const double denom = std::max(std::abs(total_exact_pnl), 1e-9);
    return 1.0 - std::abs(residual) / denom;
}

PnLAttribution compute_pnl_attribution(std::span<const Position> positions,
                                       const VolSurface& base_surface,
                                       const MarketPoint& base_market,
                                       const VolSurface& new_surface,
                                       const MarketPoint& new_market,
                                       std::span<const double> new_years) {
    PnLAttribution out{};
    const std::size_t n = positions.size();
    out.leg_exact_pnl.assign(n, 0.0);
    out.leg_residual.assign(n, 0.0);

    // Accumulated in position order -- the library-wide determinism
    // contract applies to attribution exactly as it does to any other
    // reduction: the same book in the same order gives the same numbers
    // regardless of how (or whether) this loop is later parallelised.
    for (std::size_t i = 0; i < n; ++i) {
        const Position& p = positions[i];
        const double w = p.signed_notional();

        const PositionValuation base = value_position(p, base_surface, base_market);

        // The bumped leg: same contract, but (spot, rate, carry) from the
        // new market point and vol from the *new* surface at this leg's own
        // moneyness -- which may have moved by a different amount than
        // another leg's vol did, so aggregating this across the book is what
        // lets a skew or term-structure move show up in the vol bucket even
        // though every leg uses the same single-number vol_pnl formula.
        double new_strike_years = p.years;
        if (i < new_years.size()) new_strike_years = new_years[i];
        const double new_forward =
            new_market.spot * std::exp((new_market.rate - new_market.carry) * new_strike_years);
        const double new_k = log_moneyness(new_forward, p.strike);
        const double new_vol = new_surface.vol(new_k, new_strike_years);

        const double dS = new_market.spot - base_market.spot;
        const double dVol = new_vol - base.vol_used;
        const double dR = new_market.rate - base_market.rate;
        const double dYears = new_strike_years - p.years;

        const GreekBump bump{dS, dVol, dR, dYears};
        const TaylorVsExact cmp = taylor_vs_exact_reprice(
            base.greeks, base_market.spot, p.strike, base.vol_used, p.years, base_market.rate,
            base_market.carry, p.type, bump);

        out.leg_exact_pnl[i] = w * cmp.exact_pnl;
        out.total_exact_pnl += w * cmp.exact_pnl;
        out.base_value += w * cmp.base_price;
        out.new_value += w * cmp.exact_price;

        out.spot_pnl += w * base.greeks.delta * dS;
        out.vol_pnl += w * base.greeks.vega * dVol;
        out.rate_pnl += w * base.greeks.rho * dR;
        // Taylor's own "time" axis is calendar time t = -T; see
        // taylor_vs_exact_reprice for why dt = -dYears.
        const double dt = -dYears;
        out.theta_pnl += w * base.greeks.theta * dt;

        out.gamma_pnl += w * 0.5 * base.greeks.gamma * dS * dS;
        out.vanna_pnl += w * base.greeks.vanna * dS * dVol;
        out.volga_pnl += w * 0.5 * base.greeks.volga * dVol * dVol;
        out.charm_pnl += w * base.greeks.charm * dS * dt;

        out.leg_residual[i] = w * cmp.exact_pnl - w * cmp.order2_pnl;
    }

    // Defined, not computed from an independent formula, so the
    // reconciliation identity is exact by construction rather than merely
    // approximately true: whatever the eight attributed components above
    // did not capture is, by definition, the residual.
    const double attributed = out.spot_pnl + out.vol_pnl + out.rate_pnl + out.theta_pnl +
                              out.gamma_pnl + out.vanna_pnl + out.volga_pnl + out.charm_pnl;
    out.residual = out.total_exact_pnl - attributed;
    return out;
}

std::vector<std::size_t> largest_pnl_contributors(const PnLAttribution& attr, std::size_t n) {
    std::vector<std::size_t> idx(attr.leg_exact_pnl.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return std::abs(attr.leg_exact_pnl[a]) > std::abs(attr.leg_exact_pnl[b]);
    });
    if (idx.size() > n) idx.resize(n);
    return idx;
}

}  // namespace vl
