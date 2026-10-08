// SPDX-License-Identifier: MIT
#include "volatility_lab/scenarios/scenario.hpp"

#include <algorithm>

namespace vl {

namespace {

std::vector<double> default_k_grid() {
    // 41 points over [-1.0, 1.0] (step 0.05). GridSlice's spline interpolates
    // *total variance*, so even a shock that is exactly linear in vol (a
    // pure skew_shift) becomes exactly quadratic in k once squared into
    // variance -- and CubicSpline uses natural boundary conditions (zero
    // second derivative at the ends, see math/interpolation.hpp), which a
    // true nonzero-curvature quadratic does not satisfy, so the spline
    // deviates from it near the grid's own edges. Measured directly: with
    // the previous [-0.6, 0.6]/21-point grid, that deviation was already
    // ~1.6e-6 at k=+-0.4 -- inside default_differential_grid()'s own query
    // range. Both widening the domain (so +-0.4 sits further from the
    // boundary-affected region) and raising the node density reduce it; this
    // configuration was chosen because it measures at ~1e-16 (machine
    // precision) there, not picked a priori -- see
    // tests/scenarios/scenario.cpp's exact-recovery tests. 0.05 spacing also
    // means every point of the default differential grid's moneyness axis
    // (-0.4, -0.2, 0, 0.2, 0.4) lands exactly on a node.
    std::vector<double> ks;
    ks.reserve(41);
    for (int i = 0; i <= 40; ++i) ks.push_back(-1.0 + 0.05 * static_cast<double>(i));
    return ks;
}

/// Floor so an extreme, pathological shock (e.g. a large negative
/// level_shift on an already-low-vol surface) cannot drive the shocked vol
/// to zero or negative -- which would make total_variance zero or negative,
/// not representable as a GridSlice. Realistic shock sizes never approach
/// this floor.
constexpr double kMinShockedVol = 1.0e-4;

}  // namespace

VolSurface apply_shock_to_surface(const VolSurface& base, const ShockSpec& shock,
                                  std::span<const double> k_grid) {
    static const std::vector<double> default_grid = default_k_grid();
    const std::span<const double> ks = k_grid.empty() ? default_grid : k_grid;

    const auto expiries = base.expiries();
    std::vector<SliceVariant> slices;
    slices.reserve(expiries.size());

    for (double T : expiries) {
        std::vector<double> ws(ks.size());
        for (std::size_t i = 0; i < ks.size(); ++i) {
            const double k = ks[i];
            const double base_vol = base.vol(k, T);
            const double shocked_vol = std::max(
                kMinShockedVol, base_vol + shock.level_shift + shock.skew_shift * k +
                                    shock.curvature_shift * k * k + shock.term_shift * T);
            ws[i] = shocked_vol * shocked_vol * T;
        }
        slices.emplace_back(GridSlice(std::vector<double>(ks.begin(), ks.end()), std::move(ws), T));
    }

    const TermCurve& old_forwards = base.forwards();
    const double spot_factor = 1.0 + shock.spot_pct;
    std::vector<double> fwd_nodes(old_forwards.nodes().begin(), old_forwards.nodes().end());
    std::vector<double> fwd_values;
    fwd_values.reserve(fwd_nodes.size());
    for (double T : fwd_nodes) fwd_values.push_back(old_forwards(T) * spot_factor);
    const TermCurve new_forwards =
        fwd_nodes.empty() ? TermCurve::flat(old_forwards(1.0) * spot_factor)
                         : TermCurve(std::move(fwd_nodes), std::move(fwd_values));

    return VolSurface(std::move(slices), new_forwards, base.discounts());
}

MarketPoint apply_shock_to_market(const MarketPoint& base, const ShockSpec& shock) noexcept {
    MarketPoint out = base;
    out.spot = base.spot * (1.0 + shock.spot_pct);
    return out;
}

ScenarioResult evaluate_scenario(std::span<const Position> positions,
                                 const VolSurface& base_surface, const MarketPoint& base_market,
                                 const ShockSpec& shock) {
    ScenarioResult out;
    out.label = shock.label;

    const VolSurface shocked_surface = apply_shock_to_surface(base_surface, shock);
    const MarketPoint shocked_market = apply_shock_to_market(base_market, shock);

    std::vector<double> new_years;
    if (shock.time_decay_days != 0.0) {
        new_years.reserve(positions.size());
        const double dt = shock.time_decay_days / 365.0;
        for (const auto& p : positions) new_years.push_back(p.years - dt);
    }

    out.attribution = compute_pnl_attribution(positions, base_surface, base_market,
                                              shocked_surface, shocked_market, new_years);
    out.realized_shock = compute_surface_differential(base_surface, shocked_surface);
    return out;
}

std::vector<ScenarioResult> evaluate_scenarios(std::span<const Position> positions,
                                               const VolSurface& base_surface,
                                               const MarketPoint& base_market,
                                               std::span<const ShockSpec> shocks) {
    std::vector<ScenarioResult> out;
    out.reserve(shocks.size());
    for (const auto& shock : shocks) {
        out.push_back(evaluate_scenario(positions, base_surface, base_market, shock));
    }
    return out;
}

}  // namespace vl
