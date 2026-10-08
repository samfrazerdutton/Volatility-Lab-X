// SPDX-License-Identifier: MIT
#include "volatility_lab/diagnostics/surface_differential.hpp"

#include <algorithm>
#include <cmath>
#include <map>

#include "volatility_lab/math/linalg.hpp"

namespace vl {

std::vector<DifferentialGridPoint> default_differential_grid() {
    std::vector<DifferentialGridPoint> grid;
    const double tenors[] = {1.0 / 12.0, 0.25, 0.5, 1.0, 2.0};
    const double moneyness[] = {-0.4, -0.2, 0.0, 0.2, 0.4};
    grid.reserve(std::size(tenors) * std::size(moneyness));
    for (double T : tenors) {
        for (double k : moneyness) {
            grid.push_back({k, T});
        }
    }
    return grid;
}

double SurfaceDifferential::explained_fraction() const noexcept {
    const double denom = std::max(total_delta_rms, 1e-12);
    return 1.0 - residual_rms / denom;
}

SurfaceDifferential compute_surface_differential(const VolSurface& old_surface,
                                                  const VolSurface& new_surface,
                                                  std::span<const DifferentialGridPoint> grid) {
    SurfaceDifferential out{};

    static const std::vector<DifferentialGridPoint> default_grid = default_differential_grid();
    const std::span<const DifferentialGridPoint> points = grid.empty() ? default_grid : grid;
    const std::size_t n = points.size();

    out.grid_k.resize(n);
    out.grid_years.resize(n);
    out.grid_delta_w.resize(n);
    out.grid_fitted_w.resize(n);
    out.grid_residual_w.resize(n);
    if (n == 0) return out;

    // --- the shape regression: level, skew, curvature, term ---------------
    //
    // Design row (1, k, k^2, T), accumulated into the normal equations in
    // grid order -- exactly the same NormalEquations/Cholesky machinery the
    // SVI calibrator's inner solve uses, reused here for an entirely
    // different, four-parameter regression.
    math::NormalEquations eq(4);
    for (std::size_t i = 0; i < n; ++i) {
        const double k = points[i].k;
        const double T = points[i].years;
        out.grid_k[i] = k;
        out.grid_years[i] = T;
        // Regressed in *annualised volatility*, not total variance.  A
        // uniform vol shift of (say) +4 points at every tenor -- the
        // textbook definition of a pure level move -- is, in total-variance
        // terms, dW = (sigma_new^2 - sigma_old^2)*T: linear in T, and so
        // indistinguishable from a term-structure twist under a
        // variance-space regression.  An earlier version of this function
        // regressed dW directly and a +4-vol-point parallel shift landed
        // entirely in term_shift with level_shift reading zero -- exactly
        // backwards from what the brief's own vocabulary ("did overall
        // implied volatility move") means by level.  Volatility is what
        // actually shifts uniformly for a level move, so it is what is
        // regressed.
        const double vol_old = old_surface.vol(k, T);
        const double vol_new = new_surface.vol(k, T);
        out.grid_delta_w[i] = vol_new - vol_old;

        const double row[4] = {1.0, k, k * k, T};
        eq.add(std::span<const double>(row, 4), out.grid_delta_w[i], 1.0);
    }

    math::SmallMatrix a = eq.matrix();
    std::array<double, 4> coeffs{};
    for (std::size_t i = 0; i < 4; ++i) coeffs[i] = eq.rhs()[i];
    const auto chol = math::cholesky_solve(a, std::span<double>(coeffs.data(), 4));
    out.condition_estimate = chol.condition_estimate;

    if (chol.ok()) {
        out.level_shift = coeffs[0];
        out.skew_shift = coeffs[1];
        out.curvature_shift = coeffs[2];
        out.term_shift = coeffs[3];
    }
    // If the regression is singular (e.g. a grid with fewer than 4 distinct
    // points, or one degenerate in k or T), the coefficients stay at zero
    // and *everything* shows up as residual -- an honest "could not
    // decompose this" rather than a silently wrong decomposition.

    double sum_sq_delta = 0.0;
    double sum_sq_resid = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double k = out.grid_k[i];
        const double T = out.grid_years[i];
        const double fitted =
            out.level_shift + out.skew_shift * k + out.curvature_shift * k * k +
            out.term_shift * T;
        out.grid_fitted_w[i] = fitted;
        out.grid_residual_w[i] = out.grid_delta_w[i] - fitted;
        sum_sq_delta += out.grid_delta_w[i] * out.grid_delta_w[i];
        sum_sq_resid += out.grid_residual_w[i] * out.grid_residual_w[i];
    }
    out.total_delta_rms = std::sqrt(sum_sq_delta / static_cast<double>(n));
    out.residual_rms = std::sqrt(sum_sq_resid / static_cast<double>(n));

    // --- forward: measured directly, not regressed -------------------------
    //
    // Distinct tenors present on the grid, each queried once from each
    // surface's own forward curve.  log, not raw ratio, so that the average
    // is the natural (additive, time-consistent) way to summarise a set of
    // per-tenor forward returns.
    {
        std::vector<double> distinct_tenors;
        for (const auto& pt : points) {
            if (std::find(distinct_tenors.begin(), distinct_tenors.end(), pt.years) ==
                distinct_tenors.end()) {
                distinct_tenors.push_back(pt.years);
            }
        }
        double sum_log_ratio = 0.0;
        std::size_t used = 0;
        for (double T : distinct_tenors) {
            const double f_old = old_surface.forwards()(T);
            const double f_new = new_surface.forwards()(T);
            if (f_old > 0.0 && f_new > 0.0) {
                sum_log_ratio += std::log(f_new / f_old);
                ++used;
            }
        }
        out.forward_shift = (used > 0) ? sum_log_ratio / static_cast<double>(used) : 0.0;
    }

    // --- event: leave-one-tenor-out residual --------------------------------
    //
    // An earlier version compared the *global* fit's residual at the
    // shortest tenor against its residual everywhere else.  That fails: a
    // single tenor with a large, genuinely isolated bump pulls the global
    // level/term fit toward itself, which spreads comparably-sized residual
    // (of the opposite sign) across every OTHER tenor too -- confirmed
    // directly in this engine's own test suite, where an ordinary skew
    // change (present, by construction, at every tenor, so not an "event"
    // in any sense) produced a LARGER naive front-tenor residual than a
    // genuinely isolated 15-vol-point bump at a single tenor, because the
    // global fit's own distortion polluted the "everywhere else" bucket
    // almost as much as the bumped one.
    //
    // Fixed with a leave-one-tenor-out comparison: for each distinct tenor,
    // refit level/skew/curv/term using every OTHER tenor's points only --
    // excluding exactly the points that could be the event -- then measure
    // how badly that fit (which never saw this tenor's data) predicts this
    // tenor's own observed move, relative to how well it predicts the
    // tenors it WAS fitted on.  A tenor that is well-explained by the rest
    // of the surface's smooth trend scores low; a genuinely isolated bump,
    // which the rest of the surface carries no information about, scores
    // high -- regardless of how much it would otherwise have distorted a
    // single global fit.  event_shift is the largest such excess across all
    // candidate tenors, floored at zero.
    {
        std::map<double, std::vector<std::size_t>> by_tenor;
        for (std::size_t i = 0; i < n; ++i) by_tenor[out.grid_years[i]].push_back(i);

        double best_excess = 0.0;
        if (by_tenor.size() >= 3) {
            for (const auto& [held_out_t, held_out_idx] : by_tenor) {
                math::NormalEquations reduced_eq(4);
                for (std::size_t i = 0; i < n; ++i) {
                    if (out.grid_years[i] == held_out_t) continue;
                    const double k = out.grid_k[i];
                    const double T = out.grid_years[i];
                    const double row[4] = {1.0, k, k * k, T};
                    reduced_eq.add(std::span<const double>(row, 4), out.grid_delta_w[i], 1.0);
                }
                math::SmallMatrix ra = reduced_eq.matrix();
                std::array<double, 4> rc{};
                for (std::size_t i = 0; i < 4; ++i) rc[i] = reduced_eq.rhs()[i];
                const auto rchol = math::cholesky_solve(ra, std::span<double>(rc.data(), 4));
                if (!rchol.ok()) continue;

                double held_sq = 0.0;
                for (std::size_t idx : held_out_idx) {
                    const double k = out.grid_k[idx];
                    const double T = out.grid_years[idx];
                    const double pred = rc[0] + rc[1] * k + rc[2] * k * k + rc[3] * T;
                    const double r = out.grid_delta_w[idx] - pred;
                    held_sq += r * r;
                }
                const double held_rms =
                    std::sqrt(held_sq / static_cast<double>(held_out_idx.size()));

                double rest_sq = 0.0;
                std::size_t rest_count = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    if (out.grid_years[i] == held_out_t) continue;
                    const double k = out.grid_k[i];
                    const double T = out.grid_years[i];
                    const double pred = rc[0] + rc[1] * k + rc[2] * k * k + rc[3] * T;
                    const double r = out.grid_delta_w[i] - pred;
                    rest_sq += r * r;
                    ++rest_count;
                }
                const double rest_rms =
                    (rest_count > 0) ? std::sqrt(rest_sq / static_cast<double>(rest_count)) : 0.0;

                best_excess = std::max(best_excess, held_rms - rest_rms);
            }
        }
        // Fewer than 3 distinct tenors: leaving one out would not leave
        // enough information to fit against, so no event signal is formed
        // rather than fabricating one from an underdetermined fit.
        out.event_shift = best_excess;
    }

    return out;
}

}  // namespace vl
