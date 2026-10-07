// SPDX-License-Identifier: MIT
#include "volatility_lab/calibration/svi_calibrator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "volatility_lab/math/linalg.hpp"

namespace vl {

const char* to_string(SviFitStatus s) noexcept {
    switch (s) {
        case SviFitStatus::Ok:                 return "ok";
        case SviFitStatus::TooFewQuotes:       return "too-few-quotes";
        case SviFitStatus::DegradedToFlat:     return "degraded-to-flat";
        case SviFitStatus::InnerSolveFailed:   return "inner-solve-failed";
        case SviFitStatus::NotAdmissible:      return "not-admissible";
        case SviFitStatus::ButterflyViolation: return "butterfly-violation";
    }
    return "?";
}

namespace {

struct SliceExtent {
    double k_min = 0.0;
    double k_max = 0.0;
    double w_min = 0.0;
    double w_max = 0.0;
    double years = 0.0;
    double atm_w = 0.0;
    std::size_t used = 0;
};

SliceExtent measure(std::span<const OptionQuote> quotes) {
    SliceExtent e;
    e.k_min = std::numeric_limits<double>::infinity();
    e.k_max = -std::numeric_limits<double>::infinity();
    e.w_min = std::numeric_limits<double>::infinity();
    e.w_max = 0.0;
    double best_abs_k = std::numeric_limits<double>::infinity();
    for (const auto& q : quotes) {
        if (!(q.weight > 0.0)) continue;
        e.k_min = std::min(e.k_min, q.log_moneyness);
        e.k_max = std::max(e.k_max, q.log_moneyness);
        e.w_min = std::min(e.w_min, q.total_variance);
        e.w_max = std::max(e.w_max, q.total_variance);
        e.years = q.years;
        const double ak = std::abs(q.log_moneyness);
        if (ak < best_abs_k) {
            best_abs_k = ak;
            e.atm_w = q.total_variance;
        }
        ++e.used;
    }
    if (e.used == 0) {
        e.k_min = e.k_max = 0.0;
        e.w_min = e.w_max = 0.0;
    }
    return e;
}

/// The residual for the inner (linear) problem is always in total variance,
/// because that is the quantity the reduced form is linear in -- which is the
/// entire basis of the reduction.  When the caller has asked for a
/// volatility-space fit, the weights are rescaled so that a total-variance
/// residual is equivalent to first order:
///
///     sigma = sqrt(w/T)  =>  d(sigma) = d(w) / (2 sqrt(w T))
///
/// so weighting the w-residual by 1/(2 sqrt(w T))^2 reproduces a vol-space
/// least squares exactly to first order, while keeping the problem linear.
/// Losing that linearity would mean losing the closed-form inner solve, which
/// is the whole point.
double inner_weight(const OptionQuote& q, ResidualKind kind) noexcept {
    switch (kind) {
        case ResidualKind::TotalVariance:
            return q.weight;
        case ResidualKind::Volatility:
        case ResidualKind::Price:
        case ResidualKind::VegaScaledPrice: {
            if (!(q.total_variance > 0.0) || !(q.years > 0.0)) return q.weight;
            const double dsigma_dw = 1.0 / (2.0 * std::sqrt(q.total_variance * q.years));
            return q.weight * dsigma_dw * dsigma_dw;
        }
    }
    return q.weight;
}

}  // namespace

// ===========================================================================
// The inner solve
// ===========================================================================

SviInnerSolve svi_inner_solve(std::span<const OptionQuote> quotes, double m, double sigma,
                              const SviCalibratorConfig& cfg) {
    SviInnerSolve out;
    if (!(sigma > 0.0)) return out;

    // ---- the change of variables that makes the constraints a box --------
    //
    // In reduced coordinates (adash, d, c) the admissibility conditions are
    //
    //     c >= 0,  |d| <= c,  |d| <= 4*sigma - c,  0 <= adash <= max(w_i)
    //
    // The two involving |d| couple d to c, so this is a polytope rather than a
    // box -- and a box solver applied to it can only approximate.  An earlier
    // version did exactly that, bounding |d| by the c from a first pass and
    // re-solving; it left the objective up to 3% above the true constrained
    // optimum when the constraints bound, and with the wing bound disabled it
    // was 12x worse on the high-vol regime, because the first pass was free to
    // choose an enormous c that the second pass then had to clamp.
    //
    // The fix is a substitution rather than a better solver.  Put
    //
    //     u = c + d,   v = c - d     (so c = (u+v)/2, d = (u-v)/2)
    //
    // Then:
    //     |d| <= c            <=>  u >= 0 and v >= 0
    //     |d| <= 4 sigma - c  <=>  max(u, v) <= 4 sigma
    //
    // -- both verified by cases on the sign of u - v.  So in (adash, u, v) the
    // entire admissible set is the box
    //
    //     0 <= adash <= max(w_i),   0 <= u <= 4 sigma,   0 <= v <= 4 sigma
    //
    // and the 27-active-set enumeration in solve_boxed_least_squares is
    // *exact*, not approximate.  This is the formulation in the Zeliade note,
    // and it is worth the two extra lines of algebra for an exact inner solve.
    //
    // The design row follows from substituting back:
    //     w = adash + u*(z + sqrt(z^2+1))/2 + v*(sqrt(z^2+1) - z)/2
    math::NormalEquations eq(3);
    double w_max = 0.0;
    for (const auto& q : quotes) {
        const double weight = inner_weight(q, cfg.residual);
        if (!(weight > 0.0)) continue;
        const double z = (q.log_moneyness - m) / sigma;
        const double root = std::sqrt(std::fma(z, z, 1.0));
        const double row[3] = {1.0, 0.5 * (z + root), 0.5 * (root - z)};
        eq.add(std::span<const double>(row, 3), q.total_variance, weight);
        w_max = std::max(w_max, q.total_variance);
    }
    if (eq.count() < 3) return out;

    // Lee's moment bound in these coordinates.
    //
    // The asymptotic wing slopes of total variance are
    //     right: b(1 + rho) = (c + d)/sigma = u/sigma
    //     left:  b(1 - rho) = (c - d)/sigma = v/sigma
    // and Lee's moment formula caps both at 2.  So the bound is
    // u, v <= 2*sigma.
    //
    // It was 4*sigma here at first, which is the form the constraint is often
    // written in -- |d| <= 4*sigma - c -- and that permits a slope of 4,
    // exactly twice Lee's limit.  The error was invisible on every realistic
    // regime, because the fitted slopes peak around 0.4 and the bound never
    // activated; it only surfaced on deliberately inadmissible data with a
    // slope of 5, where the "bounded" fit returned 4.  A constraint that is
    // wrong by a factor of two but never binds is still wrong, and it would
    // have bound on the first genuinely stressed surface.
    const double uv_upper = cfg.enforce_wing_bound
                                ? std::max(kLeeSlopeBound * sigma, 1e-12)
                                : std::numeric_limits<double>::infinity();
    std::array<double, 3> lower{0.0, 0.0, 0.0};
    const std::array<double, 3> upper{std::max(w_max, 1e-12), uv_upper, uv_upper};

    // The exact non-negative-variance condition, imposed by iteration.
    //
    // `adash >= 0` is sufficient but conservative: the true condition is
    // a + b*sigma*sqrt(1-rho^2) >= 0, and in these coordinates
    //
    //     b*sigma*sqrt(1-rho^2) = c*sqrt(1 - d^2/c^2) = sqrt(c^2 - d^2)
    //                           = sqrt((c-d)(c+d)) = sqrt(u*v)
    //
    // so the exact condition is simply  adash >= -sqrt(u*v).
    //
    // That couples adash to u and v, so it is not a box constraint and cannot
    // go straight into the enumeration.  But sqrt(u*v) varies slowly, so
    // solving the box with a lower bound of -sqrt(u*v) from the previous
    // iterate converges in two or three passes -- and each pass is itself
    // exact.  Imposing only `adash >= 0` instead cost up to 50% in RMS
    // volatility error on the earnings regime, where the bump drives the
    // fitted level negative.
    auto result = math::solve_boxed_least_squares(eq, lower, upper);
    if (!result.feasible) return out;
    for (int pass = 0; pass < 3; ++pass) {
        const double wing = std::sqrt(std::max(result.solution[1] * result.solution[2], 0.0));
        const double exact_lower = -wing;
        if (exact_lower >= lower[0] - 1e-18) break;  // already no tighter
        lower[0] = exact_lower;
        const auto refined = math::solve_boxed_least_squares(eq, lower, upper);
        if (!refined.feasible) break;
        if (refined.objective >= result.objective * (1.0 - 1e-15)) {
            result = refined;
            break;
        }
        result = refined;
    }

    const double u = result.solution[1];
    const double v = result.solution[2];
    out.reduced.adash = result.solution[0];
    out.reduced.c = 0.5 * (u + v);
    out.reduced.d = 0.5 * (u - v);
    out.objective = result.objective;
    out.active_constraints = result.active_constraints;
    out.feasible = true;
    return out;
}

// ===========================================================================
// The outer search
// ===========================================================================

namespace {

/// Weighted objective of a full parameter set, in the inner (total-variance)
/// metric, so that it is directly comparable with the inner solve's value.
double slice_objective(std::span<const OptionQuote> quotes, const SviParams& p,
                       ResidualKind kind) noexcept {
    double s = 0.0;
    for (const auto& q : quotes) {
        const double weight = inner_weight(q, kind);
        if (!(weight > 0.0)) continue;
        const double r = svi_total_variance(p, q.log_moneyness) - q.total_variance;
        s += weight * r * r;
    }
    return s;
}

/// Volatility-space error statistics, which are what a reader wants reported
/// regardless of which metric was minimised.
void vol_errors(std::span<const OptionQuote> quotes, const SviParams& p, double& rms,
                double& max_abs, double& rms_w) noexcept {
    double sum_sq = 0.0;
    double sum_sq_w = 0.0;
    double weight_sum = 0.0;
    max_abs = 0.0;
    for (const auto& q : quotes) {
        if (!(q.weight > 0.0) || !(q.years > 0.0)) continue;
        const double model_w = svi_total_variance(p, q.log_moneyness);
        const double model_vol = (model_w > 0.0) ? std::sqrt(model_w / q.years) : 0.0;
        const double dv = model_vol - q.implied_vol;
        const double dw = model_w - q.total_variance;
        sum_sq += q.weight * dv * dv;
        sum_sq_w += q.weight * dw * dw;
        weight_sum += q.weight;
        max_abs = std::max(max_abs, std::abs(dv));
    }
    rms = (weight_sum > 0.0) ? std::sqrt(sum_sq / weight_sum) : 0.0;
    rms_w = (weight_sum > 0.0) ? std::sqrt(sum_sq_w / weight_sum) : 0.0;
}

}  // namespace

SviFitResult calibrate_svi_slice(std::span<const OptionQuote> quotes,
                                 const SviCalibratorConfig& cfg) {
    SviFitResult out;
    out.quotes_available = quotes.size();
    const SliceExtent ext = measure(quotes);
    out.quotes_used = ext.used;
    out.params.years = ext.years;

    if (ext.used == 0) {
        out.status = SviFitStatus::TooFewQuotes;
        out.diagnostics.add(make_diag(DiagCode::SliceTooFewQuotes, Severity::Error, "slice",
                                      "count", 0.0, "no weighted quotes in the slice"));
        return out;
    }

    if (ext.used < cfg.min_quotes) {
        // Degrade to a level-only fit rather than fitting five parameters to
        // four points.  A flat slice is a poor model but an honest one; an
        // over-parameterised fit looks excellent and extrapolates to nonsense.
        double sum_w = 0.0;
        double sum = 0.0;
        for (const auto& q : quotes) {
            if (!(q.weight > 0.0)) continue;
            sum += q.weight * q.total_variance;
            sum_w += q.weight;
        }
        out.params.a = (sum_w > 0.0) ? sum / sum_w : ext.atm_w;
        out.params.b = 0.0;
        out.params.rho = 0.0;
        out.params.m = 0.0;
        out.params.sigma = 1.0;
        out.params = svi_project_to_admissible(out.params);
        out.objective = slice_objective(quotes, out.params, cfg.residual);
        vol_errors(quotes, out.params, out.rms_vol_error, out.max_vol_error,
                   out.rms_total_variance_error);
        out.status = SviFitStatus::DegradedToFlat;
        out.diagnostics.add(make_range_diag(
            DiagCode::SliceTooFewQuotes, Severity::Warning, "slice", "count",
            static_cast<double>(ext.used), static_cast<double>(cfg.min_quotes),
            std::numeric_limits<double>::infinity(),
            "too few quotes for a five-parameter fit; fitted a level only"));
        return out;
    }

    // --- the outer grid ---------------------------------------------------
    //
    // This is what the reduction buys: a *global* search over the only two
    // parameters the problem is nonlinear in.  Each grid point costs one
    // closed-form 3x3 solve, which is cheaper than a single LM iteration on
    // the full five parameters -- so the whole grid costs less than a direct
    // fit and has no dependence on a starting guess.
    const double k_span = std::max(ext.k_max - ext.k_min, 1e-6);
    const double k_mid = 0.5 * (ext.k_min + ext.k_max);
    const double m_lo = k_mid - cfg.m_range_scale * 0.5 * k_span;
    const double m_hi = k_mid + cfg.m_range_scale * 0.5 * k_span;
    const double sigma_lo = cfg.sigma_min;
    const double sigma_hi = std::max(cfg.sigma_max_scale * k_span, 2.0 * cfg.sigma_min);

    double best_obj = std::numeric_limits<double>::infinity();
    double worst_obj = 0.0;
    double best_m = k_mid;
    double best_sigma = std::max(0.1 * k_span, cfg.sigma_min);
    SviInnerSolve best_inner;

    const int nm = std::max(cfg.grid_m, 2);
    const int ns = std::max(cfg.grid_sigma, 2);
    for (int i = 0; i < nm; ++i) {
        const double m = m_lo + (m_hi - m_lo) * static_cast<double>(i) /
                                    static_cast<double>(nm - 1);
        for (int j = 0; j < ns; ++j) {
            // Geometric in sigma: the objective varies on a multiplicative
            // scale in a curvature parameter, so a linear grid wastes most of
            // its points at the wide end where the fit barely changes.
            const double u = static_cast<double>(j) / static_cast<double>(ns - 1);
            const double sigma = sigma_lo * std::pow(sigma_hi / sigma_lo, u);

            const SviInnerSolve inner = svi_inner_solve(quotes, m, sigma, cfg);
            ++out.inner_solves;
            if (!inner.feasible) continue;
            worst_obj = std::max(worst_obj, inner.objective);
            if (inner.objective < best_obj) {
                best_obj = inner.objective;
                best_m = m;
                best_sigma = sigma;
                best_inner = inner;
            }
        }
    }

    if (!best_inner.feasible) {
        out.status = SviFitStatus::InnerSolveFailed;
        out.diagnostics.add(make_diag(DiagCode::CalibrationDidNotConverge, Severity::Error,
                                      "slice", "inner", 0.0,
                                      "every active set of the inner problem was degenerate"));
        return out;
    }

    out.grid_objective = best_obj;
    out.grid_objective_range =
        (std::isfinite(worst_obj) && best_obj > 0.0) ? (worst_obj - best_obj) : 0.0;

    // --- refinement on (m, sigma) ----------------------------------------
    //
    // The grid locates the basin; two-parameter Levenberg-Marquardt finds the
    // bottom of it.  The inner solve is nested inside the residual function,
    // so the outer problem really is two-dimensional.
    if (cfg.refine) {
        // The grid locates the basin; two-parameter Levenberg-Marquardt finds
        // the bottom of it, with the inner solve nested inside the residual
        // function so the outer problem really is two-dimensional.
        //
        // The residual vector is the *per-quote* residual at the profiled
        // optimum, not the scalar objective.  That distinction is not
        // cosmetic: with a single residual the Jacobian is 1x2, so J^T J has
        // rank one and is singular at every damping level.  The first version
        // did exactly that, and the symptom was not a failure but something
        // worse -- the optimiser ran its full 100-iteration budget on every
        // slice, burning ~400 inner solves to improve the objective by 10%,
        // and never reported a convergence status at all.  With m residuals
        // the system is full rank and the refinement converges in a handful of
        // iterations.
        //
        // Differentiating through the inner solve is unnecessary.  By the
        // envelope theorem the explicit dependence on (adash, d, c) vanishes
        // at the inner optimum, so a finite difference of the profiled
        // residual is the correct derivative of the profiled objective.
        std::vector<const OptionQuote*> used;
        used.reserve(ext.used);
        for (const auto& q : quotes) {
            if (q.weight > 0.0) used.push_back(&q);
        }

        LeastSquaresProblem prob;
        prob.num_params = 2;
        prob.num_residuals = used.size();
        prob.lower = {m_lo - k_span, sigma_lo};
        prob.upper = {m_hi + k_span, sigma_hi * 4.0};
        prob.typical_scale = {std::max(k_span, 1e-3), std::max(best_sigma, 1e-3)};
        prob.weights.reserve(used.size());
        for (const auto* q : used) prob.weights.push_back(inner_weight(*q, cfg.residual));

        int inner_calls = 0;
        prob.residuals = [&](std::span<const double> p, std::span<double> r) -> std::size_t {
            const double m = p[0];
            const double sg = std::max(p[1], cfg.sigma_min);
            const SviInnerSolve inner = svi_inner_solve(quotes, m, sg, cfg);
            ++inner_calls;
            if (!inner.feasible) {
                // A large but finite residual, not infinity: the optimiser has
                // to be able to step away from an infeasible point, and an
                // infinity gives it no gradient to descend.
                const double big = std::sqrt(std::max(best_obj, 1e-12)) * 1e3 + 1.0;
                for (std::size_t i = 0; i < used.size(); ++i) r[i] = big;
                return used.size();
            }
            const SviParams sp = svi_from_reduced(inner.reduced, m, sg, ext.years);
            for (std::size_t i = 0; i < used.size(); ++i) {
                r[i] = svi_total_variance(sp, used[i]->log_moneyness) -
                       used[i]->total_variance;
            }
            return used.size();
        };

        const double start[2] = {best_m, best_sigma};
        const OptimizerResult ref = calibrate(prob, start, cfg.refine_settings);
        out.outer_iterations = ref.iterations;
        out.residual_evaluations = ref.residual_evaluations;
        out.inner_solves += inner_calls;

        const double m_ref = ref.parameters[0];
        const double sg_ref = std::max(ref.parameters[1], cfg.sigma_min);
        const SviInnerSolve inner_ref = svi_inner_solve(quotes, m_ref, sg_ref, cfg);
        ++out.inner_solves;
        if (inner_ref.feasible && inner_ref.objective <= best_obj) {
            best_obj = inner_ref.objective;
            best_m = m_ref;
            best_sigma = sg_ref;
            best_inner = inner_ref;
        }
    }

    // --- assemble ---------------------------------------------------------
    out.params = svi_from_reduced(best_inner.reduced, best_m, best_sigma, ext.years);
    const SviParams before_projection = out.params;
    out.params = svi_project_to_admissible(out.params);
    out.objective = slice_objective(quotes, out.params, cfg.residual);
    vol_errors(quotes, out.params, out.rms_vol_error, out.max_vol_error,
               out.rms_total_variance_error);

    if (!svi_parameters_admissible(out.params)) {
        out.status = SviFitStatus::NotAdmissible;
        out.diagnostics.add(make_diag(DiagCode::ParameterAtBound, Severity::Error, "slice",
                                      "params", out.params.b,
                                      "fitted parameters are not admissible even after "
                                      "projection"));
        return out;
    }
    if (std::abs(before_projection.a - out.params.a) > 1e-12 * (1.0 + std::abs(out.params.a))) {
        out.diagnostics.add(make_diag(
            DiagCode::ParameterAtBound, Severity::Warning, "slice", "a",
            before_projection.a,
            "level parameter was raised to keep total variance non-negative"));
    }

    out.status = SviFitStatus::Ok;

    // --- butterfly ---------------------------------------------------------
    if (cfg.check_butterfly) {
        // Checked over a range wider than the quotes, because the fitted slice
        // will be *used* outside them -- that is the point of having a
        // parametric model -- and a violation just outside the quoted range is
        // exactly the one a grid restricted to the data would miss.
        const double pad = 0.5 * k_span + 0.25;
        out.butterfly = svi_butterfly_check(out.params, ext.k_min - pad, ext.k_max + pad);
        if (!out.butterfly.arbitrage_free) {
            out.status = SviFitStatus::ButterflyViolation;
            out.diagnostics.add(make_range_diag(
                DiagCode::ButterflyArbitrage, Severity::Warning, "slice", "durrleman_g",
                out.butterfly.worst_g, 0.0, std::numeric_limits<double>::infinity(),
                "fitted slice has negative implied density"));
        }
    }
    return out;
}

// ===========================================================================
// The direct fit, for comparison
// ===========================================================================

std::vector<std::vector<double>> svi_default_starts(std::span<const OptionQuote> quotes) {
    const SliceExtent ext = measure(quotes);
    const double k_span = std::max(ext.k_max - ext.k_min, 1e-3);
    const double k_mid = 0.5 * (ext.k_min + ext.k_max);
    const double w = std::max(ext.atm_w, 1e-6);

    // A deterministic spread over the plausible parameter space: three skews
    // times two curvature scales times two levels.  Fixed rather than random
    // so the comparison against the quasi-explicit path is reproducible --
    // a randomised multi-start would make the local-minimum rate a function of
    // the seed, which is precisely the number being measured.
    std::vector<std::vector<double>> starts;
    for (double rho : {-0.7, -0.3, 0.0, 0.3}) {
        for (double sigma_scale : {0.2, 0.6}) {
            for (double level : {0.3, 0.8}) {
                const double sigma = std::max(sigma_scale * k_span, 1e-3);
                const double b = 0.5 * w / std::max(sigma, 1e-6);
                starts.push_back({level * w, b, rho, k_mid, sigma});
            }
        }
    }
    return starts;
}

SviFitResult calibrate_svi_slice_direct(std::span<const OptionQuote> quotes,
                                        std::span<const std::vector<double>> starts,
                                        const SviCalibratorConfig& cfg) {
    SviFitResult out;
    out.quotes_available = quotes.size();
    const SliceExtent ext = measure(quotes);
    out.quotes_used = ext.used;
    out.params.years = ext.years;

    if (ext.used < cfg.min_quotes) {
        out.status = SviFitStatus::TooFewQuotes;
        return out;
    }

    // Collect the weighted quotes once; the residual function is called
    // hundreds of times and must not re-filter.
    std::vector<const OptionQuote*> used;
    used.reserve(ext.used);
    for (const auto& q : quotes) {
        if (q.weight > 0.0) used.push_back(&q);
    }

    LeastSquaresProblem prob;
    prob.num_params = 5;
    prob.num_residuals = used.size();
    prob.weights.reserve(used.size());
    for (const auto* q : used) prob.weights.push_back(inner_weight(*q, cfg.residual));

    const double k_span = std::max(ext.k_max - ext.k_min, 1e-3);
    prob.lower = {-std::numeric_limits<double>::infinity(), 0.0, -1.0 + 1e-9,
                  ext.k_min - 2.0 * k_span, 1e-4};
    prob.upper = {std::max(ext.w_max, 1e-9),
                  std::numeric_limits<double>::infinity(), 1.0 - 1e-9,
                  ext.k_max + 2.0 * k_span, 8.0 * k_span};
    prob.typical_scale = {std::max(ext.atm_w, 1e-4), std::max(ext.atm_w / k_span, 1e-3),
                          0.5, std::max(k_span, 1e-3), std::max(k_span, 1e-3)};

    prob.residuals = [&](std::span<const double> p, std::span<double> r) -> std::size_t {
        SviParams sp;
        sp.a = p[0];
        sp.b = p[1];
        sp.rho = p[2];
        sp.m = p[3];
        sp.sigma = std::max(p[4], 1e-8);
        sp.years = ext.years;
        for (std::size_t i = 0; i < used.size(); ++i) {
            r[i] = svi_total_variance(sp, used[i]->log_moneyness) -
                   used[i]->total_variance;
        }
        return used.size();
    };

    // Analytic Jacobian.  The derivatives of w with respect to the five raw
    // parameters are all closed form, and supplying them removes five residual
    // evaluations per iteration -- which matters here precisely because this
    // is the path being benchmarked as the expensive alternative, and
    // handicapping it with finite differences would make the comparison
    // dishonest.
    prob.jacobian = [&](std::span<const double> p, std::span<double> jac) {
        const double b = p[1];
        const double rho = p[2];
        const double m = p[3];
        const double sigma = std::max(p[4], 1e-8);
        for (std::size_t i = 0; i < used.size(); ++i) {
            const double y = used[i]->log_moneyness - m;
            const double r = std::sqrt(y * y + sigma * sigma);
            const double inv_r = (r > 0.0) ? 1.0 / r : 0.0;
            double* row = &jac[i * 5];
            row[0] = 1.0;                                  // dw/da
            row[1] = rho * y + r;                          // dw/db
            row[2] = b * y;                                // dw/drho
            row[3] = -b * (rho + y * inv_r);               // dw/dm
            row[4] = b * sigma * inv_r;                    // dw/dsigma
        }
    };

    const MultiStartResult ms = calibrate_multi_start(prob, starts, cfg.refine_settings);
    out.outer_iterations = ms.best.iterations;
    out.residual_evaluations = ms.best.residual_evaluations;

    SviParams sp;
    sp.a = ms.best.parameters[0];
    sp.b = ms.best.parameters[1];
    sp.rho = ms.best.parameters[2];
    sp.m = ms.best.parameters[3];
    sp.sigma = std::max(ms.best.parameters[4], 1e-8);
    sp.years = ext.years;
    out.params = svi_project_to_admissible(sp);
    out.objective = slice_objective(quotes, out.params, cfg.residual);
    vol_errors(quotes, out.params, out.rms_vol_error, out.max_vol_error,
               out.rms_total_variance_error);
    out.status = svi_parameters_admissible(out.params) ? SviFitStatus::Ok
                                                       : SviFitStatus::NotAdmissible;

    if (cfg.check_butterfly && out.status == SviFitStatus::Ok) {
        const double pad = 0.5 * k_span + 0.25;
        out.butterfly = svi_butterfly_check(out.params, ext.k_min - pad, ext.k_max + pad);
        if (!out.butterfly.arbitrage_free) out.status = SviFitStatus::ButterflyViolation;
    }
    return out;
}

}  // namespace vl
