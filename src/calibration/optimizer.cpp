// SPDX-License-Identifier: MIT
#include "volatility_lab/calibration/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vl {

const char* to_string(OptimizerStatus s) noexcept {
    switch (s) {
        case OptimizerStatus::Converged:          return "converged";
        case OptimizerStatus::ObjectiveTolerance: return "objective-tolerance";
        case OptimizerStatus::ParameterTolerance: return "parameter-tolerance";
        case OptimizerStatus::GradientTolerance:  return "gradient-tolerance";
        case OptimizerStatus::MaxIterations:      return "max-iterations";
        case OptimizerStatus::MaxEvaluations:     return "max-evaluations";
        case OptimizerStatus::LambdaOverflow:     return "lambda-overflow";
        case OptimizerStatus::SingularSystem:     return "singular-system";
        case OptimizerStatus::InvalidProblem:     return "invalid-problem";
        case OptimizerStatus::ResidualNonFinite:  return "residual-non-finite";
    }
    return "?";
}

bool is_success(OptimizerStatus s) noexcept {
    switch (s) {
        case OptimizerStatus::Converged:
        case OptimizerStatus::ObjectiveTolerance:
        case OptimizerStatus::ParameterTolerance:
        case OptimizerStatus::GradientTolerance:
            return true;
        default:
            // MaxIterations is *not* success.  The parameters it returns are
            // often perfectly usable, and the temptation is to call it
            // converged -- but then a caller cannot distinguish "the fit is
            // good" from "the fit ran out of budget", and the surface quality
            // report would stop being able to flag the second.
            return false;
    }
}

std::size_t OptimizerResult::num_at_bound() const noexcept {
    std::size_t n = 0;
    for (auto f : parameters_at_bound) {
        if (f) ++n;
    }
    return n;
}

namespace {

struct Workspace {
    std::vector<double> params;
    std::vector<double> trial;
    std::vector<double> residual;
    std::vector<double> trial_residual;
    std::vector<double> jacobian;  ///< row-major, num_residuals x num_params
    std::vector<double> weights;
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<double> scale;
    std::vector<double> step;
};

double effective_scale(double typical, double value) noexcept {
    const double t = (typical > 0.0 && std::isfinite(typical)) ? typical : 1.0;
    return std::abs(value) + t;
}

/// Weighted sum of squares, accumulated in a fixed order so that the objective
/// is bitwise reproducible across runs.
double weighted_sum_squares(std::span<const double> r, std::span<const double> w,
                            std::size_t n) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += w[i] * r[i] * r[i];
    return s;
}

void project_into_box(std::span<double> p, std::span<const double> lo,
                      std::span<const double> hi) noexcept {
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (p[i] < lo[i]) p[i] = lo[i];
        if (p[i] > hi[i]) p[i] = hi[i];
    }
}

/// Forward differences, with a step sized from the parameter's own scale.
///
/// Forward rather than central: the Jacobian is evaluated once per accepted
/// step and costs num_params residual evaluations, so central differences
/// would double the dominant cost of the whole fit for an accuracy improvement
/// the LM step does not need -- the step direction is far more tolerant of
/// Jacobian error than a Newton step would be.  The step is sqrt(eps) times
/// the parameter scale, which balances truncation against round-off for a
/// first-order difference.
void finite_difference_jacobian(const LeastSquaresProblem& problem, Workspace& ws,
                                std::size_t m, std::size_t n, int& evaluations) {
    constexpr double kSqrtEps = 1.4901161193847656e-08;  // sqrt(2^-52)
    for (std::size_t j = 0; j < n; ++j) {
        const double scale = effective_scale(
            j < problem.typical_scale.size() ? problem.typical_scale[j] : 1.0,
            ws.params[j]);
        double h = kSqrtEps * scale;
        // Keep the perturbed point inside the box: stepping outside would
        // evaluate an inadmissible model, whose residuals carry no information
        // about the derivative at an admissible point.
        if (ws.params[j] + h > ws.upper[j]) h = -h;
        if (h == 0.0) h = kSqrtEps;

        ws.trial = ws.params;
        ws.trial[j] += h;
        project_into_box(ws.trial, ws.lower, ws.upper);
        const double actual_h = ws.trial[j] - ws.params[j];

        problem.residuals(ws.trial, ws.trial_residual);
        ++evaluations;

        if (actual_h == 0.0) {
            for (std::size_t i = 0; i < m; ++i) ws.jacobian[i * n + j] = 0.0;
            continue;
        }
        for (std::size_t i = 0; i < m; ++i) {
            ws.jacobian[i * n + j] = (ws.trial_residual[i] - ws.residual[i]) / actual_h;
        }
    }
}

}  // namespace

// ===========================================================================
// calibrate
// ===========================================================================

OptimizerResult calibrate(const LeastSquaresProblem& problem,
                          std::span<const double> initial,
                          const OptimizerSettings& settings) {
    OptimizerResult out;
    const std::size_t n = problem.num_params;
    const std::size_t m = problem.num_residuals;

    if (n == 0 || n > math::kMaxSmallDim || m == 0 || !problem.residuals ||
        initial.size() < n) {
        out.status = OptimizerStatus::InvalidProblem;
        return out;
    }

    Workspace ws;
    ws.params.assign(initial.begin(), initial.begin() + static_cast<std::ptrdiff_t>(n));
    ws.trial.resize(n);
    ws.residual.assign(m, 0.0);
    ws.trial_residual.assign(m, 0.0);
    ws.jacobian.assign(m * n, 0.0);
    ws.step.assign(n, 0.0);

    ws.weights.assign(m, 1.0);
    for (std::size_t i = 0; i < std::min(m, problem.weights.size()); ++i) {
        ws.weights[i] = problem.weights[i];
    }
    ws.lower.assign(n, -std::numeric_limits<double>::infinity());
    ws.upper.assign(n, std::numeric_limits<double>::infinity());
    for (std::size_t j = 0; j < std::min(n, problem.lower.size()); ++j) {
        ws.lower[j] = problem.lower[j];
    }
    for (std::size_t j = 0; j < std::min(n, problem.upper.size()); ++j) {
        ws.upper[j] = problem.upper[j];
    }
    ws.scale.assign(n, 1.0);
    for (std::size_t j = 0; j < n; ++j) {
        ws.scale[j] = (j < problem.typical_scale.size() && problem.typical_scale[j] > 0.0)
                          ? problem.typical_scale[j]
                          : 1.0;
    }

    project_into_box(ws.params, ws.lower, ws.upper);

    problem.residuals(ws.params, ws.residual);
    out.residual_evaluations = 1;
    double objective = weighted_sum_squares(ws.residual, ws.weights, m);
    out.initial_objective = objective;
    if (!std::isfinite(objective)) {
        out.parameters = ws.params;
        out.objective = objective;
        out.status = OptimizerStatus::ResidualNonFinite;
        return out;
    }

    double lambda = settings.initial_lambda;
    OptimizerStatus status = OptimizerStatus::MaxIterations;

    for (int iter = 1; iter <= settings.max_iterations; ++iter) {
        out.iterations = iter;

        if (problem.jacobian) {
            problem.jacobian(ws.params, ws.jacobian);
            ++out.jacobian_evaluations;
        } else {
            finite_difference_jacobian(problem, ws, m, n, out.residual_evaluations);
            ++out.jacobian_evaluations;
        }

        // Normal equations, accumulated in residual order for reproducibility.
        math::NormalEquations eq(n);
        for (std::size_t i = 0; i < m; ++i) {
            eq.add(std::span<const double>(&ws.jacobian[i * n], n), ws.residual[i],
                   ws.weights[i]);
        }

        // Scaled gradient test.  The scaling makes it dimensionless: with
        // parameters spanning four orders of magnitude, an unscaled gradient
        // norm is dominated by whichever parameter happens to have the largest
        // units and says nothing about the others.
        double grad_norm = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            grad_norm = std::max(grad_norm,
                                 std::abs(eq.rhs()[j]) *
                                     effective_scale(ws.scale[j], ws.params[j]));
        }
        out.gradient_norm = grad_norm;
        if (grad_norm <= settings.gradient_tol) {
            status = OptimizerStatus::GradientTolerance;
            break;
        }

        const math::SmallVector diag = eq.diagonal();
        bool step_accepted = false;

        // Inner loop: increase damping until a step reduces the objective.
        while (lambda <= settings.max_lambda) {
            math::SmallMatrix a = eq.matrix();
            // Marquardt's scaled damping: lambda multiplies the *diagonal*,
            // not the identity.  With SVI parameters differing by four orders
            // of magnitude in natural units, an identity-damped step damps the
            // small-scale directions into immobility while barely touching the
            // others.  The floor keeps a structurally zero diagonal entry (a
            // parameter the residuals do not depend on at this point) from
            // making the system singular.
            for (std::size_t j = 0; j < n; ++j) {
                a(j, j) += lambda * std::max(diag[j], 1e-12);
            }

            math::SmallVector delta{};
            for (std::size_t j = 0; j < n; ++j) delta[j] = -eq.rhs()[j];
            const auto chol = math::cholesky_solve(a, std::span<double>(delta.data(), n));
            if (!chol.ok()) {
                lambda *= settings.lambda_increase;
                ++out.rejected_steps;
                continue;
            }
            out.condition_estimate = chol.condition_estimate;

            // Trust region on the *relative* step.  A parameter legitimately
            // passing through zero (SVI's `a` does) has no meaningful relative
            // step, hence the additive typical scale.
            double shrink = 1.0;
            for (std::size_t j = 0; j < n; ++j) {
                const double limit =
                    settings.max_relative_step * effective_scale(ws.scale[j], ws.params[j]);
                if (std::abs(delta[j]) > limit && std::abs(delta[j]) > 0.0) {
                    shrink = std::min(shrink, limit / std::abs(delta[j]));
                }
            }
            for (std::size_t j = 0; j < n; ++j) ws.step[j] = delta[j] * shrink;

            for (std::size_t j = 0; j < n; ++j) ws.trial[j] = ws.params[j] + ws.step[j];
            project_into_box(ws.trial, ws.lower, ws.upper);

            problem.residuals(ws.trial, ws.trial_residual);
            ++out.residual_evaluations;
            if (out.residual_evaluations > settings.max_evaluations) {
                status = OptimizerStatus::MaxEvaluations;
                break;
            }

            const double trial_objective =
                weighted_sum_squares(ws.trial_residual, ws.weights, m);

            if (std::isfinite(trial_objective) && trial_objective < objective) {
                // Accept.
                const double improvement = objective - trial_objective;
                const double rel_improvement =
                    improvement / std::max(objective, 1e-300);

                // The *actual* step taken, after projection -- not the
                // requested one.  A step that was mostly clipped by a bound
                // has barely moved, and reporting the requested size would
                // hide a stall.
                double max_rel_step = 0.0;
                for (std::size_t j = 0; j < n; ++j) {
                    const double moved = std::abs(ws.trial[j] - ws.params[j]);
                    max_rel_step = std::max(
                        max_rel_step,
                        moved / effective_scale(ws.scale[j], ws.params[j]));
                }

                ws.params = ws.trial;
                ws.residual = ws.trial_residual;
                objective = trial_objective;
                lambda = std::max(lambda * settings.lambda_decrease, 1e-15);
                step_accepted = true;

                if (rel_improvement <= settings.objective_rtol) {
                    status = OptimizerStatus::ObjectiveTolerance;
                } else if (max_rel_step <= settings.param_rtol &&
                           max_rel_step * 1.0 <= settings.param_rtol) {
                    status = OptimizerStatus::ParameterTolerance;
                }
                break;
            }

            lambda *= settings.lambda_increase;
            ++out.rejected_steps;
        }

        if (status == OptimizerStatus::MaxEvaluations) break;
        if (!step_accepted) {
            status = (lambda > settings.max_lambda) ? OptimizerStatus::LambdaOverflow
                                                    : OptimizerStatus::SingularSystem;
            break;
        }
        if (status == OptimizerStatus::ObjectiveTolerance ||
            status == OptimizerStatus::ParameterTolerance) {
            break;
        }
    }

    out.parameters = ws.params;
    out.objective = objective;
    out.lambda_final = lambda;
    out.status = status;

    double total_weight = 0.0;
    double max_abs = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        total_weight += ws.weights[i];
        max_abs = std::max(max_abs, std::abs(ws.residual[i]));
    }
    out.max_abs_residual = max_abs;
    out.rms_residual = (total_weight > 0.0) ? std::sqrt(objective / total_weight) : 0.0;

    out.parameters_at_bound.assign(n, 0);
    for (std::size_t j = 0; j < n; ++j) {
        const double tol = 1e-10 * effective_scale(ws.scale[j], ws.params[j]);
        if (std::isfinite(ws.lower[j]) && ws.params[j] <= ws.lower[j] + tol) {
            out.parameters_at_bound[j] = 1;
        }
        if (std::isfinite(ws.upper[j]) && ws.params[j] >= ws.upper[j] - tol) {
            out.parameters_at_bound[j] = 1;
        }
    }
    return out;
}

// ===========================================================================
// Multi-start
// ===========================================================================

MultiStartResult calibrate_multi_start(const LeastSquaresProblem& problem,
                                       std::span<const std::vector<double>> starts,
                                       const OptimizerSettings& settings) {
    MultiStartResult out;
    out.starts_tried = starts.size();
    out.objectives.reserve(starts.size());
    double best_objective = std::numeric_limits<double>::infinity();

    for (std::size_t s = 0; s < starts.size(); ++s) {
        const OptimizerResult r = calibrate(problem, starts[s], settings);
        out.objectives.push_back(r.objective);
        if (r.ok()) ++out.starts_succeeded;

        // Strictly less, so ties go to the earlier start.  That is what makes
        // the result a function of the inputs alone, which the determinism
        // contract requires.
        const bool better = r.objective < best_objective;
        if (better || out.best.parameters.empty()) {
            best_objective = r.objective;
            out.best = r;
            out.best_start = s;
        }
    }
    return out;
}

// ===========================================================================
// Jacobian check
// ===========================================================================

JacobianCheck check_jacobian(const LeastSquaresProblem& problem,
                             std::span<const double> params, double rtol) {
    JacobianCheck out;
    const std::size_t n = problem.num_params;
    const std::size_t m = problem.num_residuals;
    if (!problem.jacobian || !problem.residuals || n == 0 || m == 0) return out;

    std::vector<double> analytic(m * n, 0.0);
    problem.jacobian(params, analytic);

    std::vector<double> base(m, 0.0);
    std::vector<double> perturbed(m, 0.0);
    std::vector<double> p(params.begin(), params.end());
    problem.residuals(p, base);

    constexpr double kCbrtEps = 6.0554544523933395e-06;  // eps^(1/3)
    for (std::size_t j = 0; j < n; ++j) {
        const double scale = effective_scale(
            j < problem.typical_scale.size() ? problem.typical_scale[j] : 1.0, p[j]);
        // Central differences with an eps^(1/3) step: this is a *validation*
        // routine, so it is worth two evaluations per parameter to get the
        // comparison accurate to ~1e-10 rather than the ~1e-8 a forward
        // difference would give.  Otherwise the check could not distinguish a
        // real Jacobian error from its own truncation.
        const double h = kCbrtEps * scale;
        p[j] = params[j] + h;
        problem.residuals(p, perturbed);
        std::vector<double> up = perturbed;
        p[j] = params[j] - h;
        problem.residuals(p, perturbed);
        p[j] = params[j];

        for (std::size_t i = 0; i < m; ++i) {
            const double fd = (up[i] - perturbed[i]) / (2.0 * h);
            const double an = analytic[i * n + j];
            const double abs_err = std::abs(an - fd);
            const double denom = std::max({std::abs(an), std::abs(fd), 1e-10});
            const double rel_err = abs_err / denom;
            if (rel_err > out.max_rel_error) {
                out.max_rel_error = rel_err;
                out.max_abs_error = abs_err;
                out.worst_residual = i;
                out.worst_param = j;
            }
        }
    }
    out.agrees = out.max_rel_error <= rtol;
    return out;
}

}  // namespace vl
