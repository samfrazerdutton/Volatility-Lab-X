// SPDX-License-Identifier: MIT
#pragma once
/// \file optimizer.hpp
/// \brief Box-constrained Levenberg-Marquardt, with multi-start.
///
/// ## What this is for
///
/// The reusable nonlinear least-squares engine behind every fit in the
/// library:
///
///     parameters -> model -> residual vector -> weighted sum of squares
///
/// The caller supplies a residual function and, optionally, its Jacobian; the
/// optimiser does the rest and reports everything it did.
///
/// ## Levenberg-Marquardt, and the two details that matter
///
/// The step solves
///
///     (J^T W J + lambda * D) delta = -J^T W r
///
/// and `lambda` interpolates between Gauss-Newton (lambda -> 0, fast but
/// unreliable far from the optimum) and gradient descent (lambda -> inf, slow
/// but safe).  Two choices in that formula are not incidental:
///
/// **1. D is the diagonal of J^T W J, not the identity.**  Marquardt's
/// original scaling.  SVI parameters differ by four orders of magnitude in
/// natural units -- `a` is around 0.01 while `m` is around 0.1 and `b` around
/// 0.1 -- so damping them all by the same absolute `lambda` damps the
/// small-scale directions into immobility while barely touching the others.
/// Scaling by the diagonal makes the damping scale-invariant, which is the
/// difference between converging in 15 iterations and stalling.
///
/// **2. The trust region is on the *relative* step.**  A parameter near zero
/// has no meaningful relative step, so the bound is `|delta_i| <= max_step *
/// (|p_i| + typical_i)` with a per-parameter `typical` scale supplied by the
/// caller.  Without that, `a` (which legitimately passes through zero) either
/// gets a trust region of zero or none at all.
///
/// ## Bounds by projection
///
/// Box constraints are handled by projecting the trial point back into the box
/// and recomputing, not by a penalty.  Penalties are the usual choice and they
/// are wrong here: the bounds encode *admissibility* (sigma > 0, |rho| < 1,
/// variance non-negative), so a point outside them is not a bad fit but an
/// invalid model whose residuals are meaningless, and a penalty computes those
/// meaningless residuals in order to decide how bad they are.
///
/// Projection has a known cost: it can stall on a bound when the unconstrained
/// descent direction points outward.  That is detected and reported through
/// `parameters_at_bound`, so a caller sees "converged, but three parameters are
/// pinned" rather than just "converged".
///
/// ## Multi-start
///
/// `calibrate_multi_start` runs the same problem from several starting points
/// and keeps the best. It exists because a five-parameter SVI fit has genuine
/// local minima -- and the single most effective answer to that is not a
/// better optimiser but a *smaller problem*, which is what the quasi-explicit
/// reduction in calibration/svi_calibrator.hpp provides.  Multi-start is the
/// fallback for the fits that cannot be reduced (SSVI's global parameters),
/// and the comparison between the two is in benchmarks/calibration.

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "volatility_lab/math/linalg.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Problem definition
// ---------------------------------------------------------------------------

/// What the optimiser needs to know about the model.
///
/// `residuals` must fill `out` with r_i(p) and return the count actually
/// written; returning fewer than `out.size()` is how a model signals that some
/// quotes became unusable at this parameter point.
///
/// `jacobian` is optional.  When absent the optimiser uses forward differences
/// with a per-parameter step derived from `typical_scale` -- adequate, and the
/// honest default, since an analytic Jacobian that disagrees with the residual
/// function is a far more expensive bug than a slightly slower fit.
/// `tests/calibration` checks the analytic Jacobians against differences for
/// every model that supplies one.
struct LeastSquaresProblem {
    std::size_t num_params = 0;
    std::size_t num_residuals = 0;

    std::function<std::size_t(std::span<const double> params, std::span<double> out)>
        residuals;

    /// Row-major (num_residuals x num_params) Jacobian, dr_i/dp_j.  Empty
    /// function means "use finite differences".
    std::function<void(std::span<const double> params, std::span<double> jac)> jacobian;

    std::vector<double> weights;  ///< per-residual; empty means all ones
    std::vector<double> lower;    ///< per-parameter; empty means unbounded
    std::vector<double> upper;

    /// Natural magnitude of each parameter.  Used for the finite-difference
    /// step and the relative trust region; a zero entry falls back to 1.
    std::vector<double> typical_scale;
};

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

struct OptimizerSettings {
    int max_iterations = 100;

    /// Stop when the weighted sum of squares improves by less than this,
    /// relatively, on a successful step.
    double objective_rtol = 1e-12;

    /// Stop when every parameter step is below `param_atol + param_rtol*|p|`.
    double param_atol = 1e-14;
    double param_rtol = 1e-10;

    /// Stop when the scaled gradient infinity-norm falls below this.  Scaled by
    /// the parameter magnitude so the test is dimensionless -- an unscaled
    /// gradient tolerance is meaningless when parameters differ by four orders
    /// of magnitude.
    double gradient_tol = 1e-12;

    /// Initial LM damping, relative to the diagonal of J^T W J.
    double initial_lambda = 1e-3;
    double lambda_increase = 10.0;
    double lambda_decrease = 0.1;
    double max_lambda = 1e12;

    /// Trust region: the largest step allowed for parameter i is
    /// `max_relative_step * (|p_i| + typical_i)`.
    double max_relative_step = 0.5;

    /// Hard cap on residual evaluations, so a pathological model cannot run
    /// forever even within the iteration budget.
    int max_evaluations = 2000;
};

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

enum class OptimizerStatus : std::uint8_t {
    Converged = 0,
    ObjectiveTolerance,
    ParameterTolerance,
    GradientTolerance,
    MaxIterations,
    MaxEvaluations,
    LambdaOverflow,   ///< damping grew past the cap without finding a descent step
    SingularSystem,   ///< the normal matrix is not positive definite at any damping
    InvalidProblem,   ///< malformed problem definition
    ResidualNonFinite
};

[[nodiscard]] const char* to_string(OptimizerStatus s) noexcept;

/// Whether the status means "we have a usable answer".
[[nodiscard]] bool is_success(OptimizerStatus s) noexcept;

struct OptimizerResult {
    std::vector<double> parameters;
    double objective = 0.0;         ///< final weighted sum of squares
    double initial_objective = 0.0;
    double rms_residual = 0.0;      ///< sqrt(objective / total_weight)
    double max_abs_residual = 0.0;
    double gradient_norm = 0.0;     ///< scaled infinity norm at the solution

    int iterations = 0;
    int residual_evaluations = 0;
    int jacobian_evaluations = 0;
    int rejected_steps = 0;         ///< steps where lambda had to increase

    /// Lower bound on the condition number of J^T W J at the solution.
    ///
    /// Reported because a fit can converge beautifully and still be
    /// meaningless: if two parameters are not separately identified by the
    /// quotes, the objective is flat along a direction and the reported
    /// parameter values are arbitrary within it.  The objective value alone
    /// cannot distinguish that from a good fit, and this can.
    double condition_estimate = 1.0;

    /// Which parameters ended pinned to a bound.  A parameter at a bound is a
    /// statement that the data wanted to go somewhere inadmissible.
    std::vector<std::uint8_t> parameters_at_bound;

    OptimizerStatus status = OptimizerStatus::InvalidProblem;
    double lambda_final = 0.0;

    [[nodiscard]] bool ok() const noexcept { return is_success(status); }
    [[nodiscard]] std::size_t num_at_bound() const noexcept;
};

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Levenberg-Marquardt from one starting point.
[[nodiscard]] OptimizerResult calibrate(const LeastSquaresProblem& problem,
                                        std::span<const double> initial,
                                        const OptimizerSettings& settings = {});

/// Run from several starting points and keep the best.
///
/// Deterministic: the starts are evaluated in the order given and ties go to
/// the earlier one, so the result is a function of the inputs alone.  There is
/// no randomised restart here for that reason -- a random restart that is not
/// seed-reproducible would break the determinism contract, and one that is
/// seed-reproducible is just a fixed list with extra steps.
struct MultiStartResult {
    OptimizerResult best;
    std::size_t best_start = 0;
    std::size_t starts_tried = 0;
    std::size_t starts_succeeded = 0;

    /// Objective of every start, so a caller can see whether the problem has
    /// local minima at all -- a spread of final objectives is the evidence,
    /// and without it "we used multi-start" is cargo cult.
    std::vector<double> objectives;
};

[[nodiscard]] MultiStartResult calibrate_multi_start(
    const LeastSquaresProblem& problem, std::span<const std::vector<double>> starts,
    const OptimizerSettings& settings = {});

/// Check an analytic Jacobian against forward differences.
///
/// Exposed as a library function rather than hidden in the tests because it
/// belongs in a caller's own test suite too: a hand-derived Jacobian is the
/// most common source of a calibration that converges to the wrong place, and
/// it fails silently -- the optimiser simply takes worse steps.
struct JacobianCheck {
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    std::size_t worst_residual = 0;
    std::size_t worst_param = 0;
    bool agrees = false;
};

[[nodiscard]] JacobianCheck check_jacobian(const LeastSquaresProblem& problem,
                                           std::span<const double> params,
                                           double rtol = 1e-6);

}  // namespace vl
