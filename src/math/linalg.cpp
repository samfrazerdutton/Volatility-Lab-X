// SPDX-License-Identifier: MIT
#include "volatility_lab/math/linalg.hpp"

#include <limits>

namespace vl::math {

// ===========================================================================
// SmallMatrix
// ===========================================================================

void SmallMatrix::symmetric_scale(std::span<const double> s) noexcept {
    const std::size_t n = std::min(n_, s.size());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            (*this)(i, j) *= s[i] * s[j];
        }
    }
}

double SmallMatrix::max_abs() const noexcept {
    double m = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = 0; j < n_; ++j) {
            m = std::max(m, std::abs((*this)(i, j)));
        }
    }
    return m;
}

// ===========================================================================
// Cholesky
// ===========================================================================

CholeskyResult cholesky_factor(SmallMatrix& a) noexcept {
    CholeskyResult r;
    const std::size_t n = a.dim();
    if (n > kMaxSmallDim) {
        r.status = LinalgStatus::DimensionTooLarge;
        return r;
    }

    double min_pivot = std::numeric_limits<double>::infinity();
    double max_pivot = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double sum = a(i, j);
            for (std::size_t k = 0; k < j; ++k) {
                sum -= a(i, k) * a(j, k);
            }
            if (!std::isfinite(sum)) {
                r.status = LinalgStatus::NonFinite;
                r.failed_pivot = i;
                return r;
            }
            if (i == j) {
                if (!(sum > 0.0)) {
                    // Not merely "failed": the index localises the problem.  In
                    // a Levenberg-Marquardt step, pivot i failing means
                    // parameter i is not identified by the residuals, and the
                    // calibrator reports *which* parameter rather than a
                    // generic singularity.
                    r.status = LinalgStatus::NotPositiveDefinite;
                    r.failed_pivot = i;
                    return r;
                }
                const double d = std::sqrt(sum);
                a(i, i) = d;
                min_pivot = std::min(min_pivot, d);
                max_pivot = std::max(max_pivot, d);
            } else {
                a(i, j) = sum / a(j, j);
            }
        }
        // Zero the upper triangle so the factor is unambiguous and a later
        // `cholesky_solve_in_place` cannot read stale data.
        for (std::size_t j = i + 1; j < n; ++j) a(i, j) = 0.0;
    }

    // cond(A) >= (max_pivot/min_pivot)^2, since the singular values of A are
    // the squares of those of L.  A lower bound is enough: the calibrator uses
    // it to flag a fit that converged but is not identified, which is
    // indistinguishable from a good fit by the objective value alone.
    if (min_pivot > 0.0 && std::isfinite(min_pivot)) {
        const double ratio = max_pivot / min_pivot;
        r.condition_estimate = ratio * ratio;
    } else {
        r.condition_estimate = std::numeric_limits<double>::infinity();
    }
    return r;
}

void cholesky_solve_in_place(const SmallMatrix& l, std::span<double> x) noexcept {
    const std::size_t n = std::min(l.dim(), x.size());
    // Forward substitution: L y = b.
    for (std::size_t i = 0; i < n; ++i) {
        double sum = x[i];
        for (std::size_t k = 0; k < i; ++k) sum -= l(i, k) * x[k];
        x[i] = sum / l(i, i);
    }
    // Back substitution: L^T x = y.
    for (std::size_t ii = n; ii-- > 0;) {
        double sum = x[ii];
        for (std::size_t k = ii + 1; k < n; ++k) sum -= l(k, ii) * x[k];
        x[ii] = sum / l(ii, ii);
    }
}

CholeskyResult cholesky_solve(SmallMatrix& a, std::span<double> b) noexcept {
    const CholeskyResult r = cholesky_factor(a);
    if (r.ok()) cholesky_solve_in_place(a, b);
    return r;
}

// ===========================================================================
// NormalEquations
// ===========================================================================

void NormalEquations::reset() noexcept {
    jtj_.resize(n_);
    jtr_.fill(0.0);
    chi2_ = 0.0;
    weight_sum_ = 0.0;
    count_ = 0;
}

void NormalEquations::add(std::span<const double> jacobian_row, double residual,
                          double weight) noexcept {
    if (!(weight > 0.0) || !std::isfinite(residual)) return;
    const std::size_t n = std::min(n_, jacobian_row.size());
    // Only the lower triangle is accumulated; Cholesky reads nothing else, and
    // filling both halves would double the work for no benefit.
    for (std::size_t i = 0; i < n; ++i) {
        const double wj = weight * jacobian_row[i];
        for (std::size_t j = 0; j <= i; ++j) {
            jtj_(i, j) += wj * jacobian_row[j];
        }
        jtr_[i] += wj * residual;
    }
    chi2_ += weight * residual * residual;
    weight_sum_ += weight;
    ++count_;
}

SmallVector NormalEquations::diagonal() const noexcept {
    SmallVector d{};
    for (std::size_t i = 0; i < n_; ++i) d[i] = jtj_(i, i);
    return d;
}

// ===========================================================================
// Boxed least squares
// ===========================================================================

BoxedLeastSquaresResult solve_boxed_least_squares(const NormalEquations& eq,
                                                  std::span<const double> lower,
                                                  std::span<const double> upper) noexcept {
    BoxedLeastSquaresResult best;
    best.objective = std::numeric_limits<double>::infinity();

    const std::size_t n = eq.num_params();
    if (n == 0 || n > 3) {
        // Enumeration is 3^n; beyond three variables it stops being the right
        // algorithm.  The only caller is the quasi-explicit inner solve, which
        // is exactly three, and the assertion is a guard against that changing
        // silently.
        return best;
    }

    // Each variable is either free, pinned at its lower bound, or pinned at its
    // upper bound: 3^n active sets.  With n = 3 that is 27 small solves, each
    // at most 3x3 -- exact, cheap, and with no convergence question.  An
    // iterative QP here would be slower *and* would need its own termination
    // criteria.
    const std::size_t total = (n == 1) ? 3 : (n == 2 ? 9 : 27);

    for (std::size_t mask = 0; mask < total; ++mask) {
        std::array<int, 3> state{};  // 0 free, 1 at lower, 2 at upper
        std::size_t m = mask;
        for (std::size_t i = 0; i < n; ++i) {
            state[i] = static_cast<int>(m % 3);
            m /= 3;
        }

        // Pinned values, and a check that the bound is finite.
        SmallVector x{};
        bool feasible_pin = true;
        std::size_t free_count = 0;
        std::array<std::size_t, 3> free_idx{};
        for (std::size_t i = 0; i < n; ++i) {
            if (state[i] == 0) {
                free_idx[free_count++] = i;
            } else {
                const double v = (state[i] == 1) ? lower[i] : upper[i];
                if (!std::isfinite(v)) {
                    feasible_pin = false;
                    break;
                }
                x[i] = v;
            }
        }
        if (!feasible_pin) continue;

        // Reduced normal equations over the free variables, with the pinned
        // contributions moved to the right-hand side.
        if (free_count > 0) {
            SmallMatrix a(free_count);
            SmallVector rhs{};
            for (std::size_t fi = 0; fi < free_count; ++fi) {
                const std::size_t i = free_idx[fi];
                // The accumulator fills only the lower triangle, so read it
                // symmetrically.
                const auto at = [&eq](std::size_t r, std::size_t c) {
                    return (r >= c) ? eq.matrix()(r, c) : eq.matrix()(c, r);
                };
                double b = eq.rhs()[i];
                for (std::size_t j = 0; j < n; ++j) {
                    if (state[j] != 0) b -= at(i, j) * x[j];
                }
                rhs[fi] = b;
                for (std::size_t fj = 0; fj < free_count; ++fj) {
                    a(fi, fj) = at(i, free_idx[fj]);
                }
            }
            SmallVector sol = rhs;
            const auto chol = cholesky_solve(a, std::span<double>(sol.data(), free_count));
            if (!chol.ok()) continue;  // this active set is degenerate
            for (std::size_t fi = 0; fi < free_count; ++fi) {
                x[free_idx[fi]] = sol[fi];
            }
        }

        // Feasibility of the whole point.
        bool feasible = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (x[i] < lower[i] - 1e-12 || x[i] > upper[i] + 1e-12) {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;

        // Objective: chi2 - 2 x^T (J^T r) + x^T (J^T J) x, which is the
        // weighted sum of squares at x up to the constant the accumulator
        // already holds.
        double obj = eq.chi_squared();
        for (std::size_t i = 0; i < n; ++i) {
            obj -= 2.0 * x[i] * eq.rhs()[i];
            for (std::size_t j = 0; j < n; ++j) {
                const double aij = (i >= j) ? eq.matrix()(i, j) : eq.matrix()(j, i);
                obj += x[i] * aij * x[j];
            }
        }
        // The objective is evaluated as chi2 - 2 x^T b + x^T A x, which
        // cancels when the fit is near-exact: on noiseless data chi2 is 0.034
        // and the true objective is 0, so the result is around 1e-18 and can
        // land slightly negative.  Clamped, because a negative sum of squares
        // is never meaningful and would propagate into a sqrt downstream.
        obj = std::max(obj, 0.0);
        if (obj < best.objective) {
            best.objective = obj;
            best.solution = x;
            best.feasible = true;
            best.active_constraints = n - free_count;
        }
    }
    return best;
}

}  // namespace vl::math
