// SPDX-License-Identifier: MIT
#pragma once
/// \file linalg.hpp
/// \brief Small dense linear algebra for the calibrator.
///
/// ## Scope, and why there is no BLAS here
///
/// The largest system this library ever solves is 5x5 -- the SVI parameter
/// count -- and the most common is 3x3, the inner solve of the quasi-explicit
/// reduction.  At that size:
///
///  * a BLAS call costs more in dispatch than the arithmetic it performs;
///  * the matrix fits in four cache lines, so blocking and tiling are
///    irrelevant;
///  * the only thing that matters is numerical care, and a fixed-size
///    hand-written Cholesky with explicit pivoting diagnostics gives more of
///    that than a generic routine whose failure mode is an error code.
///
/// So this is deliberately not a linear algebra library.  It is four
/// factorisations sized for the problem at hand, each reporting *why* it
/// failed rather than only that it did -- because a rank-deficient normal
/// matrix in a calibration is a statement about the data (two parameters are
/// not separately identified by these quotes) and the caller needs to hear it.
///
/// Dimensions are runtime values with a compile-time maximum, so everything
/// lives on the stack and no calibration step allocates.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>

#include "volatility_lab/core/config.hpp"

namespace vl::math {

/// Largest system the calibrator poses.  Five for raw SVI; the SSVI global fit
/// is three.  Fixed so that `SmallMatrix` is a stack value.
inline constexpr std::size_t kMaxSmallDim = 8;

// ---------------------------------------------------------------------------
// SmallMatrix
// ---------------------------------------------------------------------------

/// Row-major fixed-capacity square matrix with a runtime dimension.
class SmallMatrix {
  public:
    SmallMatrix() = default;
    explicit SmallMatrix(std::size_t n) : n_(n) { data_.fill(0.0); }

    [[nodiscard]] std::size_t dim() const noexcept { return n_; }
    void resize(std::size_t n) noexcept {
        n_ = n;
        data_.fill(0.0);
    }

    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * kMaxSmallDim + j];
    }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * kMaxSmallDim + j];
    }

    void set_identity() noexcept {
        data_.fill(0.0);
        for (std::size_t i = 0; i < n_; ++i) (*this)(i, i) = 1.0;
    }

    /// Add `lambda` to the diagonal -- the Levenberg-Marquardt damping step.
    void add_to_diagonal(double lambda) noexcept {
        for (std::size_t i = 0; i < n_; ++i) (*this)(i, i) += lambda;
    }

    /// Scale row i and column i by `s`, i.e. D A D with D diagonal.
    void symmetric_scale(std::span<const double> s) noexcept;

    [[nodiscard]] double max_abs() const noexcept;

  private:
    std::array<double, kMaxSmallDim * kMaxSmallDim> data_{};
    std::size_t n_ = 0;
};

using SmallVector = std::array<double, kMaxSmallDim>;

// ---------------------------------------------------------------------------
// Cholesky
// ---------------------------------------------------------------------------

enum class LinalgStatus : std::uint8_t {
    Ok = 0,
    NotPositiveDefinite,  ///< a pivot was <= 0: the matrix is singular or indefinite
    NonFinite,            ///< NaN or inf in the input
    DimensionTooLarge
};

[[nodiscard]] constexpr const char* to_string(LinalgStatus s) noexcept {
    switch (s) {
        case LinalgStatus::Ok:                  return "ok";
        case LinalgStatus::NotPositiveDefinite: return "not-positive-definite";
        case LinalgStatus::NonFinite:           return "non-finite";
        case LinalgStatus::DimensionTooLarge:   return "dimension-too-large";
    }
    return "?";
}

struct CholeskyResult {
    LinalgStatus status = LinalgStatus::Ok;

    /// Index of the pivot that failed, when the status is NotPositiveDefinite.
    ///
    /// Reported because it *localises the problem*: in a Levenberg-Marquardt
    /// step on a parameter vector, pivot j failing means parameter j is not
    /// identified by the residuals, and the calibrator can say which one
    /// rather than reporting a generic failure.
    std::size_t failed_pivot = 0;

    /// Ratio of the largest to the smallest diagonal of the factor, squared --
    /// a cheap lower bound on the condition number of the original matrix.
    /// Used by the calibrator to flag a fit that converged but is not
    /// identified, which looks identical to a good fit in the objective value
    /// alone.
    double condition_estimate = 1.0;

    [[nodiscard]] bool ok() const noexcept { return status == LinalgStatus::Ok; }
};

/// Cholesky factorisation in place: A = L L^T, L stored in the lower triangle.
///
/// No pivoting, because the matrices here are normal matrices J^T J (plus LM
/// damping), which are symmetric positive semi-definite by construction --
/// pivoting buys nothing for an SPD matrix, and the failure of a pivot is
/// itself the information the caller wants.
[[nodiscard]] CholeskyResult cholesky_factor(SmallMatrix& a) noexcept;

/// Solve L L^T x = b given the factor from `cholesky_factor`, in place on `x`.
void cholesky_solve_in_place(const SmallMatrix& l, std::span<double> x) noexcept;

/// Factor and solve in one call.  `a` is overwritten with its factor.
[[nodiscard]] CholeskyResult cholesky_solve(SmallMatrix& a, std::span<double> b) noexcept;

// ---------------------------------------------------------------------------
// Normal equations
// ---------------------------------------------------------------------------

/// Accumulate J^T W J and J^T W r for a least-squares problem, one residual at
/// a time.
///
/// Streaming rather than forming J explicitly.  A slice can have 60 quotes and
/// 5 parameters, so J is 60x5 = 2.4 kB -- small, but accumulating straight
/// into the 5x5 normal matrix means one pass over the residuals, no allocation
/// at all, and the working set stays in registers.  It also means the caller
/// can generate residuals lazily, which the quasi-explicit inner solve relies
/// on.
///
/// The accumulation is in a fixed order (residual index ascending), so the
/// result is bitwise reproducible -- required by the determinism contract,
/// since a calibration is part of the deterministic pipeline.
class NormalEquations {
  public:
    explicit NormalEquations(std::size_t num_params) : n_(num_params) { reset(); }

    void reset() noexcept;

    /// Add one weighted residual: `jacobian_row` is dr/dp, `residual` is r,
    /// `weight` is w.  Contributes w*j*j^T to the matrix and w*j*r to the rhs.
    void add(std::span<const double> jacobian_row, double residual, double weight) noexcept;

    [[nodiscard]] const SmallMatrix& matrix() const noexcept { return jtj_; }
    [[nodiscard]] SmallMatrix& matrix() noexcept { return jtj_; }
    [[nodiscard]] std::span<const double> rhs() const noexcept {
        return {jtr_.data(), n_};
    }
    [[nodiscard]] std::span<double> rhs() noexcept { return {jtr_.data(), n_}; }

    /// Sum of w*r^2 -- the objective value, accumulated alongside so it costs
    /// nothing extra.
    [[nodiscard]] double chi_squared() const noexcept { return chi2_; }
    [[nodiscard]] double total_weight() const noexcept { return weight_sum_; }
    [[nodiscard]] std::size_t count() const noexcept { return count_; }
    [[nodiscard]] std::size_t num_params() const noexcept { return n_; }

    /// The diagonal of J^T W J, used for Marquardt's scaled damping.
    [[nodiscard]] SmallVector diagonal() const noexcept;

  private:
    SmallMatrix jtj_;
    SmallVector jtr_{};
    double chi2_ = 0.0;
    double weight_sum_ = 0.0;
    std::size_t count_ = 0;
    std::size_t n_ = 0;
};

// ---------------------------------------------------------------------------
// Non-negative least squares on a box
// ---------------------------------------------------------------------------

/// Solve a small least-squares problem subject to simple bounds, by enumerating
/// active sets.
///
/// Used by the quasi-explicit SVI inner solve, where the three reduced
/// coefficients must satisfy a handful of linear inequalities.  With three
/// variables there are at most 2^3 = 8 active-set combinations, so enumeration
/// is exact and cheap -- there is no reason to run an iterative QP and hope.
///
/// `lower` and `upper` may contain infinities for unbounded directions.
/// Returns the objective value at the solution, or infinity if every active
/// set was infeasible.
struct BoxedLeastSquaresResult {
    SmallVector solution{};
    double objective = 0.0;
    std::size_t active_constraints = 0;
    bool feasible = false;
};

[[nodiscard]] BoxedLeastSquaresResult solve_boxed_least_squares(
    const NormalEquations& eq, std::span<const double> lower,
    std::span<const double> upper) noexcept;

}  // namespace vl::math
