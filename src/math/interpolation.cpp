// SPDX-License-Identifier: MIT
#include "volatility_lab/math/interpolation.hpp"

#include <cmath>
#include <utility>

namespace vl::math {

std::size_t locate_interval(std::span<const double> xs, double x) noexcept {
    const std::size_t n = xs.size();
    if (n < 2) return 0;
    if (x <= xs[0]) return 0;
    if (x >= xs[n - 1]) return n - 2;
    // upper_bound gives the first element strictly greater than x; the
    // containing interval starts one before it.
    const auto it = std::upper_bound(xs.begin(), xs.end(), x);
    const auto idx = static_cast<std::size_t>(it - xs.begin());
    return (idx == 0) ? 0 : std::min(idx - 1, n - 2);
}

bool is_strictly_increasing(std::span<const double> xs) noexcept {
    for (std::size_t i = 1; i < xs.size(); ++i) {
        if (!(xs[i] > xs[i - 1])) return false;
    }
    return true;
}

bool is_non_decreasing(std::span<const double> xs) noexcept {
    for (std::size_t i = 1; i < xs.size(); ++i) {
        if (xs[i] < xs[i - 1]) return false;
    }
    return true;
}

// ===========================================================================
// Linear
// ===========================================================================

LinearInterp::LinearInterp(std::vector<double> xs, std::vector<double> ys)
    : xs_(std::move(xs)), ys_(std::move(ys)) {
    if (ys_.size() != xs_.size()) ys_.resize(xs_.size(), 0.0);
}

double LinearInterp::operator()(double x) const noexcept {
    if (xs_.empty()) return 0.0;
    if (xs_.size() == 1) return ys_[0];
    const std::size_t i = locate_interval(xs_, x);
    const double h = xs_[i + 1] - xs_[i];
    if (!(h > 0.0)) return ys_[i];
    const double t = (x - xs_[i]) / h;
    // Held end slope outside the range: `t` simply goes outside [0, 1], which
    // continues the end segment linearly.  That is the documented behaviour.
    return std::fma(t, ys_[i + 1] - ys_[i], ys_[i]);
}

double LinearInterp::derivative(double x) const noexcept {
    if (xs_.size() < 2) return 0.0;
    const std::size_t i = locate_interval(xs_, x);
    const double h = xs_[i + 1] - xs_[i];
    return (h > 0.0) ? (ys_[i + 1] - ys_[i]) / h : 0.0;
}

// ===========================================================================
// Natural cubic spline
// ===========================================================================

CubicSpline::CubicSpline(std::vector<double> xs, std::vector<double> ys)
    : xs_(std::move(xs)), ys_(std::move(ys)) {
    const std::size_t n = xs_.size();
    if (ys_.size() != n) ys_.resize(n, 0.0);
    y2_.assign(n, 0.0);
    if (n < 3) return;  // fewer than three knots: the spline degenerates to linear

    // Thomas algorithm on the natural-spline tridiagonal system.
    //
    //   h_{i-1} y2_{i-1} + 2(h_{i-1}+h_i) y2_i + h_i y2_{i+1}
    //       = 6( (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1} )
    //
    // with y2_0 = y2_{n-1} = 0.  No pivoting: the matrix is symmetric positive
    // definite and strictly diagonally dominant (2(h_{i-1}+h_i) > h_{i-1}+h_i),
    // which is what makes Thomas stable here.  It is not stable in general.
    std::vector<double> u(n, 0.0);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const double h0 = xs_[i] - xs_[i - 1];
        const double h1 = xs_[i + 1] - xs_[i];
        if (!(h0 > 0.0) || !(h1 > 0.0)) {
            y2_.assign(n, 0.0);
            return;  // non-increasing knots: fall back to linear behaviour
        }
        const double sig = h0 / (h0 + h1);
        const double p = sig * y2_[i - 1] + 2.0;
        y2_[i] = (sig - 1.0) / p;
        const double d =
            (ys_[i + 1] - ys_[i]) / h1 - (ys_[i] - ys_[i - 1]) / h0;
        u[i] = (6.0 * d / (h0 + h1) - sig * u[i - 1]) / p;
    }
    y2_[n - 1] = 0.0;
    for (std::size_t k = n - 1; k-- > 0;) {
        y2_[k] = y2_[k] * y2_[k + 1] + u[k];
    }
    y2_[0] = 0.0;
    y2_[n - 1] = 0.0;
}

double CubicSpline::operator()(double x) const noexcept { return jet(x).y; }

CubicSpline::Jet CubicSpline::jet(double x) const noexcept {
    Jet out;
    const std::size_t n = xs_.size();
    if (n == 0) return out;
    if (n == 1) {
        out.y = ys_[0];
        return out;
    }

    const std::size_t i = locate_interval(xs_, x);
    const double h = xs_[i + 1] - xs_[i];
    if (!(h > 0.0)) {
        out.y = ys_[i];
        return out;
    }

    // Extrapolation holds the end slope rather than continuing the cubic.  A
    // continued cubic diverges and produces negative total variance a little
    // way outside the quoted strikes; a held slope grows linearly, which is
    // what Lee's moment bound asks of total variance anyway.
    if (x < xs_[0] || x > xs_[n - 1]) {
        const bool left = x < xs_[0];
        const std::size_t e = left ? 0 : n - 2;
        const double he = xs_[e + 1] - xs_[e];
        const double slope_secant = (ys_[e + 1] - ys_[e]) / he;
        // Slope of the cubic at the relevant endpoint.
        const double slope_end =
            left ? (slope_secant - he * (2.0 * y2_[e] + y2_[e + 1]) / 6.0)
                 : (slope_secant + he * (y2_[e] + 2.0 * y2_[e + 1]) / 6.0);
        const double x0 = left ? xs_[0] : xs_[n - 1];
        const double y0 = left ? ys_[0] : ys_[n - 1];
        out.y = std::fma(slope_end, x - x0, y0);
        out.dy = slope_end;
        out.d2y = 0.0;
        return out;
    }

    const double a = (xs_[i + 1] - x) / h;
    const double b = (x - xs_[i]) / h;
    const double h2_6 = h * h / 6.0;

    out.y = a * ys_[i] + b * ys_[i + 1] +
            ((a * a * a - a) * y2_[i] + (b * b * b - b) * y2_[i + 1]) * h2_6;
    out.dy = (ys_[i + 1] - ys_[i]) / h +
             ((-3.0 * a * a + 1.0) * y2_[i] + (3.0 * b * b - 1.0) * y2_[i + 1]) * h / 6.0;
    out.d2y = a * y2_[i] + b * y2_[i + 1];
    return out;
}

// ===========================================================================
// Monotone cubic (Fritsch-Carlson)
// ===========================================================================

namespace {

/// Fritsch-Carlson limiter: shrink (d0, d1) onto the circle of radius 3 in
/// units of the secant slope, which is the sufficient condition for a Hermite
/// cubic to be monotone on the interval.
void monotone_limit(double secant, double& d0, double& d1) noexcept {
    if (secant == 0.0) {
        d0 = 0.0;
        d1 = 0.0;
        return;
    }
    const double alpha = d0 / secant;
    const double beta = d1 / secant;
    // A derivative with the wrong sign would create a local extremum inside a
    // monotone interval; zero it rather than scale it.
    if (alpha < 0.0) {
        d0 = 0.0;
        return;
    }
    if (beta < 0.0) {
        d1 = 0.0;
        return;
    }
    const double tau = alpha * alpha + beta * beta;
    if (tau > 9.0) {
        const double scale = 3.0 / std::sqrt(tau);
        d0 = scale * alpha * secant;
        d1 = scale * beta * secant;
    }
}

}  // namespace

MonotoneCubic::MonotoneCubic(std::vector<double> xs, std::vector<double> ys)
    : xs_(std::move(xs)), ys_(std::move(ys)) {
    const std::size_t n = xs_.size();
    if (ys_.size() != n) ys_.resize(n, 0.0);
    m_.assign(n, 0.0);
    if (n < 2) return;

    // Secant slopes.
    std::vector<double> secant(n - 1, 0.0);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const double h = xs_[i + 1] - xs_[i];
        secant[i] = (h > 0.0) ? (ys_[i + 1] - ys_[i]) / h : 0.0;
    }

    // Initial knot slopes: the average of the adjacent secants, with the ends
    // taking the single available secant.
    m_[0] = secant[0];
    m_[n - 1] = secant[n - 2];
    for (std::size_t i = 1; i + 1 < n; ++i) {
        // Zero the slope at a local extremum of the data, so a flat spot stays
        // flat rather than wobbling.
        if (secant[i - 1] * secant[i] <= 0.0) {
            m_[i] = 0.0;
        } else {
            m_[i] = 0.5 * (secant[i - 1] + secant[i]);
        }
    }

    for (std::size_t i = 0; i + 1 < n; ++i) {
        monotone_limit(secant[i], m_[i], m_[i + 1]);
    }
}

double MonotoneCubic::operator()(double x) const noexcept {
    const std::size_t n = xs_.size();
    if (n == 0) return 0.0;
    if (n == 1) return ys_[0];

    // Held end slope outside the range, for the same reason as CubicSpline.
    if (x < xs_[0]) return std::fma(m_[0], x - xs_[0], ys_[0]);
    if (x > xs_[n - 1]) return std::fma(m_[n - 1], x - xs_[n - 1], ys_[n - 1]);

    const std::size_t i = locate_interval(xs_, x);
    const double h = xs_[i + 1] - xs_[i];
    if (!(h > 0.0)) return ys_[i];
    const double t = (x - xs_[i]) / h;
    const double t2 = t * t;
    const double t3 = t2 * t;
    // Standard cubic Hermite basis.
    const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const double h10 = t3 - 2.0 * t2 + t;
    const double h01 = -2.0 * t3 + 3.0 * t2;
    const double h11 = t3 - t2;
    return h00 * ys_[i] + h10 * h * m_[i] + h01 * ys_[i + 1] + h11 * h * m_[i + 1];
}

double MonotoneCubic::derivative(double x) const noexcept {
    const std::size_t n = xs_.size();
    if (n < 2) return 0.0;
    if (x < xs_[0]) return m_[0];
    if (x > xs_[n - 1]) return m_[n - 1];

    const std::size_t i = locate_interval(xs_, x);
    const double h = xs_[i + 1] - xs_[i];
    if (!(h > 0.0)) return 0.0;
    const double t = (x - xs_[i]) / h;
    const double t2 = t * t;
    const double d00 = 6.0 * t2 - 6.0 * t;
    const double d10 = 3.0 * t2 - 4.0 * t + 1.0;
    const double d01 = -6.0 * t2 + 6.0 * t;
    const double d11 = 3.0 * t2 - 2.0 * t;
    return (d00 * ys_[i] + d01 * ys_[i + 1]) / h + d10 * m_[i] + d11 * m_[i + 1];
}

}  // namespace vl::math
