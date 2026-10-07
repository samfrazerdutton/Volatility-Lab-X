// SPDX-License-Identifier: MIT
#include "volatility_lab/volatility/svi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "volatility_lab/math/special.hpp"

namespace vl {

// ===========================================================================
// slice.hpp implementations
// ===========================================================================

double implied_density(double k, const SliceJet& j) noexcept {
    const double g = durrleman_g(k, j);
    if (!std::isfinite(g) || !(j.w > 0.0)) return 0.0;
    const double sqrt_w = std::sqrt(j.w);
    const double d2 = -k / sqrt_w - 0.5 * sqrt_w;
    return g * math::norm_pdf(d2) / sqrt_w;
}

SliceDefect inspect_slice_point(double k, const SliceJet& j) noexcept {
    SliceDefect out = SliceDefect::None;
    if (!std::isfinite(j.w) || !std::isfinite(j.dw) || !std::isfinite(j.d2w)) {
        out = out | SliceDefect::NonFinite;
        return out;  // nothing else is meaningful
    }
    if (!(j.w > 0.0)) out = out | SliceDefect::NonPositiveVariance;
    if (j.w > 0.0 && durrleman_g(k, j) < 0.0) out = out | SliceDefect::ButterflyArbitrage;
    // Lee's bound is asymptotic, so it is only enforced where it bites.
    if (std::abs(k) >= kLeeSlopeTestFrom && std::abs(j.dw) > kLeeSlopeBound) {
        out = out | SliceDefect::SlopeOutOfBounds;
    }
    return out;
}

// ===========================================================================
// SVI
// ===========================================================================

void svi_total_variance_batch(const SviParams& p, std::span<const double> k,
                              std::span<double> w) noexcept {
    const std::size_t n = std::min(k.size(), w.size());
    const double s2 = p.sigma * p.sigma;
    const double a = p.a;
    const double b = p.b;
    const double rho = p.rho;
    const double m = p.m;
    // Locals rather than member reads so the compiler can keep them in
    // registers across the loop; `p` is a reference and could in principle
    // alias `w`.
    for (std::size_t i = 0; i < n; ++i) {
        const double y = k[i] - m;
        const double r = std::sqrt(std::fma(y, y, s2));
        w[i] = std::fma(b, std::fma(rho, y, r), a);
    }
}

bool svi_parameters_admissible(const SviParams& p) noexcept {
    if (!std::isfinite(p.a) || !std::isfinite(p.b) || !std::isfinite(p.rho) ||
        !std::isfinite(p.m) || !std::isfinite(p.sigma)) {
        return false;
    }
    if (p.b < 0.0) return false;
    if (!(p.sigma > 0.0)) return false;
    if (!(std::abs(p.rho) < 1.0)) return false;
    // Exact positivity: the minimum of w is a + b*sigma*sqrt(1-rho^2).
    return svi_min_variance(p) >= 0.0;
}

SviParams svi_project_to_admissible(SviParams p) noexcept {
    if (!std::isfinite(p.b) || p.b < 0.0) p.b = 0.0;
    if (!std::isfinite(p.sigma) || p.sigma <= 0.0) p.sigma = 1e-8;
    if (!std::isfinite(p.rho)) p.rho = 0.0;
    p.rho = std::clamp(p.rho, -1.0 + 1e-12, 1.0 - 1e-12);
    if (!std::isfinite(p.m)) p.m = 0.0;
    if (!std::isfinite(p.a)) p.a = 0.0;

    // Positivity: raise `a` rather than lower `b`.
    //
    // The two are not interchangeable.  `a` is the parameter the data
    // identifies least well -- it trades off almost one-for-one against
    // b*sigma, so the fit is nearly flat along that direction -- whereas `b` is
    // pinned by the wing slopes, which the quotes determine sharply.  Moving
    // the weakly identified parameter does least damage to the fit, and it
    // cannot introduce a new violation either, since dw_min/da = +1.
    //
    // The smallest admissible `a` is exactly -b*sigma*sqrt(1-rho^2), so it is
    // assigned directly rather than computed as `a -= min_variance`.  That is
    // not a micro-optimisation: the subtraction cancels when |a| is much larger
    // than the wing term, leaving an absolute error of ulp(a) in the result.
    // With a = -3.65 and the wing term 3.9e-14 -- which happens whenever rho
    // has just been clamped to +-(1 - 1e-12), making sqrt(1-rho^2) about 1.4e-6
    // -- that error is 4.4e-16, so the "projected" slice came out with a
    // minimum variance of -1.3e-16 and was promptly rejected as inadmissible by
    // the very predicate the projection exists to satisfy.  Assigning the floor
    // makes the minimum exactly zero.  Found by fuzzing the projection over
    // wild inputs; see tests/numerical/slice_models.cpp.
    const double wing_term = p.b * p.sigma * std::sqrt(std::max(0.0, 1.0 - p.rho * p.rho));
    if (p.a < -wing_term) p.a = -wing_term;

    // Belt and braces: two ulps of headroom in case the sqrt above rounds
    // differently from the one inside svi_min_variance on some platform.
    for (int i = 0; i < 4 && svi_min_variance(p) < 0.0; ++i) {
        p.a = std::nextafter(p.a, std::numeric_limits<double>::infinity());
    }
    return p;
}

ButterflyCheck svi_butterfly_check(const SviParams& p, double k_lo, double k_hi,
                                   int grid_points) noexcept {
    ButterflyCheck out;
    out.worst_g = std::numeric_limits<double>::infinity();
    if (grid_points < 3) grid_points = 3;
    if (!(k_hi > k_lo)) std::swap(k_lo, k_hi);

    // The grid is uniform in k rather than clustered near the money.  g is
    // smooth, and the violations SVI actually produces are in the *wings*
    // (where a steep fitted slope fights insufficient curvature), not at the
    // money -- so clustering at the money would look in the wrong place.  This
    // was checked against a 100x denser grid over the fuzz corpus; see
    // tests/calibration.
    const double step = (k_hi - k_lo) / static_cast<double>(grid_points - 1);
    for (int i = 0; i < grid_points; ++i) {
        const double k = k_lo + step * static_cast<double>(i);
        const SliceJet j = svi_jet(p, k);
        const double g = durrleman_g(k, j);
        if (g < out.worst_g) {
            out.worst_g = g;
            out.worst_k = k;
            out.worst_density = implied_density(k, j);
        }
    }
    out.arbitrage_free = out.worst_g >= 0.0;
    return out;
}

// ---------------------------------------------------------------------------
// The quasi-explicit reduction
// ---------------------------------------------------------------------------

SviParams svi_from_reduced(const SviReduced& r, double m, double sigma,
                           double years) noexcept {
    SviParams p;
    p.m = m;
    p.sigma = sigma;
    p.a = r.adash;
    // c = b*sigma  and  d = rho*b*sigma, so b = c/sigma and rho = d/c.
    p.b = (sigma > 0.0) ? (r.c / sigma) : 0.0;
    p.rho = (r.c > 0.0) ? (r.d / r.c) : 0.0;
    p.rho = std::clamp(p.rho, -1.0 + 1e-12, 1.0 - 1e-12);
    p.years = years;
    return p;
}

SviReduced svi_to_reduced(const SviParams& p) noexcept {
    SviReduced r;
    r.adash = p.a;
    r.c = p.b * p.sigma;
    r.d = p.rho * r.c;
    return r;
}

}  // namespace vl
