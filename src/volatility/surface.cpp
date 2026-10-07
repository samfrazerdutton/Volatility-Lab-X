// SPDX-License-Identifier: MIT
#include "volatility_lab/volatility/surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "volatility_lab/pricing/black.hpp"

namespace vl {

// ===========================================================================
// GridSlice
// ===========================================================================

GridSlice::GridSlice(std::vector<double> ks, std::vector<double> ws, double years)
    : years_(years) {
    if (ks.size() != ws.size() || ks.empty()) return;
    spline_ = math::CubicSpline(std::move(ks), std::move(ws));
}

double GridSlice::total_variance(double k) const noexcept {
    // Clamped at zero: a cubic through noisy quotes can dip below zero between
    // knots, and a negative total variance is not a number any downstream
    // consumer can use.  The dip itself is reported by the arbitrage engine,
    // which is the right place for it -- silently clamping *and* silently not
    // reporting would be the error.
    return std::max(0.0, spline_(k));
}

SliceJet GridSlice::jet(double k) const noexcept {
    const auto j = spline_.jet(k);
    SliceJet out;
    out.w = std::max(0.0, j.y);
    out.dw = j.dy;
    out.d2w = j.d2y;
    return out;
}

void GridSlice::total_variance_batch(std::span<const double> k,
                                     std::span<double> w) const noexcept {
    const std::size_t n = std::min(k.size(), w.size());
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = std::max(0.0, spline_(k[i]));
    }
}

double GridSlice::k_min() const noexcept {
    const auto nodes = spline_.nodes();
    return nodes.empty() ? 0.0 : nodes.front();
}

double GridSlice::k_max() const noexcept {
    const auto nodes = spline_.nodes();
    return nodes.empty() ? 0.0 : nodes.back();
}

// ===========================================================================
// SliceVariant dispatch
// ===========================================================================

double slice_total_variance(const SliceVariant& s, double k) noexcept {
    return std::visit(
        [k](const auto& v) -> double {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, FlatParams>) {
                return v.w;
            } else if constexpr (std::is_same_v<T, SviParams>) {
                return svi_total_variance(v, k);
            } else if constexpr (std::is_same_v<T, SsviSlice>) {
                return ssvi_total_variance(v.global, v.theta, k);
            } else {
                return v.total_variance(k);
            }
        },
        s);
}

SliceJet slice_jet(const SliceVariant& s, double k) noexcept {
    return std::visit(
        [k](const auto& v) -> SliceJet {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, FlatParams>) {
                SliceJet j;
                j.w = v.w;
                return j;
            } else if constexpr (std::is_same_v<T, SviParams>) {
                return svi_jet(v, k);
            } else if constexpr (std::is_same_v<T, SsviSlice>) {
                return ssvi_jet(v.global, v.theta, k);
            } else {
                return v.jet(k);
            }
        },
        s);
}

double slice_years(const SliceVariant& s) noexcept {
    // GridSlice exposes `years()` while the parameter structs expose a `years`
    // member, because the former owns storage and keeps its fields private.
    // The `if constexpr` is the cost of that; the alternative is a base class.
    return std::visit(
        [](const auto& v) -> double {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, GridSlice>) {
                return v.years();
            } else {
                return v.years;
            }
        },
        s);
}

SliceKind slice_kind(const SliceVariant& s) noexcept {
    return std::visit(
        [](const auto& v) -> SliceKind {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, FlatParams>) return SliceKind::Flat;
            else if constexpr (std::is_same_v<T, SviParams>) return SliceKind::Svi;
            else if constexpr (std::is_same_v<T, SsviSlice>) return SliceKind::Ssvi;
            else return SliceKind::Grid;
        },
        s);
}

void slice_total_variance_batch(const SliceVariant& s, std::span<const double> k,
                                std::span<double> w) noexcept {
    // The visit is here, outside the loop.  Each concrete branch then runs a
    // straight-line loop with the slice parameters in registers, which is what
    // makes the variant worth having over a virtual base class.
    std::visit(
        [k, w](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, FlatParams>) {
                std::fill(w.begin(), w.begin() + static_cast<std::ptrdiff_t>(
                                                     std::min(k.size(), w.size())),
                          v.w);
            } else if constexpr (std::is_same_v<T, SviParams>) {
                svi_total_variance_batch(v, k, w);
            } else if constexpr (std::is_same_v<T, SsviSlice>) {
                // SSVI at fixed theta *is* a raw SVI slice, exactly.  Convert
                // once and use the SVI kernel rather than re-deriving phi per
                // option: phi(theta) involves two pow() calls, and hoisting
                // them out of the loop is worth more than the conversion costs.
                const SviParams sp = ssvi_slice_params(v.global, v.theta, v.years);
                svi_total_variance_batch(sp, k, w);
            } else {
                v.total_variance_batch(k, w);
            }
        },
        s);
}

std::optional<SviParams> slice_as_svi(const SliceVariant& s) noexcept {
    return std::visit(
        [](const auto& v) -> std::optional<SviParams> {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, FlatParams>) {
                SviParams p;
                p.a = v.w;
                p.b = 0.0;
                p.rho = 0.0;
                p.m = 0.0;
                p.sigma = 1.0;
                p.years = v.years;
                return p;
            } else if constexpr (std::is_same_v<T, SviParams>) {
                return v;
            } else if constexpr (std::is_same_v<T, SsviSlice>) {
                return ssvi_slice_params(v.global, v.theta, v.years);
            } else {
                return std::nullopt;  // a grid slice has no parametric form
            }
        },
        s);
}

// ===========================================================================
// TermCurve
// ===========================================================================

TermCurve::TermCurve(std::vector<double> years, std::vector<double> values) {
    if (years.empty() || years.size() != values.size()) return;
    if (years.size() == 1) {
        flat_value_ = values[0];
        is_flat_ = true;
        return;
    }
    std::vector<double> logs;
    logs.reserve(values.size());
    for (double v : values) {
        logs.push_back((v > 0.0) ? std::log(v) : -std::numeric_limits<double>::max());
    }
    log_interp_ = math::LinearInterp(std::move(years), std::move(logs));
    is_flat_ = false;
}

TermCurve TermCurve::flat(double value) {
    TermCurve c;
    c.flat_value_ = value;
    c.is_flat_ = true;
    return c;
}

double TermCurve::operator()(double years) const noexcept {
    if (is_flat_ || log_interp_.empty()) return flat_value_;
    return std::exp(log_interp_(years));
}

// ===========================================================================
// VolSurface
// ===========================================================================

VolSurface::VolSurface(std::vector<SliceVariant> slices, TermCurve forwards,
                       TermCurve discounts)
    : slices_(std::move(slices)), forwards_(std::move(forwards)),
      discounts_(std::move(discounts)) {
    expiries_.reserve(slices_.size());
    for (const auto& s : slices_) expiries_.push_back(slice_years(s));

    if (slices_.empty()) {
        diags_.add(make_diag(DiagCode::SliceTooFewQuotes, Severity::Fatal, "surface",
                             "slices", 0.0, "a surface needs at least one slice"));
        return;
    }
    // Validation is reported, not thrown: a surface built from a bad snapshot
    // is a normal occurrence in a market feed, and the caller needs to see the
    // partial result alongside the complaint.
    for (std::size_t i = 0; i < expiries_.size(); ++i) {
        if (!(expiries_[i] > 0.0) || !std::isfinite(expiries_[i])) {
            diags_.add(make_range_diag(DiagCode::NonPositiveExpiry, Severity::Fatal,
                                       "surface", "expiry", expiries_[i], 0.0,
                                       std::numeric_limits<double>::infinity(),
                                       "slice expiry must be positive and finite",
                                       static_cast<std::uint32_t>(i)));
        }
        if (i > 0 && !(expiries_[i] > expiries_[i - 1])) {
            diags_.add(make_diag(DiagCode::DuplicateQuote, Severity::Fatal, "surface",
                                 "expiry", expiries_[i],
                                 "slices must be strictly increasing in expiry",
                                 static_cast<std::uint32_t>(i)));
        }
    }
}

VolSurface::TermLocation VolSurface::locate(double years) const noexcept {
    TermLocation loc;
    const std::size_t n = expiries_.size();
    if (n == 0) return loc;
    if (n == 1) {
        loc.lo = loc.hi = 0;
        loc.weight = 0.0;
        loc.extrapolated = (years != expiries_[0]);
        return loc;
    }
    if (years <= expiries_.front()) {
        loc.lo = loc.hi = 0;
        loc.extrapolated = years < expiries_.front();
        return loc;
    }
    if (years >= expiries_.back()) {
        loc.lo = loc.hi = n - 1;
        loc.extrapolated = years > expiries_.back();
        return loc;
    }
    const std::size_t i = math::locate_interval(expiries_, years);
    loc.lo = i;
    loc.hi = i + 1;
    const double h = expiries_[i + 1] - expiries_[i];
    loc.weight = (h > 0.0) ? (years - expiries_[i]) / h : 0.0;
    return loc;
}

namespace {

/// Interpolate total variance between two slices at the same moneyness.
///
/// Linear in total variance, which is flat forward variance over the interval.
/// That is the only choice that cannot create calendar arbitrage out of two
/// admissible slices: w is then monotone in T whenever w_lo <= w_hi, and the
/// interpolated slice is a convex combination of two convex functions of k, so
/// it stays convex.
///
/// Note that this is *not* the monotone cubic from math/interpolation.hpp.
/// The cubic is used for the ATM theta term structure, where the knots are the
/// quantity being modelled; here the two bracketing values are already
/// available and linear is both sufficient and provably safe.  Using a cubic
/// through the full set of slices at every k would also mean evaluating every
/// slice at every query, which is O(num_slices) instead of O(1).
[[nodiscard]] VL_FORCE_INLINE double blend_variance(double w_lo, double w_hi,
                                                    double weight) noexcept {
    return std::fma(weight, w_hi - w_lo, w_lo);
}

/// Extrapolation beyond the last expiry holds *implied volatility* constant,
/// i.e. total variance grows linearly in T.
///
/// Holding total variance constant instead would mean zero forward variance
/// beyond the last quote -- the surface would assert that nothing can happen
/// after the longest listed expiry, which is both wrong and, since it makes a
/// longer-dated option worth less than a shorter one at the same strike,
/// arbitrageable.
[[nodiscard]] VL_FORCE_INLINE double extrapolate_variance(double w_anchor,
                                                          double t_anchor,
                                                          double t) noexcept {
    if (!(t_anchor > 0.0)) return w_anchor;
    return w_anchor * (t / t_anchor);
}

}  // namespace

double VolSurface::total_variance(double k, double years) const noexcept {
    if (slices_.empty()) return 0.0;
    const TermLocation loc = locate(years);
    const double w_lo = slice_total_variance(slices_[loc.lo], k);
    if (loc.lo == loc.hi) {
        return loc.extrapolated ? extrapolate_variance(w_lo, expiries_[loc.lo], years)
                                : w_lo;
    }
    const double w_hi = slice_total_variance(slices_[loc.hi], k);
    return blend_variance(w_lo, w_hi, loc.weight);
}

double VolSurface::vol(double k, double years) const noexcept {
    const double w = total_variance(k, years);
    return (years > 0.0 && w > 0.0) ? std::sqrt(w / years) : 0.0;
}

SliceJet VolSurface::jet(double k, double years) const noexcept {
    if (slices_.empty()) return {};
    const TermLocation loc = locate(years);
    const SliceJet lo = slice_jet(slices_[loc.lo], k);
    if (loc.lo == loc.hi) {
        if (!loc.extrapolated) return lo;
        const double scale =
            (expiries_[loc.lo] > 0.0) ? (years / expiries_[loc.lo]) : 1.0;
        // The k-derivatives scale with the same factor as w, since the
        // extrapolation is a pure multiplication by a function of T alone.
        SliceJet out;
        out.w = lo.w * scale;
        out.dw = lo.dw * scale;
        out.d2w = lo.d2w * scale;
        return out;
    }
    const SliceJet hi = slice_jet(slices_[loc.hi], k);
    SliceJet out;
    out.w = blend_variance(lo.w, hi.w, loc.weight);
    out.dw = blend_variance(lo.dw, hi.dw, loc.weight);
    out.d2w = blend_variance(lo.d2w, hi.d2w, loc.weight);
    return out;
}

SurfacePoint VolSurface::at_strike(double strike, double years) const noexcept {
    SurfacePoint p;
    p.forward = forwards_(years);
    p.discount = discounts_(years);
    if (!(p.forward > 0.0) || !(strike > 0.0)) return p;
    p.log_moneyness = log_moneyness(p.forward, strike);
    const TermLocation loc = locate(years);
    p.extrapolated_in_time = loc.extrapolated;
    p.total_variance = total_variance(p.log_moneyness, years);
    p.vol = (years > 0.0 && p.total_variance > 0.0)
                ? std::sqrt(p.total_variance / years)
                : 0.0;
    // Strike extrapolation is only well defined for grid slices, which have a
    // quoted range; the parametric models are defined on the whole line.
    if (const auto* g = std::get_if<GridSlice>(&slices_[loc.lo])) {
        p.extrapolated_in_strike =
            p.log_moneyness < g->k_min() || p.log_moneyness > g->k_max();
    }
    return p;
}

void VolSurface::total_variance_batch_single_expiry(double years,
                                                    std::span<const double> k,
                                                    std::span<double> w) const noexcept {
    const std::size_t n = std::min(k.size(), w.size());
    if (slices_.empty() || n == 0) return;

    const TermLocation loc = locate(years);

    if (loc.lo == loc.hi) {
        slice_total_variance_batch(slices_[loc.lo], k.first(n), w.first(n));
        if (loc.extrapolated) {
            const double scale =
                (expiries_[loc.lo] > 0.0) ? (years / expiries_[loc.lo]) : 1.0;
            for (std::size_t i = 0; i < n; ++i) w[i] *= scale;
        }
        return;
    }

    // Two slices, blended.  The lower slice goes straight into the output and
    // the upper into a scratch buffer, then one fused pass combines them --
    // two sequential streaming writes instead of an interleaved per-element
    // dispatch.  The scratch allocation is the only one on this path and it is
    // amortised over the whole batch.
    std::vector<double> upper(n);
    slice_total_variance_batch(slices_[loc.lo], k.first(n), w.first(n));
    slice_total_variance_batch(slices_[loc.hi], k.first(n), upper);
    const double weight = loc.weight;
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = std::fma(weight, upper[i] - w[i], w[i]);
    }
}

void VolSurface::total_variance_batch(std::span<const double> k,
                                      std::span<const double> years,
                                      std::span<double> w) const noexcept {
    const std::size_t n = std::min({k.size(), years.size(), w.size()});
    if (slices_.empty() || n == 0) return;

    // Mixed expiries: the general path.  Grouping by expiry would be faster
    // still, but it needs a sort and the scenario engine always calls the
    // single-expiry path, so this one stays simple and correct.  The grouped
    // variant is what `price_batch` builds on -- see pricing/pricer.hpp.
    for (std::size_t i = 0; i < n; ++i) {
        w[i] = total_variance(k[i], years[i]);
    }
}

std::vector<double> VolSurface::atm_total_variance_term() const {
    std::vector<double> out;
    out.reserve(slices_.size());
    for (const auto& s : slices_) out.push_back(slice_total_variance(s, 0.0));
    return out;
}

VolSurface VolSurface::with_slice(std::size_t index, SliceVariant replacement) const {
    std::vector<SliceVariant> copy = slices_;
    if (index < copy.size()) copy[index] = std::move(replacement);
    return VolSurface(std::move(copy), forwards_, discounts_);
}

VolSurface VolSurface::with_slices(std::vector<SliceVariant> replacement) const {
    return VolSurface(std::move(replacement), forwards_, discounts_);
}

}  // namespace vl
