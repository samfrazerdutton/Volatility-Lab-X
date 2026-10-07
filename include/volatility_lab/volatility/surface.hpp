// SPDX-License-Identifier: MIT
#pragma once
/// \file surface.hpp
/// \brief The volatility surface: a term structure of slices, plus the batch
///        query API the pricing and scenario engines are built on.
///
/// ## Structure
///
///     VolSurface
///       |- forward curve      F(T)      (log-linear in T)
///       |- discount curve      DF(T)    (log-linear in T)
///       |- slices[]            sorted by expiry, one per quoted maturity
///       `- term interpolation  monotone cubic in total variance
///
/// A query at (k, T) finds the bracketing slices, evaluates each at k, and
/// interpolates the resulting *total variances* in T.  Three details in that
/// sentence are load-bearing:
///
///  1. **Interpolate in total variance, not volatility.**  Flat forward
///     variance between expiries is linear in w; linear in sigma introduces
///     calendar arbitrage between every pair of quoted maturities.
///
///  2. **Interpolate at fixed k, not fixed strike.**  The smile lives in
///     moneyness.  Interpolating at fixed strike across expiries mixes
///     different parts of the smile together as the forward curve slopes, and
///     produces a term structure of skew that is an artefact of the forward.
///
///  3. **Use a monotone scheme in T.**  Total variance must be non-decreasing
///     in T.  A natural cubic spline through non-decreasing data overshoots
///     between knots, and an overshoot downwards *is* calendar arbitrage --
///     manufactured by the interpolator out of clean data.  See
///     math/interpolation.hpp.
///
/// ## Dispatch: variant, not virtual
///
/// Slices are held in a `std::variant`.  The dispatch happens once per slice
/// per batch -- a `std::visit` *outside* the inner loop -- so the loop body is
/// a direct, inlinable, vectorisable expression.  A virtual
/// `total_variance(k)` called per option would prevent inlining and
/// vectorisation both, and it is the single most common way a
/// "well-abstracted" pricing library ends up ten times slower than it needs to
/// be.  Measured in benchmarks/pricing.
///
/// The second reason is cost of copying.  A surface of 12 SVI slices is about
/// 700 bytes with no heap nodes, so cloning one is a memcpy.  The scenario
/// engine clones and perturbs a surface per scenario and evaluates millions of
/// them; that is only affordable because a surface is a value.
///
/// ## Thread safety
///
/// A `VolSurface` is immutable once built.  All query methods are `const` and
/// touch no shared mutable state, so any number of threads may query one
/// concurrently without synchronisation.  Mutation goes through the builder,
/// which produces a new surface.  That is what makes the parallel pricing
/// backend lock-free rather than merely carefully locked.

#include <optional>
#include <span>
#include <variant>
#include <vector>

#include "volatility_lab/core/diagnostics.hpp"
#include "volatility_lab/core/types.hpp"
#include "volatility_lab/math/interpolation.hpp"
#include "volatility_lab/volatility/slice.hpp"
#include "volatility_lab/volatility/ssvi.hpp"
#include "volatility_lab/volatility/svi.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// Grid slice
// ---------------------------------------------------------------------------

/// A slice defined by interpolating total variance through quoted points.
///
/// This is "Model A" of the brief: the non-parametric option.  It will fit any
/// snapshot essentially exactly, and it guarantees nothing -- a cubic through
/// noisy quotes can easily violate the Durrleman condition between knots.
/// That is not a defect to be hidden; it is the trade, and it is why the
/// arbitrage diagnostics engine exists and why the quality report compares it
/// against the parametric fits.  A grid slice is the right choice when the
/// quotes are clean and the user wants interpolation rather than a model.
///
/// Unlike the parametric slices this one owns heap storage, so it is *not*
/// cheap to copy.  The scenario engine therefore converts a grid surface to a
/// parametric one before shocking it, and says so rather than silently
/// becoming slow.
class GridSlice {
  public:
    GridSlice() = default;

    /// `ks` must be strictly increasing; `ws` are total variances.
    GridSlice(std::vector<double> ks, std::vector<double> ws, double years);

    [[nodiscard]] double total_variance(double k) const noexcept;
    [[nodiscard]] SliceJet jet(double k) const noexcept;
    void total_variance_batch(std::span<const double> k, std::span<double> w) const noexcept;

    [[nodiscard]] double years() const noexcept { return years_; }
    [[nodiscard]] std::span<const double> nodes() const noexcept { return spline_.nodes(); }
    [[nodiscard]] bool empty() const noexcept { return spline_.empty(); }

    /// Smallest and largest quoted log-moneyness.  Queries outside this range
    /// are extrapolated with a held slope, and the surface marks them with
    /// `DiagCode::SurfaceExtrapolated` so a caller can tell.
    [[nodiscard]] double k_min() const noexcept;
    [[nodiscard]] double k_max() const noexcept;

  private:
    math::CubicSpline spline_;
    double years_ = 0.0;
};

// ---------------------------------------------------------------------------
// The slice variant
// ---------------------------------------------------------------------------

/// An SSVI slice carries the global parameters plus its own ATM variance, so
/// that it can be evaluated without reaching back to the surface.  It is
/// stored rather than collapsed to `SviParams` so that the surface knows the
/// slice came from an SSVI fit and the shock transforms can act on the global
/// parameters coherently.
struct SsviSlice {
    SsviParams global;
    double theta = 0.0;  ///< ATM total variance at this expiry
    double years = 0.0;
};

using SliceVariant = std::variant<FlatParams, SviParams, SsviSlice, GridSlice>;

/// Uniform scalar access across slice types.
[[nodiscard]] double slice_total_variance(const SliceVariant& s, double k) noexcept;
[[nodiscard]] SliceJet slice_jet(const SliceVariant& s, double k) noexcept;
[[nodiscard]] double slice_years(const SliceVariant& s) noexcept;
[[nodiscard]] SliceKind slice_kind(const SliceVariant& s) noexcept;

/// Batch access.  **One visit for the whole batch**, which is the entire point
/// of the variant: the dispatch cost is amortised over `k.size()` options
/// rather than paid per option.
void slice_total_variance_batch(const SliceVariant& s, std::span<const double> k,
                                std::span<double> w) noexcept;

/// The equivalent raw-SVI parameters, where the slice has them.  Returns
/// nullopt for grid slices, which have no parametric form -- callers that
/// require one (the scenario engine, the SIMD kernels) check and report
/// rather than approximate.
[[nodiscard]] std::optional<SviParams> slice_as_svi(const SliceVariant& s) noexcept;

// ---------------------------------------------------------------------------
// Curves
// ---------------------------------------------------------------------------

/// Forward and discount curves, interpolated log-linearly in T.
///
/// Log-linear rather than linear because both are exponentials of a rate times
/// time: log-linear interpolation of DF is piecewise-constant *forward rates*,
/// which is the standard market convention and the only choice that cannot
/// produce a negative forward rate between nodes.  Linear interpolation of DF
/// can, and then the implied forward is negative over that interval.
class TermCurve {
  public:
    TermCurve() = default;
    /// `years` strictly increasing, `values` strictly positive.
    TermCurve(std::vector<double> years, std::vector<double> values);

    /// Flat curve: the same value at every maturity.
    static TermCurve flat(double value);

    [[nodiscard]] double operator()(double years) const noexcept;
    [[nodiscard]] bool empty() const noexcept { return log_interp_.empty(); }
    [[nodiscard]] std::span<const double> nodes() const noexcept {
        return log_interp_.nodes();
    }

  private:
    math::LinearInterp log_interp_;  ///< interpolates log(value)
    double flat_value_ = 1.0;
    bool is_flat_ = true;
};

// ---------------------------------------------------------------------------
// VolSurface
// ---------------------------------------------------------------------------

/// A query result: everything a pricer needs at one (strike, expiry), plus
/// whether the point was extrapolated.
struct SurfacePoint {
    double total_variance = 0.0;
    double vol = 0.0;      ///< annualised
    double forward = 0.0;
    double discount = 1.0;
    double log_moneyness = 0.0;
    bool extrapolated_in_strike = false;
    bool extrapolated_in_time = false;
};

class VolSurface {
  public:
    VolSurface() = default;

    /// Slices must be sorted by expiry and have distinct, positive maturities.
    /// Construction validates that and reports through `diagnostics()` rather
    /// than throwing: a surface built from a bad snapshot is a normal
    /// occurrence in a market feed, not an exceptional one.
    VolSurface(std::vector<SliceVariant> slices, TermCurve forwards, TermCurve discounts);

    // -- scalar queries ---------------------------------------------------
    [[nodiscard]] double total_variance(double k, double years) const noexcept;
    [[nodiscard]] double vol(double k, double years) const noexcept;
    [[nodiscard]] SliceJet jet(double k, double years) const noexcept;

    /// Query by strike rather than moneyness, resolving the forward itself.
    [[nodiscard]] SurfacePoint at_strike(double strike, double years) const noexcept;

    // -- batch queries ----------------------------------------------------

    /// Total variance for many (k, T) pairs.
    ///
    /// Not merely a loop over the scalar path: the batch is processed
    /// slice-pair by slice-pair so that each slice is visited once, its
    /// parameters stay in registers, and the inner loop is a straight-line
    /// expression over contiguous memory.  Inputs need not be sorted; the
    /// implementation groups internally.
    void total_variance_batch(std::span<const double> k, std::span<const double> years,
                              std::span<double> w) const noexcept;

    /// The common case: one expiry, many strikes.  Avoids the grouping pass
    /// entirely and is what the slice-level benchmarks measure.
    void total_variance_batch_single_expiry(double years, std::span<const double> k,
                                            std::span<double> w) const noexcept;

    // -- structure --------------------------------------------------------
    [[nodiscard]] std::size_t num_slices() const noexcept { return slices_.size(); }
    [[nodiscard]] const SliceVariant& slice(std::size_t i) const noexcept {
        return slices_[i];
    }
    [[nodiscard]] std::span<const SliceVariant> slices() const noexcept { return slices_; }
    [[nodiscard]] std::span<const double> expiries() const noexcept { return expiries_; }
    [[nodiscard]] const TermCurve& forwards() const noexcept { return forwards_; }
    [[nodiscard]] const TermCurve& discounts() const noexcept { return discounts_; }
    [[nodiscard]] const DiagnosticSink& diagnostics() const noexcept { return diags_; }
    [[nodiscard]] bool valid() const noexcept {
        return !slices_.empty() && diags_.usable();
    }

    /// ATM total variance at each slice expiry -- the theta term structure
    /// SSVI is parameterised on, and the input to the calendar check.
    [[nodiscard]] std::vector<double> atm_total_variance_term() const;

    /// Replace one slice, returning a new surface.  The surface is immutable,
    /// so this is how the incremental recalibration path (section 31) applies
    /// a localised update without rebuilding anything it did not touch.
    [[nodiscard]] VolSurface with_slice(std::size_t index, SliceVariant replacement) const;

    /// Replace every slice, keeping the curves.  Used by the shock transforms.
    [[nodiscard]] VolSurface with_slices(std::vector<SliceVariant> replacement) const;

  private:
    /// Locate the slice interval for an expiry, and the interpolation weight.
    struct TermLocation {
        std::size_t lo = 0;
        std::size_t hi = 0;
        double weight = 0.0;  ///< 0 -> all lo, 1 -> all hi
        bool extrapolated = false;
    };
    [[nodiscard]] TermLocation locate(double years) const noexcept;

    std::vector<SliceVariant> slices_;
    std::vector<double> expiries_;
    TermCurve forwards_;
    TermCurve discounts_;
    DiagnosticSink diags_;
};

}  // namespace vl
