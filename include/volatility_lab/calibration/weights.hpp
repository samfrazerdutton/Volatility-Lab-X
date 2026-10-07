// SPDX-License-Identifier: MIT
#pragma once
/// \file weights.hpp
/// \brief The quote weighting model.
///
/// ## Why unweighted least squares is the wrong default
///
/// Fit a slice with equal weights and the deep wings win.  Not because they
/// matter more, but because there are more of them and their implied
/// volatilities are noisier -- so they carry most of the squared error, and the
/// optimiser spends its degrees of freedom fitting microstructure noise at
/// strikes nobody trades, at the cost of the near-the-money region where
/// essentially all the risk actually sits.
///
/// The fix is not a hack; it is a statement about the likelihood.  Least
/// squares is maximum likelihood under Gaussian errors of *equal* variance, so
/// if the errors have unequal variance the correct estimator weights each
/// residual by the inverse of its variance.  Everything below is an attempt to
/// estimate that variance from observables.
///
/// ## The components
///
/// Weights multiply, because the effects are roughly independent and each is
/// naturally expressed as a factor:
///
///     w = w_spread * w_liquidity * w_moneyness * w_expiry * w_staleness
///
/// **Spread.**  The dominant term.  A quote's bid/ask half-width is the
/// market's own statement about how precisely it knows the price, so
/// `w_spread = 1 / sigma_spread^2` where `sigma_spread` is the half-spread
/// converted into *volatility* points by dividing by vega.  That conversion is
/// the important part: a half-spread of 0.05 on a 2.00 near-the-money option
/// and on a 0.10 wing option are wildly different amounts of volatility
/// uncertainty, and only the vega-divided version compares them correctly.
///
/// **Liquidity.**  Volume and open interest, as a soft confirmation that the
/// quote is real.  Weak on purpose: a quote with no volume may still be a
/// perfectly good two-sided market, and down-weighting it hard would discard
/// the wings of every surface.
///
/// **Moneyness.**  A Gaussian kernel in log-moneyness scaled by the slice's
/// own standard deviation (sqrt of ATM total variance), *not* by a fixed
/// percentage.  A fixed-percentage kernel treats 10% out of the money as
/// equally far at one week and at two years, when in standard deviations it is
/// four times further at the short end.  The kernel is deliberately wide --
/// the point is to stop the wings dominating, not to ignore them.
///
/// **Expiry.**  Optional emphasis on a maturity range, for a caller who cares
/// about a particular bucket.  Defaults to uniform.
///
/// **Staleness.**  A linear decay to zero over the configured window, so a
/// quote does not fall off a cliff at an arbitrary age.
///
/// ## Normalisation
///
/// Weights are normalised so the *mean* weight over used quotes is one.  That
/// keeps the reported objective comparable between slices with different quote
/// counts and makes `rms_residual` interpretable as a typical volatility error
/// rather than an arbitrary scale.  It also means the LM damping and the
/// convergence tolerances do not need re-tuning per slice.

#include <span>
#include <vector>

#include "volatility_lab/options/quote.hpp"

namespace vl {

/// Which residual the calibrator minimises.
///
/// This choice matters as much as the weights and is often conflated with
/// them.
enum class ResidualKind : std::uint8_t {
    /// Difference in implied volatility.  The default, and the right choice
    /// for a surface fit: it is what a trader reads, it is scale-free across
    /// strikes and maturities, and it does not need a weighting scheme to stop
    /// expensive options dominating cheap ones.
    Volatility = 0,

    /// Difference in total variance, w = sigma^2 T.  The quantity the slice
    /// models are *linear* in, which is what makes the quasi-explicit SVI
    /// reduction exact.  Fitting in w rather than sigma slightly over-weights
    /// long maturities (w scales with T) and that is corrected by the expiry
    /// weight.
    TotalVariance = 1,

    /// Difference in price.  Occasionally wanted -- it is the quantity a P&L
    /// is denominated in -- but a poor default: prices vary over four orders
    /// of magnitude across a slice, so a price-space fit without weights is
    /// entirely determined by the three most expensive options.
    Price = 2,

    /// Price difference divided by vega, i.e. price error expressed in
    /// volatility points.  Equivalent to `Volatility` to first order but
    /// better behaved where vega is small, because it degrades smoothly
    /// instead of dividing by a near-zero sensitivity.
    VegaScaledPrice = 3
};

[[nodiscard]] constexpr const char* to_string(ResidualKind k) noexcept {
    switch (k) {
        case ResidualKind::Volatility:      return "volatility";
        case ResidualKind::TotalVariance:   return "total-variance";
        case ResidualKind::Price:           return "price";
        case ResidualKind::VegaScaledPrice: return "vega-scaled-price";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct WeightConfig {
    bool use_spread = true;
    bool use_liquidity = true;
    bool use_moneyness = true;
    bool use_staleness = true;

    /// Floor on the half-spread in volatility points when converting to a
    /// weight, so a locked market does not get infinite weight.  0.0005 is
    /// half a basis point of volatility: tighter than any real market quotes.
    double min_half_spread_vol = 0.0005;

    /// Cap on the resulting weight relative to the median, so one
    /// suspiciously tight quote cannot dominate a slice.  This is the guard
    /// against the failure mode inverse-variance weighting is prone to: the
    /// estimator trusts whichever observation *claims* the smallest error.
    double max_weight_ratio = 50.0;

    /// Width of the moneyness kernel, in standard deviations.  2.0 is wide:
    /// at 2 sd the weight is exp(-0.5) = 0.61, so the wings are moderated
    /// rather than suppressed.
    double moneyness_kernel_sd = 2.0;

    /// Weight multiplier for quotes marked `Degraded`.
    double degraded_multiplier = 0.25;

    /// Liquidity reference levels.  A quote at or above these gets full
    /// liquidity weight; below, it is scaled by a square root so the penalty
    /// is gentle.
    double reference_volume = 100.0;
    double reference_open_interest = 500.0;

    /// Maximum age before the staleness weight reaches zero.
    double stale_window_seconds = 300.0;

    ResidualKind residual = ResidualKind::Volatility;
};

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

/// The individual factors for one quote, so the weighting can be explained
/// rather than merely applied.
///
/// Returned and stored because "why is this quote barely affecting the fit"
/// is a question that gets asked, and a single combined number cannot answer
/// it.  `volatility-lab surface-fit --explain-weights` prints these.
struct WeightBreakdown {
    double spread = 1.0;
    double liquidity = 1.0;
    double moneyness = 1.0;
    double staleness = 1.0;
    double status = 1.0;
    double combined = 1.0;    ///< the product, before normalisation
    double normalised = 1.0;  ///< after normalisation to a unit mean

    /// Half-spread expressed in volatility points -- the quantity the spread
    /// weight is actually built from, kept for diagnostics because it is the
    /// most informative single number about a quote's quality.
    double half_spread_vol = 0.0;
};

/// Compute weights for a set of quotes from one slice.
///
/// `atm_total_variance` is used to scale the moneyness kernel; pass the slice's
/// ATM total variance, or 0 to disable moneyness weighting for this call.
///
/// Writes into `quotes[i].weight` and returns the breakdowns in the same order.
[[nodiscard]] std::vector<WeightBreakdown> assign_weights(std::span<OptionQuote> quotes,
                                                          double atm_total_variance,
                                                          const WeightConfig& cfg = {});

/// Weights for a whole book, slice by slice.
///
/// Normalisation is per slice, not global.  A global normalisation would let a
/// slice with 60 quotes outvote one with 12 simply by having more of them,
/// which is the opposite of what is wanted: each expiry should get its own
/// shape fitted on its own terms.
[[nodiscard]] std::vector<WeightBreakdown> assign_weights_by_slice(
    std::span<OptionQuote> quotes, const WeightConfig& cfg = {});

/// The residual the calibrator should minimise for one quote, given a model
/// total variance at that quote's moneyness.
///
/// Centralised here rather than written out in each calibrator so that the
/// `ResidualKind` choice is applied consistently -- and so that adding a kind
/// does not mean finding every place that fits something.
[[nodiscard]] double quote_residual(const OptionQuote& q, double model_total_variance,
                                    ResidualKind kind) noexcept;

/// d(residual)/d(model total variance), for the analytic Jacobians.
[[nodiscard]] double quote_residual_dw(const OptionQuote& q, double model_total_variance,
                                       ResidualKind kind) noexcept;

}  // namespace vl
