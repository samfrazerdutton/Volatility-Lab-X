// SPDX-License-Identifier: MIT
#pragma once
/// \file ssvi.hpp
/// \brief SSVI: the surface-level parameterisation with *provable* absence of
///        static arbitrage.
///
/// ## Why a third model
///
/// Fitting SVI expiry by expiry gives a good smile at each maturity and says
/// nothing whatsoever about the relationship between them.  Two independently
/// excellent SVI slices routinely cross -- total variance decreasing in T at
/// some strike -- which is calendar arbitrage, and no amount of per-slice
/// quality control detects it.  Repairing it afterwards means re-fitting with
/// inequality constraints coupling every pair of expiries.
///
/// SSVI (Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces*,
/// Quantitative Finance 2014) takes the opposite approach: it parameterises
/// the whole surface at once, in a form whose no-arbitrage conditions are
/// **closed-form inequalities on two global parameters**.  Satisfy them and the
/// surface is free of both butterfly and calendar arbitrage everywhere, for
/// every strike and every maturity, as a theorem rather than as a grid check.
///
/// That is the trade: SSVI has far fewer degrees of freedom than a stack of
/// independent SVI slices and will fit a given snapshot less closely, but what
/// it produces is guaranteed admissible.  Both are offered, and
/// `SurfaceQuality` reports the fit error of each so the trade is visible
/// rather than assumed.
///
/// ## The model
///
///     w(k, theta) = (theta/2) * ( 1 + rho*phi(theta)*k
///                                 + sqrt( (phi(theta)*k + rho)^2 + 1 - rho^2 ) )
///
/// where
///
///     theta(T) = w(0, T)   is the ATM total variance term structure, and
///     phi(theta)           controls how fast the smile flattens with maturity.
///
/// At each fixed T this *is* a raw SVI slice -- the mapping is exact and given
/// by `ssvi_slice_params` below -- so everything built on SVI (the pricer, the
/// Greeks, the Durrleman check) works unchanged.  What SSVI adds is that the
/// slices are generated from a shared structure instead of being independent.
///
/// Two parameters are global: `rho` (skew) and whatever parameterises `phi`.
/// The ATM term structure theta(T) is taken from the data, which is the right
/// division of labour: theta is directly observable from the ATM quotes at
/// each expiry and needs no model, while the *shape* is what needs one.
///
/// ## The no-arbitrage conditions
///
/// From Gatheral-Jacquier, Theorems 4.1 and 4.2:
///
/// **Butterfly-free** (sufficient, and the standard operating condition):
///
///     theta * phi(theta) * (1 + |rho|) < 4        and
///     theta * phi(theta)^2 * (1 + |rho|) <= 4
///
/// **Calendar-free**:
///
///     d(theta)/dT >= 0                            and
///     0 <= d( theta * phi(theta) )/d(theta) <= (1/rho^2) * (1 + sqrt(1 - rho^2)) * phi(theta)
///
/// The first calendar condition is just "total ATM variance does not fall with
/// maturity", which is a property of the data.  The second constrains how the
/// smile may steepen, and it is the one a naive term-structure interpolation
/// violates.
///
/// These are checked by `ssvi_arbitrage_check`, which reports *which*
/// inequality failed and by how much -- a calibrator that only learns "not
/// admissible" cannot steer.

#include <cmath>
#include <cstdint>
#include <span>

#include "volatility_lab/core/config.hpp"
#include "volatility_lab/volatility/slice.hpp"
#include "volatility_lab/volatility/svi.hpp"

namespace vl {

// ---------------------------------------------------------------------------
// The phi function
// ---------------------------------------------------------------------------

/// Which functional form phi(theta) takes.
///
/// Both forms are from the original paper, and both are offered because they
/// encode different beliefs about the term structure of skew:
///
///  * **PowerLaw** -- phi(theta) = eta / ( theta^gamma * (1+theta)^(1-gamma) ).
///    Skew decays like a power of maturity.  This is the form that matches
///    equity-index surfaces well and the one the calibrator defaults to.  Note
///    the (1+theta)^(1-gamma) factor: with plain theta^(-gamma) the butterfly
///    condition fails for large theta, and this is the modification
///    Gatheral-Jacquier introduce (their eq. 4.4) to keep it satisfied for all
///    theta.  It is not cosmetic.
///
///  * **Heston** -- phi(theta) = (1/(lambda*theta)) * (1 - (1 - exp(-lambda*theta))
///    /(lambda*theta)).  The shape implied by the Heston model in the
///    long-maturity limit.  Useful as a sanity check: if a surface fits this
///    materially better than the power law, that is information about the
///    underlying dynamics rather than about the fit.
enum class SsviPhiKind : std::uint8_t { PowerLaw = 0, Heston = 1 };

[[nodiscard]] constexpr const char* to_string(SsviPhiKind k) noexcept {
    return k == SsviPhiKind::PowerLaw ? "power-law" : "heston";
}

/// Global SSVI parameters.  Three numbers for the entire surface.
struct SsviParams {
    double rho = -0.4;  ///< global skew, |rho| < 1
    double eta = 1.0;   ///< phi scale (PowerLaw) or unused (Heston)
    double gamma = 0.5;   ///< phi decay exponent (PowerLaw), in [0, 1]
    double lambda = 1.0;  ///< Heston decay rate (Heston only), > 0
    SsviPhiKind phi_kind = SsviPhiKind::PowerLaw;

    /// Number of free parameters for the active phi form.  The calibrator
    /// needs this to size its Jacobian and to report degrees of freedom
    /// honestly in the quality report.
    [[nodiscard]] constexpr int num_params() const noexcept {
        return phi_kind == SsviPhiKind::PowerLaw ? 3 : 2;
    }
};

static_assert(std::is_trivially_copyable_v<SsviParams>);

/// phi(theta) and its derivative, which the calendar condition needs.
struct PhiValue {
    double phi = 0.0;
    double dphi_dtheta = 0.0;
};

[[nodiscard]] PhiValue ssvi_phi(const SsviParams& p, double theta) noexcept;

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Total variance at (k, theta).
[[nodiscard]] double ssvi_total_variance(const SsviParams& p, double theta,
                                         double k) noexcept;

/// Total variance and its k-derivatives at (k, theta).
[[nodiscard]] SliceJet ssvi_jet(const SsviParams& p, double theta, double k) noexcept;

/// The exact raw-SVI parameters of the SSVI slice at ATM total variance
/// `theta` and expiry `years`.
///
/// The correspondence (Gatheral-Jacquier section 4) is
///
///     a     = (theta/2) * (1 - rho^2)
///     b     = (theta/2) * phi
///     rho_s = rho
///     m     = -rho / phi
///     sigma = sqrt(1 - rho^2) / phi
///
/// Having this mapping means SSVI costs nothing downstream: the surface stores
/// SVI slices either way, and the only difference is where the five numbers
/// came from.  Everything that consumes a slice -- the pricer, the batch
/// kernels, the Greeks, the Durrleman check -- is shared.
[[nodiscard]] SviParams ssvi_slice_params(const SsviParams& p, double theta,
                                          double years) noexcept;

// ---------------------------------------------------------------------------
// The arbitrage conditions
// ---------------------------------------------------------------------------

/// Result of checking the Gatheral-Jacquier conditions, with the slack in each
/// inequality reported separately.
///
/// The slacks are what make this usable in a calibrator: a penalty
/// proportional to the violated slack gives a gradient that points back into
/// the admissible set, whereas a boolean gives nothing to descend.
struct SsviArbitrageCheck {
    bool butterfly_free = true;
    bool calendar_free = true;

    /// 4 - theta*phi*(1+|rho|).  Must be > 0.
    double butterfly_slack_1 = 0.0;
    /// 4 - theta*phi^2*(1+|rho|).  Must be >= 0.
    double butterfly_slack_2 = 0.0;
    /// d(theta*phi)/d(theta).  Must lie in [0, upper_calendar_bound].
    double calendar_derivative = 0.0;
    double calendar_upper_bound = 0.0;

    [[nodiscard]] bool admissible() const noexcept {
        return butterfly_free && calendar_free;
    }

    /// Total magnitude of violation, zero when admissible.  Used as the
    /// penalty term in constrained calibration.
    [[nodiscard]] double violation() const noexcept;
};

/// Check the conditions at one point of the term structure.
[[nodiscard]] SsviArbitrageCheck ssvi_arbitrage_check(const SsviParams& p,
                                                      double theta) noexcept;

/// Check across a whole term structure, returning the worst case.
[[nodiscard]] SsviArbitrageCheck ssvi_arbitrage_check_term(const SsviParams& p,
                                                           std::span<const double> thetas) noexcept;

/// Structural parameter bounds: |rho| < 1, eta > 0, gamma in [0, 1],
/// lambda > 0.
[[nodiscard]] bool ssvi_parameters_admissible(const SsviParams& p) noexcept;

/// Project onto the structural bounds (not the arbitrage conditions, which are
/// handled by penalty since they couple to theta).
[[nodiscard]] SsviParams ssvi_project_to_admissible(SsviParams p) noexcept;

}  // namespace vl
