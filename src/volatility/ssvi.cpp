// SPDX-License-Identifier: MIT
#include "volatility_lab/volatility/ssvi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vl {

PhiValue ssvi_phi(const SsviParams& p, double theta) noexcept {
    PhiValue out;
    if (!(theta > 0.0) || !std::isfinite(theta)) {
        out.phi = 0.0;
        out.dphi_dtheta = 0.0;
        return out;
    }

    if (p.phi_kind == SsviPhiKind::PowerLaw) {
        // phi(theta) = eta / ( theta^gamma * (1+theta)^(1-gamma) )
        //
        // The (1+theta)^(1-gamma) factor is Gatheral-Jacquier eq. 4.4, not a
        // free embellishment: with plain eta*theta^(-gamma) the butterfly
        // condition theta*phi^2*(1+|rho|) <= 4 fails once theta grows, i.e.
        // the "arbitrage-free" parameterisation stops being arbitrage-free at
        // long maturities.  With the factor it holds for every theta > 0.
        const double g = p.gamma;
        const double t_g = std::pow(theta, g);
        const double o_g = std::pow(1.0 + theta, 1.0 - g);
        const double denom = t_g * o_g;
        if (!(denom > 0.0) || !std::isfinite(denom)) {
            out.phi = 0.0;
            out.dphi_dtheta = 0.0;
            return out;
        }
        out.phi = p.eta / denom;
        // d(log phi)/d(theta) = -gamma/theta - (1-gamma)/(1+theta)
        const double dlog = -g / theta - (1.0 - g) / (1.0 + theta);
        out.dphi_dtheta = out.phi * dlog;
    } else {
        // phi(theta) = (1/(lambda*theta)) * (1 - (1 - exp(-lambda*theta))/(lambda*theta))
        const double u = p.lambda * theta;
        if (!(u > 0.0)) {
            out.phi = 0.0;
            out.dphi_dtheta = 0.0;
            return out;
        }
        // Both the bracket and its derivative cancel badly for small u.
        //
        //   bracket(u) = 1 - (1 - exp(-u))/u = sum_{n>=1} (-1)^(n+1) u^n/(n+1)!
        //
        // The direct form subtracts (1-exp(-u))/u, which is ~1, from 1, so it
        // loses about log10(2/u) digits -- 3 at u = 2e-3, 6 at u = 2e-6.  And
        // short-dated slices are exactly where u = lambda*theta is small, so
        // this is the normal case rather than a corner.
        //
        // The series is used up to u = 0.5 and run to convergence rather than
        // truncated at a fixed order.  A four-term truncation was the first
        // attempt and is only good to u ~ 0.01: at u = 1 it is 3.3e-3 wrong,
        // which left a band around the switch point where *neither* form was
        // accurate.  Measured crossover in tests/numerical/slice_models.cpp.
        double bracket;      // 1 - (1 - exp(-u))/u
        double dbracket_du;  // d/du of the above
        if (u < 0.5) {
            // term_n = (-1)^(n+1) u^n/(n+1)!,  dterm_n = (-1)^(n+1) n u^(n-1)/(n+1)!
            double inv_fact = 0.5;  // 1/(n+1)! at n = 1
            double u_pow = u;       // u^n at n = 1
            double sign = 1.0;
            bracket = 0.0;
            dbracket_du = 0.0;
            for (int n = 1; n <= 24; ++n) {
                const double term = sign * u_pow * inv_fact;
                bracket += term;
                dbracket_du += sign * static_cast<double>(n) *
                               (u_pow / std::max(u, 1e-300)) * inv_fact;
                if (std::abs(term) <= 1e-18 * std::abs(bracket)) break;
                sign = -sign;
                u_pow *= u;
                inv_fact /= static_cast<double>(n + 2);
            }
        } else {
            const double e = std::exp(-u);
            bracket = 1.0 - (1.0 - e) / u;
            dbracket_du = (1.0 - e) / (u * u) - e / u;
        }
        out.phi = bracket / u;
        // phi = bracket(u)/u  =>  dphi/du = (dbracket*u - bracket)/u^2
        const double dphi_du = (dbracket_du * u - bracket) / (u * u);
        out.dphi_dtheta = dphi_du * p.lambda;
    }
    return out;
}

double ssvi_total_variance(const SsviParams& p, double theta, double k) noexcept {
    if (!(theta > 0.0)) return 0.0;
    const double phi = ssvi_phi(p, theta).phi;
    const double pk = phi * k;
    const double u = pk + p.rho;
    const double root = std::sqrt(std::fma(u, u, 1.0 - p.rho * p.rho));
    return 0.5 * theta * (1.0 + p.rho * pk + root);
}

SliceJet ssvi_jet(const SsviParams& p, double theta, double k) noexcept {
    SliceJet j;
    if (!(theta > 0.0)) return j;
    const double phi = ssvi_phi(p, theta).phi;
    const double rho = p.rho;
    const double pk = phi * k;
    const double u = pk + rho;
    const double q = 1.0 - rho * rho;
    const double root = std::sqrt(std::fma(u, u, q));
    const double half_theta = 0.5 * theta;

    j.w = half_theta * (1.0 + rho * pk + root);
    if (root > 0.0) {
        // dw/dk = (theta/2) * phi * (rho + u/root)
        j.dw = half_theta * phi * (rho + u / root);
        // d2w/dk2 = (theta/2) * phi^2 * q / root^3
        j.d2w = half_theta * phi * phi * q / (root * root * root);
    }
    return j;
}

SviParams ssvi_slice_params(const SsviParams& p, double theta, double years) noexcept {
    SviParams s;
    s.years = years;
    if (!(theta > 0.0)) {
        s.a = 0.0;
        s.b = 0.0;
        s.rho = p.rho;
        s.m = 0.0;
        s.sigma = 1e-8;
        return s;
    }
    const double phi = ssvi_phi(p, theta).phi;
    const double q = std::max(0.0, 1.0 - p.rho * p.rho);
    if (!(phi > 0.0)) {
        // Degenerate: the smile has flattened completely.  A flat slice at the
        // ATM variance is the correct limit.
        s.a = theta;
        s.b = 0.0;
        s.rho = p.rho;
        s.m = 0.0;
        s.sigma = 1.0;
        return s;
    }
    s.a = 0.5 * theta * q;
    s.b = 0.5 * theta * phi;
    s.rho = p.rho;
    s.m = -p.rho / phi;
    s.sigma = std::sqrt(q) / phi;
    // sigma == 0 when |rho| == 1, which the structural bounds exclude; guard
    // anyway so a degenerate input cannot produce a division by zero
    // downstream.
    if (!(s.sigma > 0.0)) s.sigma = 1e-12;
    return s;
}

// ---------------------------------------------------------------------------
// The arbitrage conditions
// ---------------------------------------------------------------------------

double SsviArbitrageCheck::violation() const noexcept {
    double v = 0.0;
    if (butterfly_slack_1 < 0.0) v += -butterfly_slack_1;
    if (butterfly_slack_2 < 0.0) v += -butterfly_slack_2;
    if (calendar_derivative < 0.0) v += -calendar_derivative;
    if (calendar_derivative > calendar_upper_bound) {
        v += calendar_derivative - calendar_upper_bound;
    }
    return v;
}

SsviArbitrageCheck ssvi_arbitrage_check(const SsviParams& p, double theta) noexcept {
    SsviArbitrageCheck c;
    if (!(theta > 0.0) || !std::isfinite(theta)) {
        c.butterfly_free = false;
        c.calendar_free = false;
        c.butterfly_slack_1 = -1.0;
        c.butterfly_slack_2 = -1.0;
        return c;
    }

    const PhiValue pv = ssvi_phi(p, theta);
    const double phi = pv.phi;
    const double abs_rho = std::abs(p.rho);

    // --- butterfly (Gatheral-Jacquier Thm 4.2) ---------------------------
    //   theta*phi*(1+|rho|) < 4   and   theta*phi^2*(1+|rho|) <= 4
    c.butterfly_slack_1 = 4.0 - theta * phi * (1.0 + abs_rho);
    c.butterfly_slack_2 = 4.0 - theta * phi * phi * (1.0 + abs_rho);
    c.butterfly_free = (c.butterfly_slack_1 > 0.0) && (c.butterfly_slack_2 >= 0.0);

    // --- calendar (Gatheral-Jacquier Thm 4.1) -----------------------------
    //   0 <= d(theta*phi)/d(theta) <= (1/rho^2)(1 + sqrt(1-rho^2)) phi
    //
    // The left inequality says the "wing scale" theta*phi must not decrease
    // with maturity; the right bounds how fast it may grow.  d(theta*phi)/dtheta
    // = phi + theta*dphi/dtheta.
    c.calendar_derivative = phi + theta * pv.dphi_dtheta;
    const double rho2 = p.rho * p.rho;
    if (rho2 < 1e-300) {
        // rho = 0: the bound is vacuous (the 1/rho^2 factor diverges), so the
        // only constraint is the lower one.  Reported as a large finite bound
        // rather than infinity so that `violation()` stays arithmetic.
        c.calendar_upper_bound = std::numeric_limits<double>::max();
    } else {
        c.calendar_upper_bound =
            (1.0 / rho2) * (1.0 + std::sqrt(std::max(0.0, 1.0 - rho2))) * phi;
    }
    c.calendar_free =
        (c.calendar_derivative >= 0.0) && (c.calendar_derivative <= c.calendar_upper_bound);
    return c;
}

SsviArbitrageCheck ssvi_arbitrage_check_term(const SsviParams& p,
                                             std::span<const double> thetas) noexcept {
    SsviArbitrageCheck worst;
    worst.butterfly_slack_1 = std::numeric_limits<double>::infinity();
    worst.butterfly_slack_2 = std::numeric_limits<double>::infinity();
    double worst_violation = -1.0;

    for (double theta : thetas) {
        const SsviArbitrageCheck c = ssvi_arbitrage_check(p, theta);
        worst.butterfly_slack_1 = std::min(worst.butterfly_slack_1, c.butterfly_slack_1);
        worst.butterfly_slack_2 = std::min(worst.butterfly_slack_2, c.butterfly_slack_2);
        worst.butterfly_free = worst.butterfly_free && c.butterfly_free;
        worst.calendar_free = worst.calendar_free && c.calendar_free;
        const double v = c.violation();
        if (v > worst_violation) {
            worst_violation = v;
            worst.calendar_derivative = c.calendar_derivative;
            worst.calendar_upper_bound = c.calendar_upper_bound;
        }
    }
    // The first calendar condition, d(theta)/dT >= 0, is a property of the
    // supplied term structure rather than of the parameters.  Checked here so
    // that a caller handing over a non-monotone theta learns about it.
    for (std::size_t i = 1; i < thetas.size(); ++i) {
        if (thetas[i] < thetas[i - 1]) {
            worst.calendar_free = false;
            break;
        }
    }
    return worst;
}

bool ssvi_parameters_admissible(const SsviParams& p) noexcept {
    if (!std::isfinite(p.rho) || !(std::abs(p.rho) < 1.0)) return false;
    if (p.phi_kind == SsviPhiKind::PowerLaw) {
        if (!std::isfinite(p.eta) || !(p.eta > 0.0)) return false;
        if (!std::isfinite(p.gamma) || p.gamma < 0.0 || p.gamma > 1.0) return false;
    } else {
        if (!std::isfinite(p.lambda) || !(p.lambda > 0.0)) return false;
    }
    return true;
}

SsviParams ssvi_project_to_admissible(SsviParams p) noexcept {
    if (!std::isfinite(p.rho)) p.rho = 0.0;
    p.rho = std::clamp(p.rho, -1.0 + 1e-12, 1.0 - 1e-12);
    if (!std::isfinite(p.eta) || p.eta <= 0.0) p.eta = 1e-8;
    if (!std::isfinite(p.gamma)) p.gamma = 0.5;
    p.gamma = std::clamp(p.gamma, 0.0, 1.0);
    if (!std::isfinite(p.lambda) || p.lambda <= 0.0) p.lambda = 1e-8;
    return p;
}

}  // namespace vl
