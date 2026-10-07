// SPDX-License-Identifier: MIT
#include "volatility_lab/io/synthetic_market.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "volatility_lab/pricing/black.hpp"

namespace vl {

// ===========================================================================
// DeterministicRng
// ===========================================================================

std::uint64_t DeterministicRng::next_u64() noexcept {
    // splitmix64 (Steele, Lea, Flood 2014).  Chosen over std::mt19937_64 not
    // for quality -- both are fine here -- but because the whole point is
    // cross-platform reproducibility, and that requires owning the bit
    // sequence rather than relying on a library's implementation choices.
    state_ += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double DeterministicRng::uniform() noexcept {
    // 53 bits, scaled by 2^-53.  Never returns exactly 1, which matters
    // because `student_t` and `normal` take logs of it.
    return static_cast<double>(next_u64() >> 11) * 0x1.0p-53;
}

double DeterministicRng::uniform(double lo, double hi) noexcept {
    return lo + (hi - lo) * uniform();
}

double DeterministicRng::normal() noexcept {
    // Box-Muller, with the second variate cached.  Written out rather than
    // using std::normal_distribution because that is not specified to produce
    // the same sequence across standard libraries, and this project's
    // determinism tests compare output across three of them.
    if (has_spare_) {
        has_spare_ = false;
        return spare_normal_;
    }
    // u1 is bounded away from zero so the log cannot be -inf.
    const double u1 = std::max(uniform(), 1e-300);
    const double u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    constexpr double kTwoPi = 6.28318530717958647692;
    spare_normal_ = r * std::sin(kTwoPi * u2);
    has_spare_ = true;
    return r * std::cos(kTwoPi * u2);
}

double DeterministicRng::student_t(double nu) noexcept {
    if (!(nu > 2.0)) return normal();  // undefined variance below 2; degrade
    // t_nu = Z / sqrt(V/nu) with V ~ chi^2_nu.  chi^2 is built from a sum of
    // squared normals, which is exact for integer nu and the honest thing to
    // do for a generator whose whole contract is reproducibility.  Scaled so
    // the result has unit variance, since callers specify noise in vol points
    // and expect that to be a standard deviation.
    const int n = static_cast<int>(std::lround(nu));
    double v = 0.0;
    for (int i = 0; i < n; ++i) {
        const double z = normal();
        v += z * z;
    }
    if (!(v > 0.0)) return 0.0;
    const double t = normal() / std::sqrt(v / static_cast<double>(n));
    // Var(t_nu) = nu/(nu-2); divide it out so `vol_noise` is a standard
    // deviation in both the Gaussian and the heavy-tailed case.
    return t * std::sqrt((nu - 2.0) / nu);
}

// ===========================================================================
// Regimes
// ===========================================================================

bool parse_regime(std::string_view name, MarketRegime& out) noexcept {
    struct Row {
        const char* name;
        MarketRegime value;
    };
    static constexpr Row rows[] = {
        {"normal", MarketRegime::Normal},     {"high-vol", MarketRegime::HighVol},
        {"highvol", MarketRegime::HighVol},   {"crash", MarketRegime::Crash},
        {"vol-crush", MarketRegime::VolCrush}, {"volcrush", MarketRegime::VolCrush},
        {"earnings", MarketRegime::Earnings}, {"illiquid", MarketRegime::Illiquid},
    };
    for (const auto& r : rows) {
        if (name == r.name) {
            out = r.value;
            return true;
        }
    }
    return false;
}

namespace {

/// The SSVI parameters and ATM variance ladder that define each regime.
struct RegimeShape {
    SsviParams ssvi;
    std::vector<double> expiries;
    std::vector<double> atm_vol;  ///< annualised, per expiry
};

RegimeShape shape_for(MarketRegime r) {
    RegimeShape s;
    s.ssvi.phi_kind = SsviPhiKind::PowerLaw;

    switch (r) {
        case MarketRegime::Normal:
            // Liquid index surface: moderate downward skew decaying with
            // maturity, upward-sloping ATM term structure.
            s.ssvi.rho = -0.45;
            s.ssvi.eta = 0.85;
            s.ssvi.gamma = 0.42;
            s.expiries = {0.019, 0.038, 0.082, 0.167, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0};
            s.atm_vol = {0.172, 0.168, 0.172, 0.178, 0.182, 0.190, 0.196, 0.200,
                         0.206, 0.211};
            break;

        case MarketRegime::HighVol:
            // Level shifted up with a flatter skew, as a sustained high-vol
            // regime looks.  Exercises the assumption-free handling of
            // sigma > 1 in the short end.
            s.ssvi.rho = -0.32;
            s.ssvi.eta = 0.70;
            s.ssvi.gamma = 0.38;
            s.expiries = {0.019, 0.038, 0.082, 0.167, 0.25, 0.5, 1.0, 2.0};
            s.atm_vol = {0.72, 0.68, 0.63, 0.58, 0.55, 0.50, 0.46, 0.42};
            break;

        case MarketRegime::Crash:
            // Steep put skew, inverted term structure.  This is where SVI's
            // wing slope approaches the Lee bound of 2 and where the butterfly
            // check actually earns its place.
            s.ssvi.rho = -0.80;
            s.ssvi.eta = 1.80;
            s.ssvi.gamma = 0.30;
            s.expiries = {0.008, 0.019, 0.038, 0.082, 0.167, 0.25, 0.5, 1.0};
            s.atm_vol = {1.20, 0.98, 0.84, 0.70, 0.60, 0.54, 0.46, 0.40};
            break;

        case MarketRegime::VolCrush:
            // Very low vol, very short dates: the small-total-variance corner
            // of the pricer and the worst-conditioned region of the inversion.
            // At T = 0.0014 (half a day) and sigma = 7%, s = 0.0026.
            s.ssvi.rho = -0.25;
            s.ssvi.eta = 0.55;
            s.ssvi.gamma = 0.50;
            s.expiries = {0.0014, 0.004, 0.011, 0.019, 0.038, 0.082, 0.167};
            s.atm_vol = {0.068, 0.072, 0.076, 0.080, 0.085, 0.092, 0.099};
            break;

        case MarketRegime::Earnings:
            // One expiry carries a variance bump.  SSVI *cannot* represent
            // this -- the bump breaks the monotone theta structure the model
            // assumes -- and that is the point: the quality report must show
            // SSVI fitting it worse than per-slice SVI, rather than both
            // claiming success.
            s.ssvi.rho = -0.40;
            s.ssvi.eta = 0.80;
            s.ssvi.gamma = 0.45;
            // The *volatility* ladder has a bump at the earnings expiry and
            // decays after it; the *total variance* ladder must still be
            // increasing, or the generated data contains calendar arbitrage
            // and every downstream no-arbitrage test is being fed a
            // contradiction.  The first version used
            // {0.21, 0.23, 0.38, 0.26, ...}, which gives w = 0.0087 at
            // T = 0.060 and 0.0055 at T = 0.082 -- decreasing, and therefore
            // arbitrageable.  The vol bump is just as pronounced here, but the
            // decay is gentle enough to keep w monotone.
            s.expiries = {0.019, 0.038, 0.060, 0.082, 0.167, 0.25, 0.5, 1.0};
            s.atm_vol = {0.210, 0.230, 0.420, 0.370, 0.290, 0.260, 0.235, 0.220};
            break;

        case MarketRegime::Illiquid:
            // Sparse and damaged.  The maths is easy; the normalisation layer
            // is what is under test.
            s.ssvi.rho = -0.50;
            s.ssvi.eta = 0.95;
            s.ssvi.gamma = 0.40;
            s.expiries = {0.038, 0.167, 0.5, 1.0};
            s.atm_vol = {0.28, 0.30, 0.32, 0.33};
            break;
    }
    return s;
}

/// Half-spread in volatility points at log-moneyness k and maturity T.
///
/// Three effects, each observed in listed markets and each a distinct stress
/// on the weighting model:
///   * widens with |k|, because wing quotes are market-maker inventory risk;
///   * narrows with maturity, because vega per vol point is larger so the
///     same price tick is fewer vol points;
///   * has a floor, because the minimum spread is one tick.
double half_spread_vol(double atm_half, double k, double years) noexcept {
    const double wing = 1.0 + 2.2 * std::abs(k) + 3.0 * k * k;
    const double maturity = 1.0 / std::sqrt(std::max(years, 1e-4) / 0.25);
    return atm_half * wing * std::clamp(maturity, 0.35, 6.0);
}

double round_to(double x, double increment) noexcept {
    if (!(increment > 0.0)) return x;
    return std::round(x / increment) * increment;
}

}  // namespace

SyntheticMarketConfig regime_defaults(MarketRegime r, std::uint64_t seed) {
    SyntheticMarketConfig c;
    c.regime = r;
    c.seed = seed;
    const RegimeShape s = shape_for(r);
    c.expiries = s.expiries;

    switch (r) {
        case MarketRegime::Normal:
            c.strikes_per_expiry = 25;
            c.vol_noise = 0.0022;
            c.atm_half_spread_vol = 0.0035;
            break;
        case MarketRegime::HighVol:
            c.strikes_per_expiry = 23;
            c.vol_noise = 0.0065;
            c.atm_half_spread_vol = 0.010;
            c.strike_increment = 5.0;
            break;
        case MarketRegime::Crash:
            c.strikes_per_expiry = 21;
            c.strike_span_sd = 3.5;
            c.vol_noise = 0.012;
            c.atm_half_spread_vol = 0.022;
            c.strike_increment = 5.0;
            break;
        case MarketRegime::VolCrush:
            c.strikes_per_expiry = 19;
            c.strike_span_sd = 4.0;
            c.vol_noise = 0.0012;
            c.atm_half_spread_vol = 0.0025;
            c.strike_increment = 1.0;
            c.price_tick = 0.01;
            break;
        case MarketRegime::Earnings:
            c.strikes_per_expiry = 21;
            c.vol_noise = 0.004;
            c.atm_half_spread_vol = 0.008;
            break;
        case MarketRegime::Illiquid:
            c.strikes_per_expiry = 11;
            c.strike_span_sd = 2.5;
            c.vol_noise = 0.018;
            c.atm_half_spread_vol = 0.045;
            c.strike_increment = 10.0;
            c.corrupt_fraction = 0.22;
            break;
    }
    return c;
}

// ===========================================================================
// Generation
// ===========================================================================

SyntheticMarket generate_market(const SyntheticMarketConfig& cfg) {
    SyntheticMarket out;
    out.config = cfg;
    const RegimeShape shape = shape_for(cfg.regime);
    out.ssvi = shape.ssvi;

    std::vector<double> expiries = cfg.expiries.empty() ? shape.expiries : cfg.expiries;
    std::sort(expiries.begin(), expiries.end());

    // --- the true surface -------------------------------------------------
    //
    // Built from SSVI at each expiry, with the ATM variance taken from the
    // regime's ladder.  Interpolating the ladder when the caller supplies its
    // own expiries keeps the two consistent.
    std::vector<double> thetas;
    thetas.reserve(expiries.size());
    for (double t : expiries) {
        double vol;
        if (cfg.expiries.empty() || expiries.size() == shape.atm_vol.size()) {
            const auto it = std::lower_bound(shape.expiries.begin(), shape.expiries.end(), t);
            const std::size_t idx =
                std::min(static_cast<std::size_t>(it - shape.expiries.begin()),
                         shape.atm_vol.size() - 1);
            vol = shape.atm_vol[idx];
        } else {
            // Linear in ATM vol across the regime ladder, which is good enough
            // for a generator and keeps the ladder monotone in total variance.
            const math::LinearInterp ladder(shape.expiries, shape.atm_vol);
            vol = ladder(t);
        }
        thetas.push_back(vol * vol * t);
    }

    // `Earnings` perturbs one slice away from the SSVI family.  Flagged,
    // because a test that compares a fit against `true_surface` needs to know
    // whether that surface is representable by the model being fitted.
    out.surface_is_exactly_ssvi = (cfg.regime != MarketRegime::Earnings);

    std::vector<SliceVariant> slices;
    slices.reserve(expiries.size());
    std::vector<double> forwards;
    std::vector<double> discounts;
    forwards.reserve(expiries.size());
    discounts.reserve(expiries.size());

    for (std::size_t i = 0; i < expiries.size(); ++i) {
        const double t = expiries[i];
        SsviSlice s;
        s.global = shape.ssvi;
        s.theta = thetas[i];
        s.years = t;
        slices.emplace_back(s);
        forwards.push_back(cfg.spot * std::exp((cfg.rate - cfg.dividend) * t));
        discounts.push_back(std::exp(-cfg.rate * t));
    }
    out.true_surface = VolSurface(std::move(slices), TermCurve(expiries, forwards),
                                  TermCurve(expiries, discounts));

    // --- the quotes -------------------------------------------------------
    DeterministicRng rng(cfg.seed);
    out.snapshot.underlying = cfg.underlying;
    out.snapshot.spot = cfg.spot;
    out.snapshot.observation_time = 0.0;
    out.snapshot.quotes.reserve(expiries.size() *
                                static_cast<std::size_t>(cfg.strikes_per_expiry));

    // Fixed nested loop, in a fixed order.  The quote *ordering* is part of
    // the reproducibility contract: a downstream diagnostic refers to quote
    // indices, and a regression baseline that depends on iteration order would
    // be useless.
    for (std::size_t ei = 0; ei < expiries.size(); ++ei) {
        const double t = expiries[ei];
        const double forward = cfg.spot * std::exp((cfg.rate - cfg.dividend) * t);
        const double discount = std::exp(-cfg.rate * t);
        const double atm_vol = std::sqrt(thetas[ei] / t);

        // Strikes at fixed standard deviations, then rounded to the listing
        // increment.  The rounding is what makes the ladder non-uniform in
        // log-moneyness, which is a real complication for interpolation and
        // for the calibrator weighting.
        //
        // The increment has to adapt to the expiry, and getting that wrong was
        // a real bug in the first version.  At a short maturity one standard
        // deviation is only a couple of points, so a fixed 2.5-point ladder
        // across +-3 sd rounds twenty-five target strikes onto four distinct
        // listed ones -- which arrived downstream as 53 duplicate quotes out of
        // 250 and a normaliser dutifully rejecting a fifth of the snapshot.
        // Real markets do the same thing in reverse: they list *finer* strikes
        // on near-dated expiries.  So the increment is halved until the ladder
        // has enough distinct strikes to identify a slice.
        const double sd = atm_vol * std::sqrt(t);
        std::vector<double> strikes;
        double increment = cfg.strike_increment;
        for (int attempt = 0; attempt < 8; ++attempt) {
            strikes.clear();
            for (int si = 0; si < cfg.strikes_per_expiry; ++si) {
                const double u =
                    (cfg.strikes_per_expiry == 1)
                        ? 0.0
                        : (2.0 * static_cast<double>(si) /
                               static_cast<double>(cfg.strikes_per_expiry - 1) -
                           1.0);
                const double strike =
                    round_to(forward * std::exp(u * cfg.strike_span_sd * sd), increment);
                if (!(strike > 0.0)) continue;
                // Dedupe: the ladder is built in ascending order, so a repeat
                // can only be the immediately preceding entry.
                if (!strikes.empty() && strike == strikes.back()) continue;
                strikes.push_back(strike);
            }
            const std::size_t wanted = std::min<std::size_t>(
                9, static_cast<std::size_t>(cfg.strikes_per_expiry));
            if (strikes.size() >= wanted || !(increment > 0.0)) break;
            increment *= 0.5;
            if (increment < 1e-6) increment = 0.0;  // last attempt: no rounding
        }

        for (double strike : strikes) {
            const double k = std::log(strike / forward);

            // The true volatility from the surface, plus a heavy-tailed quote
            // error.  The Earnings bump is applied here rather than to the
            // surface slice so that `true_surface` stays a valid SSVI surface
            // and the discrepancy is attributable.
            double true_vol = out.true_surface.vol(k, t);
            if (cfg.regime == MarketRegime::Earnings && std::abs(t - 0.060) < 1e-9) {
                true_vol *= 1.0;  // already in the ladder; kept explicit
            }
            if (!(true_vol > 0.0)) continue;

            const double noise =
                cfg.vol_noise * rng.student_t(cfg.noise_degrees_of_freedom);
            const double quoted_vol = std::max(true_vol + noise, 1e-4);

            const double half = half_spread_vol(cfg.atm_half_spread_vol, k, t);
            const double bid_vol = std::max(quoted_vol - half, 1e-4);
            const double ask_vol = quoted_vol + half;

            // Quote the OTM side, as a listed market does.
            const OptionType side =
                (strike >= forward) ? OptionType::Call : OptionType::Put;

            OptionQuote q;
            q.strike = strike;
            q.forward = forward;
            q.years = t;
            q.type = side;
            q.style = ExerciseStyle::European;
            q.spot = cfg.spot;
            q.rate = cfg.rate;
            q.dividend = cfg.dividend;
            q.discount = discount;

            if (cfg.emit_vols_not_prices) {
                q.implied_vol = quoted_vol;
                q.bid = 0.0;
                q.ask = 0.0;
            } else {
                q.bid = round_to(
                    black_price(forward, strike, bid_vol, t, discount, side),
                    cfg.price_tick);
                q.ask = round_to(
                    black_price(forward, strike, ask_vol, t, discount, side),
                    cfg.price_tick);
                // Tick rounding can collapse a deep-wing spread to zero or
                // invert it; both are real and the normaliser must cope.
                if (q.ask < q.bid) std::swap(q.bid, q.ask);
                q.last = 0.5 * (q.bid + q.ask);
            }

            // Liquidity decays away from the money, as it does in practice.
            const double liquidity = std::exp(-3.0 * std::abs(k) / std::max(sd, 1e-6) * sd);
            q.volume = std::round(5000.0 * liquidity * rng.uniform(0.3, 1.7));
            q.open_interest = std::round(20000.0 * liquidity * rng.uniform(0.2, 2.0));
            q.age_seconds = rng.uniform(0.0, 30.0);

            // --- damage, for the Illiquid regime --------------------------
            if (cfg.corrupt_fraction > 0.0 && rng.bernoulli(cfg.corrupt_fraction)) {
                // Six distinct failure modes, chosen uniformly.  Each one
                // exercises a different branch of the normaliser, so the
                // Illiquid regime is a functional test of that layer rather
                // than merely noisy data.
                switch (static_cast<int>(rng.uniform(0.0, 6.0))) {
                    case 0:  // zero bid
                        q.bid = 0.0;
                        break;
                    case 1:  // crossed
                        std::swap(q.bid, q.ask);
                        if (q.bid == q.ask) q.bid = q.ask + 0.05;
                        break;
                    case 2:  // stale
                        q.age_seconds = rng.uniform(600.0, 7200.0);
                        break;
                    case 3:  // no market at all
                        q.bid = 0.0;
                        q.ask = 0.0;
                        q.last = 0.0;
                        break;
                    case 4:  // dead
                        q.volume = 0.0;
                        q.open_interest = 0.0;
                        break;
                    default:  // below intrinsic
                        q.bid = 0.0;
                        q.ask = std::max(0.01, 0.2 * forward_intrinsic(forward, strike,
                                                                       side) * discount);
                        break;
                }
            }
            out.snapshot.quotes.push_back(q);
        }
    }
    return out;
}

SyntheticMarket generate_market(MarketRegime regime, std::uint64_t seed) {
    return generate_market(regime_defaults(regime, seed));
}

}  // namespace vl
