// SPDX-License-Identifier: MIT
/// \file main.cpp
/// \brief volatility_lab_server -- a thin, local-only HTTP service wrapping
///        the existing C++ engine for the web frontend (Quant Compute
///        Workbench, Phase C vertical slice).
///
/// Every endpoint below calls directly into already-tested library
/// functions (`generate_market`, `normalize`, `assign_weights_by_slice`,
/// `calibrate_svi_slice`, `IncrementalEngine`, `full_rebuild`). No pricing,
/// calibration, or Greeks math is reimplemented here -- this file is
/// serialization and request handling only.
///
/// Binds to 127.0.0.1 only. There is no authentication because there is no
/// network exposure to authenticate against; do not change the bind
/// address without adding one.

#include "volatility_lab/core/build_info.hpp"
#include "volatility_lab/greeks/greeks.hpp"
#include "volatility_lab/io/synthetic_market.hpp"
#include "volatility_lab/options/normalize.hpp"
#include "volatility_lab/runtime/full_rebuild.hpp"
#include "volatility_lab/runtime/incremental_engine.hpp"
#include "volatility_lab/runtime/market_event.hpp"
#include "volatility_lab/runtime/state_hash.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <optional>

using json = nlohmann::json;
using namespace vl;

namespace {

// ---------------------------------------------------------------------------
// JSON serialization: real engine types -> wire format. Deliberately
// explicit (no reflection/macros) so every field crossing the boundary is
// visible at the call site.
// ---------------------------------------------------------------------------

json quote_to_json(const OptionQuote& q) {
    return json{
        {"years", q.years},
        {"strike", q.strike},
        {"type", q.type == OptionType::Call ? "call" : "put"},
        {"bid", q.bid},
        {"ask", q.ask},
        {"mid", q.mid},
        {"implied_vol", q.implied_vol},
        {"volume", q.volume},
        {"open_interest", q.open_interest},
        {"age_seconds", q.age_seconds},
        {"status", to_string(q.status)},
        {"weight", q.weight},
        {"log_moneyness", q.log_moneyness},
        {"forward", q.forward},
    };
}

json greeks_to_json(const OptionGreeks& g) {
    return json{{"price", g.price},   {"delta", g.delta}, {"gamma", g.gamma},
               {"vega", g.vega},     {"theta", g.theta}, {"rho", g.rho},
               {"vanna", g.vanna},   {"volga", g.volga}, {"charm", g.charm},
               {"speed", g.speed}};
}

json fit_to_json(const SviFitResult& fit) {
    return json{
        {"status", to_string(fit.status)},
        {"ok", fit.ok()},
        {"objective", fit.objective},
        {"rms_vol_error", fit.rms_vol_error},
        {"max_vol_error", fit.max_vol_error},
        {"quotes_used", fit.quotes_used},
        {"quotes_available", fit.quotes_available},
        {"inner_solves", fit.inner_solves},
        {"outer_iterations", fit.outer_iterations},
        {"active_constraints", fit.active_constraints},
    };
}

json report_to_json(const RecomputeReport& r) {
    return json{
        {"total_nodes", r.total_nodes},
        {"recomputed_nodes", r.recomputed_nodes},
        {"reused_nodes", r.reused_nodes},
        {"fraction_avoided", r.fraction_avoided()},
        {"quotes_total", r.quotes_total},
        {"quotes_examined", r.quotes_examined},
        {"calibrations_run", r.calibrations_run},
        {"latency_us", std::chrono::duration<double, std::micro>(r.latency).count()},
    };
}

json portfolio_to_json(const PortfolioGreeks& p) {
    return json{{"value", p.value}, {"delta", p.delta}, {"gamma", p.gamma},
               {"vega", p.vega},   {"theta", p.theta}, {"rho", p.rho}};
}

json pnl_to_json(const PnLAttribution& p) {
    return json{
        {"total_exact_pnl", p.total_exact_pnl}, {"base_value", p.base_value},
        {"new_value", p.new_value},             {"spot_pnl", p.spot_pnl},
        {"vol_pnl", p.vol_pnl},                 {"rate_pnl", p.rate_pnl},
        {"theta_pnl", p.theta_pnl},             {"gamma_pnl", p.gamma_pnl},
        {"residual", p.residual},
    };
}

// ---------------------------------------------------------------------------
// Application state: one engine, protected by one mutex. A real market has
// one book, not one per HTTP connection -- every request serializes on
// this lock, which is correct (not merely convenient) for a service whose
// entire point is "the same engine state, observed and mutated
// consistently" (section 8's "prevent concurrent updates from corrupting
// shared state").
// ---------------------------------------------------------------------------

VolSurface make_baseline_surface(std::span<const double> tenors) {
    std::vector<SliceVariant> slices;
    for (double T : tenors) {
        SviParams p;
        p.years = T;
        p.b = 0.08;
        p.rho = -0.4;
        p.m = 0.0;
        p.sigma = 0.13;
        p.a = 0.20 * 0.20 * T - p.b * std::sqrt(p.m * p.m + p.sigma * p.sigma);
        slices.emplace_back(svi_project_to_admissible(p));
    }
    return VolSurface(std::move(slices), TermCurve::flat(100.0), TermCurve::flat(1.0));
}

struct EventLogEntry {
    std::int64_t timestamp_ms;
    std::string type;
    std::string description;
    std::string state_hash;
};

class AppState {
  public:
    std::mutex mu;

    std::optional<SyntheticMarket> market;
    std::vector<double> distinct_years;
    IncrementalEngine::Config config;
    std::optional<IncrementalEngine> engine;
    RecomputeReport last_report;
    bool has_last_report = false;
    std::chrono::nanoseconds last_full_rebuild_latency{0};
    bool has_last_full_rebuild = false;
    std::vector<EventLogEntry> event_log;

    void log(std::string type, std::string description, std::string hash = "") {
        const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
        event_log.push_back({now, std::move(type), std::move(description), std::move(hash)});
        if (event_log.size() > 200) event_log.erase(event_log.begin());
    }

    /// Not the replay/state-hash machinery's own fingerprint (that hashes a
    /// `MarketEventStream`, a different concept) -- a display-only
    /// fingerprint of the engine's *current* quote book, built from the
    /// same `StateHasher` primitive `runtime/state_hash.hpp` already
    /// validates, so a viewer can see "the book changed" without this file
    /// inventing a second hashing scheme.
    std::string book_fingerprint() const {
        StateHasher h;
        for (const auto& q : engine->quotes()) {
            h.combine(q.years);
            h.combine(q.strike);
            h.combine(static_cast<std::int64_t>(q.type));
            h.combine(q.mid);
        }
        return h.finish().to_hex();
    }
};

AppState g_state;

void load_market(const std::string& regime_name) {
    MarketRegime regime = MarketRegime::Normal;
    (void)parse_regime(regime_name, regime);

    g_state.market = generate_market(regime);
    auto norm = normalize(g_state.market->snapshot);
    (void)assign_weights_by_slice(norm.quotes);

    g_state.distinct_years.clear();
    for (const auto& q : norm.quotes) {
        if (std::find_if(g_state.distinct_years.begin(), g_state.distinct_years.end(),
                         [&](double y) { return std::abs(y - q.years) < 1e-9; }) ==
            g_state.distinct_years.end()) {
            g_state.distinct_years.push_back(q.years);
        }
    }
    std::sort(g_state.distinct_years.begin(), g_state.distinct_years.end());

    g_state.config = IncrementalEngine::Config{};
    g_state.config.baseline_surface = make_baseline_surface(g_state.distinct_years);
    g_state.config.baseline_market =
        MarketPoint{g_state.market->snapshot.spot, 0.03, 0.0};

    // A small, deterministic sample book -- real positions the dependency
    // graph actually carries, not a decoration. Spans every expiry.
    const double spot = g_state.market->snapshot.spot;
    for (std::size_t i = 0; i < g_state.distinct_years.size(); ++i) {
        const double years = g_state.distinct_years[i];
        g_state.config.positions.push_back(
            Position{"pos_call_" + std::to_string(i), 10.0, 100.0, spot, years,
                     OptionType::Call});
        g_state.config.positions.push_back(
            Position{"pos_put_" + std::to_string(i), -5.0, 100.0, spot * 0.95, years,
                     OptionType::Put});
    }

    g_state.engine.emplace(norm.quotes, g_state.config);
    g_state.has_last_report = false;
    g_state.has_last_full_rebuild = false;
    g_state.event_log.clear();
    g_state.log("market_load", "Loaded " + std::string(to_string(regime)) + " sample market (" +
                                   std::to_string(norm.quotes.size()) + " quotes)",
               g_state.book_fingerprint());
}

bool require_loaded(const httplib::Request&, httplib::Response& res) {
    if (!g_state.engine) {
        res.status = 409;
        res.set_content(json{{"error", "no_market_loaded"},
                             {"message", "POST /api/market/load first"}}
                             .dump(),
                        "application/json");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int port = 8787;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--port") port = std::atoi(argv[i + 1]);
    }

    load_market("normal");  // ready immediately; /api/market/load can reload

    httplib::Server svr;
    svr.set_default_headers({{"Access-Control-Allow-Origin", "http://localhost:5173"},
                             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
                             {"Access-Control-Allow-Headers", "Content-Type"}});
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) { res.status = 204; });

    svr.Get("/api/health", [](const httplib::Request&, httplib::Response& res) {
        const auto& bi = build_info();
        std::lock_guard<std::mutex> lock(g_state.mu);
        res.set_content(json{{"status", "ok"},
                             {"compiler", bi.compiler_id},
                             {"compiler_version", bi.compiler_version},
                             {"build_type", bi.build_type},
                             {"simd", bi.simd_build_level},
                             {"host_cpu", bi.host_cpu},
                             {"market_loaded", g_state.engine.has_value()}}
                             .dump(),
                        "application/json");
    });

    svr.Post("/api/market/load", [](const httplib::Request& req, httplib::Response& res) {
        std::string regime = "normal";
        try {
            if (!req.body.empty()) {
                auto body = json::parse(req.body);
                if (body.contains("regime")) regime = body.at("regime").get<std::string>();
            }
        } catch (...) {
            res.status = 400;
            res.set_content(json{{"error", "invalid_request"}}.dump(), "application/json");
            return;
        }
        std::lock_guard<std::mutex> lock(g_state.mu);
        load_market(regime);
        res.set_content(json{{"loaded", true},
                             {"regime", to_string(g_state.market->config.regime)},
                             {"spot", g_state.market->snapshot.spot},
                             {"quotes_total", g_state.engine->quotes().size()},
                             {"expiries", g_state.distinct_years},
                             {"state_hash", g_state.book_fingerprint()}}
                             .dump(),
                        "application/json");
    });

    svr.Get("/api/market/quotes", [](const httplib::Request& req, httplib::Response& res) {
        if (!require_loaded(req, res)) return;
        std::lock_guard<std::mutex> lock(g_state.mu);

        std::optional<double> expiry_filter;
        if (req.has_param("expiry")) expiry_filter = std::atof(req.get_param_value("expiry").c_str());
        std::string type_filter = req.has_param("type") ? req.get_param_value("type") : "";

        const std::size_t limit =
            req.has_param("limit")
                ? std::min<std::size_t>(
                      5000, static_cast<std::size_t>(std::atoll(req.get_param_value("limit").c_str())))
                : 500;
        const std::size_t offset =
            req.has_param("offset")
                ? static_cast<std::size_t>(std::atoll(req.get_param_value("offset").c_str()))
                : 0;

        json out = json::array();
        std::size_t matched = 0, emitted = 0;
        for (const auto& q : g_state.engine->quotes()) {
            if (expiry_filter && std::abs(q.years - *expiry_filter) > 1e-6) continue;
            if (!type_filter.empty() && to_string(q.type) != type_filter) continue;
            ++matched;
            if (matched <= offset) continue;
            if (emitted >= limit) continue;
            out.push_back(quote_to_json(q));
            ++emitted;
        }
        res.set_content(json{{"quotes", out}, {"matched", matched}, {"returned", emitted}}.dump(),
                        "application/json");
    });

    svr.Get("/api/market/expiries", [](const httplib::Request& req, httplib::Response& res) {
        if (!require_loaded(req, res)) return;
        std::lock_guard<std::mutex> lock(g_state.mu);
        json out = json::array();
        for (double years : g_state.distinct_years) {
            std::size_t count = 0;
            for (const auto& q : g_state.engine->quotes()) {
                if (std::abs(q.years - years) < 1e-6) ++count;
            }
            out.push_back(json{{"years", years}, {"quote_count", count}});
        }
        res.set_content(json{{"expiries", out}}.dump(), "application/json");
    });

    svr.Get("/api/surface", [](const httplib::Request& req, httplib::Response& res) {
        if (!require_loaded(req, res)) return;
        if (!req.has_param("years")) {
            res.status = 400;
            res.set_content(json{{"error", "missing_param"}, {"param", "years"}}.dump(),
                            "application/json");
            return;
        }
        const double years = std::atof(req.get_param_value("years").c_str());
        std::lock_guard<std::mutex> lock(g_state.mu);

        std::vector<OptionQuote> slice_quotes;
        for (const auto& q : g_state.engine->quotes()) {
            if (std::abs(q.years - years) < 1e-6) slice_quotes.push_back(q);
        }
        if (slice_quotes.empty()) {
            res.status = 404;
            res.set_content(json{{"error", "unknown_expiry"}}.dump(), "application/json");
            return;
        }

        // Observed points: exactly what the book holds -- never interpolated
        // and labelled as observed.
        json observed = json::array();
        double k_min = 0.0, k_max = 0.0;
        bool first = true;
        for (const auto& q : slice_quotes) {
            observed.push_back(json{{"strike", q.strike},
                                    {"log_moneyness", q.log_moneyness},
                                    {"implied_vol", q.implied_vol},
                                    {"bid", q.bid},
                                    {"ask", q.ask},
                                    {"mid", q.mid},
                                    {"status", to_string(q.status)},
                                    {"type", to_string(q.type)}});
            if (first || q.log_moneyness < k_min) k_min = q.log_moneyness;
            if (first || q.log_moneyness > k_max) k_max = q.log_moneyness;
            first = false;
        }

        // Calibrated curve: the model's own vol() query across the observed
        // moneyness range -- a model estimate, explicitly labelled as such,
        // never substituted for an observation.
        constexpr int kGridPoints = 41;
        json calibrated = json::array();
        for (int i = 0; i < kGridPoints; ++i) {
            const double k = k_min + (k_max - k_min) * static_cast<double>(i) /
                                         static_cast<double>(kGridPoints - 1);
            calibrated.push_back(
                json{{"log_moneyness", k}, {"implied_vol", g_state.engine->surface().vol(k, years)}});
        }

        // Fit diagnostics: recomputed directly from this slice's own quotes
        // via the same calibrate_svi_slice the engine itself calls --
        // real, same code path, just not cached inside the engine (it only
        // stores the fitted SviParams, not the SviFitResult diagnostics).
        const auto fit = calibrate_svi_slice(slice_quotes, g_state.config.calibrator);

        res.set_content(json{{"years", years},
                             {"observed", observed},
                             {"calibrated", calibrated},
                             {"fit", fit_to_json(fit)}}
                             .dump(),
                        "application/json");
    });

    svr.Post("/api/event/apply", [](const httplib::Request& req, httplib::Response& res) {
        if (!require_loaded(req, res)) return;
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(json{{"error", "invalid_json"}}.dump(), "application/json");
            return;
        }

        const char* required[] = {"years", "strike", "option_type", "bid", "ask"};
        for (const char* field : required) {
            if (!body.contains(field)) {
                res.status = 400;
                res.set_content(json{{"error", "missing_field"}, {"field", field}}.dump(),
                                "application/json");
                return;
            }
        }
        const double years = body.at("years").get<double>();
        const double strike = body.at("strike").get<double>();
        const std::string type_str = body.at("option_type").get<std::string>();
        const double bid = body.at("bid").get<double>();
        const double ask = body.at("ask").get<double>();
        const double mid = body.contains("mid") ? body.at("mid").get<double>() : 0.5 * (bid + ask);

        if (!(strike > 0.0) || !(years > 0.0) || !std::isfinite(bid) || !std::isfinite(ask)) {
            res.status = 422;
            res.set_content(json{{"error", "invalid_values"},
                                 {"message", "strike and years must be positive and finite"}}
                                 .dump(),
                            "application/json");
            return;
        }
        if (bid > ask) {
            res.status = 422;
            res.set_content(
                json{{"error", "invalid_values"}, {"message", "bid must not exceed ask"}}.dump(),
                "application/json");
            return;
        }
        OptionType type;
        if (type_str == "call") {
            type = OptionType::Call;
        } else if (type_str == "put") {
            type = OptionType::Put;
        } else {
            res.status = 422;
            res.set_content(json{{"error", "invalid_values"}, {"message", "option_type must be 'call' or 'put'"}}
                                 .dump(),
                            "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(g_state.mu);
        const MarketEvent evt{
            SequenceNumber{1}, Timestamp{std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::system_clock::now().time_since_epoch())
                                             .count()},
            "SYN",             "instr",
            Years{years},      Strike{strike},
            type,              MarketEventType::Quote,
            Money{bid},        Money{ask},
            Money{mid},        0.0,
        };

        const auto applied = g_state.engine->apply_event(evt);
        if (!applied.has_value()) {
            res.status = 422;
            res.set_content(json{{"error", "unknown_expiry"},
                                 {"message", "no expiry within tolerance of the requested years"}}
                                 .dump(),
                            "application/json");
            return;
        }

        const auto report = g_state.engine->recompute();
        g_state.last_report = report;
        g_state.has_last_report = true;

        const std::string hash = g_state.book_fingerprint();
        g_state.log("apply_event",
                   "Applied tick: years=" + std::to_string(years) +
                       " strike=" + std::to_string(strike) + " type=" + type_str,
                   hash);

        json positions = json::array();
        const auto valuations = g_state.engine->position_valuations();
        for (std::size_t i = 0; i < valuations.size() && i < g_state.config.positions.size(); ++i) {
            positions.push_back(json{{"label", g_state.config.positions[i].label},
                                     {"strike", g_state.config.positions[i].strike},
                                     {"years", g_state.config.positions[i].years},
                                     {"type", to_string(g_state.config.positions[i].type)},
                                     {"vol_used", valuations[i].vol_used},
                                     {"greeks", greeks_to_json(valuations[i].greeks)}});
        }

        res.set_content(json{{"applied", true},
                             {"report", report_to_json(report)},
                             {"portfolio", portfolio_to_json(g_state.engine->portfolio())},
                             {"pnl", pnl_to_json(g_state.engine->pnl())},
                             {"positions", positions},
                             {"state_hash", hash}}
                             .dump(),
                        "application/json");
    });

    svr.Post("/api/compare/full-rebuild", [](const httplib::Request& req, httplib::Response& res) {
        if (!require_loaded(req, res)) return;
        std::lock_guard<std::mutex> lock(g_state.mu);

        std::vector<OptionQuote> current_quotes(g_state.engine->quotes().begin(),
                                                 g_state.engine->quotes().end());

        const auto t0 = std::chrono::steady_clock::now();
        const auto rebuilt = full_rebuild(current_quotes, g_state.config);
        const auto t1 = std::chrono::steady_clock::now();
        const double full_rebuild_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        g_state.last_full_rebuild_latency = t1 - t0;
        g_state.has_last_full_rebuild = true;

        const double incremental_us = g_state.has_last_report
                                          ? std::chrono::duration<double, std::micro>(
                                                g_state.last_report.latency)
                                                .count()
                                          : 0.0;

        const double pnl_diff =
            std::abs(g_state.engine->pnl().total_exact_pnl - rebuilt.pnl.total_exact_pnl);

        json surface_diffs = json::array();
        double max_vol_diff = 0.0;
        for (double years : g_state.distinct_years) {
            const double v_inc = g_state.engine->surface().vol(0.0, years);
            const double v_full = rebuilt.surface.vol(0.0, years);
            const double diff = std::abs(v_inc - v_full);
            max_vol_diff = std::max(max_vol_diff, diff);
            surface_diffs.push_back(json{{"years", years},
                                         {"incremental_vol", v_inc},
                                         {"full_rebuild_vol", v_full},
                                         {"diff", diff}});
        }

        g_state.log("compare_full_rebuild",
                   "Compared incremental vs full rebuild: pnl_diff=" + std::to_string(pnl_diff));

        const json speedup_json = (incremental_us > 0.0)
                                      ? json((full_rebuild_ms * 1000.0) / incremental_us)
                                      : json(nullptr);

        res.set_content(
            json{
                {"incremental_us", incremental_us},
                {"incremental_measured",
                 g_state.has_last_report},  // false until at least one apply_event has run
                {"full_rebuild_ms", full_rebuild_ms},
                {"speedup", speedup_json},
                {"pnl_diff", pnl_diff},
                {"max_vol_diff", max_vol_diff},
                {"surface_diffs", surface_diffs},
                {"quotes_examined", g_state.last_report.quotes_examined},
                {"quotes_total", g_state.last_report.quotes_total},
                {"nodes_total", g_state.last_report.total_nodes},
                {"nodes_recomputed", g_state.last_report.recomputed_nodes},
            }
                .dump(),
            "application/json");
    });

    svr.Get("/api/quote/greeks", [](const httplib::Request& req, httplib::Response& res) {
        if (!require_loaded(req, res)) return;
        if (!req.has_param("years") || !req.has_param("strike") || !req.has_param("type") ||
            !req.has_param("vol")) {
            res.status = 400;
            res.set_content(
                json{{"error", "missing_param"}, {"message", "years, strike, type, vol required"}}
                    .dump(),
                "application/json");
            return;
        }
        const double years = std::atof(req.get_param_value("years").c_str());
        const double strike = std::atof(req.get_param_value("strike").c_str());
        const double vol = std::atof(req.get_param_value("vol").c_str());
        const std::string type_str = req.get_param_value("type");
        const OptionType type = (type_str == "put") ? OptionType::Put : OptionType::Call;

        std::lock_guard<std::mutex> lock(g_state.mu);
        const double spot = g_state.config.baseline_market.spot;
        const double rate = g_state.config.baseline_market.rate;
        const double carry = g_state.config.baseline_market.carry;
        const OptionGreeks g = black_scholes_greeks(spot, strike, vol, years, rate, carry, type);
        res.set_content(greeks_to_json(g).dump(), "application/json");
    });

    svr.Get("/api/events", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_state.mu);
        json out = json::array();
        for (const auto& e : g_state.event_log) {
            out.push_back(json{{"timestamp_ms", e.timestamp_ms},
                               {"type", e.type},
                               {"description", e.description},
                               {"state_hash", e.state_hash}});
        }
        res.set_content(json{{"events", out}}.dump(), "application/json");
    });

    std::printf("volatility_lab_server listening on http://127.0.0.1:%d\n", port);
    svr.listen("127.0.0.1", port);
    return 0;
}
