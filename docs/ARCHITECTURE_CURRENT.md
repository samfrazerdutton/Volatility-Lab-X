# Architecture, as it actually exists today

A **living** inventory, updated as the Volatility State Compiler runtime
work (directive, 2026-10-07 onward) proceeds -- unlike `docs/BASELINE.md`,
which is a frozen snapshot of the state *before* that work began and is
deliberately not kept current. This document is the "what is real" list to
check before assuming something exists or is still scaffold-only.

## How to read this document

Every module below is marked:

- **Real** — has headers, an implementation, and a dedicated test binary
  that passes.
- **Scaffold only** — a directory exists (sometimes with a `CMakeLists.txt`
  guarded by `if(EXISTS ...)`), but it contains no source files. These are
  not broken; they are deliberately-empty slots the original project
  skeleton left for exactly the work this directive now asks for.

## Added since the frozen baseline (`docs/BASELINE.md`)

| Module | What it does |
|---|---|
| `runtime` (`market_event.hpp`, `state_hash.hpp`, `replay.hpp`, `incremental_engine.hpp`) | Deterministic market events (immutable, strictly-sequenced stream); a portable FNV-1a state fingerprint; deterministic replay (`MarketState` + per-step hashes, verified to match bit-for-bit across independent runs); and the real runtime dependency graph wiring `core/dependency_graph.hpp`'s bookkeeping to the actual pipeline (quote -> expiry slice -> surface -> {surface differential, uncertainty} -> position Greeks -> portfolio -> PnL), verified to produce results identical to a full rebuild and to recalibrate nothing on a pure market-point move. |
| `benchmarks/` | No longer scaffold-only: `bench_scalar_baseline` measures `black_scholes_price`/`black_scholes_greeks`/`erfcx`/`norm_cdf_hp`/`implied_volatility`; `bench_simd_erfcx` measures the scalar-vs-AVX2 erfcx speedup (3.6-3.8x measured). See `docs/PERFORMANCE_ANALYSIS.md`. `VL_BUILD_BENCHMARKS=ON` now configures cleanly from a fresh clone. |
| `kernels/scalar/`, `kernels/simd/` | No longer scaffold-only (both were `if(EXISTS ...)`-guarded empty placeholders): `kernels/scalar/erfcx_poly.hpp` is the branch-light Numerical Recipes erfcx approximation (~1.045e-7 measured relative error vs the dd reference, its own `math::tol::kErfcxPoly` budget); `kernels/simd/erfcx_avx2.hpp` is the AVX2+FMA batch version of the same formula, including a small internal vectorised `exp` (AVX2 has no native one) scoped explicitly to the bounded domain this one kernel needs. `math::erfcx` itself is unchanged -- these are additional, independently-validated tiers, not a replacement. `kernels/parallel/` and `kernels/cuda/` remain empty. |

## Real modules (from the frozen baseline)

| Module | Headers | What it does |
|---|---|---|
| `core` | `types.hpp`, `config.hpp`, `diagnostics.hpp`, `expected.hpp`, `aligned_buffer.hpp`, `build_info.hpp`, `dependency_graph.hpp` | Strong types, structured (never-thrown) diagnostics, `Expected<T,E>`, over-aligned SoA buffers, compiled-in toolchain/host description, and a generic dirty-propagation DAG (bookkeeping only — see Gaps). |
| `numerics` (header-only, under `math/`) | `dd_real.hpp`, `special.hpp`, `reference_special.hpp`, `root_finding.hpp`, `interpolation.hpp`, `linalg.hpp`, `compare.hpp` | Double-double (`DDouble`) high-precision reference arithmetic; `erfcx`/normal CDF/inverse CDF; Newton/Brent/safeguarded root finding; natural cubic spline and log-linear interpolation; small dense linear algebra (`NormalEquations` + Cholesky) reused by both the SVI calibrator's inner solve and the Surface Differential engine's regression; a `rel_error`/`WorstCase` comparison helper used throughout the differential-testing style tests. |
| `pricing` | `black.hpp`, `reference.hpp`, `implied_vol.hpp` | Normalised-Black (`b(x,s)`) pricing via `erfcx`, both spot- and forward-measure; an *independent* double-double reference (`black_scholes_greeks_ref`, built from the spot measure, deliberately not derived from the production forward-measure path) used to validate every new closed-form formula; implied-vol inversion (Halley with a safeguarded bisection fallback). |
| `greeks` | `greeks.hpp` | Closed-form spot-measure Greeks through third order (delta, gamma, vega, theta, rho, vanna, volga, charm, speed), portfolio dollar-Greek aggregation, and `taylor_vs_exact_reprice` (the Taylor-vs-exact comparison the directive's scenario work builds on). |
| `volatility` | `slice.hpp`, `svi.hpp`, `ssvi.hpp`, `surface.hpp` | SVI and SSVI slice parameterisations with admissibility projection and Lee-bound checking; `VolSurface` (immutable, `with_slice`/`with_slices` for incremental replacement); `GridSlice` (cubic-spline, used by the scenario engine's shock reconstruction); `TermCurve` for forward/discount curves. |
| `options` | `quote.hpp`, `normalize.hpp` | `OptionQuote` (AoS) / `QuoteBook` (SoA) with an explicit, measured rationale for both layouts; quote normalisation (deriving forward/discount/vega/log-moneyness/weight from raw bid/ask/spot/rate) with structured `DiagnosticSink` rejection, never a silent NaN. |
| `calibration` | `weights.hpp`, `optimizer.hpp`, `svi_calibrator.hpp`, `incremental.hpp` | Quote weighting (spread/liquidity/moneyness/staleness, each a measured factor, not a hand-tuned constant); the quasi-explicit SVI reduction (2-D outer grid + boxed 3-D closed-form inner solve, replacing a 5-D nonlinear fit); incremental (single-slice) recalibration via `VolSurface::with_slice`. |
| `diagnostics` | `surface_differential.hpp` | The surface differential engine: regresses an observed vol-space change onto `{level, skew, curvature, term}` plus independently-measured `forward_shift` and a leave-one-tenor-out `event_shift`. |
| `risk` | `uncertainty.hpp` | Per-point vol uncertainty from quote-level bid/ask spread (vega-converted), local Kish-effective-sample-size coverage, and quote-range-based extrapolation detection. |
| `portfolio` | `portfolio.hpp` | `Position`/`MarketPoint`, per-leg and portfolio Greeks, and `compute_pnl_attribution` — PnL attribution that reconciles *exactly* (`residual` is defined as whatever is left over, never dropped). |
| `scenarios` | `scenario.hpp` | Named `ShockSpec`s (vol-space level/skew/curvature/term + spot + time decay), realised as a new `VolSurface` via per-expiry `GridSlice` reconstruction, cross-validated against the surface differential engine's own measurement of what the shock actually did. |
| `io` | `synthetic_market.hpp` | Deterministic synthetic market generation across named regimes (Normal, HighVol, Crash, VolCrush, Earnings, Illiquid) for testing and benchmarking without needing real market data. |

## Scaffold-only (directories exist, nothing inside yet)

| Path | Guarded by | What it is for |
|---|---|---|
| `kernels/scalar/` | `kernels/CMakeLists.txt`'s `if(EXISTS .../scalar/CMakeLists.txt)` | The scalar-optimized tier between the reference path and SIMD, per the directive's `REFERENCE → SCALAR OPTIMIZED → SIMD → MULTI-THREADED → INCREMENTAL` validation ladder. |
| `kernels/simd/` | same pattern, additionally gated on `VL_ENABLE_SIMD` | AVX2+FMA kernels, compiled as a separate object library so only this translation unit carries the ISA flag — the comment in `kernels/CMakeLists.txt` is explicit about why (a binary that mixes ISA levels in one TU is how a "portable" build acquires an illegal-instruction crash on older hardware). `VL_ENABLE_SIMD=ON` and `-mavx2 -mfma` are already wired into the build (`VL_SIMD_DESCRIPTION`); there is simply no kernel source here yet. |
| `kernels/parallel/` | same pattern | Thread pool and/or OpenMP execution over the scalar/SIMD kernels. No `std::thread` or thread-pool code exists anywhere in the repository yet (`grep -rln "std::thread\|ThreadPool" src include` is empty). |
| `kernels/cuda/` | `VL_ENABLE_CUDA` (off) + `EXISTS` | Explicitly deferred; `VL_ENABLE_CUDA` defaults off and nothing requires turning it on. |
| `apps/cli/` | `VL_BUILD_CLI` (currently forced off) | Empty. No CLI exists. |
| `examples/` | `VL_BUILD_EXAMPLES` (currently forced off) | Empty. |
| `python/` | `VL_BUILD_PYTHON` (defaults off) | Empty. No pybind11 bindings exist. |
| `research/`, `tools/`, `data/` | not wired into CMake at all | Empty; no research console frontend, no auxiliary tooling, no sample data files. |
| `include/volatility_lab/{execution,numerics,optimization}/`, `src/{execution,numerics,optimization}/` | n/a | Empty directories with no files. Not currently used by anything — `calibration/optimizer.hpp` already covers what an `optimization/` module name might suggest, and the double-double/`erfcx`/root-finding code that might suggest a dedicated `numerics/` module in fact lives under `math/` and is referred to as "numerics" descriptively in this document, not as a separate real module. |

The deterministic market event model, replay engine, state hashing, and a
real runtime dependency-graph execution layer (`runtime/incremental_engine.hpp`,
wiring `core/dependency_graph.hpp`'s bookkeeping to the actual quote ->
expiry slice -> surface -> {differential, uncertainty} -> Greeks ->
portfolio -> PnL pipeline) now exist -- see "Added since the frozen
baseline" above. `core/dependency_graph.hpp` itself remains exactly what it
was designed to be: label-and-dirty-flag bookkeeping, not a scheduler;
`IncrementalEngine` is the scheduler built on top of it.

Still absent, as of this update: **a regime engine, a volatility factor
engine, a cross-underlier engine, a surface health engine, a numerical-
conditioning diagnostic layer, a thread pool, a million-state benchmark, a
performance-regression system, a CLI, Python bindings, and the research
console.** SIMD kernels exist now for exactly one primitive (`erfcx`, both
scalar-optimised and AVX2 tiers -- see "Added since the frozen baseline"
above and `docs/PERFORMANCE_ANALYSIS.md`); `norm_cdf_hp`, `black_price`,
`black_scholes_greeks`, and `implied_volatility` do not yet have SIMD
tiers.

## Protected numerical regressions

Historical failure modes this project has already found and fixed, each
with a permanent test guarding it. These are not hypothetical — every one
below is cited from an actual comment at the point in the source where it
was found, not reconstructed from memory:

- **Implied-vol step-size accounting bug** — `src/pricing/implied_vol.cpp`:
  convergence was measured from the pre-clamp Halley `step` rather than the
  post-clamp actual move `s_next - s`; when the safeguarded bisection
  fallback replaced the Halley step, the solver reported convergence on a
  move that was never taken. *"That bug cost 309 spurious failures and a
  worst relative error of 166 before it was caught by the round-trip
  sweep."*
- **Third-derivative (speed) reference-stencil precision** —
  `src/pricing/reference.cpp`: a finite-difference step of `1e-9*S` is right
  for first/second derivatives but leaves a third-derivative stencil with
  round-off `~eps_dd/h^3` dominating truncation `~h^2`, producing a ~1e-4
  relative discrepancy against an otherwise-correct production formula.
  Fixed by widening the step specifically for that stencil to `1e-6*S`.
  (Independently re-confirmed this session, for the *same* class of error in
  the *same* function, while validating `speed` and `vanna`/`volga`/`charm`.)
- **Lee moment-bound wing slope** (`slice.hpp`/`svi.cpp`) — enforced only
  where it asymptotically applies (`|k| >= kLeeSlopeTestFrom`), not
  everywhere, because the bound is asymptotic and naively applying it near
  the money rejects admissible slices.
- **Variance-space vs. volatility-space confusion** (this session, Surface
  Differential engine) — regressing a surface's *total-variance* change puts
  a uniform annualised-vol shift into the `term` bucket instead of `level`,
  because `W = sigma^2 * T` makes a level move inherently linear in `T`.
  Fixed by regressing in vol space; now a permanent test
  (`UniformVolShiftIsPureLevelWithZeroResidual`).
- **Event-detector false positives** (this session, Surface Differential
  engine) — comparing a global fit's residual at the shortest tenor against
  the rest scored an ordinary, every-tenor skew change *higher* than a
  genuinely isolated single-tenor bump, because the bump drags the global
  fit enough to contaminate the "rest" bucket. Fixed with leave-one-tenor-out
  cross-validation; permanent test
  `GenuineIsolatedEventScoresHigherThanAnOrdinaryShapeChange`.
- **Uncertainty tenor-bandwidth blow-up** (this session, Uncertainty engine)
  — a relevance kernel bandwidth proportional to the query's own tenor grows
  wide enough, far beyond the data, to treat the entire book as locally
  informative, understating uncertainty exactly where it should be highest.
  Fixed by capping the bandwidth at the book's own observed tenor span;
  permanent coverage via the far-tenor extrapolation tests in
  `tests/risk/uncertainty.cpp`.
- **Scenario reconstruction boundary artifact** (this session, Scenario
  graph) — `GridSlice` splines in total variance, so a vol-space-linear
  shock becomes variance-space-quadratic once squared, and the natural
  cubic spline's zero-curvature boundary condition does not match a true
  quadratic, producing ~1.6e-6 deviation at exactly the differential
  engine's default query points. Fixed by widening/densifying the default
  reconstruction grid to a configuration *measured* (not assumed) to reduce
  that to ~1e-16; permanent coverage via the exact-recovery tests in
  `tests/scenarios/scenario.cpp`.
- **SVI local-minimum failures** (`svi_calibrator.hpp`'s own documented
  measurement) — a direct five-parameter fit's final objective varies by a
  factor of up to 107x between the 16 starting points tested on the `Crash`
  synthetic regime, with ~3% of single-start fits landing >10% high; the
  quasi-explicit 2-D-grid-plus-closed-form-inner-solve reduction removes the
  dependency on a starting guess entirely rather than mitigating it with
  more starts.
- **`IncrementalEngine::apply_event` dropping quote metadata on update**
  (`src/runtime/incremental_engine.cpp`) — an update to an *already-known*
  instrument built a brand-new, default-constructed `OptionQuote`, silently
  zeroing `volume`/`open_interest`/`age_seconds` (fields a `MarketEvent`
  has no data for at all, but which `assign_weights_by_slice`'s liquidity
  factor reads directly). The touched quote's calibration weight then
  diverged from a fresh rebuild of the same final quotes, by enough to move
  portfolio PnL ~$20 — found by `apps/cli`'s engineering demo, not by the
  existing test suite, because the pre-existing exact-match test's fixed
  quote index happened to have zero volume/open_interest already. Fixed by
  starting from the existing stored quote and overwriting only the fields a
  `MarketEvent` actually carries; permanent coverage via
  `ApplyEventOnAnExistingQuotePreservesVolumeAndOpenInterest`, which
  deliberately picks a quote with nonzero volume/open_interest so an
  unlucky index cannot hide this class of bug again.

Future work on this codebase must not regress any of these — in several
cases (variance/vol-space confusion, event-detector specificity, the
scenario boundary artifact) the *test that would catch the regression*
exists specifically because the original mistake was made once already
inside this same project.

## What "turn the dependency graph into a real runtime" turned out to mean

Done: `runtime/incremental_engine.hpp`'s `IncrementalEngine` builds one
`DependencyGraph` from an initial quote book (one `ExpirySlice` node per
expiry, a `Surface` node depending on all of them, `SurfaceDifferential`/
`Uncertainty`/one-per-`Position` Greeks nodes depending on `Surface`, a
`Portfolio` node depending on every position, a `Pnl` node depending on
`Portfolio`), and for each node the graph reports dirty, calls the one
already-tested function responsible for that piece of the pipeline
(`calibrate_svi_slice`, `VolSurface`'s constructor,
`compute_surface_differential`, `estimate_point_uncertainty`,
`value_position`, `aggregate_greeks`, `compute_pnl_attribution`) --
exactly the "wire the existing functions into nodes of the existing
DependencyGraph" plan this section originally described, not a new graph
structure or new math.
