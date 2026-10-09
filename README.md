# Volatility-Lab-X

A C++20 volatility-surface analytics engine, and a local web workbench
built on top of it for investigating and validating that engine's
computations interactively.

## What this actually is

**The engine** (`include/`, `src/`, `kernels/`): quote normalisation and
weighting, quasi-explicit SVI calibration, a volatility surface with
scalar/batch queries, Greeks and portfolio PnL attribution, an AVX2 erfcx
kernel, a deterministic incremental runtime (`IncrementalEngine`) that
recomputes only the parts of a dependency graph a market event actually
touches, and a deliberately independent `full_rebuild()` reference path
used to prove the incremental path produces the *same* answer, not
merely a fast one.

**The workbench** (`apps/server`, `frontend`): a thin, local-only HTTP
service (`volatility_lab_server`) wrapping the engine, and a React/
TypeScript frontend that lets you load a synthetic market, inspect the
calibrated volatility smile for any expiry, edit a quote's bid/ask,
apply that edit as a real market event through the C++ engine, and
compare the resulting incremental computation against an independent
full rebuild -- numerically and in timing.

**What the workbench currently covers, honestly**: the Overview, Surface
Lab (2D smile, not yet the 3D surface), and Market Explorer screens are
real and wired to the live engine. Compute Graph, Performance Lab,
Validation Center, and Replay Studio are not yet built -- see "Known
limitations" below; the navigation bar marks them accordingly rather
than hiding them or pretending they work.

## Architecture

```
Browser (React + TypeScript, Vite dev server / static build)
    |
    |  fetch('/api/...')  (JSON over HTTP, localhost only)
    v
volatility_lab_server (apps/server, cpp-httplib + nlohmann/json)
    |
    |  calls directly into, no logic duplicated here:
    v
volatility_lab (the C++ library: calibration, pricing, Greeks,
                 IncrementalEngine, full_rebuild, state hashing)
```

The server holds one `IncrementalEngine` instance, protected by one
mutex. Every HTTP request either reads that engine's current state or
mutates it through the engine's own `apply_event`/`recompute()` API --
the frontend never computes a price, a Greek, or a calibration itself.

## Prerequisites

- A C++20 compiler. Developed and measured with Clang 22.1.0 targeting
  `x86_64-pc-windows-msvc` on Windows 11; the CMake presets are portable
  (see `docs/BASELINE.md`), but only that exact combination has been run.
- CMake >= 3.24, Ninja.
- Node.js (developed with v25.6.1) and npm, for the frontend.
- Network access the first time you configure the build: `tests/` fetches
  GoogleTest and `apps/server/` fetches cpp-httplib and nlohmann/json via
  `FetchContent`, all pinned to specific tags.

## Build and run the C++ engine + service

```bash
cmake --preset relwithdebinfo
cmake --build build/relwithdebinfo
./build/relwithdebinfo/bin/volatility_lab_server          # listens on 127.0.0.1:8787
```

(If you configured into the flat `build/` directory used during
development instead of the preset's `build/relwithdebinfo/`, the binary
is at `build/bin/volatility_lab_server.exe`.)

The service binds to `127.0.0.1` only. It loads a synthetic sample
market automatically on startup, so `GET http://127.0.0.1:8787/api/health`
should immediately report `"market_loaded": true`.

## Build and run the frontend

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173 (or the next free port)
```

The dev server proxies `/api/*` to `http://127.0.0.1:8787` (see
`frontend/vite.config.ts`), so the C++ service must already be running.

Production build: `npm run build` (output in `frontend/dist/`); it still
expects whatever serves it to proxy `/api/*` to the C++ service.

## Running the C++ test suite

```bash
cmake --preset relwithdebinfo
cmake --build build/relwithdebinfo
ctest --test-dir build/relwithdebinfo
```

Last run on this machine: **32/32 passing**. AddressSanitizer+
UndefinedBehaviorSanitizer (`cmake --preset asan`) also run clean on all
32 tests -- see `docs/VALIDATION.md` for how that preset is configured
and a real toolchain issue (Debug-CRT/ASan incompatibility on this exact
Clang/Windows combination) that was found and fixed to get there.

Frontend: `cd frontend && npm run lint && npx tsc --noEmit && npm run build`.
No frontend unit/e2e test suite exists yet -- see "Known limitations".

## Running a deterministic demo

```bash
./build/relwithdebinfo/bin/volatility_lab_demo
```

Prints a synthetic market, per-expiry calibration, one incremental market
event, a full-rebuild comparison (should read `PnL difference:
$0.000000000`), and a deterministic-replay check -- all computed live,
nothing hardcoded. See `apps/cli/main.cpp`.

## Reproducing benchmarks

```bash
./build/relwithdebinfo/bin/bench_scalar_baseline
./build/relwithdebinfo/bin/bench_simd_erfcx
./build/relwithdebinfo/bin/bench_incremental_scaling [--million]
```

See `docs/BENCHMARKS.md` for the actual numbers these produced on this
machine (AMD Ryzen 9 4900HS, Clang 22.1.0, RelWithDebInfo) and what is
and is not yet measured.

## Documentation

- `docs/ARCHITECTURE_CURRENT.md` -- module-by-module inventory, what's
  real vs. scaffold-only, the protected-regression catalog.
- `docs/INCREMENTAL_RUNTIME.md` -- the dependency-graph runtime, the
  indexed quote store, the independent full-rebuild reference path.
- `docs/PERFORMANCE.md` / `docs/BENCHMARKS.md` / `docs/PERFORMANCE_BASELINE.md`
  -- the performance-regression mechanism, measured numbers, and how to
  reproduce them with this project's own tooling or an external profiler.
- `docs/VALIDATION.md` / `docs/DETERMINISM.md` -- what is tested, what is
  proven bitwise-deterministic vs. tolerance-bounded, and where each claim
  is backed by a named, runnable test.

## Known limitations

- **Compute Graph, Performance Lab, Validation Center, and Replay
  Studio are not implemented.** The navigation bar shows them, disabled,
  labelled "not built" -- they were not silently omitted.
- **The Surface Lab view is 2D** (an SVG smile chart per expiry), not the
  3D WebGL surface described in the original product brief. The engine's
  own batch surface-query API (`VolSurface::total_variance_batch`) would
  support building one; it has not been wired to a 3D view yet.
- **No automated frontend or integration test suite.** The backend was
  verified by curling every endpoint directly against the real engine
  (not a mock) and inspecting the responses; the frontend was verified
  via `tsc --noEmit`, `npm run lint`, and a production `npm run build`,
  all passing. No browser-based visual or interaction test was run in
  this environment (no browser automation tool was available when this
  was built) -- open `http://localhost:5173` yourself to confirm the UI
  renders and behaves as described.
- **No CSV/JSON market import.** The workbench only loads the bundled
  synthetic generator (`io/synthetic_market.hpp`); there is no upload
  path yet.
- **Single market book, single engine instance.** The service holds one
  `IncrementalEngine` behind one mutex -- correct for one user inspecting
  one book, not a multi-tenant design.
- **No authentication.** Appropriate only because the service binds to
  `127.0.0.1` and is never exposed to a network; do not change the bind
  address without adding one.
- Python bindings, a regime/factor engine, and SIMD kernels beyond
  `erfcx` remain out of scope -- see `docs/ARCHITECTURE_CURRENT.md`'s
  "Phase 2" section for what was deliberately deferred and why.
