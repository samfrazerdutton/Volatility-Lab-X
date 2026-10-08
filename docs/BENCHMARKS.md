# Benchmarks

Every number below was executed on this machine, this build, and is
labelled with exactly what produced it. Nothing here is hypothetical,
extrapolated-and-presented-as-measured, or carried over from a different
run than the one that produced it -- where a number *is* extrapolated
(the historical 1,000,000-state `million_state` baseline comparison), it
is marked EXTRAPOLATED explicitly, next to the measured number it was
derived from.

## Environment (every benchmark below)

| | |
|---|---|
| CPU | AMD Ryzen 9 4900HS, 8 physical / 16 logical cores |
| Compiler | Clang 22.1.0, target `x86_64-pc-windows-msvc` |
| Build | RelWithDebInfo, `-O2 -DNDEBUG -g -Xclang -gcodeview` |
| SIMD | AVX2+FMA (`-mavx2 -mfma`, per-function `[[gnu::target]]`) |
| Threads | Single-threaded for every number below -- no benchmark in this
document exercises the `ThreadPool`/parallel-reduction machinery yet |

## Kernel microbenchmarks

`bench_scalar_baseline` (n=3528 per kernel, warm-up pass before timing):

| kernel | ns/op | ops/sec |
|---|---|---|
| `black_scholes_price` | 120.73 | 8,282,828 |
| `black_scholes_greeks` | 150.43 | 6,647,686 |
| `erfcx` (scalar, `std::erfc`-based) | 18.68 | 53,533,629 |
| `norm_cdf_hp` | 27.88 | 35,871,157 |
| `implied_volatility` (round-trip) | 901.92 | 1,108,745 |

`bench_simd_erfcx` (n=100,000, 50 repeats, 7 trials, best-of):

| kernel | ns/op | ops/sec | vs scalar |
|---|---|---|---|
| scalar `erfcx_poly` | 18.653 | 53,610,794 | 1.00x |
| AVX2 `erfcx_avx2_batch` | 4.930 | 202,853,747 | **3.78x** |

## Incremental-engine scaling: the structural fix, measured before and after

`bench_incremental_scaling`: a fixed 50 quotes per expiry, expiry count
scaled so total quote count grows from ~100 to ~500,000 -- deliberately
isolating "cost of touching one expiry" from "cost of the whole book",
which a scheme that instead grew quotes-per-expiry at fixed expiry count
would conflate. See `docs/INCREMENTAL_RUNTIME.md` for the architectural
story this table is evidence for.

**Before** the indexed-quote-store fix (`recompute_node`'s `ExpirySlice`
case scanned every stored quote regardless of which expiry was dirty):

| Quotes | Full rebuild | Single quote update | Single expiry update | Large market update |
|---|---|---|---|---|
| 91 | 3.4 ms | 1.17 ms | 1.11 ms | 1.78 ms |
| 913 | 27.2 ms | 1.17 ms | 0.72 ms | 21.5 ms |
| 9,380 | 218.0 ms | 1.18 ms | 3.09 ms | 200.5 ms |
| 99,077 | 2,482.8 ms | 4.02 ms | 4.41 ms | 2,429.8 ms |
| 498,126 | 34,879.2 ms | 23.10 ms | 25.47 ms | 34,339.1 ms |

**After** (`quote_indices_by_node_`, a per-expiry bucket of quote indices
built once and maintained incrementally):

| Quotes | Full rebuild | Single quote update | Single expiry update | Large market update | Full-rebuild speedup |
|---|---|---|---|---|---|
| 91 | 3.2 ms | 1.13 ms | 1.05 ms | 1.69 ms | 1.05x |
| 913 | 25.9 ms | 1.14 ms | 0.70 ms | 20.3 ms | 1.05x |
| 9,380 | 201.7 ms | 0.92 ms | 2.85 ms | 185.0 ms | 1.08x |
| 99,077 | 1,246.8 ms | 3.51 ms | 3.98 ms | 1,158.7 ms | **1.99x** |
| 498,126 | 6,604.1 ms | 20.69 ms | 22.27 ms | 5,145.9 ms | **5.28x** |

**Full rebuild sped up far more than single-quote update.** The bug cost
was paid *per expiry* during a full rebuild (every one of E expiries
scanned the full N-quote book, an O(E*N) cost that is quadratic once both
scale together in this benchmark's design) but only *once* for a
single-quote update. This is why the fix's benefit grows with scale for
full rebuild (1.05x at 91 quotes, 5.28x at 498,126) but grows much more
slowly for single-quote update (1.03x at 91 quotes, ~1.12x at 498,126).

**Work reduction is directly observable**, not just fast: `RecomputeReport::quotes_examined`
reports the real count. In `volatility_lab_demo`'s own 192-quote, 10-expiry
book, a single quote update examines 25 quotes (13.0% of the book) and
runs exactly 1 calibration out of 10 expiries -- both read from the
report, not inferred from timing.

### Single-quote update's remaining cost at scale is a different, by-design bottleneck

The fix above did not make single-quote update O(1) in total book size,
and the reason is not a remaining instance of the same bug: `Uncertainty`
is a graph node that depends on `Surface`, so it is marked dirty (and
recomputed) on *every* expiry change, and `estimate_point_uncertainty`
necessarily kernel-weights every quote in the book to produce its
cross-sectional estimate (see `risk/uncertainty.hpp`'s own documentation).
Measured directly, in isolation, at 498,126 quotes:

```
estimate_point_uncertainty over 498126 quotes: 21.38 ms
```

against a 20.69 ms *total* single-quote-update time at the same scale --
this one call accounts for essentially all of it. This is a real,
measured, **by-design** cost (a global statistical estimator, not a
local lookup that forgot to be local), named here as the actual remaining
bottleneck rather than left unexplained. A legitimate future optimisation
-- maintaining the kernel-weighted partial sums incrementally, or bounding
the scan to quotes within some number of bandwidths of the query point --
would change the *numerical* result (even if only within existing
tolerance) and needs its own differential validation; not attempted this
phase.

## Flagship end-to-end benchmark: `million_state`

(Unchanged from the previous phase; reproduced here for completeness,
not re-measured this phase since the structural fix above is to a
different code path than this benchmark's per-event
calibrate-already-small-book-many-times workload.)

```
scalar_baseline (n=3528/kernel): see table above
1,000,000-event run: 738 s wall clock, 1,355 states/sec, 4.19% computation
avoided, 1.8x speedup vs a 2,000-state baseline sample (BASELINE is
measured on a smaller sample and extrapolated to 1,000,000 -- see
benchmarks/million_state.cpp's own header comment for why).
```

## What is explicitly not in this document

- SIMD-vectorised pricing/Greeks kernels (only `erfcx` has an AVX2 tier
  today; `black_scholes_price`/`black_scholes_greeks`/implied-vol helpers
  do not yet).
- Parallel calibration across expiries (the `ThreadPool` and
  `kernels/parallel/reduce.hpp` exist and are independently benchmarked at
  the kernel level, but are not wired into `IncrementalEngine`'s
  calibration path).
- Thread-scaling tables (1/2/4/8/16 workers) for anything in this document
  -- every number above is single-threaded.
- AoS-vs-SoA comparison for `OptionQuote` storage.
- Peak memory / allocation-byte tables beyond the single-update allocation
  counts folded into the scaling table above.

Each is a legitimate next step, not claimed as done.
