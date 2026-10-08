# Performance baseline (Phase 2 start)

Captured before any performance-sensitive code in this phase was touched,
per the phase directive's own ordering (measure first, then change). Every
number below is from an actual run on this machine, this build, this
commit (`05965da`) -- none invented, none carried over from a different
run than the one that produced it.

## Environment

| | |
|---|---|
| CPU | AMD Ryzen 9 4900HS with Radeon Graphics -- 8 physical cores, 16 logical (SMT) |
| RAM | 16,556,589,056 bytes (~15.4 GiB) |
| OS | Windows 11 Home, build 10.0.26200 |
| Compiler | Clang 22.1.0, target `x86_64-pc-windows-msvc` |
| Build type | RelWithDebInfo (`-O2 -DNDEBUG -g -Xclang -gcodeview`) |
| SIMD | AVX2+FMA enabled (`-mavx2 -mfma`), per-function `[[gnu::target]]` attributes, not global flags |
| OpenMP | Not found by this toolchain -- the thread-pool backend is used instead |
| Thread count | 16 logical; `ThreadPool` tests exercise 1/2/4/8/16-worker pools (`tests/core/thread_pool.cpp`) |

## Existing kernel benchmarks, re-run as the Phase 2 starting point

`bench_scalar_baseline` (n=3528 per kernel):

| kernel | ns/op | ops/sec |
|---|---|---|
| `black_scholes_price` | 120.73 | 8,282,828 |
| `black_scholes_greeks` | 150.43 | 6,647,686 |
| `erfcx` (scalar, `std::erfc`-based) | 18.68 | 53,533,629 |
| `norm_cdf_hp` | 27.88 | 35,871,157 |
| `implied_volatility` (round-trip) | 901.92 | 1,108,745 |

`bench_simd_erfcx` (n=100,000, 50 repeats, 7 trials, best-of):

| kernel | ns/op | ops/sec |
|---|---|---|
| scalar `erfcx_poly` | 18.653 | 53,610,794 |
| AVX2 `erfcx_avx2_batch` | 4.930 | 202,853,747 |
| **measured speedup** | **3.78x** | (4 lanes/instruction; overhead keeps it off the theoretical 4x) |

## Full test suite

```
100% tests passed, 0 tests failed out of 31
Total Test time (real) = 13.15 sec
```

## What this phase starts from, qualitatively

- `InstrumentKey` (strongly typed, allocation-free, `std::hash` specialised)
  already replaced string-concatenated instrument identity in
  `IncrementalEngine` in the previous phase -- Phase 2 section 4 is already
  substantially satisfied; this phase re-verifies rather than re-does it.
- The known, not-yet-fixed structural issue going into this phase:
  `IncrementalEngine::recompute_node`'s `NodeKind::ExpirySlice` case scans
  **every** stored quote (`for (const auto& q : quotes_)`) to find the
  ones belonging to the one expiry being recomputed, regardless of how
  many quotes that expiry actually has or how large the rest of the book
  is. This is exactly the pattern Phase 2 section 3 asks to find and fix;
  see `docs/PERFORMANCE.md`'s scaling benchmark for a measured before/after
  of this specific change.
- `apply_event`'s own expiry-node lookup (`std::find_if` over
  `expiry_node_by_years_`, a `std::map`) is also a linear scan, not a
  binary search, though expiry counts are normally far smaller than quote
  counts so this is the secondary, not primary, target.
- No allocation counting, no SoA layout, no SIMD pricing/Greeks kernels, no
  parallel calibration, and no dedicated `FullRebuildEngine` reference path
  exist yet at the start of this phase -- each is addressed, or explicitly
  deferred with a stated reason, below.

## Methodology notes carried into this phase

- Every "before/after" benchmark number in this phase's docs is captured
  by running the *same* benchmark binary and *same* dataset against the
  pre-change and post-change code, not estimated.
- `RelWithDebInfo` (not `Debug`) is the build used for every measurement
  below, per the project's existing `relwithdebinfo` CMake preset default.
- Timing uses `std::chrono::steady_clock` throughout, consistent with the
  existing `RecomputeReport`/benchmark code; no wall-clock `time` command
  timing is treated as a precise measurement.
