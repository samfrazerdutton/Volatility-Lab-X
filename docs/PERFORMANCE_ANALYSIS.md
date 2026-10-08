# Performance analysis: scalar baseline

Measured on the same machine and build as `docs/BASELINE.md` (Ryzen 9
4900HS, 8C/16T, Clang 22.1.0, RelWithDebInfo), via
`benchmarks/bench_scalar_baseline`. Methodology: a fixed, deterministic
sweep (9 strikes x 7 vols x 7 tenors x 4 rates x 2 types = 3,528 points,
the same domain shape `tests/greeks/greeks.cpp` already validates
numerically across), one excluded warm-up pass, then 7 trials of 200
repeats each, reporting the **minimum** total time across trials -- a
trial can only be slowed down by scheduling noise, never sped up below
what the code actually costs, so the minimum is the best estimate of the
code's own cost (standard microbenchmark practice).

```
kernel                                      ns/op      ops/sec         total_ms
black_scholes_price                        119.22      8387558           84.125
black_scholes_greeks                       147.13      6796909          103.812
erfcx                                       18.69     53515358           13.185
norm_cdf_hp                                 28.39     35217841           20.035
implied_volatility (round-trip)            911.29      1097348          643.005
```

Raw JSON: `benchmarks/results/scalar_baseline.json` (regenerate with
`bench_scalar_baseline <path>`).

## What this says about where to spend SIMD effort

**`implied_volatility` dominates by a wide margin** -- 911 ns/op against
119-147 ns/op for direct pricing/Greeks, i.e. roughly 6-8x the cost of a
single Black evaluation. That is exactly what an iterative (Halley +
safeguarded bisection) solver doing several evaluations per inversion
should cost, and it means **implied-vol inversion is the single most
expensive thing this library does per option**, by far. A calibration
pass over a quote book of any size spends most of its scalar time here,
not in pricing or Greeks.

This has a direct, honest consequence for SIMD prioritisation (directive
section 8's ordered list): vectorising `black_price`/`black_scholes_greeks`
directly helps every caller of those functions, but the *dominant* cost
(implied vol) is dominated by **how many Newton/Halley iterations it
takes**, which is data-dependent (different inputs converge in different
numbers of steps) -- the classic case where naive lane-synchronous SIMD
(every lane iterates until the *slowest* lane in the batch converges)
gives a smaller speedup than the raw per-evaluation vector width would
suggest, because some lanes do wasted extra work waiting for others. That
is not a reason to skip vectorising the implied-vol solver; it is a reason
to measure its *actual* SIMD speedup rather than assume it matches
`black_scholes_price`'s, and to report that honestly once it exists rather
than before.

**`erfcx` (18.69 ns/op) and `norm_cdf_hp` (28.39 ns/op) are the cheap,
high-volume primitives** underneath pricing and Greeks -- every
`black_scholes_price`/`black_scholes_greeks` call makes several calls into
one or the other. Their absolute cost is small, but their call *volume* is
the largest of anything measured here, which is the standard argument for
vectorising a primitive even when its own per-call cost looks modest in
isolation: the win compounds across every caller.

## Priority order for the SIMD kernel layer (directive Phase 3)

Based on the measurements above, not assumed:

1. `erfcx` / `norm_cdf_hp` -- the shared primitives, highest call volume,
   and the simplest control flow (no iteration, no data-dependent
   branching), so the SIMD-vs-scalar error contract (directive section 9)
   is also the simplest to establish correctly first. **Done for erfcx**
   (see "AVX2 erfcx" below); `norm_cdf_hp` is next and is largely free
   once `erfcx_avx2_batch` exists, since the production scalar
   `norm_cdf_hp` is itself `0.5*erfcx(z)*exp(-x^2/2)`.
2. `black_price` / `black_scholes_greeks` -- straight-line, no iteration,
   builds directly on (1). This is where most of the directive's
   "scenario evaluation" and "portfolio valuation" SIMD targets actually
   spend their time, since both ultimately call these per leg.
3. `implied_volatility` -- iterative, data-dependent convergence, the
   hardest kernel to vectorise *and get an honest speedup number for*.
   Attempted last, and reported with whatever speedup is actually
   measured, including "less than the vector width would suggest" if
   that is what the lane-synchronisation cost turns out to be -- see
   section 31 of the runtime directive: an honest negative or partial
   result is a required deliverable, not a failure to hide.

## AVX2 erfcx: the first SIMD kernel, measured

`kernels/scalar/erfcx_poly.hpp` (the branch-light Numerical Recipes formula,
~1.045e-7 relative error vs the dd reference) is now also implemented as
`kernels/simd/erfcx_avx2.hpp`'s AVX2+FMA batch kernel. Measured via
`bench_simd_erfcx` (same methodology as the scalar baseline: excluded
warm-up, 7 trials of 50 repeats over n=100,000, minimum reported):

```
kernel                                ns/op      ops/sec
scalar erfcx_poly                    18.5-30.5    33-54M
AVX2 erfcx_avx2_batch                  4.9-8.4   118-205M

measured speedup: 3.6x-3.8x across repeated runs
```

(The absolute ns/op numbers vary run to run on this laptop by close to 2x --
thermal/scheduling noise on shared hardware, not a measurement bug; the
*ratio* between scalar and AVX2 stays in the 3.6-3.8x band across runs,
which is the number that matters.) 3.6-3.8x against a theoretical ceiling
of 4x (one AVX2 register holds 4 doubles) means the kernel is well
vectorised with little overhead -- not a number inflated by comparing
against an artificially slow scalar baseline, since the scalar kernel
being compared against is the exact same formula.

One real obstacle, documented rather than glossed over: AVX2 has no `exp`
instruction, so the kernel needed its own vectorised `exp`
(`exp_avx2_bounded`, single-constant range reduction + a 14-term Taylor
series, deliberately scoped to the `[-1, 710]` domain this one kernel
needs, not offered as a general-purpose primitive). That introduced a
second, independent approximation of `exp` alongside the scalar kernel's
`std::exp` call, so "AVX2 matches scalar" is not bitwise and not pure
FMA-rounding noise the way a shared primitive would produce. Measured (not
guessed) at up to 690 ulps worst case -- `math::tol::kSimdEquivalence` was
corrected from a pre-existing, never-yet-measured "4 ulps" guess to this
real number (with ~3x headroom) once it existed to check against. 690 ulps
is `~1.5e-13` relative error, six orders of magnitude inside the formula's
own ~1e-7 ceiling, so it has no practical effect -- but reporting the
*measured* number rather than leaving the old guess standing uncorrected is
exactly the "verify, don't assume" discipline this project applies
everywhere else.

## What has not been profiled yet

- **Calibration** (`calibrate_svi_slice`): not included in this pass.
  `svi_calibrator.hpp`'s own documentation already reports its per-regime
  timing against a direct multi-start LM fit (262-710 us per slice,
  depending on regime) from this project's earlier benchmarking; that is
  a calibration-level measurement, not a kernel-level one, and is left as
  that study rather than re-measured here.
- **Memory/allocation/cache behaviour**: no profiler is wired in yet
  (`docs/BASELINE.md`'s own gap list). The kernels measured here are all
  pure-arithmetic, no heap allocation in the hot loop, so allocation
  profiling is unlikely to change the SIMD priority order above, but that
  is an expectation, not a measurement, and is stated as such.
- **The incremental runtime's own recompute cost** at realistic scale
  (many expiries, many positions): `runtime/incremental_engine.hpp`'s
  correctness was verified directly, but its *performance* has not been
  benchmarked yet -- that is the directive's Phase 5 (million-state
  benchmark), not this scalar-kernel pass.
