# Validation

How this project proves a numerical claim rather than asserting it, and
where to find each kind of check. This is an index into the test suite's
own validation strategy, not a restatement of individual test bodies.

## The validation ladder

Every optimized implementation in this codebase sits below a reference it
was checked against, never the other way around:

```
REFERENCE (double-double, independent derivation)
        v
SCALAR OPTIMIZED (production formula, library calls: std::erfc, std::exp)
        v
SIMD (AVX2+FMA, its own vectorised exp where the ISA has no native one)
        v
PARALLEL (thread pool, deterministic or fast reduction)
        v
INCREMENTAL (dependency-graph runtime, recomputes only what changed)
```

A failure at any tier is checked against the tier below it, not patched in
place -- see `docs/PERFORMANCE_ANALYSIS.md` and the AVX2 erfcx section of
`docs/ARCHITECTURE_CURRENT.md` for two worked examples (a reference-stencil
precision bug found by cross-checking against a *third* independent
method, and an event-detection heuristic that failed its own differential
test before being redesigned).

## Reference implementations: what they are built from, independently

- **`math::reference::erfcx_ref`** (`math/reference_special.hpp`): a
  double-double (`DDouble`) continued-fraction evaluation, built from a
  different derivation than the production `math::erfcx`'s
  `std::erfc`-based fast path, specifically so a shared bug in the
  derivation cannot hide from both at once.
- **`pricing::reference::black_scholes_greeks_ref`**
  (`pricing/reference.hpp`): built from the *spot* measure (S, sigma, T, r)
  via double-double central differences, deliberately not derived from the
  production *forward*-measure closed-form Greeks -- an independent path
  to the same numbers, not a second copy of the same formula.
- **`forward_intrinsic`-based no-arbitrage bounds**
  (`tests/property/core_invariants.cpp`): a model-free identity (put-call
  parity, the discounted-forward intrinsic lower bound) true regardless of
  how `black_price` happens to be implemented -- an independent *check*,
  not an independent *implementation*, which is its own useful category:
  it cannot share a derivation bug with the code under test because it
  has no derivation of the option-pricing formula at all.

## Differential testing: production vs reference, by kernel

| Kernel | Reference | Where | Measured bound |
|---|---|---|---|
| `erfcx` (production, `std::erfc`-based) | dd continued fraction | `tests/numerical/special.cpp` | machine precision (near-exact) |
| `erfcx_poly` (scalar, Numerical Recipes rational form) | dd continued fraction | `tests/kernels/erfcx_poly.cpp` | ~1.045e-7 relative (`math::tol::kErfcxPoly`) |
| `erfcx_avx2_batch` (AVX2+FMA) | dd reference, *and* `erfcx_poly` | `tests/kernels/erfcx_avx2.cpp` | inherits `kErfcxPoly` vs the dd reference; up to 690 ulps vs `erfcx_poly` (`math::tol::kSimdEquivalence`) |
| `black_scholes_greeks` | spot-measure dd central differences | `tests/greeks/greeks.cpp` | per-Greek budgets tuned from measured dd-reference noise floors |
| `implied_volatility` | dd bisection (`BisectionAlwaysConvergesAndActsAsTheOracle`) | `tests/pricing/implied_vol.cpp` | `math::tol::kImpliedVol` |
| `calibrate_svi_slice` (quasi-explicit) | direct 5-parameter LM, multi-start | `tests/calibration/svi_calibrator.cpp`, `benchmarks/calibration` | objective within noise floor; 1.7-2.9x faster, measured per regime |

Every tolerance in that table is a named `math::tol::*` constant with a
one-line rationale in `math/compare.hpp` -- not a bare number in the test
file that asserts it.

## Property-based testing

`tests/property/core_invariants.cpp` checks mathematical invariants across
a wide domain sweep (deep ITM/OTM, near-expiry, extreme vol, nonzero rate
and carry), not individual hand-picked points: Gamma and Vega
non-negativity, price never falling below the discounted-forward
no-arbitrage bound, price strictly increasing in volatility, and put-call
parity as an independent, formula-free identity. It deliberately does not
duplicate this project's many *other* property-style tests scattered
through their own module files (`NormCdfSymmetry`, `ErfcxIsMonotoneDecreasing`,
`NormInvIsMonotone`, `NormalisedBlackIsStrictlyIncreasingInTotalVolatility`,
`RoundTripRecoversTotalVolatilityOverTheFullDomain`) -- see that file's own
header comment for the full list of what is covered elsewhere.

One property-test failure became a real finding rather than a loosened
assertion: an earlier version of `PriceNeverFallsBelowIntrinsicValue`
asserted the bound against naive *spot* intrinsic and failed on deep-ITM
puts -- correctly, because a European put's price is textbook-permitted to
fall below its undiscounted spot intrinsic. Fixed by asserting against the
discounted *forward* intrinsic instead (the actual no-arbitrage bound),
reusing the existing `forward_intrinsic` function rather than re-deriving
it a second time.

## Fuzz testing

`tests/fuzz/numerical_fuzz.cpp`: seeded (`std::mt19937_64`, a fixed
constant, not a time-based seed -- an unreproducible fuzz failure is far
less useful than a reproducible one), mixing plausible values with
zero/negative/infinite/NaN/astronomically-large draws across
`black_scholes_greeks`, `implied_volatility`, and `calibrate_svi_slice`.

This found two real production issues (not test bugs), both fixed rather
than worked around:

1. `black_scholes_greeks`'s degenerate-axis guard (`!(vol > 0.0)`, written
   that way specifically to also catch NaN) silently treated `vol = NaN`
   as the vol-to-zero *limit* and returned a confident discounted-intrinsic
   price instead of propagating "unknown". Fixed with an explicit
   `isnan` check ahead of the limit logic.
2. `calibrate_svi_slice` could report `Ok` with a non-finite objective,
   because `svi_project_to_admissible` unconditionally projects fitted
   parameters into the admissible region regardless of how adversarial
   the input quotes' weights were. Fixed by adding
   `SviFitStatus::NonFiniteObjective`, checked immediately after the
   objective is computed.

It also found two bugs in the fuzz test's *own* assumptions -- treating
"IEEE-finite" as synonymous with "well-posed" for astronomically large
(e.g. `vol = 5e299`) inputs, where ordinary, correct floating-point
arithmetic legitimately overflows -- documented and fixed at the point in
the test file where the wrong assumption was made, following the same
discipline as every other test-bug fix in this project's history: fix the
test, say so, and say why, rather than silently relaxing it.

## Sanitizers

- **AddressSanitizer**: `core/thread_pool.hpp`'s full test suite
  (`tests/core/thread_pool.cpp`) runs clean (no memory errors, no leaks)
  under `-DVL_ENABLE_ASAN=ON` (the `asan` CMake preset). Verified directly
  this session, not assumed.
- **ThreadSanitizer**: attempted (`-DVL_ENABLE_TSAN=ON`, the `tsan`
  preset) and found **unavailable** for the `x86_64-pc-windows-msvc`
  target on this Clang build (`clang++: error: unsupported option
  '-fsanitize=thread' for target 'x86_64-pc-windows-msvc'`) -- a real
  toolchain limitation, stated here rather than silently skipped or
  claimed as passing.
- **UndefinedBehaviorSanitizer**: wired into the same `asan` preset
  (ASan+UBSan combined); not yet run as a dedicated pass separate from
  ASan at the time of writing.

## Reproducing a validation claim

Every row in the differential-testing table and every fuzz/property claim
above corresponds to a named test; `ctest --test-dir build -R <name>
--output-on-failure` reproduces it directly. None of the numbers in this
document were computed by hand or carried over from a different run than
the one that produced the test file they describe.
