# Determinism

What this project actually guarantees to be reproducible, what it does
not, and how each claim was checked -- not asserted.

## The two kinds of "same result"

Two different claims get conflated under "deterministic" if this is not
stated precisely, and this project makes both, for different things:

1. **Bitwise reproducibility**: the same inputs produce the exact same
   bits, every time, on this build. Claimed only where it is cheap to
   guarantee and actually checked.
2. **Numerical-tolerance reproducibility**: the same inputs produce results
   within a stated, reviewed tolerance (`math::compare.hpp`'s `Tolerance`
   records). This is what almost all of the numerical layer promises --
   `erfcx`, `black_scholes_price`, calibration -- because the underlying
   floating-point operations (`exp`, `erfc`, the compiler's own
   instruction selection) are not themselves bit-for-bit portable across
   compilers or ISA extensions, and claiming otherwise would be false.

**Never claimed**: bitwise reproducibility across different CPUs,
compilers, or compiler versions. Every determinism claim below is scoped
to "this build, this toolchain" unless stated otherwise.

## What is bitwise deterministic, and how it was checked

### Replay: identical event streams produce identical state hashes

`runtime/replay.hpp`'s `replay()` folds a `MarketEventStream` into a
`MarketState` and fingerprints it (`StateHash`, a fixed 64-bit FNV-1a
accumulator -- deliberately not `std::hash`, whose implementation is not
specified to be stable across standard library versions or build
configurations, which would make a replay fingerprint depend on *that*
rather than on the state being hashed).

Checked directly in `tests/runtime/replay.cpp`:
`TwoIndependentlyBuiltStreamsWithTheSameDataFingerprintIdenticallyAtEveryStep`
builds two separate `MarketEventStream`s from the same event data and
asserts every one of N per-step state hashes matches exactly, not just the
final one. `CallingReplayTwiceOnTheSameStreamGivesTheSameResult` confirms
the same stream replayed twice agrees. `APerturbedFieldInOneEventChangesTheFinalFingerprint`
confirms the hash is actually sensitive to state (a passing "always equal"
hash would trivially satisfy the other two tests for the wrong reason).

Why this is bitwise, not tolerance-based: `MarketState::apply` and
`fingerprint()` are pure integer/bit operations over the event's own
already-given `double` fields (`StateHasher::combine(double)` hashes the
IEEE-754 bit pattern directly) -- no `exp`/`erfc`/transcendental function
is in this path, so there is no floating-point-reassociation question to
hedge on.

### Deterministic parallel reduction: fixed chunking, in-order combine

`kernels/parallel/reduce.hpp`'s `reduce_deterministic` partitions `[0, n)`
into `num_chunks` ranges whose boundaries depend only on `(n, num_chunks)`
-- never on which worker thread picks up which chunk, or in what order
chunks finish -- and combines the resulting partials in chunk-index order.

Checked directly in `tests/kernels/parallel_reduce.cpp`:
`DeterministicResultIsBitwiseIdenticalAcrossThreadCounts` runs the same
reduction through 1/2/4/8/16-thread pools and asserts bit-for-bit equality;
`DeterministicRepeatedCallsOnTheSamePoolAreBitwiseIdentical` does the same
across repeated calls on one pool.

The honest counterpoint, in the same file: `reduce_fast` combines partials
as each chunk *finishes*, under a mutex, with no fixed order -- floating-
point addition is not associative, so this is explicitly **not** claimed
bitwise-reproducible, and `FastIsNotClaimedOrRequiredToBeBitwiseReproducible`
checks only that repeated runs agree to ordinary reassociation error, not
exactly. This pairing -- one function with the guarantee, proven; one
without it, proven not to need it -- is `math::tol::kParallelReduction`'s
own stated rationale ("fixed chunking + in-order combine => bitwise"),
written into the codebase before `reduce_deterministic` existed to check
it against.

### Incremental vs full rebuild: identical results, not merely "close"

`runtime/incremental_engine.hpp`'s `IncrementalEngine` recomputes only the
dirty subset of its dependency graph on each event. Its defining
correctness claim -- that doing *less* work produces the *same* answer as
doing all of it -- is checked in
`tests/runtime/incremental_engine.cpp`'s `IncrementalUpdateMatchesAFreshFullRebuildExactly`:
an incrementally-updated engine's PnL, portfolio delta, and surface vol
are compared (`EXPECT_NEAR` at `1e-9`/`1e-12`, the floating-point-noise
floor, not a loophole) against a second engine built from scratch with
the same final quotes. This is the central "impossible to fake"
demonstration this kind of engine lives or dies on.

## What is NOT bitwise deterministic, by design

- **`reduce_fast`** (above) -- deliberately, and tested as such.
- **The AVX2 erfcx kernel vs the scalar kernel**
  (`kernels/simd/erfcx_avx2.hpp`): both compute the same formula, but the
  AVX2 kernel's own vectorised `exp` is a different approximation from the
  scalar kernel's `std::exp` call (`math::tol::kSimdEquivalence`, measured
  at up to 690 ulps -- see `docs/PERFORMANCE_ANALYSIS.md`). This is a
  tolerance claim, not a bitwise one, and the tolerance constant's own
  comment states the real, measured number rather than a round guess.
- **Calibration** (`calibrate_svi_slice`): deterministic in the sense that
  the same quotes always reach the same optimum (the quasi-explicit
  reduction has no random starting guess to vary run to run), but not
  bit-for-bit promised against a different compiler's `exp`/`log`
  implementations.

## How to reproduce a determinism check yourself

```bash
ctest --test-dir build -R "replay|parallel_reduce|incremental_engine" --output-on-failure
```

Every claim in this document is backed by a test in that filter, not a
separate, unverifiable assertion in this file alone.
