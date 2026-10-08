# Performance: the regression system

This is the directive's section 13 deliverable: a way to catch a
performance regression automatically, from real measurements, never from
a hardcoded expectation. For the *analysis* of where time actually goes
and why (scalar baseline, SIMD priority order, the AVX2 erfcx numbers),
see `docs/PERFORMANCE_ANALYSIS.md` -- this document is about the
mechanism that keeps those numbers honest over time, not the numbers
themselves.

## Layout

```
benchmarks/           the benchmark programs themselves (bench_scalar_baseline,
                       bench_simd_erfcx, bench_million_state)
benchmarks/results/   JSON output from the most recent run of each, regenerated
                       every time a benchmark is run -- not committed as "the
                       truth", just the latest observation
perf/
  check_regression.py the regression detector
  baselines/          the stored "last known good" JSON per benchmark --
                       committed to git, updated deliberately (see below),
                       never silently
```

## How a regression is detected

`perf/check_regression.py <baseline.json> <current.json>` walks both JSON
documents leaf by leaf, matches paths that exist in both, classifies each
as "lower is better" (any `ns`/`latency`/`_ms`/`runtime_s` in the path) or
"higher is better" (`ops_per_sec`/`throughput`/`speedup`/
`states_per_second`/`fraction_avoided`), and fails (exit code 1) if any
metric moved against its own direction by more than a threshold (default
20%, overridable per-path).

That default is not arbitrary: `docs/BASELINE.md` and
`docs/PERFORMANCE_ANALYSIS.md` both measured run-to-run variance on this
project's own (shared, laptop-class) development machine of up to ~2x on
some kernels between otherwise-identical runs. A tighter default would
flag normal machine noise as a regression constantly; the 20% default is
loose enough to absorb that noise and still catch something that actually
regressed by a meaningful amount.

```bash
python perf/check_regression.py perf/baselines/scalar_baseline.json \
                                benchmarks/results/scalar_baseline.json
```

### A real bug this tool caught in itself, immediately

The first version's "informational field" filter (`n`, meaning sample
count, excluded so it is never compared as a performance metric) matched
by substring, and the single letter `"n"` is a substring of almost every
field name (`"ns_per_op"`, `"kernels"`, ...). Every leaf in every
benchmark file matched the informational filter and was silently excluded
-- the tool ran, printed "OK", and had checked precisely zero metrics.
Caught immediately by running it against a baseline and that exact same
file as "current": a tool whose job is "compare two runs" should report
every shared leaf as unchanged when given identical inputs, not zero
leaves examined, and "zero checked" was the signal something was wrong
well before any real regression could have been missed by it. Fixed by
matching the informational filter against the path's exact final segment,
not a substring anywhere in the path.

## Updating a stored baseline

A baseline in `perf/baselines/` should change only when a change is
*understood* to improve or legitimately alter performance (a new SIMD
kernel, an algorithmic change) -- never to make a regression stop being
reported. Updating it is: run the benchmark, inspect the new numbers
against the analysis in `docs/PERFORMANCE_ANALYSIS.md`, and only then copy
the fresh `benchmarks/results/*.json` over the corresponding
`perf/baselines/*.json` file, as its own reviewable commit explaining why.

## What has a baseline today

- `scalar_baseline` (`bench_scalar_baseline`): `black_scholes_price`,
  `black_scholes_greeks`, `erfcx`, `norm_cdf_hp`, `implied_volatility`.
- `simd_erfcx` (`bench_simd_erfcx`): scalar vs AVX2 erfcx throughput and
  measured speedup.

`million_state` (`bench_million_state`, the flagship benchmark) does not
have a stored baseline yet -- it has only been run once so far (see
`docs/PERFORMANCE_ANALYSIS.md`'s million-state section once that run
completes), and a baseline from a single observation, before knowing this
project's own run-to-run variance on *this* specific benchmark, would be a
number nobody has actually validated as "known good" yet.
