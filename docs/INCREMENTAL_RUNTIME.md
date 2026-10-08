# The incremental runtime

What `runtime/incremental_engine.hpp`'s `IncrementalEngine` actually is,
what it guarantees, what it does not, and the one real bug its own
dogfooding (`apps/cli`'s engineering demo) found in it. Not a tutorial on
`core/dependency_graph.hpp` itself -- see that header's own comments for
the bookkeeping-only DAG underneath this.

## What problem this solves

A full surface rebuild recalibrates every expiry from scratch, even when
one quote in one expiry moved. `IncrementalEngine` wires the existing,
separately-tested pipeline functions --
`calibrate_svi_slice`, `VolSurface`'s constructor,
`compute_surface_differential`, `estimate_point_uncertainty`,
`value_position`/`aggregate_greeks`, `compute_pnl_attribution` -- onto the
nodes of a `DependencyGraph`, so that one quote tick dirties exactly the
one `ExpirySlice` node it belongs to, plus everything downstream of the
`Surface` node (which must still reassemble, since it depends on every
slice), and nothing else. The *math* is unchanged from the non-incremental
path; the only thing this module adds is not re-doing work whose inputs
did not change.

## The node graph

One `ExpirySlice` node per distinct expiry in the initial quote book, a
`Surface` node depending on all of them, `SurfaceDifferential` and
`Uncertainty` nodes depending on `Surface`, one `PositionGreeks` node per
position depending on `Surface`, a `Portfolio` node depending on every
position, and a `Pnl` node depending on `Portfolio`:

```
ExpirySlice[0] --\
ExpirySlice[1] ---+--> Surface --+--> SurfaceDifferential
   ...            |              +--> Uncertainty
ExpirySlice[N] --/               +--> PositionGreeks[0] --\
                                  +--> PositionGreeks[1] ---+--> Portfolio --> Pnl
                                  +--> ...                 /
                                  +--> PositionGreeks[M] --/
```

Two kinds of update, with deliberately different blast radii:

- **`apply_event`** (a quote tick): dirties exactly the one `ExpirySlice`
  node whose expiry (within `kYearsMatchTolerance = 1e-9`) matches the
  event, and nothing else directly -- the graph's own propagation handles
  the rest. This is the operation the "incremental" half of the name is
  about.
- **`update_market_point`** (spot/rate/carry move): dirties the `Surface`
  node directly, skipping every `ExpirySlice` node entirely. This is not
  an optimisation shortcut; it is a documented finding, independently
  reached in three other modules (`portfolio/portfolio.hpp`'s
  sticky-moneyness tests, `diagnostics/surface_differential.hpp`'s
  forward-orthogonality guarantee, `scenarios/scenario.hpp`'s spot-only
  shock test) and reused here: a pure spot/forward move leaves a
  log-moneyness-parameterised surface's *shape* unchanged, only where the
  forward sits, so recalibrating any slice in response to it would be
  wasted, shape-preserving work.

## A deliberate v1 scope limit

`core/dependency_graph.hpp`'s `add_node` can create a new node with
dependencies on existing nodes, but there is no operation to add a new
*dependency* to a node that already exists. So the set of expiries this
engine will ever know about is fixed at construction time, from
`initial_quotes`. An event for an expiry outside that set is rejected with
`RuntimeError::UnknownExpiry` (checked by
`UnknownExpiryIsRejectedAndChangesNothing`) rather than silently dropped
or triggering an undocumented full rebuild. Supporting a graph that can
grow new expiry nodes at runtime is a real, separate extension, not
attempted here speculatively.

## The central correctness claim, and how it is checked

The guarantee this engine lives or dies on: recomputing only the dirty
subset must produce the *same* answer as recomputing everything, not
merely a close one.
`tests/runtime/incremental_engine.cpp`'s
`IncrementalUpdateMatchesAFreshFullRebuildExactly` checks this directly --
an incrementally-updated engine's PnL, portfolio delta/vega, and surface
vol are compared (`EXPECT_NEAR` at `1e-9`/`1e-12`, the float-noise floor,
not a loophole) against a second engine built from scratch with the same
final quotes. `docs/DETERMINISM.md` discusses the same test under "what is
bitwise deterministic."

## A real bug this guarantee caught, via dogfooding rather than a unit test

`apply_event` used to build a brand-new, default-constructed `OptionQuote`
for an instrument that already existed in `quotes_`, overwriting the slot
in full. `MarketEvent` carries no opinion on `volume`, `open_interest`, or
`age_seconds` -- a tick only ever reports bid/ask/mid -- so those fields
silently reset to their defaults on every update to a known instrument.
`calibration/weights.hpp`'s `assign_weights_by_slice` reads
volume/open_interest directly as part of the liquidity factor, so the
touched quote's calibration weight diverged from a fresh rebuild of the
same final quotes, moving that expiry's calibrated slice -- and therefore
portfolio PnL, by about $20 in `apps/cli`'s 200-position demo book --
measurably outside the 1e-9 guarantee above.

`IncrementalUpdateMatchesAFreshFullRebuildExactly` did not catch this: its
fixed quote index (`f.quotes[10]`) happens to have zero volume and zero
open_interest in that fixture's synthetic market, so resetting zero to
zero is invisible. This is exactly why the engineering demo exists as more
than a presentation layer -- it runs the same pipeline at a different
scale (200 positions, a quote drawn from the middle of a 199-quote book
rather than a fixed small index) and its own printed, computed numbers
disagreed with an already-stated guarantee, which is what this project
treats as a finding to chase down, not round off.

Fixed by having `apply_event` start from the *existing* stored quote (when
one exists for that instrument) and overwrite only the fields a
`MarketEvent` actually carries -- `strike`/`years`/`type`/`spot`/`bid`/
`ask`/`mid`/`rate`/`dividend`/`status` -- leaving `volume`,
`open_interest`, `age_seconds`, and every other field untouched. A new
instrument (no existing entry) still gets a fresh, default-constructed
quote, since there is nothing to preserve.

Permanent coverage: `ApplyEventOnAnExistingQuotePreservesVolumeAndOpenInterest`
deliberately searches the fixture for a quote with nonzero volume *and*
open_interest before running the same incremental-vs-full-rebuild
comparison, specifically so this bug class cannot hide behind an unlucky
fixed index again. Verified to actually exercise the bug (not merely pass
vacuously) by temporarily reverting the fix and confirming the test fails
with a reported discrepancy of ~2.3e-3 in surface vol, then restoring the
fix and confirming it passes.

## Reproducing this yourself

```bash
ctest --test-dir build -R incremental_engine --output-on-failure
./build/bin/volatility_lab_demo.exe
```

The demo's step 6 ("FULL REBUILD, for comparison") prints the same
incremental-vs-full-rebuild PnL difference the test above checks, computed
from the demo's own 200-position synthetic book rather than the test
fixture's three positions -- a second, independent scale at which the
guarantee is exercised every time the demo runs.
