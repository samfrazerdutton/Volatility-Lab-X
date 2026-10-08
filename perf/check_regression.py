#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Performance regression detector (directive section 13).

Compares a freshly-produced benchmark JSON (from `benchmarks/bench_*`)
against the stored baseline for that same benchmark (`perf/baselines/`) and
exits non-zero if any metric regressed by more than its threshold. Never
hardcodes expected numbers itself -- every number it compares against comes
from a benchmark's own prior, committed run.

## Design

Benchmark JSON files have different shapes (a flat set of named metrics, or
a nested "kernels" list, or nested "incremental"/"baseline_full_rebuild"
objects). Rather than writing a bespoke comparator per benchmark, this walks
both JSON documents in parallel and compares every leaf path the two share,
classifying each leaf as "lower is better" (any key containing "ns",
"latency", "_ms", or "runtime_s") or "higher is better" (any key containing
"ops_per_sec", "throughput", "speedup", "states_per_second",
"fraction_avoided"); any other leaf (host description strings, booleans,
counts that are not themselves a timing) is compared for informational
display only, never for pass/fail.

A leaf's own threshold can be overridden per-path in `--threshold-overrides`
(see --help); the default is 20% -- loose enough to absorb this project's
own measured run-to-run noise (BASELINE.md and PERFORMANCE_ANALYSIS.md both
note run-to-run variance up to ~2x on the shared laptop this was developed
on), tight enough to still catch a real regression.
"""

import argparse
import json
import sys
from pathlib import Path

LOWER_IS_BETTER_MARKERS = ("ns", "latency", "_ms", "runtime_s", "mean_ns", "p50", "p95", "p99")
HIGHER_IS_BETTER_MARKERS = (
    "ops_per_sec",
    "throughput",
    "speedup",
    "states_per_second",
    "fraction_avoided",
)
# Keys that are numeric but not a performance claim -- never compared.
# Matched against the path's own LAST segment, exactly (not a substring
# check): an early version of this used substring matching for these too,
# and the single-letter entry "n" then matched almost every path in the
# file ("ns_per_op", "kernels", ...), silently excluding everything and
# reporting "0 metrics checked" with no error -- caught by running this
# script against a baseline and its own identical current run, which
# should report every leaf unchanged, not zero leaves examined.
INFORMATIONAL_EXACT_LEAF_NAMES = {
    "n",
    "total_nodes",
    "avg_nodes_recomputed",
    "avg_nodes_reused",
    "peak_working_set_bytes",
    "measured",
}


def classify(path: str):
    lower_path = path.lower()
    last_segment = lower_path.rsplit(".", 1)[-1]
    if last_segment in INFORMATIONAL_EXACT_LEAF_NAMES:
        return None
    if any(m in lower_path for m in LOWER_IS_BETTER_MARKERS):
        return "lower_is_better"
    if any(m in lower_path for m in HIGHER_IS_BETTER_MARKERS):
        return "higher_is_better"
    return None


def walk_leaves(obj, prefix=""):
    """Yield (dotted_path, value) for every numeric leaf in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            # List-of-objects (e.g. scalar_baseline's "kernels"): use the
            # entry's own "name" field as the path component if present,
            # so "kernels.erfcx.ns_per_op" rather than "kernels.2.ns_per_op"
            # (which would silently stop matching if the list were
            # reordered between baseline and current run).
            if isinstance(v, dict) and "name" in v:
                yield from walk_leaves(v, f"{prefix}[{v['name']}]")
            else:
                yield from walk_leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def compare(baseline_path: Path, current_path: Path, default_threshold: float,
           overrides: dict) -> int:
    baseline = json.loads(baseline_path.read_text())
    current = json.loads(current_path.read_text())

    baseline_leaves = dict(walk_leaves(baseline))
    current_leaves = dict(walk_leaves(current))

    failures = []
    checked = 0
    for path, base_value in baseline_leaves.items():
        kind = classify(path)
        if kind is None:
            continue
        if path not in current_leaves:
            continue  # benchmark shape changed; not this tool's job to flag
        cur_value = current_leaves[path]
        threshold = overrides.get(path, default_threshold)
        checked += 1

        if kind == "lower_is_better":
            # Regression = current is worse (higher) than baseline by more
            # than `threshold` fraction.
            if base_value <= 0:
                continue
            ratio = (cur_value - base_value) / base_value
            if ratio > threshold:
                failures.append((path, base_value, cur_value, ratio, kind))
        else:  # higher_is_better
            if base_value <= 0:
                continue
            ratio = (base_value - cur_value) / base_value
            if ratio > threshold:
                failures.append((path, base_value, cur_value, ratio, kind))

    print(f"{baseline.get('benchmark', baseline_path.stem)}: checked {checked} metrics")
    for path, base_value, cur_value, ratio, kind in failures:
        arrow = "worse" if kind == "lower_is_better" else "worse"
        print(
            f"  REGRESSION  {path}: baseline={base_value:.4g}  current={cur_value:.4g}  "
            f"({ratio * 100:+.1f}% {arrow})"
        )
    if not failures:
        print("  OK -- no metric regressed past its threshold")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="stored baseline JSON (perf/baselines/*.json)")
    parser.add_argument("current", type=Path, help="freshly-produced benchmark JSON to check")
    parser.add_argument(
        "--threshold", type=float, default=0.20, help="default regression threshold (fraction, default 0.20)"
    )
    parser.add_argument(
        "--threshold-overrides",
        type=str,
        default="",
        help="comma-separated path=threshold overrides, e.g. 'erfcx.ns_per_op=0.3'",
    )
    args = parser.parse_args()

    overrides = {}
    if args.threshold_overrides:
        for pair in args.threshold_overrides.split(","):
            k, v = pair.split("=")
            overrides[k] = float(v)

    return compare(args.baseline, args.current, args.threshold, overrides)


if __name__ == "__main__":
    sys.exit(main())
