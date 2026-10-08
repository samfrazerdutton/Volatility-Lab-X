# ---------------------------------------------------------------------------
# Benchmarks.
#
# No external benchmark library (Google Benchmark etc.) is fetched here: a
# hand-rolled std::chrono timing loop, with a warm-up pass and a fixed,
# deterministic, reproducible input domain, is enough to get honest numbers
# and keeps this directory's only dependency on the library it is measuring.
# ---------------------------------------------------------------------------

add_executable(bench_scalar_baseline scalar_baseline.cpp)
target_link_libraries(bench_scalar_baseline PRIVATE volatility_lab vl_compiler_flags)
