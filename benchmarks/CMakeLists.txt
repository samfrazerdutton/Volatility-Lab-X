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

if(VL_ENABLE_SIMD)
    add_executable(bench_simd_erfcx simd_erfcx.cpp)
    target_link_libraries(bench_simd_erfcx PRIVATE volatility_lab vl_compiler_flags)
endif()

add_executable(bench_million_state million_state.cpp)
target_link_libraries(bench_million_state PRIVATE volatility_lab vl_compiler_flags)
if(WIN32)
    target_link_libraries(bench_million_state PRIVATE psapi)
endif()
