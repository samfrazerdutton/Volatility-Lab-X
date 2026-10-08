# ---------------------------------------------------------------------------
# The library itself.
#
# Sources are listed explicitly rather than globbed: a glob makes the build
# silently stale when a file is added, and in a project where a missing
# translation unit means a missing *kernel* that is a correctness problem, not
# just an inconvenience.
# ---------------------------------------------------------------------------

# Phase-ordered.  Files are appended as each phase lands, so that `main` is
# always buildable and always tested -- see docs/design-decisions.md (D-01).
set(VL_SOURCES
    # --- phase 1: core -----------------------------------------------------
    core/diagnostics.cpp
    core/dependency_graph.cpp
    ${PROJECT_BINARY_DIR}/generated/build_info.cpp

    # --- phase 2: special functions, Black, implied volatility -------------
    math/normal.cpp
    pricing/black.cpp
    pricing/reference.cpp
    pricing/implied_vol.cpp

    # --- phase 3: slice models and the surface -----------------------------
    math/interpolation.cpp
    volatility/svi.cpp
    volatility/ssvi.cpp
    volatility/surface.cpp

    # --- phase 4: market data, normalisation, synthetic generator ----------
    options/quote.cpp
    options/normalize.cpp
    io/synthetic_market.cpp
    math/linalg.cpp
    calibration/optimizer.cpp
    calibration/weights.cpp
    calibration/svi_calibrator.cpp

    # --- phase 5: Greeks and exact-vs-Taylor repricing ----------------------
    greeks/greeks.cpp

    # --- phase 6: portfolio and PnL attribution ------------------------------
    portfolio/portfolio.cpp

    # --- phase 5-flagship: surface differential engine -----------------------
    diagnostics/surface_differential.cpp

    # --- phase 7: uncertainty-aware volatility -------------------------------
    risk/uncertainty.cpp

    # --- phase 8: incremental (online) calibration ---------------------------
    calibration/incremental.cpp

    # --- phase 10: scenario graph ---------------------------------------------
    scenarios/scenario.cpp

    # --- runtime: deterministic market event model ----------------------------
    runtime/market_event.cpp
    runtime/state_hash.cpp
    runtime/replay.cpp
    runtime/incremental_engine.cpp
)

add_library(volatility_lab STATIC ${VL_SOURCES})
add_library(volatility_lab::volatility_lab ALIAS volatility_lab)

target_include_directories(volatility_lab
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(volatility_lab
    PUBLIC  vl_compiler_flags
    PRIVATE vl_sanitizers
)

target_compile_features(volatility_lab PUBLIC cxx_std_20)

find_package(Threads REQUIRED)
target_link_libraries(volatility_lab PUBLIC Threads::Threads)

if(VL_HAVE_OPENMP)
    target_link_libraries(volatility_lab PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(volatility_lab PUBLIC VL_WITH_OPENMP=1)
else()
    target_compile_definitions(volatility_lab PUBLIC VL_WITH_OPENMP=0)
endif()

if(VL_ENABLE_SIMD)
    target_compile_definitions(volatility_lab PUBLIC VL_WITH_SIMD=1)
else()
    target_compile_definitions(volatility_lab PUBLIC VL_WITH_SIMD=0)
endif()

if(VL_ENABLE_CUDA)
    target_compile_definitions(volatility_lab PUBLIC VL_WITH_CUDA=1)
else()
    target_compile_definitions(volatility_lab PUBLIC VL_WITH_CUDA=0)
endif()

set_target_properties(volatility_lab PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)
