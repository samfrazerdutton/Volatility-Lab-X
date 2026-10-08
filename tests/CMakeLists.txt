# ---------------------------------------------------------------------------
# Tests.
#
# GoogleTest is fetched rather than vendored or required pre-installed: the
# project must build from a clean checkout on three toolchains, and a system
# package is the one dependency most likely to be absent or mismatched.  The
# version is pinned -- a floating dependency makes a red CI run ambiguous
# between "my change broke it" and "upstream changed".
# ---------------------------------------------------------------------------
include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)

# GoogleTest is not clean under this project's warning set, and its warnings
# are not ours to fix.
if(TARGET gtest)
    if(MSVC AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
        target_compile_options(gtest PRIVATE /W0)
    else()
        target_compile_options(gtest PRIVATE -w)
    endif()
endif()

# ---------------------------------------------------------------------------
# vl_add_test(<dir> <name>) -- one binary per test file, registered with ctest.
#
# One binary per file rather than one monolith: a crash or a hang in a
# numerical edge case then takes down a single test, and the fuzz and
# sanitizer runs can target individual areas.
# ---------------------------------------------------------------------------
function(vl_add_test dir name)
    set(target test_${name})
    add_executable(${target} ${dir}/${name}.cpp)
    target_link_libraries(${target} PRIVATE
        volatility_lab
        vl_compiler_flags
        vl_sanitizers
        GTest::gtest_main
    )
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
    add_test(NAME ${name} COMMAND ${target})
    set_tests_properties(${name} PROPERTIES
        LABELS ${dir}
        TIMEOUT 300
    )
endfunction()

# --- unit ------------------------------------------------------------------
vl_add_test(unit types)
vl_add_test(unit diagnostics)
vl_add_test(unit aligned_buffer)
vl_add_test(unit expected)

# --- numerical -------------------------------------------------------------
vl_add_test(numerical dd_real)
vl_add_test(numerical special)
vl_add_test(numerical root_finding)
vl_add_test(numerical interpolation)
vl_add_test(numerical slice_models)
vl_add_test(numerical surface)

# --- core -------------------------------------------------------------------
vl_add_test(core dependency_graph)

# --- calibration -----------------------------------------------------------
vl_add_test(calibration market_data)
vl_add_test(calibration svi_calibrator)
vl_add_test(calibration incremental)

# --- pricing ---------------------------------------------------------------
vl_add_test(pricing black)
vl_add_test(pricing implied_vol)

# --- greeks ------------------------------------------------------------
vl_add_test(greeks greeks)

# --- portfolio ---------------------------------------------------------
vl_add_test(portfolio portfolio)

# --- diagnostics ---------------------------------------------------------
vl_add_test(diagnostics surface_differential)

# --- risk ------------------------------------------------------------------
vl_add_test(risk uncertainty)

# --- scenarios ---------------------------------------------------------------
vl_add_test(scenarios scenario)

# --- runtime -----------------------------------------------------------------
vl_add_test(runtime market_event)
vl_add_test(runtime replay)
vl_add_test(runtime incremental_engine)
