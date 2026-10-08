# ---------------------------------------------------------------------------
# AVX2+FMA kernel tier.
#
# Every AVX2-using function in erfcx_avx2.cpp carries its own
# `[[gnu::target("avx2,fma")]]` attribute -- that is the mechanism actually
# relied on for correct codegen, not this file. A per-source
# COMPILE_OPTIONS override was tried first (hardcoding -mavx2 -mfma for
# just this one source file, so the rest of `volatility_lab` keeps the
# project's ordinary flags and the binary still loads on a machine without
# AVX2), but inspecting the actual compiler invocation Ninja ran showed
# those flags were not present on the command line -- CMake's
# per-source-file COMPILE_OPTIONS did not take effect as expected for this
# generator/toolchain combination, for reasons not chased down further
# once the function-attribute fix (which does not depend on it) resolved
# the build. The override is kept below in case it starts working on a
# different CMake/generator combination; it is redundant with, not a
# substitute for, the function attributes.
#
# An OBJECT library + install-export set was tried before either of those:
# a STATIC `volatility_lab` propagates an OBJECT library's objects all the
# way to the final consuming executable's link line, which means CMake's
# `install(EXPORT ...)` wants that OBJECT library in the same export set --
# real, but unnecessary plumbing for one source file.
# ---------------------------------------------------------------------------

target_sources(volatility_lab PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/erfcx_avx2.cpp")
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/erfcx_avx2.cpp" PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
else()
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/erfcx_avx2.cpp" PROPERTIES
        COMPILE_OPTIONS "-mavx2;-mfma")
endif()
target_include_directories(volatility_lab PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
