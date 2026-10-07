// SPDX-License-Identifier: MIT
#pragma once
/// \file build_info.hpp
/// \brief Toolchain and host description captured at configure time.
///
/// Benchmarks are only meaningful alongside the machine they ran on.  Rather
/// than asking the author to keep a markdown table in sync with reality, the
/// build system bakes the real values in and every benchmark report prints
/// them from here.

#include <cstdint>

namespace vl {

struct BuildInfo {
    const char* version;
    const char* git_sha;
    const char* compiler_id;
    const char* compiler_version;
    const char* build_type;
    const char* cxx_standard;
    const char* simd_build_level;   ///< ISA the SIMD kernels were compiled for
    const char* target_system;
    const char* host_cpu;           ///< as reported at configure time
    const char* host_name;
    int         physical_cores;
    int         logical_cores;
    std::int64_t total_ram_mib;
    bool        openmp_enabled;
    bool        cuda_enabled;
};

/// Immutable description of how this binary was built.
const BuildInfo& build_info() noexcept;

}  // namespace vl
