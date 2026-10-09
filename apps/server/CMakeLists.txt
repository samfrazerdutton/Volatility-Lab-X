# ---------------------------------------------------------------------------
# volatility_lab_server: a thin, local-only HTTP service wrapping the
# existing volatility_lab library for the web frontend. Binds to 127.0.0.1
# only -- never exposed to the network. Every endpoint calls into real,
# already-tested library functions; no numerical logic is duplicated here.
#
# cpp-httplib and nlohmann/json are both single-header, permissively
# licensed, and fetched the same way tests/CMakeLists.txt already fetches
# GoogleTest -- pinned tags, not a floating dependency.
# ---------------------------------------------------------------------------

include(FetchContent)

FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.6
    GIT_SHALLOW    TRUE
)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(HTTPLIB_COMPILE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_ZLIB OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_BROTLI OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(httplib nlohmann_json)

add_executable(volatility_lab_server main.cpp)
target_link_libraries(volatility_lab_server PRIVATE
    volatility_lab
    vl_compiler_flags
    httplib::httplib
    nlohmann_json::nlohmann_json
)
