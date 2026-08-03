cmake_minimum_required(VERSION 3.24)

project(astronomical_runtime_integration LANGUAGES C CXX)

include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/../../../third-party/pins/mlx-v0.32.0.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../../third-party/pins/mlx-c-v0.6.0.cmake")
find_program(PATCH_EXECUTABLE patch REQUIRED)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    message(FATAL_ERROR "Astronomical's MLX runtime integration may only build on macOS")
endif()
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR "Astronomical's MLX runtime integration requires Apple Silicon")
endif()

if(NOT DEFINED ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR OR "${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}" STREQUAL "")
    if(NOT DEFINED ENV{HOME} OR "$ENV{HOME}" STREQUAL "")
        message(FATAL_ERROR "Set ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR or HOME, then run scripts/bootstrap-native-dependencies.sh")
    endif()
    set(ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR "$ENV{HOME}/Library/Caches/Astronomical/native-dependencies")
endif()
if(NOT IS_ABSOLUTE "${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}")
    message(FATAL_ERROR "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR must be an absolute path: ${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}")
endif()

function(require_pinned_archive variable_name dependency_description archive_file_name expected_sha256)
    if(NOT DEFINED ${variable_name} OR "${${variable_name}}" STREQUAL "")
        set(archive_path "${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}/${archive_file_name}")
    else()
        set(archive_path "${${variable_name}}")
    endif()
    if(NOT IS_ABSOLUTE "${archive_path}")
        message(FATAL_ERROR "${variable_name} must be an absolute path: ${archive_path}")
    endif()
    if(NOT EXISTS "${archive_path}")
        message(FATAL_ERROR "Missing ${dependency_description} archive: ${archive_path}. Run scripts/bootstrap-native-dependencies.sh to provision the verified cache.")
    endif()
    file(SHA256 "${archive_path}" detected_archive_sha256)
    if(NOT detected_archive_sha256 STREQUAL expected_sha256)
        message(FATAL_ERROR "${dependency_description} archive SHA-256 ${detected_archive_sha256} does not match required ${expected_sha256}")
    endif()
    set(${variable_name} "${archive_path}" PARENT_SCOPE)
endfunction()

require_pinned_archive(ASTRONOMICAL_MLX_SOURCE_ARCHIVE "MLX ${ASTRONOMICAL_MLX_VERSION}" "${ASTRONOMICAL_MLX_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_MLX_SOURCE_ARCHIVE_SHA256}")
require_pinned_archive(ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE "MLX C ${ASTRONOMICAL_MLX_C_VERSION}" "${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_SHA256}")
require_pinned_archive(ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE "metal-cpp ${ASTRONOMICAL_METAL_CPP_VERSION}" "${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_SHA256}")
require_pinned_archive(ASTRONOMICAL_JSON_SOURCE_ARCHIVE "nlohmann/json ${ASTRONOMICAL_NLOHMANN_JSON_VERSION}" "${ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_SHA256}")
require_pinned_archive(ASTRONOMICAL_FMT_SOURCE_ARCHIVE "fmt ${ASTRONOMICAL_FMT_VERSION}" "${ASTRONOMICAL_FMT_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_FMT_SOURCE_ARCHIVE_SHA256}")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build one static runtime image" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "Disable upstream MLX tests" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "Disable upstream MLX examples" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "Disable upstream MLX benchmarks" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Exclude Python from the worker runtime" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "Disable unneeded GGUF support" FORCE)
set(MLX_BUILD_METAL ON CACHE BOOL "Require the Metal backend" FORCE)
set(MLX_BUILD_CPU ON CACHE BOOL "Retain bounded CPU fallback operations" FORCE)
set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "Require descriptor-backed safetensors loading" FORCE)
set(MLX_METAL_JIT ON CACHE BOOL "Select and cache Metal kernels for the current Apple GPU" FORCE)
set(MLX_ENABLE_X64_MAC OFF CACHE BOOL "Reject unsupported Intel macOS builds" FORCE)
set(MLX_USE_CCACHE OFF CACHE BOOL "Keep the native artifact graph machine-independent" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "Disable upstream MLX C examples" FORCE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OSX_DEPLOYMENT_TARGET "${ASTRONOMICAL_MLX_MACOS_DEPLOYMENT_TARGET}" CACHE STRING "Pinned deployment target" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# These declarations intentionally use the dependency names expected by MLX
# and MLX C. FetchContent's first declaration wins, so every nested request is
# satisfied by a verified local archive rather than an upstream URL.
FetchContent_Declare(metal_cpp URL "${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_SHA256}")
FetchContent_Declare(json URL "${ASTRONOMICAL_JSON_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_SHA256}")
FetchContent_Declare(fmt URL "${ASTRONOMICAL_FMT_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_FMT_SOURCE_ARCHIVE_SHA256}")
FetchContent_Declare(
    mlx
    URL "${ASTRONOMICAL_MLX_SOURCE_ARCHIVE}"
    URL_HASH "SHA256=${ASTRONOMICAL_MLX_SOURCE_ARCHIVE_SHA256}"
    PATCH_COMMAND
        "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../third-party/patches/mlx-0.32.0-nax-head-dim-256-attention.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../third-party/patches/mlx-0.32.0-stable-prefill-split-k.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../third-party/patches/mlx-0.32.0-astronomical-active-memory-ceiling.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../third-party/patches/mlx-0.32.0-adaptive-io-thread-pool.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
)
FetchContent_Declare(
    mlx_c
    URL "${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE}"
    URL_HASH "SHA256=${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_SHA256}"
    PATCH_COMMAND
        "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../third-party/patches/mlx-c-0.6.0-metallib-path.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../third-party/patches/mlx-c-0.6.0-mlx-0.32.0-fft-compatibility.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
)

FetchContent_MakeAvailable(metal_cpp json fmt mlx mlx_c)

if(NOT TARGET mlx OR NOT TARGET mlxc)
    message(FATAL_ERROR "Pinned archives did not define the expected mlx and mlxc targets")
endif()

add_library(
    astronomical_metal_expert_loader
    STATIC
    astronomical_metal_expert_loader.cpp
)
target_compile_features(astronomical_metal_expert_loader PRIVATE cxx_std_20)
target_link_libraries(astronomical_metal_expert_loader PRIVATE mlx mlxc)
target_include_directories(
    astronomical_metal_expert_loader
    PRIVATE
    "${mlx_SOURCE_DIR}"
    "${mlx_c_SOURCE_DIR}"
)
set_target_properties(
    astronomical_metal_expert_loader
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    )

add_executable(
    astronomical_metal_expert_loader_native_tests
    EXCLUDE_FROM_ALL
    tests/astronomical_metal_expert_loader_native_tests.cpp
)
target_compile_features(astronomical_metal_expert_loader_native_tests PRIVATE cxx_std_20)
target_link_libraries(
    astronomical_metal_expert_loader_native_tests
    PRIVATE
    astronomical_metal_expert_loader
    mlx
    mlxc
)
target_include_directories(
    astronomical_metal_expert_loader_native_tests
    PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${mlx_SOURCE_DIR}"
    "${mlx_c_SOURCE_DIR}"
)
set_target_properties(
    astronomical_metal_expert_loader_native_tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if(ASTRONOMICAL_BUILD_MEMORY_CONTRACT_PROBE)
    add_executable(
        mlx_memory_contract_probe
        EXCLUDE_FROM_ALL
        tests/mlx_memory_contract_probe.cpp
    )
    target_compile_features(mlx_memory_contract_probe PRIVATE cxx_std_20)
    target_link_libraries(mlx_memory_contract_probe PRIVATE mlx)
    set_target_properties(
        mlx_memory_contract_probe
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
