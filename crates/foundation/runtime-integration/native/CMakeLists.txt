# The pinned native runtime build for Astronomical.
#
# Architecture in one paragraph: this project does NOT fetch MLX from the
# network. Five verified source archives (MLX, MLX-C, metal-cpp,
# nlohmann/json, fmt) are provisioned ahead of time by
# `cargo xtask bootstrap-native-dependencies` into a machine-wide astro native build cache,
# referenced here by absolute path, and re-verified by SHA-256 at configure
# time (require_pinned_archive below) so a corrupted or swapped archive
# fails loudly before any compile. MLX and MLX-C are then patched with the
# repository-owned patch set under third-party/patches (each patch's purpose
# and retirement condition are documented in third-party/README.md) and
# built as one static, Release-mode image: arm64 with Metal on macOS, the
# host architecture with the CPU backend on Linux, or, when the Rust side
# selects the Linux CUDA recipe (ASTRONOMICAL_MLX_BUILD_CUDA), the CUDA
# backend beside the CPU one (issue #2699).
#
# Consumers: the Cargo build script (build-orchestration/native_compile.rs)
# and the standalone native-build tool drive this file identically; the
# products are published into the identity-keyed native build store
# (build-orchestration/native_store.rs). This file is a fingerprinted native
# identity input — a comment or code change here re-keys the store entry and
# rebuilds the native runtime by design.
cmake_minimum_required(VERSION 3.24)

project(astronomical_runtime_integration LANGUAGES C CXX)

include(FetchContent)
include("${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/pins/mlx-v0.32.5.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/pins/mlx-c-v0.8.0.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/pins/mlx-cuda-sources-v0.32.5.cmake")
find_program(PATCH_EXECUTABLE patch REQUIRED)

# macOS builds the Metal backend on Apple silicon. Linux builds MLX's CPU
# backend, which is the first step towards GPU backends beyond Metal
# (issue #2424); it links the BLAS and LAPACK that MLX's configure finds.
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        message(FATAL_ERROR "Astronomical's MLX runtime integration requires Apple Silicon")
    endif()
    set(ASTRONOMICAL_MLX_BUILD_METAL ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ASTRONOMICAL_MLX_BUILD_METAL OFF)
else()
    message(FATAL_ERROR "Astronomical's MLX runtime integration may only build on macOS or Linux")
endif()

# The CUDA backend is chosen by the Rust native build (the LinuxCuda recipe in
# native-identity), never by default: it needs the CUDA toolkit, cuDNN and an
# NVIDIA GPU, which the Rust side has probed and fingerprinted.
option(ASTRONOMICAL_MLX_BUILD_CUDA "Build MLX's CUDA backend (Linux only)" OFF)
if(ASTRONOMICAL_MLX_BUILD_CUDA AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Astronomical builds MLX's CUDA backend on Linux only")
endif()

if(NOT DEFINED ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR OR "${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}" STREQUAL "")
    if(NOT DEFINED ENV{HOME} OR "$ENV{HOME}" STREQUAL "")
        message(FATAL_ERROR "Set ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR or HOME, then run cargo xtask bootstrap-native-dependencies")
    endif()
    set(ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR "$ENV{HOME}/Library/Caches/Astronomical/native-dependencies")
endif()
if(NOT IS_ABSOLUTE "${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}")
    message(FATAL_ERROR "ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR must be an absolute path: ${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}")
endif()

function(require_pinned_archive variable_name dependency_description archive_file_name expected_sha256)
    set(archive_path "${ASTRONOMICAL_NATIVE_DEPENDENCY_CACHE_DIR}/${archive_file_name}")
    if(NOT IS_ABSOLUTE "${archive_path}")
        message(FATAL_ERROR "${variable_name} must be an absolute path: ${archive_path}")
    endif()
    if(NOT EXISTS "${archive_path}")
        message(FATAL_ERROR "Missing ${dependency_description} archive: ${archive_path}. Run cargo xtask bootstrap-native-dependencies to provision the verified astro native build cache.")
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
if(ASTRONOMICAL_MLX_BUILD_CUDA)
    require_pinned_archive(ASTRONOMICAL_CCCL_SOURCE_ARCHIVE "CCCL ${ASTRONOMICAL_CCCL_VERSION}" "${ASTRONOMICAL_CCCL_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_CCCL_SOURCE_ARCHIVE_SHA256}")
    require_pinned_archive(ASTRONOMICAL_NVTX_SOURCE_ARCHIVE "NVTX ${ASTRONOMICAL_NVTX_VERSION}" "${ASTRONOMICAL_NVTX_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_NVTX_SOURCE_ARCHIVE_SHA256}")
    require_pinned_archive(ASTRONOMICAL_CUDNN_FRONTEND_SOURCE_ARCHIVE "cuDNN frontend ${ASTRONOMICAL_CUDNN_FRONTEND_VERSION}" "${ASTRONOMICAL_CUDNN_FRONTEND_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_CUDNN_FRONTEND_SOURCE_ARCHIVE_SHA256}")
    require_pinned_archive(ASTRONOMICAL_CUTLASS_SOURCE_ARCHIVE "CUTLASS ${ASTRONOMICAL_CUTLASS_VERSION}" "${ASTRONOMICAL_CUTLASS_SOURCE_ARCHIVE_FILE_NAME}" "${ASTRONOMICAL_CUTLASS_SOURCE_ARCHIVE_SHA256}")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build one static runtime image" FORCE)
set(MLX_BUILD_TESTS OFF CACHE BOOL "Disable upstream MLX tests" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "Disable upstream MLX examples" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "Disable upstream MLX benchmarks" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Exclude Python from the worker runtime" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "Disable unneeded GGUF support" FORCE)
set(MLX_BUILD_METAL ${ASTRONOMICAL_MLX_BUILD_METAL} CACHE BOOL "Require the Metal backend on macOS" FORCE)
set(MLX_BUILD_CPU ON CACHE BOOL "Retain bounded CPU fallback operations" FORCE)
set(MLX_BUILD_CUDA ${ASTRONOMICAL_MLX_BUILD_CUDA} CACHE BOOL "Build the CUDA backend for the Linux CUDA recipe" FORCE)
set(MLX_BUILD_SAFETENSORS ON CACHE BOOL "Require descriptor-backed safetensors loading" FORCE)
set(MLX_METAL_JIT ${ASTRONOMICAL_MLX_BUILD_METAL} CACHE BOOL "Select Metal kernels for the current Apple GPU and keep them in the metal_kernel_cache" FORCE)
set(MLX_ENABLE_X64_MAC OFF CACHE BOOL "Reject unsupported Intel macOS builds" FORCE)
set(MLX_USE_CCACHE OFF CACHE BOOL "Keep the native artifact graph machine-independent" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "Disable upstream MLX C examples" FORCE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OSX_DEPLOYMENT_TARGET "${ASTRONOMICAL_MLX_MACOS_DEPLOYMENT_TARGET}" CACHE STRING "Pinned deployment target" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
option(
    ASTRONOMICAL_BUILD_MEMORY_CONTRACT_PROBE
    "Build the explicit MLX allocator memory-contract probe"
    OFF
)

# These declarations intentionally use the dependency names expected by MLX
# and MLX C. FetchContent's first declaration wins, so every nested request is
# satisfied by a verified local archive rather than an upstream URL.
FetchContent_Declare(metal_cpp URL "${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_METAL_CPP_SOURCE_ARCHIVE_SHA256}")
FetchContent_Declare(json URL "${ASTRONOMICAL_JSON_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_NLOHMANN_JSON_SOURCE_ARCHIVE_SHA256}")
FetchContent_Declare(fmt URL "${ASTRONOMICAL_FMT_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_FMT_SOURCE_ARCHIVE_SHA256}")
# The four sources MLX's CUDA backend declares, under its names and with its
# SOURCE_SUBDIR and EXCLUDE_FROM_ALL settings, so its own declarations
# (mlx/backend/cuda/CMakeLists.txt) resolve to these verified archives.
if(ASTRONOMICAL_MLX_BUILD_CUDA)
    FetchContent_Declare(cccl URL "${ASTRONOMICAL_CCCL_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_CCCL_SOURCE_ARCHIVE_SHA256}")
    FetchContent_Declare(nvtx3 URL "${ASTRONOMICAL_NVTX_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_NVTX_SOURCE_ARCHIVE_SHA256}" SOURCE_SUBDIR c EXCLUDE_FROM_ALL)
    FetchContent_Declare(cudnn URL "${ASTRONOMICAL_CUDNN_FRONTEND_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_CUDNN_FRONTEND_SOURCE_ARCHIVE_SHA256}" EXCLUDE_FROM_ALL)
    FetchContent_Declare(cutlass URL "${ASTRONOMICAL_CUTLASS_SOURCE_ARCHIVE}" URL_HASH "SHA256=${ASTRONOMICAL_CUTLASS_SOURCE_ARCHIVE_SHA256}" SOURCE_SUBDIR include EXCLUDE_FROM_ALL)
endif()
# Every patch in the two PATCH_COMMAND lists below is documented with its
# purpose and retirement condition in the patch registry of
# third-party/README.md. Two patches retired at the v0.32.5 bump:
# jit-sdpa-dsplit-template-arity, whose defect upstream fixed in the same
# generator, and nax-qmm-tile-variants, whose 4x1 simdgroup tile for
# transposed prefill upstream now selects itself through its staged
# qmm_t_nax kernel.
FetchContent_Declare(
    mlx
    URL "${ASTRONOMICAL_MLX_SOURCE_ARCHIVE}"
    URL_HASH "SHA256=${ASTRONOMICAL_MLX_SOURCE_ARCHIVE_SHA256}"
    PATCH_COMMAND
        "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-0.32.5-stable-prefill-split-k.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-0.32.5-astronomical-active-memory-ceiling.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-0.32.5-streaming-safetensors-writer.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-0.32.5-fast-synch-setting.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-0.32.5-evaluation-counts.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-0.32.5-cuda-jit-header-directories.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
)
# No mlx-c release pairs with MLX v0.32.5 yet, so the pinned mlx-c v0.8.0
# carries two version-gap bridges, each deleted at the mlx-c release that makes
# it unnecessary. One passes MLX's own default for the window_size MLX inserted
# into scaled_dot_product_attention, so the C signature is unchanged. The
# other adds the <format> include that mlx-c's io.h needs for std::format:
# MLX v0.32.4's headers reached it through <chrono>, v0.32.5's do not, and
# libstdc++ (GCC on Linux) then stops compiling mlx-c (issue #2682). The
# remaining patches add what the C API does not offer: mlx_array_detach, and
# the two CUDA run-time settings the Linux CUDA recipe needs (the folders of
# MLX's run-time compiler headers, and TF32 for float32 matmuls).
FetchContent_Declare(
    mlx_c
    URL "${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE}"
    URL_HASH "SHA256=${ASTRONOMICAL_MLX_C_SOURCE_ARCHIVE_SHA256}"
    PATCH_COMMAND
        "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-c-0.8.0-expose-array-detach.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-c-0.8.0-attention-window-size-for-mlx-0.32.5.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-c-0.8.0-include-format-in-io-header.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
        COMMAND "${CMAKE_COMMAND}"
        "-DPATCH_EXECUTABLE=${PATCH_EXECUTABLE}"
        "-DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/../../../../third-party/patches/mlx-c-0.8.0-cuda-runtime-settings.patch"
        -P "${CMAKE_CURRENT_LIST_DIR}/apply_patch_if_needed.cmake"
)

FetchContent_MakeAvailable(metal_cpp json fmt mlx mlx_c)

if(NOT TARGET mlx OR NOT TARGET mlxc)
    message(FATAL_ERROR "Pinned archives did not define the expected mlx and mlxc targets")
endif()

if(ASTRONOMICAL_MLX_BUILD_CUDA)
    include("${CMAKE_CURRENT_LIST_DIR}/cuda_runtime_facts.cmake")
endif()

if(ASTRONOMICAL_BUILD_MEMORY_CONTRACT_PROBE)
    # The probe measures the Metal allocator through Mach and MLX's Metal
    # headers, so it cannot exist without Metal. Failing at configure time
    # spares a full MLX compile that would only fail at the probe.
    if(NOT ASTRONOMICAL_MLX_BUILD_METAL)
        message(FATAL_ERROR "The MLX memory contract probe measures the Metal allocator and builds on macOS only")
    endif()
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
