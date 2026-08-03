#!/usr/bin/env sh

set -eu

readonly MAXIMUM_BUILD_SECONDS=600
readonly MAXIMUM_TEST_SECONDS=120

print_error() {
    printf '%s\n' "Error: $1" >&2
}

if command -v timeout >/dev/null 2>&1; then
    timeout_executable="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
    timeout_executable="$(command -v gtimeout)"
else
    print_error "GNU timeout is required; install Homebrew coreutils"
    exit 1
fi

if ! command -v sccache >/dev/null 2>&1; then
    print_error "sccache is required for commit verification"
    exit 1
fi

logical_cpu_count="$(sysctl -n hw.logicalcpu)"
case "${logical_cpu_count}" in
    ''|*[!0-9]*|0)
        print_error "sysctl did not return a positive logical CPU count"
        exit 1
        ;;
esac

export CARGO_BUILD_JOBS="${logical_cpu_count}"
export RUSTC_WRAPPER=sccache

printf '%s\n' "[commit-verification] compiler_cache=${RUSTC_WRAPPER} build_jobs=${CARGO_BUILD_JOBS} test_threads=${logical_cpu_count}"
printf '%s\n' "[commit-verification] included_tests=hermetic,rest_api excluded_tests=direct_mlx,mlx_memory_contract,model_artifact_qualification,persistent_prompt_cache_qualification,performance_measurement,macos_menu_contract,native_metal_contract,structural_guard"
printf '%s\n' "[commit-verification] sccache stats before verification:"
sccache --show-stats

printf '\n%s\n' "[commit-verification] step=rust-dependency-notices timeout_seconds=${MAXIMUM_TEST_SECONDS} started_at=$(date +%H:%M:%S)"
"${timeout_executable}" --foreground -k 5s "${MAXIMUM_TEST_SECONDS}s" scripts/generate-rust-dependency-notices.sh --check
printf '%s\n' "[commit-verification] PASSED step=rust-dependency-notices"

run_cargo_step() {
    step_name="$1"
    step_timeout_seconds="$2"
    shift 2

    printf '\n%s\n' "[commit-verification] step=${step_name} timeout_seconds=${step_timeout_seconds} started_at=$(date +%H:%M:%S)"
    step_started_at_seconds="$(date +%s)"
    "${timeout_executable}" --foreground -k 5s "${step_timeout_seconds}s" cargo --verbose "$@"
    printf '%s\n' "[commit-verification] PASSED step=${step_name} elapsed_seconds=$(( $(date +%s) - step_started_at_seconds ))"
}

run_cargo_step format "${MAXIMUM_TEST_SECONDS}" fmt --all -- --check
run_cargo_step compile-hermetic "${MAXIMUM_BUILD_SECONDS}" test --no-fail-fast --no-run --jobs "${logical_cpu_count}" -p astronomical-config -p astronomical-ipc-protocol -p astronomical-runtime-integration -p astronomical-model-serving -p astronomical-inference-worker -p astronomical-supervisor --test hermetic_tests
run_cargo_step compile-rest-api "${MAXIMUM_BUILD_SECONDS}" test --no-fail-fast --no-run --jobs "${logical_cpu_count}" -p astronomical-rest-contract -p astronomical-supervisor --test rest_api_tests
run_cargo_step run-hermetic "${MAXIMUM_TEST_SECONDS}" test --no-fail-fast --jobs "${logical_cpu_count}" -p astronomical-config -p astronomical-ipc-protocol -p astronomical-runtime-integration -p astronomical-model-serving -p astronomical-inference-worker -p astronomical-supervisor --test hermetic_tests -- --test-threads "${logical_cpu_count}"
run_cargo_step run-rest-api "${MAXIMUM_TEST_SECONDS}" test --no-fail-fast --jobs "${logical_cpu_count}" -p astronomical-rest-contract -p astronomical-supervisor --test rest_api_tests -- --test-threads "${logical_cpu_count}"

printf '\n%s\n' "[commit-verification] sccache stats after verification:"
sccache --show-stats
printf '\n%s\n' "[commit-verification] ALL STEPS PASSED"
