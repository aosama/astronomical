#!/usr/bin/env sh

set -eu

readonly MAXIMUM_TEST_SECONDS=120
readonly MLX_PROCESS_QUIESCENCE_SECONDS=5

print_error() {
    printf '%s\n' "Error: $1" >&2
}

if [ "$#" -ne 0 ]; then
    print_error "test-mlx-memory-contracts.sh does not accept arguments"
    exit 2
fi

if command -v timeout >/dev/null 2>&1; then
    timeout_executable="$(command -v timeout)"
elif command -v gtimeout >/dev/null 2>&1; then
    timeout_executable="$(command -v gtimeout)"
else
    print_error "GNU timeout is required; install Homebrew coreutils"
    exit 1
fi

if ! command -v sccache >/dev/null 2>&1; then
    print_error "sccache is required for MLX memory-contract qualification"
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

printf '%s\n' "[mlx-memory-contracts] status=compiler_cache compiler_wrapper=${RUSTC_WRAPPER} build_jobs=${CARGO_BUILD_JOBS}"
sccache --show-stats

run_cargo_test_phase() {
    test_phase_name="$1"
    shift

    phase_started_at_seconds="$(date +%s)"
    printf '%s\n' "[mlx-memory-contracts] status=start phase=${test_phase_name} timeout_seconds=${MAXIMUM_TEST_SECONDS} started_at=$(date +%H:%M:%S)"
    if "${timeout_executable}" --foreground -k 5s "${MAXIMUM_TEST_SECONDS}s" cargo --verbose test "$@"; then
        phase_finished_at_seconds="$(date +%s)"
        printf '%s\n' "[mlx-memory-contracts] status=success phase=${test_phase_name} elapsed_seconds=$((phase_finished_at_seconds - phase_started_at_seconds))"
        return
    else
        cargo_test_exit_status=$?
    fi

    if [ "${cargo_test_exit_status}" -eq 124 ] || [ "${cargo_test_exit_status}" -eq 137 ]; then
        print_error "${test_phase_name} exceeded the ${MAXIMUM_TEST_SECONDS}-second safety timeout"
    else
        print_error "${test_phase_name} failed with status ${cargo_test_exit_status}"
    fi
    exit "${cargo_test_exit_status}"
}

run_cargo_test_phase model_swap_allocator \
    --package astronomical-model-serving \
    --test mlx_memory_contract_tests \
    --features direct-mlx \
    mlx_memory_contract::model_swap_allocator::should_clear_stale_mlx_allocator_memory_before_loading_the_replacement_model \
    -- --ignored --exact --nocapture --test-threads 1

quiescence_started_at_seconds="$(date +%s)"
printf '%s\n' "[mlx-memory-contracts] status=start phase=metal_quiescence delay_seconds=${MLX_PROCESS_QUIESCENCE_SECONDS}"
sleep "${MLX_PROCESS_QUIESCENCE_SECONDS}"
quiescence_finished_at_seconds="$(date +%s)"
printf '%s\n' "[mlx-memory-contracts] status=success phase=metal_quiescence elapsed_seconds=$((quiescence_finished_at_seconds - quiescence_started_at_seconds))"

run_cargo_test_phase native_mlx_allocator_probe \
    --package astronomical-runtime-integration \
    --test mlx_memory_contract_tests \
    --features mlx-memory-contract-probe \
    mlx_memory_contract::native_probe::should_pass_the_pinned_native_mlx_memory_contract_probe \
    -- --ignored --exact --nocapture --test-threads 1

run_cargo_test_phase mlx_c_memory_transitions \
    --package astronomical-runtime-integration \
    --test mlx_memory_contract_tests \
    --features mlx-memory-contract-probe \
    mlx_memory_contract::mlx_c_memory::should_preserve_native_memory_transitions_through_the_mlx_c_boundary \
    -- --ignored --exact --nocapture --test-threads 1

run_cargo_test_phase mlx_c_typed_capacity_rejections \
    --package astronomical-runtime-integration \
    --test mlx_memory_contract_tests \
    --features mlx-memory-contract-probe \
    mlx_memory_contract::mlx_c_memory::should_map_new_and_cached_mlx_capacity_rejections_to_typed_errors \
    -- --ignored --exact --nocapture --test-threads 1

run_cargo_test_phase mlx_c_host_backed_capacity_rejection \
    --package astronomical-runtime-integration \
    --test mlx_memory_contract_tests \
    --features mlx-memory-contract-probe \
    mlx_memory_contract::mlx_c_memory::should_preserve_host_backed_capacity_rejection_error_details \
    -- --ignored --exact --nocapture --test-threads 1

printf '%s\n' "[mlx-memory-contracts] status=success phase=complete"
