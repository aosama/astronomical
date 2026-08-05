#!/usr/bin/env sh

set -eu

REPOSITORY_ROOT="$(CDPATH='' cd -- "$(dirname -- "$0")/../../.." && pwd -P)"
readonly REPOSITORY_ROOT
readonly NATIVE_BUILD_DIRECTORY="${REPOSITORY_ROOT}/target/experimental-native-metal-expert-loader-tests"
readonly NATIVE_EXECUTABLE_PATH="${NATIVE_BUILD_DIRECTORY}/bin/astronomical_metal_expert_loader_native_tests"
readonly NATIVE_TEST_TIMEOUT_SECONDS=120

print_error() {
    printf '%s\n' "Error: $1" >&2
}

elapsed_seconds() {
    started_at_seconds="$1"
    finished_at_seconds="$(date +%s)"
    printf '%s' "$((finished_at_seconds - started_at_seconds))"
}

run_step() {
    step_name="$1"
    shift
    started_at_seconds="$(date +%s)"
    printf '%s\n' "[native-metal-expert-loader-script] status=start step=${step_name} timestamp_seconds=${started_at_seconds}"
    if "$@"; then
        printf '%s\n' "[native-metal-expert-loader-script] status=success step=${step_name} elapsed_seconds=$(elapsed_seconds "${started_at_seconds}")"
        return
    else
        step_status=$?
    fi

    print_error "step=${step_name} failed status=${step_status} elapsed_seconds=$(elapsed_seconds "${started_at_seconds}")"
    exit "${step_status}"
}

require_timeout_executable() {
    if command -v timeout >/dev/null 2>&1; then
        timeout_executable="$(command -v timeout)"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_executable="$(command -v gtimeout)"
    else
        print_error "GNU timeout is required; install Homebrew coreutils"
        exit 1
    fi
}

require_logical_cpu_count() {
    logical_cpu_count="$(sysctl -n hw.logicalcpu)"
    case "${logical_cpu_count}" in
        ''|*[!0-9]*)
            print_error "sysctl did not return a positive logical CPU count"
            exit 1
            ;;
    esac
    if [ "${logical_cpu_count}" -eq 0 ]; then
        print_error "sysctl returned zero logical CPUs"
        exit 1
    fi
}

parse_arguments() {
    if [ "$#" -eq 1 ] && [ "$1" = "contracts" ]; then
        return
    fi

    print_error "usage: experimental/aligned-expert-packs/scripts/test-experimental-native-metal-loader.sh contracts"
    exit 2
}

main() {
    parse_arguments "$@"
    require_timeout_executable
    require_logical_cpu_count

    compiler_launcher_path=""
    if command -v sccache >/dev/null 2>&1; then
        compiler_launcher_path="$(command -v sccache)"
        export SCCACHE_CLIENT_SIDE="${SCCACHE_CLIENT_SIDE:-1}"
        printf '%s\n' "[native-metal-expert-loader-script] status=enabled compiler_launcher=${compiler_launcher_path}"
    fi

    run_step configure cmake \
        -S "${REPOSITORY_ROOT}/crates/runtime-integration/native" \
        -B "${NATIVE_BUILD_DIRECTORY}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DASTRONOMICAL_BUILD_EXPERIMENTAL_ALIGNED_EXPERT_PACKS=ON \
        "-DCMAKE_C_COMPILER_LAUNCHER=${compiler_launcher_path}" \
        "-DCMAKE_CXX_COMPILER_LAUNCHER=${compiler_launcher_path}"
    run_step build cmake \
        --build "${NATIVE_BUILD_DIRECTORY}" \
        --target astronomical_metal_expert_loader_native_tests \
        --parallel "${logical_cpu_count}"

    run_step execute \
        "${timeout_executable}" -k 5s "${NATIVE_TEST_TIMEOUT_SECONDS}s" \
        "${NATIVE_EXECUTABLE_PATH}" contracts
}

main "$@"
