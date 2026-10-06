#!/usr/bin/env sh

# Runs the complete local commit gate on Cargo's stable routine artifact graph.
# Release and model acceptance remain isolated behind disposable journeys.
#
# The gate executes three independent phases concurrently (repository shell
# contracts, Swift/Node contracts, and the Cargo core) because they share no
# state: contract lanes exercise fixture scripts, the Swift lane builds its own
# package world, and the Cargo core owns the shared target directory. Within
# the Cargo core, the direct-MLX lane launches as soon as the combined
# compile-only invocation succeeds so its disposable-target graph builds while
# the hermetic suites run; the ordering still keeps the compile invocation as
# the second Cargo call, which the verification contract pins.
#
# Every step reports start, end, and elapsed seconds; every phase reports its
# own elapsed time; the run closes with the slowest steps and the Cargo timing
# report so slowness is attributable without re-reading the whole log.
#
# Cargo invocations the Cargo core phase runs, in order:
#   1. cargo fmt --all -- --check
#   2. cargo test-hermetic-and-rest --timings --no-run --jobs N   (compile only;
#      the alias lists the hermetic + REST packages and test binaries)
#   3. scripts/compile-mlx-memory-contract-lane.sh                (feature-gated
#      memory-contract world, compile only)
#   4. scripts/test-direct-mlx.sh (background lane; one cargo test invocation
#      for the direct-MLX contract binaries in an owned disposable target)
#   5. cargo test-hermetic-and-rest --jobs N -- --quiet --test-threads N
# Shell contracts run before these; Swift/Node contracts run alongside.

set -eu
# pipefail is Bash/Zsh; the subshell probe keeps this script POSIX-runnable.
# shellcheck disable=SC3040
if (set -o pipefail) 2>/dev/null; then
    set -o pipefail
fi

readonly COMPILE_TIMEOUT_SECONDS=600
readonly TEST_TIMEOUT_SECONDS=120
# The direct-MLX lane compiles its feature world in an owned Cargo target before
# its own bounded run, so it needs the compilation timeout class.
readonly DIRECT_MLX_TIMEOUT_SECONDS=600
# The Thin Talk package compiles its own Swift world before its hermetic canvas
# contracts run in a web view, so it needs the compilation timeout class rather
# than the 120-second test bound.
readonly THIN_TALK_TIMEOUT_SECONDS=600
# The Swift migration skeleton builds its own package world (including the
# MLX dependencies) before its hermetic journeys run, so it shares the
# compilation timeout class; its journeys stay individually bounded through
# their own Swift Testing time limits.
readonly SWIFT_SKELETON_TIMEOUT_SECONDS=600
readonly TOTAL_STEP_COUNT=26
readonly REPOSITORY_CONTRACT_STEP_COUNT=14
readonly SWIFT_NODE_CONTRACT_STEP_COUNT=6
readonly CARGO_CORE_STEP_COUNT=6
readonly PHASE_PROGRESS_INTERVAL_SECONDS=2
readonly FAILED_PHASE_LOG_TAIL_LINES=40
# Grace window for phase process groups to exit after a termination signal
# before the gate stops waiting on them; lane coordinators own their cleanup.
readonly PHASE_TERMINATION_GRACE_SECONDS=5

PHASE_LOG_DIRECTORY=""
VERIFICATION_FAILED="false"
REPOSITORY_CONTRACTS_PROCESS_ID=""
SWIFT_NODE_CONTRACTS_PROCESS_ID=""
CARGO_CORE_PROCESS_ID=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

print_usage() {
    printf '%s\n' "Usage: scripts/verify-before-commit.sh"
    printf '%s\n' ""
    printf '%s\n' "Runs formatting, repository contracts, application journeys, and the required Rust test boundaries in parallel phases."
}

require_command() {
    required_command_name="$1"
    command -v "$required_command_name" >/dev/null 2>&1 || {
        print_error "required command is unavailable: ${required_command_name}"
        exit 2
    }
}

resolve_timeout_executable() {
    if command -v timeout >/dev/null 2>&1; then
        timeout_executable="$(command -v timeout)"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_executable="$(command -v gtimeout)"
    else
        print_error "GNU timeout is required; install Homebrew coreutils"
        exit 2
    fi
}

line_count() {
    counted_file="$1"
    counted_lines="$(wc -l < "$counted_file" 2>/dev/null | tr -d '[:space:]')" || counted_lines=0
    case "$counted_lines" in
        ''|*[!0-9]*) counted_lines=0 ;;
    esac
    printf '%s\n' "$counted_lines"
}

run_step() {
    step_name="$1"
    step_timeout_seconds="$2"
    shift 2

    step_started_at_seconds="$(date +%s)"
    printf '[commit-verification] step=%s status=start timeout_seconds=%s started_at=%s phase=%s\n' \
        "$step_name" "$step_timeout_seconds" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$PHASE_NAME"
    if "$timeout_executable" --foreground -k 5s "${step_timeout_seconds}s" "$@"; then
        step_exit_code=0
    else
        step_exit_code=$?
        printf '[commit-verification] step=%s status=failed exit_code=%s elapsed_seconds=%s ended_at=%s phase=%s\n' \
            "$step_name" "$step_exit_code" "$(( $(date +%s) - step_started_at_seconds ))" \
            "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$PHASE_NAME" >&2
        return "$step_exit_code"
    fi

    COMPLETED_STEP_COUNT=$((COMPLETED_STEP_COUNT + 1))
    printf '[commit-verification] step=%s status=success elapsed_seconds=%s ended_at=%s phase=%s\n' \
        "$step_name" "$(( $(date +%s) - step_started_at_seconds ))" \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$PHASE_NAME"
}

run_phase() {
    phase_label="$1"
    phase_step_total="$2"
    phase_entry_point="$3"

    PHASE_NAME="$phase_label"
    COMPLETED_STEP_COUNT=0
    phase_started_at_seconds="$(date +%s)"
    printf '[commit-verification] phase=%s status=start steps=%s started_at=%s\n' \
        "$phase_label" "$phase_step_total" "$(date '+%Y-%m-%dT%H:%M:%S%z')"
    phase_exit_status=0
    "$phase_entry_point" || phase_exit_status=$?
    if [ "$phase_exit_status" -eq 0 ]; then
        [ "$COMPLETED_STEP_COUNT" -eq "$phase_step_total" ] || {
            print_error "phase ${phase_label} completed ${COMPLETED_STEP_COUNT} of ${phase_step_total} steps"
            return 1
        }
        printf '[commit-verification] phase=%s status=success steps=%s elapsed_seconds=%s\n' \
            "$phase_label" "$COMPLETED_STEP_COUNT" "$(( $(date +%s) - phase_started_at_seconds ))"
        return 0
    fi
    printf '[commit-verification] phase=%s status=failed completed=%s/%s elapsed_seconds=%s\n' \
        "$phase_label" "$COMPLETED_STEP_COUNT" "$phase_step_total" \
        "$(( $(date +%s) - phase_started_at_seconds ))" >&2
    return "$phase_exit_status"
}

# Starts one phase as its own process group whose output lands in the phase
# log; the phase writes its exit status to a marker file so the monitor can
# detect completion without reaping the process from the monitoring loop.
start_phase() {
    phase_status_directory="$1"
    phase_label="$2"
    phase_step_total="$3"
    phase_log_path="$4"
    phase_entry_point="$5"

    (
        phase_status=0
        run_phase "$phase_label" "$phase_step_total" "$phase_entry_point" || phase_status=$?
        printf '%s\n' "$phase_status" > "${phase_status_directory}/${phase_label}.status"
        exit "$phase_status"
    ) > "$phase_log_path" 2>&1 &
}

phase_repository_contracts() {
    # Explicit || return on every step: these phases run inside a checked
    # context (run_phase ... || status=$?), where set -e is suppressed, so a
    # failed step would otherwise be skipped over and the phase would report
    # success or keep running later steps.
    run_step rust-dependency-notices "$TEST_TIMEOUT_SECONDS" scripts/generate-rust-dependency-notices.sh --check || return $?
    run_step rust-dependency-notices-contract "$TEST_TIMEOUT_SECONDS" scripts/test-rust-dependency-notices-contract.sh || return $?
    run_step thin-talk-canvas-assets "$TEST_TIMEOUT_SECONDS" scripts/vendor-thin-talk-canvas-assets.sh --verify-only || return $?
    run_step commit-release-isolation "$TEST_TIMEOUT_SECONDS" scripts/test-commit-release-isolation.sh || return $?
    run_step ci-native-cache-contract "$TEST_TIMEOUT_SECONDS" scripts/test-ci-native-cache-coordination.sh || return $?
    run_step cache-prune-contract "$TEST_TIMEOUT_SECONDS" scripts/test-prune-ci-caches-contract.sh || return $?
    run_step sccache-save-contract "$TEST_TIMEOUT_SECONDS" scripts/test-save-sccache-cache-contract.sh || return $?
    run_step cargo-artifact-lifecycle-contract "$TEST_TIMEOUT_SECONDS" scripts/test-cargo-artifact-lifecycle-contract.sh || return $?
    run_step bounded-cargo-test-lock-contract "$TEST_TIMEOUT_SECONDS" scripts/test-bounded-cargo-test-lock-contract.sh || return $?
    run_step cargo-artifact-cleanup-signal-contract "$TEST_TIMEOUT_SECONDS" scripts/test-cargo-artifact-cleanup-signal-contract.sh || return $?
    run_step legacy-native-output-cleanup-contract "$TEST_TIMEOUT_SECONDS" scripts/test-retired-cargo-native-output-cleanup.sh || return $?
    run_step commit-verification-contract "$TEST_TIMEOUT_SECONDS" scripts/test-verify-before-commit-contract.sh || return $?
    run_step test-channel-isolation-checker "$TEST_TIMEOUT_SECONDS" scripts/test-channel-isolation-checker-contract.sh || return $?
    run_step test-channel-isolation "$TEST_TIMEOUT_SECONDS" scripts/check-test-channel-isolation.sh || return $?
}

phase_swift_node_contracts() {
    run_step test-macos-app-validation-contract "$TEST_TIMEOUT_SECONDS" scripts/test-validate-macos-app-contract.sh || return $?
    run_step test-macos-menu-contracts "$TEST_TIMEOUT_SECONDS" scripts/test-macos-menu-contracts.sh || return $?
    run_step thin-talk-contracts "$THIN_TALK_TIMEOUT_SECONDS" swift test --package-path apps/thin-talk || return $?
    # The Rust-to-Swift migration journeys: plain `swift test` streams every
    # journey's name, verdict, and duration through this phase's log, which
    # the monitor relays live, so progress and per-journey timing stay
    # visible without any wrapper-owned selection or silencing.
    run_step swift-skeleton-journeys "$SWIFT_SKELETON_TIMEOUT_SECONDS" swift test --package-path swift-skeleton || return $?
    run_step test-pull-request-policy-contracts "$TEST_TIMEOUT_SECONDS" node \
        --test --test-reporter=spec .github/scripts/pull-request-issue-compliance.test.js || return $?
    run_step test-observatory-contracts "$TEST_TIMEOUT_SECONDS" node --test --test-reporter=spec \
        apps/supervisor/console/console.test.js \
        apps/supervisor/console/library.test.js \
        apps/supervisor/console/library-fetch.test.js \
        apps/supervisor/console/connect.test.js \
        apps/supervisor/console/playground.test.js || return $?
}

launch_direct_mlx_lane() {
    run_step test-direct-mlx "$DIRECT_MLX_TIMEOUT_SECONDS" scripts/test-direct-mlx.sh &
    DIRECT_MLX_LANE_PROCESS_ID=$!
}

reap_direct_mlx_lane() {
    direct_mlx_status=0
    wait "$DIRECT_MLX_LANE_PROCESS_ID" || direct_mlx_status=$?
    return "$direct_mlx_status"
}

# The lane runs in a forked subshell, so its step completion can only be
# counted by this shell once the reap observes success; a forked counter
# would never propagate back.
count_direct_mlx_lane_step() {
    lane_exit_status=0
    reap_direct_mlx_lane || lane_exit_status=$?
    if [ "$lane_exit_status" -eq 0 ]; then
        COMPLETED_STEP_COUNT=$((COMPLETED_STEP_COUNT + 1))
    fi
    return "$lane_exit_status"
}

phase_cargo_core() {
    run_step format "$TEST_TIMEOUT_SECONDS" cargo fmt --all -- --check || return $?
    # The combined compile-only invocation must stay the second Cargo call: the
    # verification contract pins that ordering, and the direct-MLX lane may only
    # start once the shared graph has compiled so a compile failure stops the
    # journey before the lane burns its disposable-target build.
    # The native CMake builds run alone before any Rust compilation so the
    # two never compete for cores; every later Cargo step reuses the store.
    # The memory-contract profile is warmed here too because its store entry
    # carries the probe binary that the feature-gated lane gate compiles
    # against, and CMake must never overlap the Rust compile steps.
    run_step prewarm-native-build "$COMPILE_TIMEOUT_SECONDS" \
        scripts/prewarm-native-build.sh --profile core --profile core+memory-contract || return $?
    run_step compile-rust "$COMPILE_TIMEOUT_SECONDS" cargo test-hermetic-and-rest \
        --timings --no-run --jobs "$logical_cpu_count" || return $?
    # The memory-contract lane is feature-gated off every routine graph, so an
    # API move under its test binary used to ship silently; this compile-only
    # gate recompiles that feature world while the store is warm. The lane's
    # GPU journeys stay behind #[ignore] and run through
    # scripts/test-mlx-memory-contracts.sh, one process per limit profile.
    run_step compile-memory-contract-lane "$COMPILE_TIMEOUT_SECONDS" \
        scripts/compile-mlx-memory-contract-lane.sh || return $?
    # Hosted CI cannot execute the direct-MLX lane; the 2026-08-26 residency
    # regression proved behavioral breaks ship silently without it.
    # The lane owns a disposable target separate from the shared graph, so it
    # overlaps the hermetic suite run instead of queueing behind it. Its one
    # Cargo invocation also runs the hermetic MLX-C coverage contract binary,
    # which is why this gate carries no separate coverage-contract step: a
    # plain Cargo step here would block on the target lock the core phase
    # holds and die at its 120-second bound.
    DIRECT_MLX_LANE_PROCESS_ID=""
    launch_direct_mlx_lane
    run_rust_exit_status=0
    run_step run-rust "$TEST_TIMEOUT_SECONDS" cargo test-hermetic-and-rest \
        --jobs "$logical_cpu_count" -- --quiet --test-threads "$logical_cpu_count" || run_rust_exit_status=$?
    if [ "$run_rust_exit_status" -ne 0 ]; then
        # The lane coordinator owns its target cleanup through its own signal
        # traps; TERM lets it reap children and remove the disposable target,
        # and the bounded grace window keeps a wedged lane from stalling the
        # failure report.
        kill -TERM "--$DIRECT_MLX_LANE_PROCESS_ID" 2>/dev/null || kill -TERM "$DIRECT_MLX_LANE_PROCESS_ID" 2>/dev/null || true
        grace_seconds_remaining="$PHASE_TERMINATION_GRACE_SECONDS"
        while kill -0 "$DIRECT_MLX_LANE_PROCESS_ID" 2>/dev/null \
            && [ "$grace_seconds_remaining" -gt 0 ]; do
            sleep 1
            grace_seconds_remaining=$((grace_seconds_remaining - 1))
        done
        reap_direct_mlx_lane || true
        return "$run_rust_exit_status"
    fi
    count_direct_mlx_lane_step
}

relay_phase_output() {
    relay_phase_log="$1"
    relay_state_file="${relay_phase_log}.relayed"
    if [ -f "$relay_state_file" ]; then
        relayed_lines="$(cat "$relay_state_file")"
    else
        relayed_lines=0
    fi
    phase_lines="$(line_count "$relay_phase_log")"
    if [ "$phase_lines" -gt "$relayed_lines" ]; then
        sed -n "$((relayed_lines + 1)),${phase_lines}p" "$relay_phase_log"
        printf '%s\n' "$phase_lines" > "$relay_state_file"
    fi
}

monitor_phases() {
    phase_status_directory="$1"
    repository_log="$2"
    swift_node_log="$3"
    cargo_core_log="$4"
    while :; do
        relay_phase_output "$repository_log"
        relay_phase_output "$swift_node_log"
        relay_phase_output "$cargo_core_log"
        [ -f "${phase_status_directory}/repository-contracts.status" ] \
            && [ -f "${phase_status_directory}/swift-node-contracts.status" ] \
            && [ -f "${phase_status_directory}/cargo-core.status" ] && break
        sleep "$PHASE_PROGRESS_INTERVAL_SECONDS"
    done
    relay_phase_output "$repository_log"
    relay_phase_output "$swift_node_log"
    relay_phase_output "$cargo_core_log"
}

report_slowest_steps() {
    slowest_repository_log="$1"
    slowest_swift_node_log="$2"
    slowest_cargo_core_log="$3"
    for phase_log in "$slowest_repository_log" "$slowest_swift_node_log" "$slowest_cargo_core_log"; do
        [ -f "$phase_log" ] || continue
        sed -n 's/^.*step=\([^ ]*\) status=[a-z]* .*elapsed_seconds=\([0-9][0-9]*\) .* phase=\([^ ]*\).*$/\2 \1 \3/p' "$phase_log"
    done | sort -rn | head -5 | while IFS=' ' read -r slow_elapsed slow_step slow_phase; do
        printf '[commit-verification] slowest elapsed_seconds=%s step=%s phase=%s\n' \
            "$slow_elapsed" "$slow_step" "$slow_phase"
    done
}

report_cargo_timings_location() {
    if [ -f target/cargo-timings/cargo-timing.html ]; then
        printf '[commit-verification] cargo-timings report=%s\n' \
            "$repository_root/target/cargo-timings/cargo-timing.html"
    fi
}

cleanup_phase_logs() {
    if [ -z "$PHASE_LOG_DIRECTORY" ]; then
        return
    fi
    case "$PHASE_LOG_DIRECTORY" in
        /|.|..)
            print_error "refusing to remove unsafe phase log directory: ${PHASE_LOG_DIRECTORY}"
            return
            ;;
    esac
    if [ "$VERIFICATION_FAILED" = true ]; then
        printf '%s\n' "[commit-verification] phase-logs retained path=${PHASE_LOG_DIRECTORY}" >&2
        return
    fi
    rm -rf "$PHASE_LOG_DIRECTORY"
}

terminate_running_phases() {
    for running_process_id in "$REPOSITORY_CONTRACTS_PROCESS_ID" "$SWIFT_NODE_CONTRACTS_PROCESS_ID" "$CARGO_CORE_PROCESS_ID"; do
        [ -n "$running_process_id" ] || continue
        kill -TERM "--$running_process_id" 2>/dev/null || kill -TERM "$running_process_id" 2>/dev/null || true
    done
}

main() {
    if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
        print_usage
        return
    fi
    if [ "$#" -ne 0 ]; then
        print_error "verify-before-commit.sh does not accept arguments"
        print_usage >&2
        exit 2
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    CDPATH='' cd -- "$repository_root"
    require_command cargo
    require_command date
    require_command node
    require_command swift
    require_command sysctl
    resolve_timeout_executable

    logical_cpu_count="$(sysctl -n hw.logicalcpu)"
    case "$logical_cpu_count" in
        ''|*[!0-9]*|0)
            print_error "sysctl did not return a positive logical CPU count"
            exit 2
            ;;
    esac
    export CARGO_BUILD_JOBS="$logical_cpu_count"

    verification_started_at_seconds="$(date +%s)"
    printf '[commit-verification] status=start steps=%s cargo_target=%s rustc_wrapper=%s build_jobs=%s started_at=%s\n' \
        "$TOTAL_STEP_COUNT" "${CARGO_TARGET_DIR:-target}" "${RUSTC_WRAPPER:-none}" \
        "$CARGO_BUILD_JOBS" "$(date '+%Y-%m-%dT%H:%M:%S%z')"

    PHASE_LOG_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-commit-verification.XXXXXX")"
    REPOSITORY_CONTRACTS_LOG="${PHASE_LOG_DIRECTORY}/repository-contracts.log"
    SWIFT_NODE_CONTRACTS_LOG="${PHASE_LOG_DIRECTORY}/swift-node-contracts.log"
    CARGO_CORE_LOG="${PHASE_LOG_DIRECTORY}/cargo-core.log"
    trap cleanup_phase_logs EXIT
    trap 'terminate_running_phases; exit 130' INT
    trap 'terminate_running_phases; exit 143' TERM

    # Monitor mode gives each phase its own process group so an interruption
    # or a failed run can signal the phase's whole command tree.
    set -m
    start_phase "$PHASE_LOG_DIRECTORY" repository-contracts "$REPOSITORY_CONTRACT_STEP_COUNT" \
        "$REPOSITORY_CONTRACTS_LOG" phase_repository_contracts
    REPOSITORY_CONTRACTS_PROCESS_ID=$!
    start_phase "$PHASE_LOG_DIRECTORY" swift-node-contracts "$SWIFT_NODE_CONTRACT_STEP_COUNT" \
        "$SWIFT_NODE_CONTRACTS_LOG" phase_swift_node_contracts
    SWIFT_NODE_CONTRACTS_PROCESS_ID=$!
    start_phase "$PHASE_LOG_DIRECTORY" cargo-core "$CARGO_CORE_STEP_COUNT" \
        "$CARGO_CORE_LOG" phase_cargo_core
    CARGO_CORE_PROCESS_ID=$!
    set +m

    monitor_phases "$PHASE_LOG_DIRECTORY" "$REPOSITORY_CONTRACTS_LOG" "$SWIFT_NODE_CONTRACTS_LOG" "$CARGO_CORE_LOG"

    repository_exit_status=0
    wait "$REPOSITORY_CONTRACTS_PROCESS_ID" || repository_exit_status=$?
    swift_node_exit_status=0
    wait "$SWIFT_NODE_CONTRACTS_PROCESS_ID" || swift_node_exit_status=$?
    cargo_core_exit_status=0
    wait "$CARGO_CORE_PROCESS_ID" || cargo_core_exit_status=$?

    report_slowest_steps "$REPOSITORY_CONTRACTS_LOG" "$SWIFT_NODE_CONTRACTS_LOG" "$CARGO_CORE_LOG"
    report_cargo_timings_location

    total_exit_status=0
    for phase_result in "repository-contracts:$repository_exit_status" \
        "swift-node-contracts:$swift_node_exit_status" \
        "cargo-core:$cargo_core_exit_status"; do
        phase_result_name="${phase_result%%:*}"
        phase_result_status="${phase_result##*:}"
        [ "$phase_result_status" -eq 0 ] || {
            VERIFICATION_FAILED=true
            printf '[commit-verification] status=failed phase=%s exit_code=%s\n' \
                "$phase_result_name" "$phase_result_status" >&2
            case "$phase_result_name" in
                repository-contracts) failed_phase_log="$REPOSITORY_CONTRACTS_LOG" ;;
                swift-node-contracts) failed_phase_log="$SWIFT_NODE_CONTRACTS_LOG" ;;
                *) failed_phase_log="$CARGO_CORE_LOG" ;;
            esac
            tail -n "$FAILED_PHASE_LOG_TAIL_LINES" "$failed_phase_log" >&2 || true
            [ "$total_exit_status" -ne 0 ] || total_exit_status="$phase_result_status"
        }
    done
    [ "$total_exit_status" -eq 0 ] || exit "$total_exit_status"

    printf '[commit-verification] status=success steps=%s elapsed_seconds=%s ended_at=%s\n' \
        "$TOTAL_STEP_COUNT" "$(( $(date +%s) - verification_started_at_seconds ))" \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')"
}

main "$@"
