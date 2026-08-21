#!/usr/bin/env sh

# Proves exact legacy-directory selection, preview safety, Cargo-lock exclusion,
# and preservation of generated bindings and user-selected diagnostics.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=10
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${LOCK_HOLDER_PROCESS_ID:-}" ]; then
        kill "$LOCK_HOLDER_PROCESS_ID" 2>/dev/null || true
        wait "$LOCK_HOLDER_PROCESS_ID" 2>/dev/null || true
    fi
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        rm -rf "$SANDBOX_DIRECTORY"
    fi
}
trap cleanup 0

main() {
    for required_command in mktemp python3; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done
    if command -v timeout >/dev/null 2>&1; then
        timeout_executable="$(command -v timeout)"
    elif command -v gtimeout >/dev/null 2>&1; then
        timeout_executable="$(command -v gtimeout)"
    else
        print_error "GNU timeout is required; install Homebrew coreutils"
        exit 2
    fi

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    subject="${repository_root}/scripts/clean-retired-cargo-native-output.sh"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-native-cleanup.XXXXXX")"
    target_directory="${SANDBOX_DIRECTORY}/target"
    first_legacy_directory="${target_directory}/debug/build/astronomical-runtime-integration-first/out/mlx-c-runtime-build"
    second_legacy_directory="${target_directory}/aarch64-apple-darwin/release/build/astronomical-runtime-integration-second/out/mlx-c-runtime-build"
    interrupted_legacy_directory="${target_directory}/debug/build/astronomical-runtime-integration-interrupted/out/.mlx-c-runtime-build.astronomical-removing-100-1"
    retained_bindings="${target_directory}/debug/build/astronomical-runtime-integration-first/out/mlx_c_bindings.rs"
    retained_diagnostic="${target_directory}/full-debug/astronomicald.dSYM"
    mkdir -p "$first_legacy_directory" "$second_legacy_directory" \
        "$interrupted_legacy_directory" "$retained_diagnostic"
    printf '%s\n' '{}' > "${target_directory}/.rustc_info.json"
    printf '%s\n' legacy > "${first_legacy_directory}/CMakeCache.txt"
    printf '%s\n' legacy > "${second_legacy_directory}/CMakeCache.txt"
    printf '%s\n' legacy > "${interrupted_legacy_directory}/CMakeCache.txt"
    printf '%s\n' bindings > "$retained_bindings"

    printf '%s\n' '[legacy-native-output-cleanup-test] case=dry-run-preserves-output status=start'
    "$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" \
        "$subject" --dry-run --target-directory "$target_directory"
    [ -d "$first_legacy_directory" ] && [ -d "$second_legacy_directory" ] \
        && [ -d "$interrupted_legacy_directory" ] || {
        print_error "dry-run removed legacy native output"
        exit 1
    }
    printf '%s\n' '[legacy-native-output-cleanup-test] case=dry-run-preserves-output status=success'

    printf '%s\n' '[legacy-native-output-cleanup-test] case=apply-requires-cargo-ownership status=start'
    unowned_target_directory="${SANDBOX_DIRECTORY}/unowned-target"
    unowned_legacy_directory="${unowned_target_directory}/debug/build/astronomical-runtime-integration-unowned/out/mlx-c-runtime-build"
    mkdir -p "$unowned_legacy_directory"
    printf '%s\n' preserve > "${unowned_legacy_directory}/evidence"
    unowned_status=0
    "$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" \
        "$subject" --apply --target-directory "$unowned_target_directory" || unowned_status=$?
    [ "$unowned_status" -eq 1 ] && [ -f "${unowned_legacy_directory}/evidence" ] || {
        print_error "cleanup modified a directory without Cargo target ownership evidence"
        exit 1
    }
    printf '%s\n' '[legacy-native-output-cleanup-test] case=apply-requires-cargo-ownership status=success'

    printf '%s\n' '[legacy-native-output-cleanup-test] case=active-cargo-lock-refuses-cleanup status=start'
    lock_ready_path="${SANDBOX_DIRECTORY}/lock-ready"
    python3 - "$target_directory" "$lock_ready_path" <<'PYTHON' &
import fcntl
import pathlib
import sys
import time

target_directory = pathlib.Path(sys.argv[1])
lock_ready_path = pathlib.Path(sys.argv[2])
with (target_directory / "debug" / ".cargo-lock").open("a+b") as target_lock:
    fcntl.flock(target_lock.fileno(), fcntl.LOCK_EX)
    lock_ready_path.write_text("ready\n")
    time.sleep(30)
PYTHON
    LOCK_HOLDER_PROCESS_ID="$!"
    readiness_attempts=0
    while [ ! -f "$lock_ready_path" ] && [ "$readiness_attempts" -lt 50 ]; do
        sleep 0.1
        readiness_attempts=$((readiness_attempts + 1))
    done
    [ -f "$lock_ready_path" ] || {
        print_error "Cargo-lock fixture did not become ready"
        exit 1
    }
    locked_status=0
    "$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" \
        "$subject" --apply --target-directory "$target_directory" || locked_status=$?
    [ "$locked_status" -eq 1 ] || {
        print_error "cleanup did not refuse an active Cargo target lock"
        exit 1
    }
    kill "$LOCK_HOLDER_PROCESS_ID"
    wait "$LOCK_HOLDER_PROCESS_ID" 2>/dev/null || true
    LOCK_HOLDER_PROCESS_ID=""
    printf '%s\n' '[legacy-native-output-cleanup-test] case=active-cargo-lock-refuses-cleanup status=success'

    printf '%s\n' '[legacy-native-output-cleanup-test] case=symlink-candidate-refuses-cleanup status=start'
    unowned_directory="${SANDBOX_DIRECTORY}/unowned"
    symlink_candidate="${target_directory}/test/build/astronomical-runtime-integration-symlink/out/mlx-c-runtime-build"
    mkdir -p "$unowned_directory" "$(dirname -- "$symlink_candidate")"
    printf '%s\n' preserve > "${unowned_directory}/evidence"
    ln -s "$unowned_directory" "$symlink_candidate"
    symlink_status=0
    "$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" \
        "$subject" --apply --target-directory "$target_directory" || symlink_status=$?
    [ "$symlink_status" -eq 1 ] || {
        print_error "cleanup accepted a symbolic-link legacy directory"
        exit 1
    }
    [ -f "${unowned_directory}/evidence" ] && [ -d "$first_legacy_directory" ] || {
        print_error "cleanup changed content after symbolic-link validation failed"
        exit 1
    }
    rm "$symlink_candidate"
    printf '%s\n' '[legacy-native-output-cleanup-test] case=symlink-candidate-refuses-cleanup status=success'

    printf '%s\n' '[legacy-native-output-cleanup-test] case=apply-removes-only-legacy-output status=start'
    "$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" \
        "$subject" --apply --target-directory "$target_directory"
    [ ! -e "$first_legacy_directory" ] && [ ! -e "$second_legacy_directory" ] \
        && [ ! -e "$interrupted_legacy_directory" ] || {
        print_error "apply retained legacy native output"
        exit 1
    }
    [ -f "$retained_bindings" ] && [ -d "$retained_diagnostic" ] || {
        print_error "apply removed generated bindings or user-selected diagnostics"
        exit 1
    }
    printf '%s\n' '[legacy-native-output-cleanup-test] case=apply-removes-only-legacy-output status=success'
}

main "$@"
