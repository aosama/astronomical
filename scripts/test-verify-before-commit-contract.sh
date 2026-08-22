#!/usr/bin/env sh

# Proves the commit gate reuses the caller's Cargo graph, selects both Rust
# boundaries once, and preserves bounded fail-fast execution from any directory.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=20
SANDBOX_DIRECTORY=""
SUBJECT_TIMEOUT_EXECUTABLE=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe verification sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

create_fake_commands() {
    fake_command_directory="$1"
    mkdir -p "$fake_command_directory"

    cat > "${fake_command_directory}/timeout" <<'TIMEOUT'
#!/usr/bin/env sh
set -eu
if [ "${1:-}" = "--foreground" ]; then shift; fi
if [ "${1:-}" = "-k" ]; then shift 2; fi
timeout_duration="$1"
shift
printf '%s|%s\n' "$timeout_duration" "$1" >> "${ASTRONOMICAL_TEST_TIMEOUT_LOG:?}"
exec "$@"
TIMEOUT
    cat > "${fake_command_directory}/sysctl" <<'SYSCTL'
#!/usr/bin/env sh
set -eu
printf '%s\n' 8
SYSCTL
    cat > "${fake_command_directory}/cargo" <<'CARGO'
#!/usr/bin/env sh
set -eu
invocation_count=0
if [ -f "${ASTRONOMICAL_TEST_CARGO_COUNT:?}" ]; then
    invocation_count="$(cat "$ASTRONOMICAL_TEST_CARGO_COUNT")"
fi
invocation_count=$((invocation_count + 1))
printf '%s\n' "$invocation_count" > "$ASTRONOMICAL_TEST_CARGO_COUNT"
printf '%s|pwd=%s|target=%s|wrapper=%s\n' "$*" "$PWD" "${CARGO_TARGET_DIR:-}" "${RUSTC_WRAPPER:-}" >> "${ASTRONOMICAL_TEST_CARGO_LOG:?}"
if [ "${ASTRONOMICAL_TEST_FAIL_CARGO_INVOCATION:-0}" -eq "$invocation_count" ]; then
    exit 37
fi
CARGO
    cat > "${fake_command_directory}/node" <<'NODE'
#!/usr/bin/env sh
set -eu
printf '%s\n' "$*" >> "${ASTRONOMICAL_TEST_NODE_LOG:?}"
NODE
    cat > "${fake_command_directory}/sccache" <<'SCCACHE'
#!/usr/bin/env sh
set -eu
printf '%s\n' "$*" >> "${ASTRONOMICAL_TEST_SCCACHE_LOG:?}"
exit 91
SCCACHE
    chmod +x "${fake_command_directory}/"*
}

create_fake_repository_scripts() {
    sandbox_scripts_directory="$1"
    for script_name in \
        generate-rust-dependency-notices.sh \
        test-commit-release-isolation.sh \
        test-ci-native-cache-coordination.sh \
        test-cargo-artifact-lifecycle-contract.sh \
        test-cargo-artifact-cleanup-signal-contract.sh \
        test-retired-cargo-native-output-cleanup.sh \
        test-verify-before-commit-contract.sh \
        test-channel-isolation-checker-contract.sh \
        test-validate-macos-app-contract.sh \
        check-test-channel-isolation.sh \
        test-macos-menu-contracts.sh
    do
        cat > "${sandbox_scripts_directory}/${script_name}" <<'SCRIPT'
#!/usr/bin/env sh
set -eu
printf '%s\n' "$(basename "$0") $*" >> "${ASTRONOMICAL_TEST_SCRIPT_LOG:?}"
SCRIPT
        chmod +x "${sandbox_scripts_directory}/${script_name}"
    done
}

assert_cargo_alias_contract() {
    repository_root="$1"
    REPOSITORY_ROOT="$repository_root" python3 <<'PYTHON'
import os
import pathlib
import tomllib

repository_root = pathlib.Path(os.environ["REPOSITORY_ROOT"])
aliases = tomllib.loads((repository_root / ".cargo/config.toml").read_text())["alias"]
verification_alias_arguments = aliases["verify-commit-rust"]
expected_alias_arguments = [
    "test",
    "--no-fail-fast",
    "-p", "astronomical-config",
    "-p", "astronomical-ipc-protocol",
    "-p", "astronomical-runtime-integration",
    "-p", "astronomical-model-serving",
    "-p", "astronomical-inference-worker",
    "-p", "astronomical-supervisor",
    "-p", "astronomical-rest-contract",
    "--test", "hermetic_tests",
    "--test", "rest_api_tests",
]
assert verification_alias_arguments == expected_alias_arguments
PYTHON
}

resolve_subject_timeout_executable() {
    if command -v timeout >/dev/null 2>&1; then
        SUBJECT_TIMEOUT_EXECUTABLE="$(command -v timeout)"
    elif command -v gtimeout >/dev/null 2>&1; then
        SUBJECT_TIMEOUT_EXECUTABLE="$(command -v gtimeout)"
    else
        print_error "GNU timeout is required for the verification contract"
        exit 2
    fi
}

run_verification_script() {
    verification_script_path="$1"
    output_path="$2"
    shift 2
    "$SUBJECT_TIMEOUT_EXECUTABLE" --foreground -k 5s "$SUBJECT_TIMEOUT_SECONDS" \
        "$verification_script_path" "$@" > "$output_path" 2>&1
}

main() {
    for required_command in basename cat chmod cp dirname grep mkdir mktemp python3 rm tr wc; do
        command -v "$required_command" >/dev/null 2>&1 || {
            print_error "required command is unavailable: ${required_command}"
            exit 2
        }
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    resolve_subject_timeout_executable
    assert_cargo_alias_contract "$repository_root"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-commit-verification.XXXXXX")"
    SANDBOX_DIRECTORY="$(CDPATH='' cd -- "$SANDBOX_DIRECTORY" && pwd -P)"
    sandbox_repository="${SANDBOX_DIRECTORY}/repository"
    sandbox_scripts_directory="${sandbox_repository}/scripts"
    sandbox_cargo_directory="${sandbox_repository}/.cargo"
    external_working_directory="${SANDBOX_DIRECTORY}/external"
    fake_command_directory="${SANDBOX_DIRECTORY}/fake-bin"
    mkdir -p "$sandbox_scripts_directory" "$sandbox_cargo_directory" "$external_working_directory"
    cp "${repository_root}/scripts/verify-before-commit.sh" "${sandbox_scripts_directory}/verify-before-commit.sh"
    cp "${repository_root}/.cargo/config.toml" "${sandbox_cargo_directory}/config.toml"
    chmod +x "${sandbox_scripts_directory}/verify-before-commit.sh"
    create_fake_commands "$fake_command_directory"
    create_fake_repository_scripts "$sandbox_scripts_directory"

    cargo_log="${SANDBOX_DIRECTORY}/cargo.log"
    cargo_count="${SANDBOX_DIRECTORY}/cargo.count"
    node_log="${SANDBOX_DIRECTORY}/node.log"
    script_log="${SANDBOX_DIRECTORY}/script.log"
    sccache_log="${SANDBOX_DIRECTORY}/sccache.log"
    timeout_log="${SANDBOX_DIRECTORY}/timeout.log"
    verification_output="${SANDBOX_DIRECTORY}/verification.log"
    custom_target_directory="${SANDBOX_DIRECTORY}/caller-cargo-target"
    mkdir -p "$custom_target_directory"

    printf '%s\n' '[commit-verification-contract] case=complete-external-journey status=start'
    (
        CDPATH='' cd -- "$external_working_directory"
        PATH="${fake_command_directory}:${PATH}" \
            CARGO_TARGET_DIR="$custom_target_directory" \
            RUSTC_WRAPPER=caller-selected-wrapper \
            ASTRONOMICAL_TEST_CARGO_LOG="$cargo_log" \
            ASTRONOMICAL_TEST_CARGO_COUNT="$cargo_count" \
            ASTRONOMICAL_TEST_NODE_LOG="$node_log" \
            ASTRONOMICAL_TEST_SCRIPT_LOG="$script_log" \
            ASTRONOMICAL_TEST_SCCACHE_LOG="$sccache_log" \
            ASTRONOMICAL_TEST_TIMEOUT_LOG="$timeout_log" \
            run_verification_script "${sandbox_scripts_directory}/verify-before-commit.sh" "$verification_output"
    )

    [ "$(wc -l < "$cargo_log" | tr -d '[:space:]')" -eq 3 ] || {
        print_error "verification did not use exactly three Cargo invocations including formatting"
        exit 1
    }
    grep -F "fmt --all -- --check|pwd=${sandbox_repository}|target=${custom_target_directory}|wrapper=caller-selected-wrapper" "$cargo_log" >/dev/null || {
        print_error "formatting did not preserve the caller Cargo context"
        exit 1
    }
    grep -F 'verify-commit-rust --timings --no-run --jobs 8' "$cargo_log" >/dev/null || {
        print_error "Rust compilation was not one combined compile-only invocation"
        exit 1
    }
    grep -F 'verify-commit-rust --jobs 8 -- --quiet --test-threads 8' "$cargo_log" >/dev/null || {
        print_error "Rust tests were not one combined bounded execution invocation"
        exit 1
    }
    [ ! -e "$sccache_log" ] || {
        print_error "verification invoked sccache"
        exit 1
    }
    grep -F '600s|cargo' "$timeout_log" >/dev/null || {
        print_error "Rust compilation did not retain its separate timeout"
        exit 1
    }
    [ "$(grep -c '^120s|' "$timeout_log")" -eq 15 ] || {
        print_error "verification did not bound every non-compilation step to 120 seconds"
        exit 1
    }
    [ "$(wc -l < "$node_log" | tr -d '[:space:]')" -eq 2 ] || {
        print_error "verification did not retain separate JavaScript attribution boundaries"
        exit 1
    }
    grep -Fx -- '--test --test-reporter=spec .github/scripts/pull-request-issue-compliance.test.js' "$node_log" >/dev/null
    grep -Fx -- '--test --test-reporter=spec apps/supervisor/console/console.test.js apps/supervisor/console/library.test.js' "$node_log" >/dev/null
    for expected_script_name in \
        generate-rust-dependency-notices.sh \
        test-commit-release-isolation.sh \
        test-ci-native-cache-coordination.sh \
        test-cargo-artifact-lifecycle-contract.sh \
        test-cargo-artifact-cleanup-signal-contract.sh \
        test-retired-cargo-native-output-cleanup.sh \
        test-verify-before-commit-contract.sh \
        test-channel-isolation-checker-contract.sh \
        test-validate-macos-app-contract.sh \
        check-test-channel-isolation.sh \
        test-macos-menu-contracts.sh
    do
        [ "$(grep -c "^${expected_script_name} " "$script_log")" -eq 1 ] || {
            print_error "verification did not run ${expected_script_name} exactly once"
            exit 1
        }
    done
    grep -F '[commit-verification] status=success' "$verification_output" >/dev/null || {
        print_error "verification did not report journey success"
        exit 1
    }
    printf '%s\n' '[commit-verification-contract] case=complete-external-journey status=success'

    printf '%s\n' '[commit-verification-contract] case=compile-failure-stops-journey status=start'
    rm -f "$cargo_log" "$cargo_count" "$timeout_log"
    failure_output="${SANDBOX_DIRECTORY}/failure.log"
    failure_exit_status=0
    (
        CDPATH='' cd -- "$external_working_directory"
        PATH="${fake_command_directory}:${PATH}" \
            ASTRONOMICAL_TEST_FAIL_CARGO_INVOCATION=2 \
            ASTRONOMICAL_TEST_CARGO_LOG="$cargo_log" \
            ASTRONOMICAL_TEST_CARGO_COUNT="$cargo_count" \
            ASTRONOMICAL_TEST_NODE_LOG="$node_log" \
            ASTRONOMICAL_TEST_SCRIPT_LOG="$script_log" \
            ASTRONOMICAL_TEST_SCCACHE_LOG="$sccache_log" \
            ASTRONOMICAL_TEST_TIMEOUT_LOG="$timeout_log" \
            run_verification_script "${sandbox_scripts_directory}/verify-before-commit.sh" "$failure_output"
    ) || failure_exit_status=$?
    [ "$failure_exit_status" -eq 37 ] || {
        print_error "compile failure returned ${failure_exit_status} instead of 37"
        exit 1
    }
    [ "$(wc -l < "$cargo_log" | tr -d '[:space:]')" -eq 2 ] || {
        print_error "verification continued after Rust compilation failed"
        exit 1
    }
    grep -F '[commit-verification] step=compile-rust status=failed exit_code=37' "$failure_output" >/dev/null || {
        print_error "verification did not attribute the failed Rust compilation"
        exit 1
    }
    printf '%s\n' '[commit-verification-contract] case=compile-failure-stops-journey status=success'
    printf '%s\n' '[commit-verification-contract] status=success'
}

main "$@"
