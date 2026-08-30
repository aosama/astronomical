#!/usr/bin/env sh

# Proves the complete disposable Cargo-target journey with one minimal real
# Cargo build: callers enter from any directory, nested lanes share one owner,
# and every owned target is removed after success, failure, or interruption.

set -eu

readonly SUBJECT_TIMEOUT_SECONDS=10
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -z "${SANDBOX_DIRECTORY:-}" ] || [ ! -d "$SANDBOX_DIRECTORY" ]; then
        return
    fi

    case "$SANDBOX_DIRECTORY" in
        /|.|..)
            print_error "refusing to remove unsafe Cargo lifecycle test sandbox"
            ;;
        *)
            rm -rf "$SANDBOX_DIRECTORY"
            ;;
    esac
}
trap cleanup 0

assert_path_is_absent() {
    unexpected_path="$1"
    [ ! -e "$unexpected_path" ] && [ ! -L "$unexpected_path" ] || {
        print_error "expected lifecycle-owned path to be absent: ${unexpected_path}"
        exit 1
    }
}

assert_lane_root_is_empty() {
    asserted_lane_root="$1"
    [ -z "$(ls -A "$asserted_lane_root")" ] || {
        print_error "disposable Cargo lane root retained an artifact"
        exit 1
    }
}

run_subject() {
    timeout_executable="$1"
    shift
    "$timeout_executable" --foreground -k 1s "${SUBJECT_TIMEOUT_SECONDS}s" "$@"
}

main() {
    for required_command in mktemp cp chmod python3; do
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

    python3 -c 'import tomllib' >/dev/null 2>&1 || {
        print_error "Python 3.11 or newer is required for TOML contract validation"
        exit 2
    }

    # This contract owns isolated targets even when commit verification itself
    # is running inside an outer disposable lifecycle.
    unset ASTRONOMICAL_CARGO_TARGET_LIFECYCLE
    unset ASTRONOMICAL_CARGO_TARGET_LANE
    unset CARGO_TARGET_DIR

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-cargo-lifecycle.XXXXXX")"
    sandbox_repository="${SANDBOX_DIRECTORY}/repository"
    sandbox_scripts_directory="${sandbox_repository}/scripts"
    lane_root="${SANDBOX_DIRECTORY}/cargo-lanes"
    mkdir -p "$sandbox_scripts_directory" "$lane_root"
    lane_root="$(CDPATH='' cd -- "$lane_root" && pwd -P)"

    cp "${repository_root}/scripts/run-in-disposable-cargo-target.sh" \
        "${sandbox_scripts_directory}/run-in-disposable-cargo-target.sh"
    cp "${repository_root}/scripts/run-disposable-cargo-journey.sh" \
        "${sandbox_scripts_directory}/run-disposable-cargo-journey.sh"
    cp "${repository_root}/scripts/run-bounded-cargo-test.sh" \
        "${sandbox_scripts_directory}/run-bounded-cargo-test.sh"
    chmod +x "${sandbox_scripts_directory}/run-in-disposable-cargo-target.sh" \
        "${sandbox_scripts_directory}/run-disposable-cargo-journey.sh" \
        "${sandbox_scripts_directory}/run-bounded-cargo-test.sh"
    lifecycle_subject="${sandbox_scripts_directory}/run-in-disposable-cargo-target.sh"
    journey_subject="${sandbox_scripts_directory}/run-disposable-cargo-journey.sh"

    cat > "${sandbox_scripts_directory}/cargo-lifecycle-fixture.sh" <<'FIXTURE'
#!/usr/bin/env sh
set -eu

fixture_operation="$1"
target_record_path="$2"
printf '%s\n' "${CARGO_TARGET_DIR:?disposable Cargo target is required}" > "$target_record_path"
printf '%s\n' "${RUSTC_WRAPPER:-none}" > "${target_record_path}.rustc-wrapper"

case "$fixture_operation" in
    capture)
        printf '%s\n' "$PWD" > "${target_record_path}.working-directory"
        mkdir -p "${CARGO_TARGET_DIR}/debug"
        printf '%s\n' artifact > "${CARGO_TARGET_DIR}/debug/fixture"
        ;;
    capture-package-version)
        python3 - "$PWD/Cargo.toml" "${target_record_path}.package-version" <<'PYTHON'
import pathlib
import sys
import tomllib

manifest_path = pathlib.Path(sys.argv[1])
version_record_path = pathlib.Path(sys.argv[2])
package_version = tomllib.loads(manifest_path.read_text())["package"]["version"]
version_record_path.write_text(f"{package_version}\n")
PYTHON
        exec cargo build --verbose
        ;;
    fail)
        mkdir -p "${CARGO_TARGET_DIR}/debug"
        exit 23
        ;;
    nested)
        "${ASTRONOMICAL_TEST_REPOSITORY_ROOT}/scripts/run-in-disposable-cargo-target.sh" \
            --lane nested-acceptance -- \
            "${ASTRONOMICAL_TEST_REPOSITORY_ROOT}/scripts/cargo-lifecycle-fixture.sh" \
            capture "${target_record_path}.nested"
        ;;
    interrupt)
        mkdir -p "${CARGO_TARGET_DIR}/debug"
        trap '' TERM
        sleep 30 &
        descendant_process_id="$!"
        printf '%s\n' "$descendant_process_id" > "${target_record_path}.descendant-process"
        wait "$descendant_process_id"
        ;;
    background-success)
        mkdir -p "${CARGO_TARGET_DIR}/debug"
        sleep 30 &
        descendant_process_id="$!"
        printf '%s\n' "$descendant_process_id" > "${target_record_path}.descendant-process"
        ;;
    replace-with-symlink)
        rm -rf "$CARGO_TARGET_DIR"
        ln -s "${ASTRONOMICAL_TEST_UNOWNED_DIRECTORY}" "$CARGO_TARGET_DIR"
        ;;
    *)
        printf '%s\n' "unknown fixture operation: ${fixture_operation}" >&2
        exit 2
        ;;
esac
FIXTURE
    chmod +x "${sandbox_scripts_directory}/cargo-lifecycle-fixture.sh"
    lifecycle_fixture="${sandbox_scripts_directory}/cargo-lifecycle-fixture.sh"

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=success-from-external-directory status=start'
    success_target_record="${SANDBOX_DIRECTORY}/success-target"
    success_log="${SANDBOX_DIRECTORY}/success.log"
    (
        CDPATH='' cd -- "$SANDBOX_DIRECTORY"
        ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
            RUSTC_WRAPPER='' \
            run_subject "$timeout_executable" "$lifecycle_subject" --lane direct-mlx -- \
            "$lifecycle_fixture" capture "$success_target_record"
    ) > "$success_log"
    success_target="$(cat "$success_target_record")"
    canonical_sandbox_repository="$(CDPATH='' cd -- "$sandbox_repository" && pwd -P)"
    recorded_working_directory="$(cat "${success_target_record}.working-directory")"
    [ "$recorded_working_directory" = "$canonical_sandbox_repository" ] || {
        print_error "disposable Cargo command used ${recorded_working_directory} instead of ${canonical_sandbox_repository}"
        exit 1
    }
    case "$success_target" in
        "${lane_root}/astronomical-cargo-direct-mlx."*) ;;
        *) print_error "disposable Cargo target escaped its lane root"; exit 1 ;;
    esac
    assert_path_is_absent "$success_target"
    assert_lane_root_is_empty "$lane_root"
    recorded_rustc_wrapper="$(cat "${success_target_record}.rustc-wrapper")"
    if command -v sccache >/dev/null 2>&1; then
        [ "$recorded_rustc_wrapper" = "sccache" ] || {
            print_error "disposable Cargo lifecycle did not default to available sccache"
            exit 1
        }
    else
        [ "$recorded_rustc_wrapper" = "none" ] || {
            print_error "disposable Cargo lifecycle did not preserve the no-wrapper fallback"
            exit 1
        }
    fi
    grep -F 'status=start' "$success_log" >/dev/null || {
        print_error "lifecycle did not report operation start"
        exit 1
    }
    grep -F 'status=cleanup-start' "$success_log" >/dev/null || {
        print_error "lifecycle did not report cleanup start"
        exit 1
    }
    grep -F 'status=removing' "$success_log" >/dev/null || {
        print_error "lifecycle did not report measured removal start"
        exit 1
    }
    grep -F 'status=removed' "$success_log" >/dev/null || {
        print_error "lifecycle did not attribute artifact removal"
        exit 1
    }
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=success-from-external-directory status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=default-macos-temporary-root-is-canonical status=start'
    default_root_target_record="${SANDBOX_DIRECTORY}/default-root-target"
    (
        unset TMPDIR ASTRONOMICAL_CARGO_LANE_ROOT CARGO_TARGET_DIR
        run_subject "$timeout_executable" "$lifecycle_subject" --lane default-root -- \
            "$lifecycle_fixture" capture "$default_root_target_record"
    )
    default_root_target="$(cat "$default_root_target_record")"
    case "$default_root_target" in
        /*/astronomical-cargo-default-root.*) ;;
        *) print_error "default temporary Cargo target was not canonical and absolute"; exit 1 ;;
    esac
    assert_path_is_absent "$default_root_target"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=default-macos-temporary-root-is-canonical status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=package-version-change-retains-no-generation status=start'
    package_manifest="${sandbox_repository}/Cargo.toml"
    package_source_directory="${sandbox_repository}/src"
    first_version_target_record="${SANDBOX_DIRECTORY}/first-version-target"
    second_version_target_record="${SANDBOX_DIRECTORY}/second-version-target"
    mkdir -p "$package_source_directory"
    cat > "${package_source_directory}/main.rs" <<'RUST'
fn main() {
    println!("Romeo and Juliet");
}
RUST
    cat > "$package_manifest" <<'TOML'
[package]
name = "cargo-lifecycle-fixture"
version = "0.2.10"
edition = "2024"
TOML
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane version-churn -- \
        "$lifecycle_fixture" capture-package-version "$first_version_target_record"
    cat > "$package_manifest" <<'TOML'
[package]
name = "cargo-lifecycle-fixture"
version = "0.2.11"
edition = "2024"
TOML
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane version-churn -- \
        "$lifecycle_fixture" capture-package-version "$second_version_target_record"
    [ "$(cat "${first_version_target_record}.package-version")" = "0.2.10" ] || {
        print_error "first disposable journey did not observe its package version"
        exit 1
    }
    [ "$(cat "${second_version_target_record}.package-version")" = "0.2.11" ] || {
        print_error "second disposable journey did not observe its changed package version"
        exit 1
    }
    [ "$(cat "$first_version_target_record")" != "$(cat "$second_version_target_record")" ] || {
        print_error "package-version journeys unexpectedly reused one disposable target"
        exit 1
    }
    assert_path_is_absent "$(cat "$first_version_target_record")"
    assert_path_is_absent "$(cat "$second_version_target_record")"
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=package-version-change-retains-no-generation status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=steady-state-storage-remains-bounded status=start'
    bounded_storage_root="${SANDBOX_DIRECTORY}/bounded-storage"
    retained_target_directory="${bounded_storage_root}/retained-target"
    bounded_lane_root="${bounded_storage_root}/cargo-lanes"
    first_bounded_target_record="${SANDBOX_DIRECTORY}/first-bounded-target"
    second_bounded_target_record="${SANDBOX_DIRECTORY}/second-bounded-target"
    mkdir -p "${retained_target_directory}/debug"
    mkdir -p "$bounded_lane_root"
    printf '%s\n' retained-artifact > "${retained_target_directory}/debug/artifact"
    baseline_allocated_kibibytes="$(du -sk "$bounded_storage_root")"
    baseline_allocated_kibibytes="${baseline_allocated_kibibytes%%[[:space:]]*}"
    ASTRONOMICAL_CARGO_LANE_ROOT="$bounded_lane_root" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane bounded-storage -- \
        "$lifecycle_fixture" capture "$first_bounded_target_record"
    ASTRONOMICAL_CARGO_LANE_ROOT="$bounded_lane_root" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane bounded-storage -- \
        "$lifecycle_fixture" capture "$second_bounded_target_record"
    retained_allocated_kibibytes="$(du -sk "$bounded_storage_root")"
    retained_allocated_kibibytes="${retained_allocated_kibibytes%%[[:space:]]*}"
    [ "$retained_allocated_kibibytes" -le $((baseline_allocated_kibibytes * 2)) ] || {
        print_error "retained artifacts exceeded twice the clean steady-state baseline"
        exit 1
    }
    assert_path_is_absent "$(cat "$first_bounded_target_record")"
    assert_path_is_absent "$(cat "$second_bounded_target_record")"
    assert_lane_root_is_empty "$bounded_lane_root"
    printf '%s\n' "[cargo-artifact-lifecycle-test] case=steady-state-storage-remains-bounded status=success baseline_bytes=$((baseline_allocated_kibibytes * 1024)) retained_bytes=$((retained_allocated_kibibytes * 1024))"

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=nested-lane-reuses-owner status=start'
    nested_target_record="${SANDBOX_DIRECTORY}/nested-target"
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        ASTRONOMICAL_TEST_REPOSITORY_ROOT="$sandbox_repository" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane outer-acceptance -- \
        "$lifecycle_fixture" nested "$nested_target_record"
    [ "$(cat "$nested_target_record")" = "$(cat "${nested_target_record}.nested")" ] || {
        print_error "nested Cargo lane created a second target owner"
        exit 1
    }
    assert_path_is_absent "$(cat "$nested_target_record")"
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=nested-lane-reuses-owner status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=failure-preserves-status-and-cleans status=start'
    failure_target_record="${SANDBOX_DIRECTORY}/failure-target"
    failure_status=0
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane failed-acceptance -- \
        "$lifecycle_fixture" fail "$failure_target_record" || failure_status=$?
    [ "$failure_status" -eq 23 ] || {
        print_error "lifecycle did not preserve the child failure status"
        exit 1
    }
    assert_path_is_absent "$(cat "$failure_target_record")"
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=failure-preserves-status-and-cleans status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=successful-parent-stops-background-descendant status=start'
    background_target_record="${SANDBOX_DIRECTORY}/background-target"
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane background-descendant -- \
        "$lifecycle_fixture" background-success "$background_target_record"
    background_descendant_process_id="$(cat "${background_target_record}.descendant-process")"
    if kill -0 "$background_descendant_process_id" 2>/dev/null; then
        print_error "successful lifecycle left descendant process ${background_descendant_process_id} running"
        exit 1
    fi
    assert_path_is_absent "$(cat "$background_target_record")"
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=successful-parent-stops-background-descendant status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=interruption-cleans status=start'
    interruption_target_record="${SANDBOX_DIRECTORY}/interruption-target"
    interruption_status=0
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        "$timeout_executable" --foreground -k 5s 2s "$lifecycle_subject" --lane interrupted-acceptance -- \
        "$lifecycle_fixture" interrupt "$interruption_target_record" || interruption_status=$?
    [ "$interruption_status" -eq 124 ] || [ "$interruption_status" -eq 143 ] || {
        print_error "interrupted lifecycle returned unexpected status ${interruption_status}"
        exit 1
    }
    interrupted_descendant_process_id="$(cat "${interruption_target_record}.descendant-process")"
    if kill -0 "$interrupted_descendant_process_id" 2>/dev/null; then
        print_error "interrupted lifecycle left descendant process ${interrupted_descendant_process_id} running"
        exit 1
    fi
    assert_path_is_absent "$(cat "$interruption_target_record")"
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=interruption-cleans status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=unowned-symlink-is-refused status=start'
    symlink_target_record="${SANDBOX_DIRECTORY}/symlink-target"
    unowned_directory="${SANDBOX_DIRECTORY}/unowned"
    mkdir -p "$unowned_directory"
    printf '%s\n' preserve > "${unowned_directory}/evidence"
    symlink_status=0
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        ASTRONOMICAL_TEST_UNOWNED_DIRECTORY="$unowned_directory" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane unsafe-cleanup -- \
        "$lifecycle_fixture" replace-with-symlink "$symlink_target_record" || symlink_status=$?
    [ "$symlink_status" -eq 1 ] || {
        print_error "lifecycle accepted an unowned replacement path"
        exit 1
    }
    [ -f "${unowned_directory}/evidence" ] || {
        print_error "lifecycle removed content through an unowned symlink"
        exit 1
    }
    replaced_target="$(cat "$symlink_target_record")"
    [ -L "$replaced_target" ] || {
        print_error "symlink refusal fixture did not replace the owned target"
        exit 1
    }
    rm "$replaced_target"
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=unowned-symlink-is-refused status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=invalid-lane-is-rejected status=start'
    invalid_lane_status=0
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        run_subject "$timeout_executable" "$lifecycle_subject" --lane '../escape' -- \
        "$lifecycle_fixture" capture "${SANDBOX_DIRECTORY}/invalid-target" || invalid_lane_status=$?
    [ "$invalid_lane_status" -eq 2 ] || {
        print_error "invalid lane name returned unexpected status ${invalid_lane_status}"
        exit 1
    }
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=invalid-lane-is-rejected status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=direct-mlx-journey-is-disposable status=start'
    fake_command_directory="${SANDBOX_DIRECTORY}/fake-bin"
    direct_mlx_target_record="${SANDBOX_DIRECTORY}/direct-mlx-target"
    mkdir -p "$fake_command_directory"
    cat > "${fake_command_directory}/cargo" <<'CARGO'
#!/usr/bin/env sh
set -eu
printf '%s\n' "${CARGO_TARGET_DIR:?}" > "${ASTRONOMICAL_TEST_CARGO_TARGET_RECORD:?}"
mkdir -p "${CARGO_TARGET_DIR}/debug"
exit 0
CARGO
    chmod +x "${fake_command_directory}/cargo"
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        ASTRONOMICAL_TEST_CARGO_TARGET_RECORD="$direct_mlx_target_record" \
        PATH="${fake_command_directory}:${PATH}" \
        run_subject "$timeout_executable" "${repository_root}/scripts/test-direct-mlx.sh"
    assert_path_is_absent "$(cat "$direct_mlx_target_record")"
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=direct-mlx-journey-is-disposable status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=named-acceptance-journey-is-disposable status=start'
    named_journey_target_record="${SANDBOX_DIRECTORY}/named-journey-target"
    ASTRONOMICAL_CARGO_LANE_ROOT="$lane_root" \
        ASTRONOMICAL_TEST_CARGO_TARGET_RECORD="$named_journey_target_record" \
        PATH="${fake_command_directory}:${PATH}" \
        run_subject "$timeout_executable" "$journey_subject" test-model-ssd-streaming-support
    assert_path_is_absent "$(cat "$named_journey_target_record")"
    assert_lane_root_is_empty "$lane_root"
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=named-acceptance-journey-is-disposable status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] case=profile-and-command-ownership status=start'
    journey_list_path="${SANDBOX_DIRECTORY}/journey-list"
    "$journey_subject" --list > "$journey_list_path"
    ASTRONOMICAL_TEST_REPOSITORY_ROOT="$repository_root" \
        ASTRONOMICAL_TEST_JOURNEY_LIST="$journey_list_path" python3 <<'PYTHON'
import os
import pathlib
import tomllib

repository_root = pathlib.Path(os.environ["ASTRONOMICAL_TEST_REPOSITORY_ROOT"])
journey_list_path = pathlib.Path(os.environ["ASTRONOMICAL_TEST_JOURNEY_LIST"])
workspace_manifest = tomllib.loads((repository_root / "Cargo.toml").read_text())
profiles = workspace_manifest["profile"]
assert profiles["dev"]["debug"] == "line-tables-only"
assert profiles["dev"]["split-debuginfo"] == "off"
assert profiles["test"]["debug"] == "line-tables-only"
assert profiles["test"]["split-debuginfo"] == "off"
assert profiles["full-debug"]["debug"] is True
assert profiles["full-debug"]["split-debuginfo"] == "packed"

cargo_configuration = tomllib.loads((repository_root / ".cargo/config.toml").read_text())
aliases = cargo_configuration["alias"]
assert set(aliases) == {"test-hermetic", "test-rest-api", "verify-commit-rust"}
assert all(isinstance(alias_command, list) for alias_command in aliases.values())

bounded_test_runner = (repository_root / "scripts/run-bounded-cargo-test.sh").read_text()
assert "readonly COMPILE_TIMEOUT_SECONDS=600" in bounded_test_runner
assert "readonly DEFAULT_TEST_TIMEOUT_SECONDS=120" in bounded_test_runner
journey_dispatcher = (repository_root / "scripts/run-disposable-cargo-journey.sh").read_text()
assert "run-bounded-cargo-test.sh" in journey_dispatcher
ignored_suite_runner = (repository_root / "scripts/run-ignored-serving-acceptance.sh").read_text()
assert "run-bounded-cargo-test.sh" in ignored_suite_runner
assert "--ignored --list" in ignored_suite_runner
assert "--foreground" in (repository_root / "scripts/test-direct-mlx.sh").read_text()
assert "--foreground" in (
    repository_root / "scripts/accept-prompt-cache-interactions.sh"
).read_text()

expected_journeys = {
    "accept-model-ssd-streaming",
    "accept-serving",
    "accept-cache-disabled-generation",
    "accept-cached-reverse-model-swap",
    "accept-tool-call-reuse",
    "accept-laguna-family-swap",
    "accept-thinking-seed",
    "accept-hard-thinking-budget",
    "accept-speculative-prefill",
    "accept-prompt-cache",
    "test-model-ssd-streaming-support",
    "test-persistent-prompt-cache-performance-support",
    "measure-model-ssd-streaming-summary",
    "measure-persistent-prompt-cache-warmup",
    "measure-persistent-prompt-cache-warmup-50k",
    "measure-persistent-prompt-cache-warmup-100k",
    "measure-model-ssd-streaming-cold-prefill-50k",
    "measure-model-ssd-streaming-prefill-1024",
    "measure-model-ssd-streaming-prefill-2048",
    "measure-model-ssd-streaming-prefill-4096",
    "measure-model-ssd-streaming-prefill-8192",
    "measure-model-ssd-streaming-read-concurrency",
    "measure-model-ssd-streaming-complete-expert-residency",
    "measure-model-ssd-streaming-leftover-complete-layer-seating",
    "measure-model-ssd-streaming-live-memory-ceiling-round-trip",
    "measure-model-ssd-streaming-decode-expert-retention",
    "measure-model-ssd-streaming-cached-suffix-streaming-prefill",
    "measure-model-ssd-streaming-high-ram-cached-suffix-prefill",
    "measure-model-ssd-streaming-large-sparse-moe-tight-ceiling-prefill",
    "measure-model-ssd-streaming-prefill-memory-progress",
    "measure-model-ssd-streaming-laguna-paging",
    "measure-experimental-aligned-expert-packs-large-sparse-moe-generation",
    "measure-experimental-aligned-expert-packs-large-sparse-moe-prompt-processing",
    "measure-experimental-aligned-expert-packs-large-sparse-moe-data-plane",
}
assert set(journey_list_path.read_text().splitlines()) == expected_journeys
PYTHON
    printf '%s\n' '[cargo-artifact-lifecycle-test] case=profile-and-command-ownership status=success'

    printf '%s\n' '[cargo-artifact-lifecycle-test] status=success'
}

main "$@"
