#!/usr/bin/env sh

# Proves the CI journey where equivalent native builds wait for one cache
# producer and every verification run explains the cache state it received.

set -eu

readonly EXPECTED_FINGERPRINT_LENGTH=40
SANDBOX_DIRECTORY=""

print_error() {
    printf '%s\n' "Error: $1" >&2
}

cleanup() {
    if [ -n "${SANDBOX_DIRECTORY:-}" ] && [ -d "$SANDBOX_DIRECTORY" ]; then
        case "$SANDBOX_DIRECTORY" in
            /|.|..) print_error "refusing to remove unsafe CI cache-test sandbox" ;;
            *) rm -rf "$SANDBOX_DIRECTORY" ;;
        esac
    fi
}
trap cleanup 0

require_command() {
    command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || {
        print_error "required command is unavailable: ${command_name}"
        exit 2
    }
}

create_fingerprint_fixture() {
    fixture_root="$1"
    mkdir -p \
        "${fixture_root}/crates/runtime-integration/native" \
        "${fixture_root}/third-party/pins" \
        "${fixture_root}/third-party/patches" \
        "${fixture_root}/scripts"
    cp "${repository_root}/scripts/native-build-cache-fingerprint.sh" \
        "${fixture_root}/scripts/native-build-cache-fingerprint.sh"
    printf '%s\n' '[workspace]' > "${fixture_root}/Cargo.toml"
    printf '%s\n' 'version = 4' > "${fixture_root}/Cargo.lock"
    printf '%s\n' '[toolchain]' > "${fixture_root}/rust-toolchain.toml"
    printf '%s\n' '[package]' > "${fixture_root}/crates/runtime-integration/Cargo.toml"
    printf '%s\n' 'fn main() {}' > "${fixture_root}/crates/runtime-integration/build.rs"
    printf '%s\n' 'fn generate() {}' > "${fixture_root}/crates/runtime-integration/build_bindings.rs"
    printf '%s\n' 'project(runtime)' > "${fixture_root}/crates/runtime-integration/native/CMakeLists.txt"
    printf '%s\n' 'set(MLX_VERSION 1)' > "${fixture_root}/third-party/native-dependency-manifest.cmake"
    printf '%s\n' 'set(MLX_PIN 1)' > "${fixture_root}/third-party/pins/mlx.cmake"
    printf '%s\n' 'native patch' > "${fixture_root}/third-party/patches/mlx.patch"
    printf '%s\n' 'unrelated documentation' > "${fixture_root}/README.md"
    git -C "$fixture_root" init --quiet
    git -C "$fixture_root" add .
}

assert_fingerprint_shape() {
    fingerprint="$1"
    [ "${#fingerprint}" -eq "$EXPECTED_FINGERPRINT_LENGTH" ] || {
        print_error "fingerprint length was ${#fingerprint}, expected ${EXPECTED_FINGERPRINT_LENGTH}"
        exit 1
    }
    case "$fingerprint" in
        *[!0-9a-f]*)
            print_error "fingerprint was not lowercase hexadecimal"
            exit 1
            ;;
    esac
}

assert_cache_classification() {
    expected_classification="$1"
    cache_hit="$2"
    matched_key="$3"
    report_output="$(
        CACHE_HIT="$cache_hit" \
        CACHE_MATCHED_KEY="$matched_key" \
        CACHE_PRIMARY_KEY='macOS-ARM64-rust-current' \
        GITHUB_STEP_SUMMARY="${SANDBOX_DIRECTORY}/step-summary.md" \
        "${repository_root}/scripts/report-build-cache-restoration.sh"
    )"
    case "$report_output" in
        *"classification=${expected_classification}"*) ;;
        *)
            print_error "cache state was not classified as ${expected_classification}: ${report_output}"
            exit 1
            ;;
    esac
}

assert_workflow_contract() {
    workflow_path="$1"
    # GitHub expressions must reach Ruby unchanged so the contract compares the
    # workflow's actual expression strings rather than shell-expanded values.
    # shellcheck disable=SC2016
    ruby -ryaml -e '
        workflow = YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)
        workflow_concurrency = workflow.fetch("concurrency")
        unless workflow_concurrency.fetch("cancel-in-progress") == false
          raise "same-ref workflow updates may cancel an active cache producer"
        end
        jobs = workflow.fetch("jobs")
        detection_job = jobs.fetch("detect-changes")
        fingerprint_output = detection_job.fetch("outputs").fetch("native_build_cache_fingerprint")
        expected_output = "${{ steps.native-build-fingerprint.outputs.fingerprint }}"
        raise "native fingerprint output is not published" unless fingerprint_output == expected_output

        verification_job = jobs.fetch("verify")
        concurrency = verification_job.fetch("concurrency")
        concurrency_group = concurrency.fetch("group")
        unless concurrency_group.include?("needs.detect-changes.outputs.native_build_cache_fingerprint")
          raise "verification jobs are not coordinated by native fingerprint"
        end
        raise "a waiting native build may cancel its cache producer" unless concurrency.fetch("cancel-in-progress") == false

        steps = verification_job.fetch("steps")
        cache_step = steps.find { |step| step["id"] == "build-state-cache" }
        raise "build cache step has no observable identifier" unless cache_step
        unless cache_step.fetch("uses").start_with?("actions/cache/restore@")
          raise "cache restoration does not expose primary and matched keys"
        end
        cache_key = cache_step.fetch("with").fetch("key")
        unless cache_key.include?("needs.detect-changes.outputs.native_build_cache_fingerprint")
          raise "exact build cache key omits the native fingerprint"
        end

        save_step = steps.find { |step| step["name"] == "Save Rust and native build state" }
        raise "successful cache production is not published" unless save_step
        unless save_step.fetch("uses").start_with?("actions/cache/save@")
          raise "cache publication does not use the dedicated save action"
        end
        unless save_step.fetch("with").fetch("key").include?("steps.build-state-cache.outputs.cache-primary-key")
          raise "cache publication does not use the attempted primary key"
        end

        report_step = steps.find { |step| step["name"] == "Report build cache restoration" }
        raise "cache restoration classification is absent" unless report_step
        report_environment = report_step.fetch("env")
        raise "cache hit output is not reported" unless report_environment.fetch("CACHE_HIT").include?("steps.build-state-cache.outputs.cache-hit")
        raise "matched cache key is not reported" unless report_environment.fetch("CACHE_MATCHED_KEY").include?("steps.build-state-cache.outputs.cache-matched-key")
        raise "primary cache key is not reported" unless report_environment.fetch("CACHE_PRIMARY_KEY").include?("steps.build-state-cache.outputs.cache-primary-key")
    ' "$workflow_path"
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "test-ci-native-cache-coordination.sh does not accept arguments"
        exit 2
    fi
    for required_command in git mktemp ruby; do
        require_command "$required_command"
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-ci-native-cache.XXXXXX")"
    fixture_root="${SANDBOX_DIRECTORY}/repository"
    create_fingerprint_fixture "$fixture_root"

    printf '%s\n' '[ci-native-cache-test] case=stable-fingerprint status=start'
    baseline_fingerprint="$("${fixture_root}/scripts/native-build-cache-fingerprint.sh" "$fixture_root")"
    repeated_fingerprint="$("${fixture_root}/scripts/native-build-cache-fingerprint.sh" "$fixture_root")"
    assert_fingerprint_shape "$baseline_fingerprint"
    [ "$baseline_fingerprint" = "$repeated_fingerprint" ] || {
        print_error "unchanged native inputs produced different fingerprints"
        exit 1
    }
    printf '%s\n' '[ci-native-cache-test] case=stable-fingerprint status=success'

    printf '%s\n' '[ci-native-cache-test] case=unrelated-change status=start'
    printf '%s\n' 'updated unrelated documentation' > "${fixture_root}/README.md"
    git -C "$fixture_root" add README.md
    unrelated_change_fingerprint="$("${fixture_root}/scripts/native-build-cache-fingerprint.sh" "$fixture_root")"
    [ "$baseline_fingerprint" = "$unrelated_change_fingerprint" ] || {
        print_error "an unrelated file invalidated the native fingerprint"
        exit 1
    }
    printf '%s\n' '[ci-native-cache-test] case=unrelated-change status=success'

    printf '%s\n' '[ci-native-cache-test] case=rust-graph-change status=start'
    printf '%s\n' 'version = 5' > "${fixture_root}/Cargo.lock"
    git -C "$fixture_root" add Cargo.lock
    rust_change_fingerprint="$("${fixture_root}/scripts/native-build-cache-fingerprint.sh" "$fixture_root")"
    [ "$baseline_fingerprint" = "$rust_change_fingerprint" ] || {
        print_error "a Rust-only graph change invalidated the native fingerprint"
        exit 1
    }
    printf '%s\n' '[ci-native-cache-test] case=rust-graph-change status=success'

    printf '%s\n' '[ci-native-cache-test] case=native-change status=start'
    printf '%s\n' 'updated native patch' > "${fixture_root}/third-party/patches/mlx.patch"
    git -C "$fixture_root" add third-party/patches/mlx.patch
    native_change_fingerprint="$("${fixture_root}/scripts/native-build-cache-fingerprint.sh" "$fixture_root")"
    [ "$baseline_fingerprint" != "$native_change_fingerprint" ] || {
        print_error "a native patch change retained the old fingerprint"
        exit 1
    }
    printf '%s\n' '[ci-native-cache-test] case=native-change status=success'

    printf '%s\n' '[ci-native-cache-test] case=cache-classification status=start'
    assert_cache_classification primary true 'macOS-ARM64-rust-current'
    assert_cache_classification fallback false 'macOS-ARM64-rust-previous'
    assert_cache_classification miss '' ''
    if CACHE_HIT=unexpected \
        CACHE_MATCHED_KEY='' \
        CACHE_PRIMARY_KEY='macOS-ARM64-rust-current' \
        "${repository_root}/scripts/report-build-cache-restoration.sh" >/dev/null 2>&1
    then
        print_error "an unsupported cache-hit state was accepted"
        exit 1
    fi
    printf '%s\n' '[ci-native-cache-test] case=cache-classification status=success'

    printf '%s\n' '[ci-native-cache-test] case=workflow-coordination status=start'
    assert_workflow_contract "${repository_root}/.github/workflows/ci.yml"
    printf '%s\n' '[ci-native-cache-test] case=workflow-coordination status=success'
}

main "$@"
