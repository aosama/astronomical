#!/usr/bin/env sh

# Proves the required CI journey uses stable native compatibility identity,
# disjoint cache owners, and independently attributable cache operations.

set -eu

readonly EXPECTED_FINGERPRINT_LENGTH=64
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
    printf '%s\n' '[workspace]' 'version = "1.0.0"' > "${fixture_root}/Cargo.toml"
    printf '%s\n' 'version = 4' > "${fixture_root}/Cargo.lock"
    printf '%s\n' '[toolchain]' > "${fixture_root}/rust-toolchain.toml"
    printf '%s\n' '[package]' > "${fixture_root}/crates/runtime-integration/Cargo.toml"
    printf '%s\n' 'fn main() {}' > "${fixture_root}/crates/runtime-integration/build.rs"
    printf '%s\n' 'fn store() {}' > "${fixture_root}/crates/runtime-integration/build_native_store.rs"
    printf '%s\n' 'fn generate() {}' > "${fixture_root}/crates/runtime-integration/build_bindings.rs"
    printf '%s\n' 'fn link() {}' > "${fixture_root}/crates/runtime-integration/build_native_linking.rs"
    printf '%s\n' '1' > "${fixture_root}/crates/runtime-integration/native-build-store-schema-version"
    printf '%s\n' 'project(runtime)' > "${fixture_root}/crates/runtime-integration/native/CMakeLists.txt"
    printf '%s\n' 'set(MLX_VERSION 1)' > "${fixture_root}/third-party/native-dependency-manifest.cmake"
    printf '%s\n' 'set(MLX_PIN 1)' > "${fixture_root}/third-party/pins/mlx.cmake"
    printf '%s\n' 'native patch' > "${fixture_root}/third-party/patches/mlx.patch"
    printf '%s\n' 'unrelated documentation' > "${fixture_root}/README.md"
    git -C "$fixture_root" init --quiet
    git -C "$fixture_root" add .
}

full_fingerprint() {
    fixture_root="$1"
    native_profile="${2:-core}"
    target_identity="${3:-aarch64-apple-darwin}"
    ASTRONOMICAL_NATIVE_IDENTITY_XCODE='Xcode 26.0 Build 17A1' \
    ASTRONOMICAL_NATIVE_IDENTITY_SDK='macOS 26.0 Build 25A1' \
    ASTRONOMICAL_NATIVE_IDENTITY_CLANG='Apple clang 17.0.0 aarch64-apple-darwin' \
    ASTRONOMICAL_NATIVE_IDENTITY_CMAKE='cmake version 4.0.0' \
    ASTRONOMICAL_NATIVE_IDENTITY_RUSTC='rustc 1.97.1 stable aarch64-apple-darwin' \
    ASTRONOMICAL_NATIVE_IDENTITY_TARGET="$target_identity" \
    ASTRONOMICAL_NATIVE_BUILD_TYPE='Release' \
        "${fixture_root}/scripts/native-build-cache-fingerprint.sh" \
        --profile "$native_profile" "$fixture_root"
}

source_fingerprint() {
    fixture_root="$1"
    "${fixture_root}/scripts/native-build-cache-fingerprint.sh" \
        --source-only --profile core "$fixture_root"
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
    cache_step_outcome="$2"
    cache_hit="$3"
    matched_key="$4"
    report_output="$(
        CACHE_OWNER='native-build' \
        CACHE_OPERATION='restore' \
        CACHE_STEP_OUTCOME="$cache_step_outcome" \
        CACHE_HIT="$cache_hit" \
        CACHE_MATCHED_KEY="$matched_key" \
        CACHE_PRIMARY_KEY='astronomical-v2-native-build-current' \
        CACHE_STARTED_AT_EPOCH_SECONDS='100' \
        CACHE_FINISHED_AT_EPOCH_SECONDS='112' \
        GITHUB_STEP_SUMMARY="${SANDBOX_DIRECTORY}/step-summary.md" \
        "${repository_root}/scripts/report-build-cache-restoration.sh"
    )"
    case "$report_output" in
        *"owner=native-build operation=restore classification=${expected_classification} elapsed_seconds=12"*) ;;
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
        verification_job = workflow.fetch("jobs").fetch("verify")
        raise "required check name changed" unless verification_job.fetch("name") == "10-minute macOS hermetic verification"
        raise "required check exceeded its hard cap" unless verification_job.fetch("timeout-minutes") == 10
        steps = verification_job.fetch("steps")
        cache_owners = {
          "cargo-downloads-cache" => ["~/.cargo/registry", "~/.cargo/git"],
          "native-archives-cache" => ["~/Library/Caches/Astronomical/native-dependencies"],
          "native-build-cache" => ["env.NATIVE_BUILD_ENTRY_DIRECTORY"],
          "sccache-cache" => ["~/Library/Caches/Astronomical/sccache"],
          "swiftpm-cache" => ["apps/astronomical-menu/.build"],
        }
        restored_paths = []
        cache_owners.each do |owner_id, expected_paths|
          restore_step = steps.find { |step| step["id"] == owner_id }
          raise "missing cache owner #{owner_id}" unless restore_step
          cache_action = restore_step.fetch("uses")
          raise "cache owner #{owner_id} is not a pinned restore action" unless cache_action.match?(%r{\Aactions/cache/restore@[0-9a-f]{40}\z})
          path_text = restore_step.fetch("with").fetch("path")
          expected_paths.each { |expected_path| raise "#{owner_id} omits #{expected_path}" unless path_text.include?(expected_path) }
          raise "target must not be cached" if path_text.lines.map(&:strip).include?("target")
          restored_paths.concat(path_text.lines.map(&:strip).reject(&:empty?))
          report_step = steps.find { |step| step["name"] == "Report #{owner_id} restoration" }
          raise "#{owner_id} has no immediate owner report" unless report_step
          report_environment = report_step.fetch("env")
          raise "#{owner_id} report has the wrong owner" unless report_environment.fetch("CACHE_OWNER") == owner_id.delete_suffix("-cache")
          raise "#{owner_id} report omits cache outcome" unless report_environment.fetch("CACHE_STEP_OUTCOME").include?("steps.#{owner_id}.outcome")
          save_step = steps.find { |step| step["name"] == "Save #{owner_id}" }
          raise "#{owner_id} has no save owner" unless save_step
          save_action = save_step.fetch("uses")
          raise "cache owner #{owner_id} is not a pinned save action" unless save_action.match?(%r{\Aactions/cache/save@[0-9a-f]{40}\z})
          raise "#{owner_id} save is unconditional" unless save_step.fetch("if").include?("success()")
        end
        raise "cache owners overlap paths" unless restored_paths.uniq.length == restored_paths.length

        native_restore = steps.find { |step| step["id"] == "native-build-cache" }
        native_configuration = native_restore.fetch("with")
        raise "native build cache must not cross identities" if native_configuration.key?("restore-keys") && !native_configuration.fetch("restore-keys").include?("NATIVE_BUILD_IDENTITY")
        raise "native entry path is not identity-specific" unless native_configuration.fetch("path").include?("NATIVE_BUILD_ENTRY_DIRECTORY")
        raise "native key omits full identity" unless native_configuration.fetch("key").include?("NATIVE_BUILD_IDENTITY")

        sccache_restore = steps.find { |step| step["id"] == "sccache-cache" }
        raise "sccache key is not rolling by source" unless sccache_restore.fetch("with").fetch("key").include?("github.sha")
        swift_restore = steps.find { |step| step["id"] == "swiftpm-cache" }
        raise "Swift state is coupled to Rust" if swift_restore.fetch("with").fetch("key").include?("Cargo")
        raise "Swift cache omits toolchain compatibility" unless swift_restore.fetch("with").fetch("key").include?("SWIFT_TOOLCHAIN_IDENTITY")
        cache_owners.each_value do |_|
          # Owner-specific action names expose compressed bytes in hosted logs.
        end
    ' "$workflow_path"
}

main() {
    if [ "$#" -ne 0 ]; then
        print_error "test-ci-native-cache-coordination.sh does not accept arguments"
        exit 2
    fi
    for required_command in git mktemp ruby shasum; do
        require_command "$required_command"
    done

    repository_root="$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd -P)"
    SANDBOX_DIRECTORY="$(mktemp -d "${TMPDIR:-/tmp}/astronomical-ci-native-cache.XXXXXX")"
    fixture_root="${SANDBOX_DIRECTORY}/repository"
    create_fingerprint_fixture "$fixture_root"

    printf '%s\n' '[ci-native-cache-test] case=stable-fingerprint status=start'
    baseline_fingerprint="$(full_fingerprint "$fixture_root")"
    repeated_fingerprint="$(full_fingerprint "$fixture_root")"
    assert_fingerprint_shape "$baseline_fingerprint"
    [ "$baseline_fingerprint" = "$repeated_fingerprint" ] || {
        print_error "unchanged native inputs produced different fingerprints"
        exit 1
    }
    printf '%s\n' '[ci-native-cache-test] case=stable-fingerprint status=success'

    printf '%s\n' '[ci-native-cache-test] case=unrelated-and-version-changes status=start'
    printf '%s\n' 'updated unrelated documentation' > "${fixture_root}/README.md"
    printf '%s\n' '[workspace]' 'version = "1.0.1"' > "${fixture_root}/Cargo.toml"
    printf '%s\n' 'version = 5' > "${fixture_root}/Cargo.lock"
    printf '%s\n' '[package]' 'rust-only-dependency = "2"' > \
        "${fixture_root}/crates/runtime-integration/Cargo.toml"
    unrelated_fingerprint="$(full_fingerprint "$fixture_root")"
    [ "$baseline_fingerprint" = "$unrelated_fingerprint" ] || {
        print_error "an unrelated or workspace-version change invalidated native identity"
        exit 1
    }
    printf '%s\n' '[ci-native-cache-test] case=unrelated-and-version-changes status=success'

    printf '%s\n' '[ci-native-cache-test] case=unstaged-native-change status=start'
    printf '%s\n' 'updated native patch' > "${fixture_root}/third-party/patches/mlx.patch"
    native_change_fingerprint="$(full_fingerprint "$fixture_root")"
    [ "$baseline_fingerprint" != "$native_change_fingerprint" ] || {
        print_error "an unstaged native change retained the old identity"
        exit 1
    }
    printf '%s\n' 'new native source' > \
        "${fixture_root}/crates/runtime-integration/native/new_native_source.cpp"
    untracked_native_fingerprint="$(full_fingerprint "$fixture_root")"
    [ "$baseline_fingerprint" != "$untracked_native_fingerprint" ] || {
        print_error "an untracked native input retained the old identity"
        exit 1
    }
    rm -f "${fixture_root}/crates/runtime-integration/native/new_native_source.cpp"
    printf '%s\n' '[ci-native-cache-test] case=unstaged-native-change status=success'

    printf '%s\n' '[ci-native-cache-test] case=compatibility-and-profile status=start'
    git -C "$fixture_root" checkout -- third-party/patches/mlx.patch
    alternate_target_fingerprint="$(full_fingerprint "$fixture_root" core arm64-apple-darwin26.0)"
    probe_profile_fingerprint="$(full_fingerprint "$fixture_root" core+memory-contract)"
    [ "$baseline_fingerprint" != "$alternate_target_fingerprint" ] || {
        print_error "a target compatibility change retained the old identity"
        exit 1
    }
    [ "$baseline_fingerprint" != "$probe_profile_fingerprint" ] || {
        print_error "a native feature profile change retained the old identity"
        exit 1
    }
    source_identity="$(source_fingerprint "$fixture_root")"
    assert_fingerprint_shape "$source_identity"
    if ASTRONOMICAL_NATIVE_IDENTITY_XCODE='Xcode 26.0 Build 17A1' \
        ASTRONOMICAL_NATIVE_IDENTITY_SDK='macOS 26.0 Build 25A1' \
        ASTRONOMICAL_NATIVE_IDENTITY_CLANG='Apple clang 17.0.0 aarch64-apple-darwin' \
        ASTRONOMICAL_NATIVE_IDENTITY_CMAKE='cmake version 4.0.0' \
        ASTRONOMICAL_NATIVE_IDENTITY_RUSTC='rustc 1.97.1 stable aarch64-apple-darwin' \
        ASTRONOMICAL_NATIVE_IDENTITY_TARGET='aarch64-apple-darwin' \
        ASTRONOMICAL_NATIVE_BUILD_TYPE='Debug' \
        "${fixture_root}/scripts/native-build-cache-fingerprint.sh" \
        --profile core "$fixture_root" >/dev/null 2>&1
    then
        print_error "an unsupported native build type was accepted"
        exit 1
    fi
    printf '%s\n' '[ci-native-cache-test] case=compatibility-and-profile status=success'

    printf '%s\n' '[ci-native-cache-test] case=cache-classification status=start'
    assert_cache_classification primary success true 'astronomical-v2-native-build-current'
    assert_cache_classification fallback success false 'astronomical-v2-native-build-previous'
    assert_cache_classification miss success '' ''
    assert_cache_classification error failure '' ''
    if CACHE_OWNER='native-build' \
        CACHE_OPERATION='restore' \
        CACHE_STEP_OUTCOME='success' \
        CACHE_HIT='true' \
        CACHE_MATCHED_KEY='different-key' \
        CACHE_PRIMARY_KEY='astronomical-v2-native-build-current' \
        CACHE_STARTED_AT_EPOCH_SECONDS='100' \
        CACHE_FINISHED_AT_EPOCH_SECONDS='112' \
        "${repository_root}/scripts/report-build-cache-restoration.sh" >/dev/null 2>&1
    then
        print_error "an inconsistent primary cache state was accepted"
        exit 1
    fi
    printf '%s\n' '[ci-native-cache-test] case=cache-classification status=success'

    printf '%s\n' '[ci-native-cache-test] case=workflow-cache-ownership status=start'
    assert_workflow_contract "${repository_root}/.github/workflows/ci.yml"
    printf '%s\n' '[ci-native-cache-test] case=workflow-cache-ownership status=success'
}

main "$@"
