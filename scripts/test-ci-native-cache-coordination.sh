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

commit_change_scope_fixture() {
    change_scope_repository="$1"
    commit_message="$2"
    git -C "$change_scope_repository" add .
    git -C "$change_scope_repository" \
        -c user.name='Astronomical Test' \
        -c user.email='astronomical-test@example.invalid' \
        commit --quiet -m "$commit_message"
    git -C "$change_scope_repository" rev-parse HEAD
}

create_change_scope_fixture() {
    change_scope_repository="$1"
    mkdir -p \
        "${change_scope_repository}/.github/workflows" \
        "${change_scope_repository}/crates/model-serving/src" \
        "${change_scope_repository}/crates/runtime-integration/native" \
        "${change_scope_repository}/scripts" \
        "${change_scope_repository}/third-party/pins" \
        "${change_scope_repository}/third-party/patches"
    printf '%s\n' '# Project' > "${change_scope_repository}/README.md"
    printf '%s\n' 'pub fn serve() {}' > \
        "${change_scope_repository}/crates/model-serving/src/lib.rs"
    printf '%s\n' 'fn main() {}' > \
        "${change_scope_repository}/crates/runtime-integration/build.rs"
    printf '%s\n' 'fn generate() {}' > \
        "${change_scope_repository}/crates/runtime-integration/build_bindings.rs"
    printf '%s\n' 'fn link() {}' > \
        "${change_scope_repository}/crates/runtime-integration/build_native_linking.rs"
    printf '%s\n' 'fn store() {}' > \
        "${change_scope_repository}/crates/runtime-integration/build_native_store.rs"
    printf '%s\n' 'fn manifest() {}' > \
        "${change_scope_repository}/crates/runtime-integration/build_native_store_manifest.rs"
    printf '%s\n' '1' > \
        "${change_scope_repository}/crates/runtime-integration/native-build-store-schema-version"
    printf '%s\n' 'project(runtime)' > \
        "${change_scope_repository}/crates/runtime-integration/native/CMakeLists.txt"
    printf '%s\n' '#!/usr/bin/env sh' > \
        "${change_scope_repository}/scripts/native-build-cache-fingerprint.sh"
    printf '%s\n' 'set(MLX_VERSION 1)' > \
        "${change_scope_repository}/third-party/native-dependency-manifest.cmake"
    printf '%s\n' 'set(MLX_PIN 1)' > "${change_scope_repository}/third-party/pins/mlx.cmake"
    printf '%s\n' 'native patch' > "${change_scope_repository}/third-party/patches/mlx.patch"
    printf '%s\n' '[toolchain]' 'channel = "stable"' > \
        "${change_scope_repository}/rust-toolchain.toml"
    printf '%s\n' 'name: CI' > "${change_scope_repository}/.github/workflows/ci.yml"
    git -C "$change_scope_repository" init --quiet
}

assert_change_scope() {
    change_scope_repository="$1"
    event_name="$2"
    base_sha="$3"
    head_sha="$4"
    expected_code_changed="$5"
    expected_native_inputs_changed="$6"
    expected_macos_verification_required="$7"
    output_file="${SANDBOX_DIRECTORY}/change-scope-output"
    : > "$output_file"

    EVENT_NAME="$event_name" \
    PULL_REQUEST_BASE_SHA="$base_sha" \
    PULL_REQUEST_HEAD_SHA="$head_sha" \
    PUSH_BEFORE_SHA="$base_sha" \
    CURRENT_SHA="$head_sha" \
    GITHUB_OUTPUT="$output_file" \
    REPOSITORY_ROOT="$change_scope_repository" \
        "${repository_root}/scripts/classify-ci-change-scope.sh"

    actual_code_changed=""
    actual_native_inputs_changed=""
    actual_macos_verification_required=""
    while IFS= read -r output_line; do
        case "$output_line" in
            code_changed=*) actual_code_changed="${output_line#code_changed=}" ;;
            native_inputs_changed=*) actual_native_inputs_changed="${output_line#native_inputs_changed=}" ;;
            macos_verification_required=*)
                actual_macos_verification_required="${output_line#macos_verification_required=}"
                ;;
        esac
    done < "$output_file"

    [ "$actual_code_changed" = "$expected_code_changed" ] || {
        print_error "${event_name} code_changed was ${actual_code_changed}, expected ${expected_code_changed}"
        exit 1
    }
    [ "$actual_native_inputs_changed" = "$expected_native_inputs_changed" ] || {
        print_error "${event_name} native_inputs_changed was ${actual_native_inputs_changed}, expected ${expected_native_inputs_changed}"
        exit 1
    }
    [ "$actual_macos_verification_required" = "$expected_macos_verification_required" ] || {
        print_error "${event_name} macos_verification_required was ${actual_macos_verification_required}, expected ${expected_macos_verification_required}"
        exit 1
    }
}

assert_workflow_contract() {
    workflow_path="$1"
    # GitHub expressions must reach Ruby unchanged so the contract compares the
    # workflow's actual expression strings rather than shell-expanded values.
    # shellcheck disable=SC2016
    ruby -ryaml -rshellwords -e '
        workflow = YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)
        triggers = workflow.fetch(true)
        raise "pull-request verification trigger is missing" unless triggers.key?("pull_request")
        raise "manual verification trigger is missing" unless triggers.key?("workflow_dispatch")
        push_trigger = triggers.fetch("push")
        push_branches = push_trigger.fetch("branches")
        raise "main classification trigger changed" unless push_branches == ["main"]
        raise "push trigger must not hide classifier runs with path filters" if push_trigger.key?("paths") || push_trigger.key?("paths-ignore")
        raise "classifier-only runs must not wait behind macOS work" if workflow.key?("concurrency")
        detection_job = workflow.fetch("jobs").fetch("detect-changes")
        detection_outputs = detection_job.fetch("outputs")
        raise "native change output is missing" unless detection_outputs.key?("native_inputs_changed")
        raise "macOS authority output is missing" unless detection_outputs.key?("macos_verification_required")
        classification_step = detection_job.fetch("steps").find { |step| step["id"] == "classify" }
        raise "change-scope classifier step is missing" unless classification_step
        raise "change-scope policy is not script-owned" unless classification_step.fetch("run").include?("classify-ci-change-scope.sh")
        raise "classifier job still computes unused native identity" if detection_job.fetch("steps").any? { |step| step["run"]&.include?("native-build-cache-fingerprint.sh") }
        verification_job = workflow.fetch("jobs").fetch("verify")
        raise "required check name changed" unless verification_job.fetch("name") == "10-minute macOS hermetic verification"
        raise "required check exceeded its hard cap" unless verification_job.fetch("timeout-minutes") == 15
        expected_authority = "${{ always() && (needs.detect-changes.result != '\''success'\'' || needs.detect-changes.outputs.macos_verification_required == '\''true'\'') }}"
        raise "macOS authority does not fail closed" unless verification_job.fetch("if") == expected_authority
        verification_concurrency = verification_job.fetch("concurrency")
        expected_group = "macos-hermetic-${{ github.event_name }}-${{ github.ref }}"
        raise "macOS concurrency is not event-and-ref scoped" unless verification_concurrency.fetch("group") == expected_group
        raise "required macOS verification must not cancel in-flight runs" unless verification_concurrency.fetch("cancel-in-progress") == false
        steps = verification_job.fetch("steps")
        classification_guard = steps.find { |step| step["name"] == "Require successful change classification" }
        raise "classification failure guard is missing" unless classification_guard
        raise "classification guard condition changed" unless classification_guard.fetch("if").include?("detect-changes.result != '\''success'\''")
        raise "classification guard does not fail the required check" unless classification_guard.fetch("run").include?("exit 1")
        observatory_step = steps.find { |step| step["name"] == "Run Observatory contracts" }
        raise "Observatory contracts are missing from required CI" unless observatory_step
        expected_observatory_command = [
          "node", "--test", "--test-reporter=spec",
          "apps/supervisor/console/console.test.js",
          "apps/supervisor/console/library.test.js",
        ]
        raise "Observatory required-CI command changed" unless Shellwords.split(observatory_step.fetch("run")) == expected_observatory_command
        raise "Observatory contracts exceeded their bounded timeout" unless observatory_step.fetch("timeout-minutes") <= 2
        library_rest_step = steps.find { |step| step["name"] == "Run Library REST contracts" }
        raise "Library REST contracts are missing from required CI" unless library_rest_step
        expected_library_rest_command = [
          "cargo", "test", "--timings", "-p", "astronomical-supervisor",
          "--test", "rest_api_tests", "library", "--", "--nocapture",
        ]
        raise "Library REST required-CI command changed" unless Shellwords.split(library_rest_step.fetch("run")) == expected_library_rest_command
        raise "Library REST contracts exceeded their bounded timeout" unless library_rest_step.fetch("timeout-minutes") <= 2
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
          save_condition = save_step.fetch("if")
          is_conditional = save_condition.include?("success()") || save_condition.include?("!cancelled()")
          raise "#{owner_id} save is unconditional" unless is_conditional
        end
        raise "cache owners overlap paths" unless restored_paths.uniq.length == restored_paths.length

        native_restore = steps.find { |step| step["id"] == "native-build-cache" }
        native_configuration = native_restore.fetch("with")
        raise "native build cache must not cross identities" if native_configuration.key?("restore-keys") && !native_configuration.fetch("restore-keys").include?("NATIVE_BUILD_IDENTITY")
        raise "native entry path is not identity-specific" unless native_configuration.fetch("path").include?("NATIVE_BUILD_ENTRY_DIRECTORY")
        raise "native key omits full identity" unless native_configuration.fetch("key").include?("NATIVE_BUILD_IDENTITY")

        sccache_restore = steps.find { |step| step["id"] == "sccache-cache" }
        raise "sccache key is not stable by dependency graph" unless sccache_restore.fetch("with").fetch("key").include?("Cargo.lock")
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

    printf '%s\n' '[ci-native-cache-test] case=event-topology status=start'
    change_scope_fixture_root="${SANDBOX_DIRECTORY}/change-scope-repository"
    create_change_scope_fixture "$change_scope_fixture_root"
    baseline_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" baseline)"
    printf '%s\n' '# Static update' >> "${change_scope_fixture_root}/README.md"
    static_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" static)"
    printf '%s\n' 'pub fn serve_updated() {}' > \
        "${change_scope_fixture_root}/crates/model-serving/src/lib.rs"
    rust_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" rust)"
    mkdir -p "${change_scope_fixture_root}/docs"
    git -C "$change_scope_fixture_root" mv \
        crates/model-serving/src/lib.rs docs/served-api.md
    renamed_code_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" code-to-static-rename)"
    printf '%s\n' 'project(runtime_updated)' > \
        "${change_scope_fixture_root}/crates/runtime-integration/native/CMakeLists.txt"
    native_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" native)"
    printf '%s\n' '[toolchain]' 'channel = "next"' > \
        "${change_scope_fixture_root}/rust-toolchain.toml"
    toolchain_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" toolchain)"
    previous_build_owner_scope_sha="$toolchain_scope_sha"
    for build_owner_path in build.rs build_bindings.rs build_native_linking.rs \
        build_native_store.rs build_native_store_manifest.rs
    do
        printf '%s\n' '// changed owner' >> \
            "${change_scope_fixture_root}/crates/runtime-integration/${build_owner_path}"
        current_build_owner_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" "${build_owner_path}-owner")"
        assert_change_scope "$change_scope_fixture_root" push \
            "$previous_build_owner_scope_sha" "$current_build_owner_scope_sha" true true true
        previous_build_owner_scope_sha="$current_build_owner_scope_sha"
    done
    build_owner_scope_sha="$previous_build_owner_scope_sha"
    printf '%s\n' '2' > \
        "${change_scope_fixture_root}/crates/runtime-integration/native-build-store-schema-version"
    store_schema_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" store-schema)"
    printf '%s\n' '#!/usr/bin/env sh' '# updated identity policy' > \
        "${change_scope_fixture_root}/scripts/native-build-cache-fingerprint.sh"
    fingerprint_policy_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" fingerprint-policy)"
    printf '%s\n' 'set(MLX_VERSION 2)' > \
        "${change_scope_fixture_root}/third-party/native-dependency-manifest.cmake"
    dependency_manifest_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" dependency-manifest)"
    printf '%s\n' 'set(MLX_PIN 2)' > \
        "${change_scope_fixture_root}/third-party/pins/mlx.cmake"
    dependency_pin_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" dependency-pin)"
    printf '%s\n' 'updated native patch' > \
        "${change_scope_fixture_root}/third-party/patches/mlx.patch"
    dependency_patch_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" dependency-patch)"
    git -C "$change_scope_fixture_root" mv \
        crates/runtime-integration/native/CMakeLists.txt NATIVE-NOTES.md
    renamed_native_scope_sha="$(commit_change_scope_fixture "$change_scope_fixture_root" native-to-static-rename)"

    assert_change_scope "$change_scope_fixture_root" pull_request \
        "$baseline_scope_sha" "$static_scope_sha" false false false
    assert_change_scope "$change_scope_fixture_root" pull_request \
        "$static_scope_sha" "$rust_scope_sha" true false true
    assert_change_scope "$change_scope_fixture_root" push \
        "$static_scope_sha" "$rust_scope_sha" true false true
    assert_change_scope "$change_scope_fixture_root" pull_request \
        "$rust_scope_sha" "$renamed_code_scope_sha" true false true
    assert_change_scope "$change_scope_fixture_root" push \
        "$rust_scope_sha" "$renamed_code_scope_sha" true false true
    assert_change_scope "$change_scope_fixture_root" push \
        "$renamed_code_scope_sha" "$native_scope_sha" true true true
    assert_change_scope "$change_scope_fixture_root" push \
        "$native_scope_sha" "$toolchain_scope_sha" true true true
    assert_change_scope "$change_scope_fixture_root" push \
        "$build_owner_scope_sha" "$store_schema_scope_sha" true true true
    assert_change_scope "$change_scope_fixture_root" push \
        "$store_schema_scope_sha" "$fingerprint_policy_scope_sha" true true true
    assert_change_scope "$change_scope_fixture_root" push \
        "$fingerprint_policy_scope_sha" "$dependency_manifest_scope_sha" true true true
    assert_change_scope "$change_scope_fixture_root" push \
        "$dependency_manifest_scope_sha" "$dependency_pin_scope_sha" true true true
    assert_change_scope "$change_scope_fixture_root" push \
        "$dependency_pin_scope_sha" "$dependency_patch_scope_sha" true true true
    assert_change_scope "$change_scope_fixture_root" push \
        "$dependency_patch_scope_sha" "$renamed_native_scope_sha" true true true
    assert_change_scope "$change_scope_fixture_root" workflow_dispatch '' '' true false true
    assert_change_scope "$change_scope_fixture_root" push \
        '0000000000000000000000000000000000000000' "$renamed_native_scope_sha" true true true
    printf '%s\n' '[ci-native-cache-test] case=event-topology status=success'

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
