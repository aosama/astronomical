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
    mkdir -p "${fixture_root}/crates/mlx-c-rust"
    printf '%s\n' 'fn main() {}' > "${fixture_root}/crates/mlx-c-rust/build.rs"
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
    mkdir -p "${change_scope_repository}/crates/mlx-c-rust"
    printf '%s\n' 'fn main() {}' > \
        "${change_scope_repository}/crates/mlx-c-rust/build.rs"
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
    composite_action_path="$2"
    # GitHub expressions must reach Ruby unchanged so the contract compares the
    # workflow's actual expression strings rather than shell-expanded values.
    # shellcheck disable=SC2016
    ruby -ryaml -rshellwords -e '
        workflow = YAML.safe_load(File.read(ARGV.fetch(0)), aliases: true)
        composite = YAML.safe_load(File.read(ARGV.fetch(1)), aliases: true)
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
        verification_job = workflow.fetch("jobs").fetch("rust-core")
        swift_node_job = workflow.fetch("jobs").fetch("swift-node")
        static_job = workflow.fetch("jobs").fetch("static")
        raise "required check name changed" unless verification_job.fetch("name") == "macOS hermetic verification"
        raise "required check exceeded its hard cap" unless verification_job.fetch("timeout-minutes") == 15
        raise "swift-node exceeded its hard cap" unless swift_node_job.fetch("timeout-minutes") == 12
        expected_authority = "${{ always() && (needs.detect-changes.result != '\''success'\'' || needs.detect-changes.outputs.macos_verification_required == '\''true'\'') }}"
        raise "macOS authority does not fail closed" unless verification_job.fetch("if") == expected_authority
        raise "swift-node authority does not fail closed" unless swift_node_job.fetch("if") == expected_authority
        verification_concurrency = verification_job.fetch("concurrency")
        expected_group = "macos-hermetic-${{ github.event_name }}-${{ github.ref }}"
        raise "macOS concurrency is not event-and-ref scoped" unless verification_concurrency.fetch("group") == expected_group
        raise "required macOS verification must not cancel in-flight runs" unless verification_concurrency.fetch("cancel-in-progress") == false
        swift_concurrency = swift_node_job.fetch("concurrency")
        raise "swift-node concurrency is not event-and-ref scoped" unless swift_concurrency.fetch("group") == "macos-swift-node-${{ github.event_name }}-${{ github.ref }}"
        raise "swift-node must not cancel in-flight runs" unless swift_concurrency.fetch("cancel-in-progress") == false
        raise "static checks must require successful classification" unless static_job.fetch("if").include?("needs.detect-changes.result ==")
        steps = verification_job.fetch("steps")
        swift_steps = swift_node_job.fetch("steps")
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
          "apps/supervisor/console/library-fetch.test.js",
          "apps/supervisor/console/connect.test.js",
          "apps/supervisor/console/playground.test.js",
        ]
        # The required-CI command now tees to a durable log for rerun-safe
        # debugging, so assert the test invocation is present rather than the
        # exact shell wrapper, which would couple the guard to log plumbing.
        raise "Observatory required-CI command changed" unless observatory_step.fetch("run").include?(expected_observatory_command.join(" "))
        raise "Observatory contracts exceeded their bounded timeout" unless observatory_step.fetch("timeout-minutes") <= 2
        library_rest_step = steps.find { |step| step["name"] == "Run Library REST contracts" }
        raise "Library REST contracts are missing from required CI" unless library_rest_step
        expected_library_rest_command = [
          "cargo", "test", "--timings", "-p", "astronomical-supervisor",
          "--test", "rest_api_tests", "library", "--", "--nocapture",
        ]
        raise "Library REST required-CI command changed" unless library_rest_step.fetch("run").include?(expected_library_rest_command.join(" "))
        raise "Library REST contracts exceeded their bounded timeout" unless library_rest_step.fetch("timeout-minutes") <= 2
        compile_step = steps.find { |step| step["name"] == "Compile hermetic tests" }
        raise "hermetic compile step is missing from required CI" unless compile_step
        compile_command = compile_step.fetch("run")
        raise "hermetic compile does not use --no-run" unless compile_command.include?("--no-run")
        raise "hermetic compile omits hermetic_tests" unless compile_command.include?("--test hermetic_tests")
        raise "hermetic compile omits rest_api_tests" unless compile_command.include?("--test rest_api_tests")
        identity_step = steps.find { |step| step["id"] == "native-build-identity" }
        raise "native identity step is missing from required CI" unless identity_step
        raise "native build progress file is not exported for CI tailing" unless compile_command.include?("ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE=")
        raise "hermetic compile does not tail the native build progress stream" unless compile_command.include?("tail -f \"$ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE\"")
        raise "hermetic compile does not stop the progress tail on exit" unless compile_command.include?("kill \"$progress_tail_pid\"")
        swiftpm_timestamp_step = swift_steps.find { |step| step["name"] == "Keep restored SwiftPM artifacts newer than checkout" }
        raise "SwiftPM timestamp reuse step is missing from required CI" unless swiftpm_timestamp_step
        raise "SwiftPM timestamp reuse is not gated on a cache hit" unless swiftpm_timestamp_step.fetch("if").include?("swiftpm-cache.outputs.cache-hit")
        raise "SwiftPM timestamp reuse does not touch restored artifacts" unless swiftpm_timestamp_step.fetch("run").include?("touch -c")
        assert_composite_cache_owner = lambda do |job_steps, owner_id, expected_paths, restored_paths|
          restore_step = job_steps.find { |step| step["id"] == owner_id }
          raise "missing cache owner #{owner_id}" unless restore_step
          raise "cache owner #{owner_id} must restore through the shared composite action" unless restore_step.fetch("uses") == "./.github/actions/astronomical-cache"
          restore_configuration = restore_step.fetch("with")
          raise "cache owner #{owner_id} must declare the restore operation" unless restore_configuration.fetch("operation") == "restore"
          path_text = restore_configuration.fetch("path")
          expected_paths.each { |expected_path| raise "#{owner_id} omits #{expected_path}" unless path_text.include?(expected_path) }
          raise "target must not be cached" if path_text.lines.map(&:strip).include?("target")
          restored_paths.concat(path_text.lines.map(&:strip).reject(&:empty?))
          raise "#{owner_id} has the wrong report owner" unless restore_configuration.fetch("owner") == owner_id.delete_suffix("-cache")
          save_step = job_steps.find { |step| step["id"] == "#{owner_id}-save" }
          raise "#{owner_id} has no save owner" unless save_step
          raise "cache owner #{owner_id} must save through the shared composite action" unless save_step.fetch("uses") == "./.github/actions/astronomical-cache"
          save_configuration = save_step.fetch("with")
          raise "cache owner #{owner_id} must declare the save operation" unless save_configuration.fetch("operation") == "save"
          save_condition = save_step.fetch("if")
          is_conditional = save_condition.include?("success()") || save_condition.include?("!cancelled()")
          raise "#{owner_id} save is unconditional" unless is_conditional
          raise "#{owner_id} save path diverges from its restore path" unless save_configuration.fetch("path") == path_text
          raise "#{owner_id} save key diverges from its restore key" unless save_configuration.fetch("key") == restore_configuration.fetch("key")
        end
        cache_owners = {
          "cargo-downloads-cache" => ["~/.cargo/registry", "~/.cargo/git"],
          "native-archives-cache" => ["~/Library/Caches/Astronomical/native-dependencies"],
          "native-build-cache" => ["env.NATIVE_BUILD_ENTRY_DIRECTORY"],
          "sccache-cache" => ["~/Library/Caches/Astronomical/sccache"],
        }
        restored_paths = []
        cache_owners.each { |owner_id, expected_paths| assert_composite_cache_owner.call(steps, owner_id, expected_paths, restored_paths) }
        assert_composite_cache_owner.call(swift_steps, "swiftpm-cache", ["apps/astronomical-menu/.build"], restored_paths)
        raise "cache owners overlap paths" unless restored_paths.uniq.length == restored_paths.length

        native_restore = steps.find { |step| step["id"] == "native-build-cache" }
        native_configuration = native_restore.fetch("with")
        raise "native build cache must not cross identities" if native_configuration.key?("restore-keys") && !native_configuration.fetch("restore-keys").include?("NATIVE_BUILD_IDENTITY")
        raise "native entry path is not identity-specific" unless native_configuration.fetch("path").include?("NATIVE_BUILD_ENTRY_DIRECTORY")
        raise "native key omits full identity" unless native_configuration.fetch("key").include?("NATIVE_BUILD_IDENTITY")

        sccache_restore = steps.find { |step| step["id"] == "sccache-cache" }
        raise "sccache key is not stable by dependency graph" unless sccache_restore.fetch("with").fetch("key").include?("Cargo.lock")
        raise "sccache key couples to the native identity; the native build product cache owns that identity" if sccache_restore.fetch("with").fetch("key").include?("NATIVE_BUILD_IDENTITY")
        sccache_fallbacks = sccache_restore.fetch("with").fetch("restore-keys").to_s.split(/\s*\n\s*/).reject(&:empty?)
        raise "sccache restore omits an identity-free fallback" unless sccache_fallbacks.any? { |fallback| !fallback.include?("NATIVE_BUILD_IDENTITY") }
        prune_step = steps.find { |step| step["name"] == "Prune surplus CI caches" }
        raise "surplus cache prune step is missing" unless prune_step
        raise "cache prune must run even when earlier steps fail" unless prune_step.fetch("if").include?("!cancelled()")
        raise "cache prune lacks the workflow token" unless prune_step.fetch("env").fetch("GH_TOKEN") == "${{ github.token }}"
        raise "cache prune does not call the prune script" unless prune_step.fetch("run").include?("prune-ci-caches.sh")
        raise "cache prune exceeded its bounded timeout" unless prune_step.fetch("timeout-minutes") <= 2
        final_save_index = steps.index { |step| step["id"] == "sccache-cache-save" }
        prune_step_index = steps.index { |step| step["name"] == "Prune surplus CI caches" }
        raise "cache prune must run after the final cache save" unless prune_step_index > final_save_index
        verification_permissions = verification_job.fetch("permissions")
        raise "required check must keep contents read-only" unless verification_permissions.fetch("contents") == "read"
        raise "cache prune needs actions write permission" unless verification_permissions.fetch("actions") == "write"
        swift_restore = swift_steps.find { |step| step["id"] == "swiftpm-cache" }
        raise "Swift state is coupled to Rust" if swift_restore.fetch("with").fetch("key").include?("Cargo")
        raise "Swift cache omits toolchain compatibility" unless swift_restore.fetch("with").fetch("key").include?("SWIFT_TOOLCHAIN_IDENTITY")
        composite_runs = composite.fetch("runs")
        raise "shared cache action must run as a composite" unless composite_runs.fetch("using") == "composite"
        composite_steps = composite_runs.fetch("steps")
        composite_restore = composite_steps.find { |step| step["id"] == "restore" }
        raise "shared cache action restore step is missing" unless composite_restore
        raise "shared cache action restore is not a pinned actions/cache/restore" unless composite_restore.fetch("uses").match?(%r{\Aactions/cache/restore@[0-9a-f]{40}\z})
        composite_save = composite_steps.find { |step| step["id"] == "save" }
        raise "shared cache action save step is missing" unless composite_save
        raise "shared cache action save is not a pinned actions/cache/save" unless composite_save.fetch("uses").match?(%r{\Aactions/cache/save@[0-9a-f]{40}\z})
        composite_report = composite_steps.find { |step| step["id"] == "report" }
        raise "shared cache action report step is missing" unless composite_report
        report_environment = composite_report.fetch("env")
        raise "shared cache action report does not forward the owner" unless report_environment.fetch("CACHE_OWNER").include?("inputs.owner")
        raise "shared cache action report does not forward the restore outcome" unless report_environment.fetch("CACHE_STEP_OUTCOME").include?("steps.restore.outcome")
        raise "shared cache action report does not forward the save outcome" unless report_environment.fetch("CACHE_STEP_OUTCOME").include?("steps.save.outcome")
        raise "shared cache action report must call the restoration reporter" unless composite_report.fetch("run").include?("report-build-cache-restoration.sh")
        raise "shared cache action must surface cache hits to callers" unless composite.fetch("outputs").fetch("cache-hit").fetch("value").include?("steps.restore.outputs.cache-hit")
        timed_steps = [steps, swift_steps, static_job.fetch("steps")].flatten
            .select { |step| step["run"].to_s.include?("ci-step-timing.sh end") }
        raise "workflow has no timed steps to attribute" if timed_steps.empty?
        timed_steps.each do |timed_step|
            end_segment = timed_step["run"][/ci-step-timing\.sh end ([a-z0-9-]+)/, 1]
            raise "timed step #{timed_step["name"].inspect} does not name its end segment" unless end_segment
            raise "timed step #{timed_step["name"].inspect} ends #{end_segment} without a begin call" unless timed_step["run"].include?("ci-step-timing.sh begin #{end_segment}")
        end
        [verification_job, swift_node_job].each do |macos_job|
            publish_step = macos_job.fetch("steps").find { |step| step["run"].to_s.include?("publish-ci-timing-summary.sh") }
            raise "macOS job #{macos_job.fetch("name").inspect} never publishes the step timing summary" unless publish_step
            raise "step timing summary must publish even for failed runs" unless publish_step.fetch("if") == "always()"
        end
    ' "$workflow_path" "$composite_action_path"
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
    for build_owner_path in crates/runtime-integration/build.rs \
        crates/mlx-c-rust/build.rs crates/runtime-integration/build_native_linking.rs \
        crates/runtime-integration/build_native_store.rs \
        crates/runtime-integration/build_native_store_manifest.rs
    do
        printf '%s\n' '// changed owner' >> \
            "${change_scope_fixture_root}/${build_owner_path}"
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
    assert_workflow_contract "${repository_root}/.github/workflows/ci.yml" "${repository_root}/.github/actions/astronomical-cache/action.yml"
    printf '%s\n' '[ci-native-cache-test] case=workflow-cache-ownership status=success'
}

main "$@"
