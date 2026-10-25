//! The workflow structural contract, as typed Rust: parses
//! `.github/workflows/ci.yml` and pins the required check's triggers,
//! fail-closed macOS authority, bounded timeouts, integration-lane compile
//! contract, and attributable timing. Cache-ownership structure lives in the
//! companion `workflow_cache_owner_contract` test; both replace the
//! Ruby-in-shell section of
//! `scripts/ci/tests/test-ci-native-cache-coordination.sh` (issue #1250).

mod contract_support;

use contract_support::{
    EXPECTED_LIBRARY_REST_COMMAND, EXPECTED_MACOS_AUTHORITY, EXPECTED_MACOS_CONCURRENCY_GROUP,
    EXPECTED_OBSERVATORY_COMMAND, EXPECTED_SWIFT_CONCURRENCY_GROUP, find_step_by_id,
    find_step_by_name, required_bool, required_field, required_job, required_text, required_u64,
    run_text_of, steps_of, triggers_of, workflow_contract,
};
use serde_yaml::Value;

#[test]
fn should_use_the_canonical_integration_package_selection_for_compile_and_execution() {
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    for step_name in [
        "Compile integration lane tests",
        "Run integration lane tests",
    ] {
        let step = find_step_by_name(&verification_steps, step_name);
        let command = run_text_of(step, step_name);
        assert!(
            command
                .lines()
                .any(|line| line.trim_start().starts_with("cargo test-integration ")),
            "{step_name} must use cargo test-integration so extracted policy packages cannot disappear"
        );
        assert!(
            !command.lines().any(|line| {
                line.trim_start().starts_with("cargo test ")
                    && line.contains("--test integration_tests")
            }),
            "{step_name} duplicates the canonical integration package selection"
        );
        assert!(
            command.contains("cargo test-test-harness-fixtures"),
            "{step_name} must cover the shared real-model evidence assertions without a network"
        );
    }
    let configuration =
        std::fs::read_to_string(contract_support::repository_root().join(".cargo/config.toml"))
            .expect("the canonical Cargo configuration should be readable");
    let integration_alias = configuration
        .lines()
        .find(|line| line.starts_with("test-integration = "))
        .expect("the canonical integration alias should exist");
    assert!(integration_alias.contains("-p astronomical-memory-policy "));
    assert!(integration_alias.contains("--lib --test integration_tests"));
}

#[test]
fn workflow_triggers_keep_the_classification_surface_open() {
    let workflow = workflow_contract();
    let triggers = triggers_of(&workflow);
    assert!(
        triggers.get("pull_request").is_some(),
        "pull-request verification trigger is missing"
    );
    assert!(
        triggers.get("workflow_dispatch").is_some(),
        "manual verification trigger is missing"
    );
    let push_trigger = required_field(triggers, "push", "the workflow triggers");
    let push_branches: Vec<&str> = required_field(push_trigger, "branches", "the push trigger")
        .as_sequence()
        .expect("push branches must be a sequence")
        .iter()
        .map(|branch| branch.as_str().expect("push branches must be strings"))
        .collect();
    assert_eq!(
        push_branches,
        ["main"],
        "main classification trigger changed"
    );
    assert!(
        push_trigger.get("paths").is_none() && push_trigger.get("paths-ignore").is_none(),
        "push trigger must not hide classifier runs with path filters"
    );
    assert!(
        workflow.get("concurrency").is_none(),
        "classifier-only runs must not wait behind macOS work"
    );
}

#[test]
fn detect_changes_outputs_drive_the_authority() {
    let workflow = workflow_contract();
    let detection_job = required_job(&workflow, "detect-changes");
    let detection_outputs = required_field(detection_job, "outputs", "job detect-changes");
    assert!(
        detection_outputs.get("native_inputs_changed").is_some(),
        "native change output is missing"
    );
    assert!(
        detection_outputs
            .get("macos_verification_required")
            .is_some(),
        "macOS authority output is missing"
    );
    let detection_steps = steps_of(detection_job, "detect-changes");
    let classification_step = find_step_by_id(&detection_steps, "classify");
    assert!(
        run_text_of(classification_step, "classify").contains("classify-scope"),
        "change-scope policy is not xtask-owned"
    );
    assert!(
        detection_steps.iter().all(|step| step
            .get("run")
            .and_then(Value::as_str)
            .map(|run_text| !run_text.contains("native-fingerprint"))
            .unwrap_or(true)),
        "classifier job still computes unused native identity"
    );
}

#[test]
fn required_jobs_keep_names_timeouts_and_fail_closed_authority() {
    let workflow = workflow_contract();
    let verification_job = required_job(&workflow, "rust-core");
    let swift_node_job = required_job(&workflow, "swift-node");
    let static_job = required_job(&workflow, "static");
    assert_eq!(
        required_text(verification_job, "name", "job rust-core"),
        "macOS offline verification",
        "required check name changed"
    );
    assert_eq!(
        required_u64(verification_job, "timeout-minutes", "job rust-core"),
        15,
        "required check exceeded its hard cap"
    );
    assert_eq!(
        required_u64(swift_node_job, "timeout-minutes", "job swift-node"),
        12,
        "swift-node exceeded its hard cap"
    );
    assert_eq!(
        required_text(verification_job, "if", "job rust-core"),
        EXPECTED_MACOS_AUTHORITY,
        "macOS authority does not fail closed"
    );
    assert_eq!(
        required_text(swift_node_job, "if", "job swift-node"),
        EXPECTED_MACOS_AUTHORITY,
        "swift-node authority does not fail closed"
    );
    let verification_concurrency = required_field(verification_job, "concurrency", "job rust-core");
    assert_eq!(
        required_text(verification_concurrency, "group", "rust-core concurrency"),
        EXPECTED_MACOS_CONCURRENCY_GROUP,
        "macOS concurrency is not event-and-ref scoped"
    );
    assert!(
        !required_bool(
            verification_concurrency,
            "cancel-in-progress",
            "rust-core concurrency"
        ),
        "required macOS verification must not cancel in-flight runs"
    );
    let swift_concurrency = required_field(swift_node_job, "concurrency", "job swift-node");
    assert_eq!(
        required_text(swift_concurrency, "group", "swift-node concurrency"),
        EXPECTED_SWIFT_CONCURRENCY_GROUP,
        "swift-node concurrency is not event-and-ref scoped"
    );
    assert!(
        !required_bool(
            swift_concurrency,
            "cancel-in-progress",
            "swift-node concurrency"
        ),
        "swift-node must not cancel in-flight runs"
    );
    assert!(
        required_text(static_job, "if", "job static").contains("needs.detect-changes.result =="),
        "static checks must require successful classification"
    );
}

#[test]
fn required_check_steps_keep_their_bounded_contracts() {
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");

    let classification_guard = find_step_by_name(
        &verification_steps,
        "Require successful change classification",
    );
    assert!(
        required_text(classification_guard, "if", "classification guard")
            .contains("detect-changes.result != 'success'"),
        "classification guard condition changed"
    );
    assert!(
        run_text_of(classification_guard, "classification-guard").contains("exit 1"),
        "classification guard does not fail the required check"
    );

    let observatory_step = find_step_by_name(&verification_steps, "Run Observatory contracts");
    // The required-CI command tees to a durable log for rerun-safe debugging,
    // so assert the test invocation is present rather than the exact shell
    // wrapper, which would couple the guard to log plumbing.
    assert!(
        run_text_of(observatory_step, "observatory").contains(EXPECTED_OBSERVATORY_COMMAND),
        "Observatory required-CI command changed"
    );
    assert!(
        required_u64(observatory_step, "timeout-minutes", "Observatory contracts") <= 2,
        "Observatory contracts exceeded their bounded timeout"
    );

    let library_rest_step = find_step_by_name(&verification_steps, "Run Library REST contracts");
    assert!(
        run_text_of(library_rest_step, "library-rest").contains(EXPECTED_LIBRARY_REST_COMMAND),
        "Library REST required-CI command changed"
    );
    assert!(
        required_u64(
            library_rest_step,
            "timeout-minutes",
            "Library REST contracts"
        ) <= 2,
        "Library REST contracts exceeded their bounded timeout"
    );

    let compile_step = find_step_by_name(&verification_steps, "Compile integration lane tests");
    let compile_command = run_text_of(compile_step, "compile-integration");
    assert!(
        compile_command.contains("--no-run"),
        "integration compile does not use --no-run"
    );
    assert!(
        compile_command.contains("cargo test-integration "),
        "integration compile omits the canonical integration alias"
    );
    assert!(
        compile_command.contains("--test rest_api_tests"),
        "integration compile omits rest_api_tests"
    );
    let timing_step = find_step_by_name(&verification_steps, "Configure step timing attribution");
    assert!(
        run_text_of(timing_step, "ci-timing").contains("ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE="),
        "native build progress file is not exported job-wide before the first cargo step"
    );
    // The runtime-integration build script declares rerun-if-env-changed for
    // the progress file variable: exporting it in the compile step alone flips
    // the build script fingerprint and re-runs the script in every later cargo
    // step of the job (one of two measured-rebuild triggers; the other, a
    // phantom rerun-if-changed path, is removed in that build script). The
    // job-wide export above keeps the value constant, and the compile step
    // must keep hands off so the flip cannot return.
    assert!(
        !compile_command.contains("export ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE="),
        "the compile step must not re-export the native build progress file variable"
    );
    assert!(
        compile_command.contains(r#"tail -f "$ASTRONOMICAL_NATIVE_BUILD_PROGRESS_FILE""#),
        "integration compile does not tail the native build progress stream"
    );
    assert!(
        compile_command.contains(r#"kill "$progress_tail_pid""#),
        "integration compile does not stop the progress tail on exit"
    );
    find_step_by_id(&verification_steps, "native-build-identity");

    let swiftpm_timestamp_step = find_step_by_name(
        &swift_steps,
        "Keep restored SwiftPM artifacts newer than checkout",
    );
    assert!(
        required_text(swiftpm_timestamp_step, "if", "SwiftPM timestamp step")
            .contains("swiftpm-cache.outputs.cache-hit"),
        "SwiftPM timestamp reuse is not gated on a cache hit"
    );
    assert!(
        run_text_of(swiftpm_timestamp_step, "swiftpm-timestamp").contains("touch -c"),
        "SwiftPM timestamp reuse does not touch restored artifacts"
    );
}

#[test]
fn dependency_notices_steps_stay_out_of_the_pull_request_lane() {
    // The dependency-notices lane is main-only: pull requests run no
    // notices work (the freshness gate repeatedly failed pull requests
    // whose authors had not regenerated locally, and freshness of the
    // committed copy is repository hygiene, not a property of the change
    // under review), while default-branch and dispatch builds regenerate
    // so the bundled app never ships stale notices. The verify step must
    // not come back silently.
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let verification_step_names: Vec<&str> = verification_steps
        .iter()
        .filter_map(|step| step.get("name").and_then(|value| value.as_str()))
        .collect();
    assert!(
        !verification_step_names
            .iter()
            .any(|&step_name| step_name.contains("dependency notices")
                && !step_name.contains("for the bundled app")),
        "a dependency-notices step outside the bundled-app refresh crept back into the pull-request lane"
    );
    let refresh_notices_step = find_step_by_name(
        &verification_steps,
        "Refresh Rust dependency notices for the bundled app",
    );
    assert!(
        required_text(refresh_notices_step, "if", "notices-refresh-gate")
            .contains("github.event_name != 'pull_request'"),
        "notices generation runs inside pull-request builds"
    );
}

#[test]
fn packaging_contracts_run_on_required_checks() {
    // The packaging journeys only stop rotting if they run on required
    // checks; a deleted step would otherwise fail silently until release
    // day, which is exactly what issue #1346 set out to end.
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let cache_contract_step =
        find_step_by_name(&verification_steps, "Verify CI native cache contracts");
    assert!(
        run_text_of(cache_contract_step, "ci-cache-contracts")
            .contains("--test integration_app_builder_journeys"),
        "the app-builder journey is not enforced on the required verification check"
    );

    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");
    let release_contracts_step =
        find_step_by_name(&swift_steps, "Run macOS packaging release contracts");
    let release_contracts_text = run_text_of(release_contracts_step, "release-contracts");
    assert!(
        release_contracts_text.contains("scripts/release/tests/test-release-contracts.sh"),
        "the release packaging contracts are not enforced on the required swift-node check"
    );
    assert!(
        required_u64(
            release_contracts_step,
            "timeout-minutes",
            "release contracts"
        ) <= 4,
        "release contracts exceeded their bounded timeout"
    );
}

/// Collects one job's timed steps and asserts the begin call lives in the
/// same run text, mirroring the shell contract's
/// `ci-step-timing.sh end ([a-z0-9-]+)` extraction.
fn assert_timed_steps_pair_begin_and_end(job_steps: &[&Value], job_id: &str) -> usize {
    let mut timed_step_count: usize = 0;
    for step in job_steps {
        let Some(run_text) = step.get("run").and_then(Value::as_str) else {
            continue;
        };
        let Some(segment_marker) = run_text.find("ci-step-timing.sh end ") else {
            continue;
        };
        let segment_start = segment_marker + "ci-step-timing.sh end ".len();
        let end_segment: String = run_text[segment_start..]
            .chars()
            .take_while(|character| {
                character.is_ascii_lowercase() || character.is_ascii_digit() || *character == '-'
            })
            .collect();
        let step_name = step
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<unnamed>");
        assert!(
            !end_segment.is_empty(),
            "timed step {step_name} in {job_id} does not name its end segment"
        );
        assert!(
            run_text.contains(&format!("ci-step-timing.sh begin {end_segment}")),
            "timed step {step_name} ends {end_segment} without a begin call"
        );
        timed_step_count += 1;
    }
    timed_step_count
}

#[test]
fn every_timed_step_pairs_its_begin_and_end_calls() {
    let workflow = workflow_contract();
    let mut total_timed_steps: usize = 0;
    for (job_id, job) in [
        ("rust-core", required_job(&workflow, "rust-core")),
        ("swift-node", required_job(&workflow, "swift-node")),
        ("static", required_job(&workflow, "static")),
    ] {
        total_timed_steps += assert_timed_steps_pair_begin_and_end(&steps_of(job, job_id), job_id);
    }
    assert!(
        total_timed_steps > 0,
        "workflow has no timed steps to attribute"
    );
}

// The macOS toolchain pin contract: one workflow-level Xcode version keeps
// the xcodebuild/xcrun/swift identity inputs stable across runner-image
// rollouts, so the native-build and SwiftPM caches stay valid until the pin
// is bumped deliberately. Each macOS job resolves the version to an
// installed Xcode and exports DEVELOPER_DIR for every later step; the
// resolution step makes a stale pin fail the job — printing the image's
// installed Xcode inventory — instead of silently falling back to the
// image's moving default.
#[test]
fn macos_toolchain_identity_is_pinned_to_a_versioned_xcode() {
    let workflow = workflow_contract();
    let workflow_env = workflow
        .get("env")
        .expect("workflow-level env block is missing");
    let pinned_xcode_version = required_text(
        workflow_env,
        "ASTRONOMICAL_PINNED_XCODE_VERSION",
        "workflow env",
    );
    assert!(
        !pinned_xcode_version.is_empty()
            && pinned_xcode_version
                .chars()
                .all(|character| character.is_ascii_digit() || character == '.'),
        "the Xcode pin must be a bare version like 26.6, got {pinned_xcode_version}"
    );
    assert!(
        workflow_env.get("DEVELOPER_DIR").is_none(),
        "DEVELOPER_DIR must be resolved per job from the version pin, not frozen as a workflow path"
    );
    for job_id in ["rust-core", "swift-node"] {
        let job_steps = steps_of(required_job(&workflow, job_id), job_id);
        let assert_step = find_step_by_name(&job_steps, "Assert pinned Xcode toolchain");
        let assert_text = run_text_of(assert_step, "pinned-xcode");
        assert!(
            assert_text.contains(r#"$ASTRONOMICAL_PINNED_XCODE_VERSION"#),
            "job {job_id} does not resolve the workflow-level Xcode version pin"
        );
        assert!(
            assert_text.contains(r#""/Applications/Xcode_${pinned_xcode_version}.app""#),
            "job {job_id} does not probe the readme primary Xcode path"
        );
        assert!(
            assert_text.contains("/Contents/Developer/usr/bin/xcodebuild"),
            "job {job_id} does not probe the canonical xcodebuild path inside the bundle"
        );
        assert!(
            assert_text
                .contains(r#"resolved_developer_dir="$xcode_bundle_candidate/Contents/Developer""#),
            "job {job_id} does not export a developer directory, not a bundle path"
        );
        assert!(
            assert_text.contains(
                r#"printf 'DEVELOPER_DIR=%s\n' "$resolved_developer_dir" >> "$GITHUB_ENV""#
            ),
            "job {job_id} does not export the resolved developer directory"
        );
        assert!(
            assert_text.contains("exit 1"),
            "job {job_id} does not fail loudly when the pinned Xcode is missing"
        );
        assert!(
            assert_text.contains(r#"ls -d /Applications/Xcode*.app"#),
            "job {job_id} does not print the installed Xcode inventory on failure"
        );
        assert!(
            assert_text.contains("xcodebuild -version"),
            "job {job_id} does not log the pinned toolchain version"
        );
    }
}

#[test]
fn macos_jobs_publish_the_timing_summary_even_when_failing() {
    let workflow = workflow_contract();
    for job_id in ["rust-core", "swift-node"] {
        let job = required_job(&workflow, job_id);
        let job_name = required_text(job, "name", &format!("job {job_id}"));
        let publish_steps: Vec<&Value> = steps_of(job, job_id)
            .into_iter()
            .filter(|step| {
                step.get("run")
                    .and_then(Value::as_str)
                    .map(|run_text| run_text.contains("publish-ci-timing-summary.sh"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            !publish_steps.is_empty(),
            "macOS job {job_name} never publishes the step timing summary"
        );
        for publish_step in publish_steps {
            assert_eq!(
                required_text(publish_step, "if", "timing summary step"),
                "always()",
                "step timing summary must publish even for failed runs"
            );
        }
    }
}
