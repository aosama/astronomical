//! The workflow structural contract, as typed Rust: parses
//! `.github/workflows/ci.yml` and pins the required check's triggers,
//! fail-closed macOS authority, bounded timeouts, integration-lane compile
//! contract, and attributable timing. The verdict fan-in that owns the
//! merge gate is pinned by the companion `workflow_verdict_contract` test;
//! cache-ownership structure lives in the companion
//! `workflow_cache_owner_contract` test; both replace the Ruby-in-shell
//! section of `scripts/ci/tests/test-ci-native-cache-coordination.sh`
//! (issue #1250). The macOS Xcode pin and its shared resolution action are
//! pinned by the companion `workflow_xcode_pin_contract` test.

mod contract_support;

use contract_support::{
    EXPECTED_MACOS_AUTHORITY, EXPECTED_MACOS_CONCURRENCY_GROUP, EXPECTED_OBSERVATORY_COMMAND,
    EXPECTED_SWIFT_CONCURRENCY_GROUP, EXPECTED_SWIFT_NODE_AUTHORITY, find_step_by_id,
    find_step_by_name, is_pinned_action_reference, notices_freshness_workflow_contract,
    required_bool, required_field, required_job, required_text, required_u64, run_text_of,
    steps_of, triggers_of, workflow_contract,
};
use serde_yaml::Value;

#[test]
fn should_use_the_canonical_integration_package_selection_for_compile_and_execution() {
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    // The compile step builds through cargo's own alias; the archive step
    // packs what was built and the per-tier lanes run it (pinned by the
    // workflow_lane_split_contract binary). This pin keeps the canonical
    // build selection so extracted policy packages cannot disappear.
    let compile_step = find_step_by_name(&verification_steps, "Compile integration lane tests");
    let compile_text = run_text_of(compile_step, "compile-integration");
    assert!(
        compile_text
            .lines()
            .any(|line| line.trim_start().starts_with("cargo test-integration ")),
        "the compile step must use cargo test-integration so extracted policy packages cannot disappear"
    );
    assert!(
        !compile_text.lines().any(|line| {
            line.trim_start().starts_with("cargo test ")
                && line.contains("--test integration_tests")
        }),
        "the compile step duplicates the canonical integration package selection"
    );
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
    assert!(
        detection_outputs.get("notices_inputs_changed").is_some(),
        "notices freshness authority output is missing"
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
        "macOS build",
        "the build job name changed"
    );
    for (macos_job_id, hard_cap_minutes, expected_authority) in [
        ("rust-core", 15, EXPECTED_MACOS_AUTHORITY),
        ("swift-node", 12, EXPECTED_SWIFT_NODE_AUTHORITY),
        ("neural-engine", 6, EXPECTED_MACOS_AUTHORITY),
    ] {
        let job_context = format!("job {macos_job_id}");
        let macos_job = required_job(&workflow, macos_job_id);
        assert_eq!(
            required_u64(macos_job, "timeout-minutes", &job_context),
            hard_cap_minutes,
            "{macos_job_id} exceeded its hard cap"
        );
        assert_eq!(
            required_text(macos_job, "if", &job_context),
            expected_authority,
            "{macos_job_id} authority does not fail closed"
        );
    }
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
    let neural_engine_job = required_job(&workflow, "neural-engine");
    assert_eq!(
        required_text(neural_engine_job, "name", "job neural-engine"),
        "Neural Engine, xtask and standalone builds",
        "the neural-engine lane name changed"
    );
    assert!(
        required_text(static_job, "if", "job static").contains("needs.detect-changes.result =="),
        "static checks must require successful classification"
    );
    let config_contracts_job = required_job(&workflow, "config-contracts");
    assert_eq!(
        required_text(config_contracts_job, "name", "job config-contracts"),
        "Configuration contracts",
        "the config-contracts lane name changed"
    );
    assert_eq!(
        required_u64(
            config_contracts_job,
            "timeout-minutes",
            "job config-contracts"
        ),
        8,
        "config-contracts exceeded its hard cap"
    );
    assert!(
        required_text(config_contracts_job, "if", "job config-contracts")
            .contains("needs.detect-changes.result =="),
        "config contracts must require successful classification"
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

    // Observatory lives in swift-node since the lane split: it is a Node
    // contract suite and that job owns the Node toolchain and test-log
    // artifact it tees into.
    let observatory_step = find_step_by_name(&swift_steps, "Run Observatory contracts");
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

    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");
    let release_contracts_step =
        find_step_by_name(&swift_steps, "Run macOS packaging release contracts");
    let release_contracts_text = run_text_of(release_contracts_step, "release-contracts");
    assert!(
        release_contracts_text.contains("scripts/internal/render-macos-app-icon-contract.swift"),
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

#[test]
fn macos_build_lane_runs_no_dependency_notices_work() {
    // The in-run refresh left the macOS build job (issue #1684): it
    // regenerated third-party/RUST_DEPENDENCY_NOTICES into the runner's
    // ephemeral working tree, where no bundle ever consumed it, on the CI
    // critical path. Notices freshness belongs to the classifier-gated
    // ubuntu notices-freshness lane and the nightly backstop, never to the
    // macOS build job.
    let workflow = workflow_contract();
    let verification_steps = steps_of(required_job(&workflow, "rust-core"), "rust-core");
    let mut step_names: Vec<&str> = Vec::new();
    for step in &verification_steps {
        if let Some(step_name) = step.get("name").and_then(Value::as_str) {
            step_names.push(step_name);
        }
    }
    for step_name in &step_names {
        assert!(
            !step_name.to_lowercase().contains("dependency notices"),
            "a dependency-notices step crept back into the ci.yml macOS lane; freshness belongs to the ubuntu notices-freshness lane"
        );
    }
    for step in &verification_steps {
        if let Some(run_text) = step.get("run").and_then(Value::as_str) {
            assert!(
                !run_text.contains("xtask notices"),
                "notices generation crept back into the ci.yml macOS lane; freshness belongs to the ubuntu notices-freshness lane"
            );
        }
    }
}

#[test]
fn should_check_committed_notices_on_ubuntu_when_classifier_flags_notices_inputs() {
    // A Dependabot lockfile bump once merged green and turned the nightly
    // notices check red. The pull-request lane closes that gap only while it
    // stays classifier-gated (other pull requests pay one skipped job), off
    // the macOS runners (the critical path and their concurrency pool), and
    // a read-only check of the committed copy through the locked xtask
    // runner. The verdict contract pins that its result reaches the gate.
    let workflow = workflow_contract();
    let notices_lane = required_job(&workflow, "notices-freshness");
    assert_eq!(
        required_text(notices_lane, "needs", "job notices-freshness"),
        "detect-changes",
        "the notices lane must start with the classifier, not wait for the macOS build"
    );
    let lane_condition = required_text(notices_lane, "if", "job notices-freshness");
    assert!(
        lane_condition.contains("needs.detect-changes.result == 'success'")
            && lane_condition
                .contains("needs.detect-changes.outputs.notices_inputs_changed == 'true'"),
        "the notices lane lost its classifier gate"
    );
    assert_eq!(
        required_text(notices_lane, "runs-on", "job notices-freshness"),
        "ubuntu-24.04",
        "the notices lane moved onto a macOS runner"
    );
    assert!(
        required_u64(notices_lane, "timeout-minutes", "job notices-freshness") <= 10,
        "the notices lane exceeded its bounded timeout"
    );
    let workflow_permissions = required_field(&workflow, "permissions", "the ci workflow");
    assert_eq!(
        required_text(workflow_permissions, "contents", "the ci permissions"),
        "read",
        "the ci workflow gained write permissions"
    );
    assert!(
        notices_lane.get("permissions").is_none(),
        "the notices lane must stay a read-only check under the workflow permissions"
    );
    let notices_steps = steps_of(notices_lane, "notices-freshness");
    let notices_check_step = find_step_by_name(
        &notices_steps,
        "Check committed Rust dependency notices freshness",
    );
    let notices_check_text = run_text_of(notices_check_step, "notices-freshness");
    assert!(
        notices_check_text.contains("--locked")
            && notices_check_text.contains("-p xtask")
            && notices_check_text.contains("notices --check"),
        "the notices lane no longer checks the committed copy through the locked xtask runner"
    );
}

#[test]
fn nightly_notices_freshness_lane_stays_pinned() {
    // The freshness contract that replaced the retired in-run refresh
    // (issue #1684): without this lane, a bot-merged dependency update can
    // leave the committed notices one regeneration behind with nothing
    // failing. The lane must stay scheduled nightly, keep the on-demand
    // dispatch, stay read-only, and check the committed copy.
    let freshness_lane = notices_freshness_workflow_contract();
    let freshness_triggers = triggers_of(&freshness_lane);
    assert!(
        freshness_triggers.get("workflow_dispatch").is_some(),
        "the notices-freshness lane lost its on-demand dispatch trigger"
    );
    let schedule_entries = required_field(
        freshness_triggers,
        "schedule",
        "the notices-freshness triggers",
    )
    .as_sequence()
    .expect("the notices-freshness schedule must be a sequence");
    assert!(
        !schedule_entries.is_empty(),
        "the notices-freshness lane lost its schedule"
    );
    let cron_expression = required_text(
        &schedule_entries[0],
        "cron",
        "the notices-freshness schedule entry",
    );
    let cron_fields: Vec<&str> = cron_expression.split_whitespace().collect();
    assert_eq!(
        cron_fields.len(),
        5,
        "the notices-freshness cron expression must stay a five-field cron"
    );
    // Day-of-month, month, and day-of-week must stay wildcard so the check
    // runs every night; drift beyond a day is what the lane exists to catch.
    assert!(
        cron_fields[2] == "*" && cron_fields[3] == "*" && cron_fields[4] == "*",
        "the notices-freshness schedule must stay nightly"
    );
    let lane_permissions = required_field(
        &freshness_lane,
        "permissions",
        "the notices-freshness workflow",
    );
    assert!(
        required_text(
            lane_permissions,
            "contents",
            "the notices-freshness permissions"
        ) == "read",
        "the read-only freshness lane gained write permissions"
    );
    let freshness_steps = steps_of(required_job(&freshness_lane, "check"), "check");
    let freshness_check_step = find_step_by_name(
        &freshness_steps,
        "Check committed Rust dependency notices freshness",
    );
    let freshness_check_run_text = run_text_of(freshness_check_step, "notices-freshness-check");
    assert!(
        freshness_check_run_text.contains("-p xtask")
            && freshness_check_run_text.contains("notices --check"),
        "the notices-freshness lane no longer checks the committed copy through the locked xtask runner"
    );
}

#[test]
fn packaging_contracts_run_on_required_checks() {
    // The packaging journeys only stop rotting if they run on required
    // checks; a deleted step would otherwise fail silently until release
    // day, which is exactly what issue #1346 set out to end.
    let workflow = workflow_contract();
    let neural_engine_job = required_job(&workflow, "neural-engine");
    assert_eq!(
        required_text(neural_engine_job, "runs-on", "job neural-engine"),
        "macos-26",
        "the packaging contracts exercise macOS tooling, so they must run on macOS"
    );
    assert_eq!(
        required_text(neural_engine_job, "if", "job neural-engine"),
        EXPECTED_MACOS_AUTHORITY,
        "the packaging contracts must run whenever macOS verification is required"
    );
    let neural_engine_steps = steps_of(neural_engine_job, "neural-engine");
    let xtask_tests_step = find_step_by_name(&neural_engine_steps, "Run the xtask unit tests");
    assert!(
        run_text_of(xtask_tests_step, "xtask unit tests")
            .contains("cargo test --locked -p xtask --bin xtask"),
        "the xtask unit tests are not enforced on the Neural Engine lane"
    );

    // The host-agnostic workflow contracts live in the config-contracts lane
    // (issue #1553 phase 2a); they must stay enforced somewhere the verdict
    // fans in, so a lane deletion cannot silently retire them.
    let config_contracts_steps = steps_of(
        required_job(&workflow, "config-contracts"),
        "config-contracts",
    );
    let contracts_step =
        find_step_by_name(&config_contracts_steps, "Run workflow and cache contracts");
    let contracts_text = run_text_of(contracts_step, "config-contracts");
    for contract_binary in [
        "--test workflow_structural_contract",
        "--test workflow_verdict_contract",
        "--test workflow_lane_split_contract",
        "--test workflow_cache_owner_contract",
        "--test workflow_xcode_pin_contract",
        "--test workflow_macos_packaging_contract",
        "--test workflow_xtask_binary_contract",
    ] {
        assert!(
            contracts_text.contains(contract_binary),
            "the {contract_binary} binary is not enforced on the config-contracts lane"
        );
    }

    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");
    let release_contracts_step =
        find_step_by_name(&swift_steps, "Run macOS packaging release contracts");
    let release_contracts_text = run_text_of(release_contracts_step, "release-contracts");
    assert!(
        release_contracts_text.contains("scripts/internal/render-macos-app-icon-contract.swift"),
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
        ("neural-engine", required_job(&workflow, "neural-engine")),
        ("static", required_job(&workflow, "static")),
        (
            "config-contracts",
            required_job(&workflow, "config-contracts"),
        ),
    ] {
        total_timed_steps += assert_timed_steps_pair_begin_and_end(&steps_of(job, job_id), job_id);
    }
    assert!(
        total_timed_steps > 0,
        "workflow has no timed steps to attribute"
    );
}

#[test]
fn macos_jobs_publish_the_timing_summary_even_when_failing() {
    let workflow = workflow_contract();
    for job_id in ["rust-core", "swift-node", "neural-engine"] {
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

const SWIFT_MTIME_ACTION: &str = "chetan/git-restore-mtime-action";
const SWIFT_MTIME_STEP_NAME: &str = "Restore Swift input mtimes from their last commits";
const SWIFT_PACKAGE_INPUT_PATHS: [&str; 7] = [
    "apps/astronomical-menu/Sources",
    "apps/astronomical-menu/tests",
    "apps/astronomical-menu/Package.swift",
    "apps/astronomical-menu/Package.resolved",
    "apps/thin-talk/Sources",
    "apps/thin-talk/Tests",
    "apps/thin-talk/Package.swift",
];

/// Position of the first step whose run text or uses reference contains the
/// marker, so ordering contracts can compare step indices.
fn first_step_index_mentioning(job_steps: &[&Value], marker: &str) -> Option<usize> {
    for (step_index, step) in job_steps.iter().enumerate() {
        let run_text = step.get("run").and_then(Value::as_str).unwrap_or_default();
        let uses_text = step.get("uses").and_then(Value::as_str).unwrap_or_default();
        if run_text.contains(marker) || uses_text.contains(marker) {
            return Some(step_index);
        }
    }
    None
}

// The SwiftPM build database rebuilds any input whose mtime differs from the
// recorded one, so a restored .build is only a no-op when unchanged Swift
// inputs present the same mtime on every commit. Per-file last-commit times
// need full history (a shallow clone stamps every file with the one fetched
// commit) and must be applied before the first Swift build; a dropped input
// path silently brings back the full recompile on every new commit.
#[test]
fn should_restore_swift_input_mtimes_from_full_history_before_any_swift_build() {
    let workflow = workflow_contract();
    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");
    let checkout_step = find_step_by_name(&swift_steps, "Check out source");
    let checkout_inputs = required_field(checkout_step, "with", "swift-node checkout");
    assert_eq!(
        required_u64(checkout_inputs, "fetch-depth", "swift-node checkout"),
        0,
        "the swift-node checkout must fetch full history for per-file commit times"
    );
    let mtime_step = find_step_by_name(&swift_steps, SWIFT_MTIME_STEP_NAME);
    let uses_text = required_text(mtime_step, "uses", "swift mtime restore");
    assert!(
        is_pinned_action_reference(uses_text, SWIFT_MTIME_ACTION),
        "the Swift mtime restore must use a SHA-pinned {SWIFT_MTIME_ACTION}, got {uses_text}"
    );
    let mtime_inputs = required_field(mtime_step, "with", "swift mtime restore");
    let mtime_arguments = required_text(mtime_inputs, "args", "swift mtime restore");
    for swift_input_path in SWIFT_PACKAGE_INPUT_PATHS {
        assert!(
            mtime_arguments.contains(swift_input_path),
            "the Swift mtime restore no longer covers {swift_input_path}"
        );
    }
    let mtime_step_index = first_step_index_mentioning(&swift_steps, SWIFT_MTIME_ACTION)
        .expect("the Swift mtime restore step should exist");
    let first_swift_build_index = first_step_index_mentioning(
        &swift_steps,
        "swift test --package-path apps/astronomical-menu",
    )
    .expect("the menu contracts step should exist");
    assert!(
        mtime_step_index < first_swift_build_index,
        "the Swift mtime restore must run before the first Swift build"
    );
}

#[test]
fn should_find_no_step_for_an_unmentioned_marker() {
    let workflow = workflow_contract();
    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");
    assert_eq!(
        first_step_index_mentioning(&swift_steps, "marker-that-no-step-mentions"),
        None
    );
    assert_eq!(first_step_index_mentioning(&[], SWIFT_MTIME_ACTION), None);
}

/// True when a step's run text rewrites mtimes inside a SwiftPM build
/// directory, which marks every recorded output stale.
fn touches_swiftpm_build_directory(run_text: &str) -> bool {
    for run_line in run_text.lines() {
        if run_line.contains("touch") && run_line.contains(".build") {
            return true;
        }
    }
    false
}

#[test]
fn should_detect_only_touches_aimed_at_a_build_directory() {
    assert!(touches_swiftpm_build_directory(
        "find apps/astronomical-menu/.build -type f -exec touch -c {} +"
    ));
    assert!(!touches_swiftpm_build_directory(
        "touch \"$RUNNER_TEMP/marker\""
    ));
    assert!(!touches_swiftpm_build_directory("ls apps/thin-talk/.build"));
    assert!(!touches_swiftpm_build_directory(""));
}

// The build database records output mtimes, so touching a restored .build
// re-runs copy tasks whose products cascade into recompiling every
// Sparkle-dependent target. Per-file commit mtimes make the touch
// unnecessary; this keeps it from coming back.
#[test]
fn should_leave_restored_swiftpm_build_directories_untouched() {
    let workflow = workflow_contract();
    let swift_steps = steps_of(required_job(&workflow, "swift-node"), "swift-node");
    for step in &swift_steps {
        let step_name = step.get("name").and_then(Value::as_str).unwrap_or_default();
        let run_text = step.get("run").and_then(Value::as_str).unwrap_or_default();
        assert!(
            !touches_swiftpm_build_directory(run_text),
            "swift-node step {step_name:?} touches a restored SwiftPM build directory"
        );
    }
}
