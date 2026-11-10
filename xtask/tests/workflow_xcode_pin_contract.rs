//! The macOS toolchain pin contract, as typed Rust: parses
//! `.github/workflows/ci.yml` and the shared
//! `.github/actions/assert-pinned-xcode` composite action, and pins the one
//! workflow-level Xcode version, the one resolution script every macOS job
//! calls, and the absence of inline copies of that script. Split from the
//! companion `workflow_structural_contract` test the way the verdict and
//! cache-owner contracts were.

mod contract_support;

use contract_support::{
    MACOS_RUNNER_LABEL, every_workflow_contract, find_step_by_name, pinned_xcode_action_contract,
    required_field, required_job, required_text, run_text_of, steps_of, workflow_contract,
};
use serde_yaml::Value;

// The macOS toolchain pin contract: one workflow-level Xcode version keeps
// the xcodebuild/xcrun/swift identity inputs stable across runner-image
// rollouts, so the native-build and SwiftPM caches stay valid until the pin
// is bumped deliberately. Each macOS job resolves the version to an
// installed Xcode through the one shared assert-pinned-xcode action, which
// exports DEVELOPER_DIR for every later step and makes a stale pin fail the
// job — printing the image's installed Xcode inventory — instead of silently
// falling back to the image's moving default.
const PINNED_XCODE_ACTION: &str = "./.github/actions/assert-pinned-xcode";
const PINNED_XCODE_STEP_NAME: &str = "Assert pinned Xcode toolchain";
const MACOS_TOOLCHAIN_JOB_IDS: [&str; 3] = ["rust-core", "swift-node", "neural-engine"];
/// A token only the resolution script uses; seeing it in a job's own run
/// text means an inline copy of the shared action came back.
const PINNED_XCODE_SCRIPT_MARKER: &str = "xcode_bundle_candidate";

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
        "the Xcode pin must be a bare version like 27.2, got {pinned_xcode_version}"
    );
    assert!(
        workflow_env.get("DEVELOPER_DIR").is_none(),
        "DEVELOPER_DIR must be resolved per job from the version pin, not frozen as a workflow path"
    );
    for job_id in MACOS_TOOLCHAIN_JOB_IDS {
        assert_job_resolves_xcode_through_the_shared_action(&workflow, job_id);
    }
}

/// One runner image for every macOS job in every workflow, and a pin whose
/// major version is the one that image ships. A job on another macOS label
/// would resolve no pinned Xcode at all (the assert action fails) or, where
/// it never calls the action, quietly compile with that image's default.
#[test]
fn should_run_every_macos_job_on_the_image_that_ships_the_pinned_xcode() {
    for (workflow_file_name, workflow) in every_workflow_contract() {
        let workflow_jobs = required_field(&workflow, "jobs", &workflow_file_name)
            .as_mapping()
            .expect("a workflow's jobs must be a mapping");
        for (job_id, job) in workflow_jobs {
            // A job that calls a reusable workflow names no runner of its own.
            let Some(runner_label) = job.get("runs-on").and_then(Value::as_str) else {
                continue;
            };
            assert!(
                runner_label == MACOS_RUNNER_LABEL || runner_label.starts_with("ubuntu-"),
                "{workflow_file_name}: job {job_id:?} runs on {runner_label}; macOS jobs must all \
                 run on {MACOS_RUNNER_LABEL}"
            );
        }
    }
    let workflow = workflow_contract();
    let pinned_xcode_version = required_text(
        required_field(&workflow, "env", "ci workflow"),
        "ASTRONOMICAL_PINNED_XCODE_VERSION",
        "workflow env",
    );
    let pinned_xcode_major = pinned_xcode_version
        .split('.')
        .next()
        .expect("a version has a first component");
    assert_eq!(
        format!("xcode-{pinned_xcode_major}"),
        MACOS_RUNNER_LABEL,
        "the pinned Xcode {pinned_xcode_version} does not ship on the {MACOS_RUNNER_LABEL} image"
    );
}

/// The job calls the shared action with the workflow-level pin, after the
/// checkout that makes the local action readable.
fn assert_job_resolves_xcode_through_the_shared_action(workflow: &Value, job_id: &str) {
    let job_steps = steps_of(required_job(workflow, job_id), job_id);
    let assert_step = find_step_by_name(&job_steps, PINNED_XCODE_STEP_NAME);
    assert_eq!(
        required_text(assert_step, "uses", "pinned-xcode"),
        PINNED_XCODE_ACTION,
        "job {job_id} does not resolve Xcode through the shared action"
    );
    assert_eq!(
        required_text(
            required_field(assert_step, "with", "pinned-xcode"),
            "xcode-version",
            "pinned-xcode"
        ),
        "${{ env.ASTRONOMICAL_PINNED_XCODE_VERSION }}",
        "job {job_id} does not pass the workflow-level Xcode version pin"
    );
    let mut checkout_index = None;
    let mut assert_index = None;
    for (step_index, step) in job_steps.iter().enumerate() {
        let step_name = step.get("name").and_then(Value::as_str);
        if step_name == Some("Check out source") && checkout_index.is_none() {
            checkout_index = Some(step_index);
        }
        if step_name == Some(PINNED_XCODE_STEP_NAME) {
            assert_index = Some(step_index);
        }
    }
    assert!(
        checkout_index.is_some() && checkout_index < assert_index,
        "job {job_id} calls the local Xcode action before checking out the repository"
    );
}

/// Each fragment the resolution script must contain, with the behavior its
/// absence would lose.
const REQUIRED_RESOLUTION_FRAGMENTS: [(&str, &str); 8] = [
    (
        r#"$ASTRONOMICAL_PINNED_XCODE_VERSION"#,
        "resolve the version pin",
    ),
    (
        r#""/Applications/Xcode_${pinned_xcode_version}.app""#,
        "probe the readme primary Xcode path",
    ),
    (
        "/Contents/Developer/usr/bin/xcodebuild",
        "probe the canonical xcodebuild path",
    ),
    (
        r#"resolved_developer_dir="$xcode_bundle_candidate/Contents/Developer""#,
        "export a developer directory, not a bundle path",
    ),
    (
        r#"printf 'DEVELOPER_DIR=%s\n' "$resolved_developer_dir" >> "$GITHUB_ENV""#,
        "export the resolved developer directory",
    ),
    ("exit 1", "fail loudly when the pinned Xcode is missing"),
    (
        r#"ls -d /Applications/Xcode*.app"#,
        "print the installed Xcode inventory on failure",
    ),
    ("xcodebuild -version", "log the pinned toolchain version"),
];

/// The action's single step; a second step would split the resolution and
/// the DEVELOPER_DIR export the callers rely on into separate shells.
fn single_resolution_step_of(action: &Value) -> &Value {
    let action_steps = required_field(
        required_field(action, "runs", "pinned-xcode action"),
        "steps",
        "pinned-xcode action",
    )
    .as_sequence()
    .expect("the pinned-xcode action steps must be a sequence");
    assert_eq!(
        action_steps.len(),
        1,
        "the pinned-xcode action grew extra steps"
    );
    &action_steps[0]
}

#[test]
fn should_resolve_export_and_fail_loudly_inside_the_shared_xcode_action() {
    let action = pinned_xcode_action_contract();
    let resolve_step = single_resolution_step_of(&action);
    assert_eq!(
        required_text(
            required_field(resolve_step, "env", "pinned-xcode action"),
            "ASTRONOMICAL_PINNED_XCODE_VERSION",
            "pinned-xcode action"
        ),
        "${{ inputs.xcode-version }}",
        "the action does not hand its input to the script through the environment"
    );
    let script_text = run_text_of(resolve_step, "pinned-xcode action");
    assert!(
        !script_text.contains("${{"),
        "the action interpolates an expression into its script body"
    );
    for (required_fragment, missing_behavior) in REQUIRED_RESOLUTION_FRAGMENTS {
        assert!(
            script_text.contains(required_fragment),
            "the pinned-xcode action does not {missing_behavior}"
        );
    }
}

#[test]
fn should_keep_no_inline_copy_of_the_xcode_resolution_in_any_job() {
    let workflow = workflow_contract();
    let jobs = required_field(&workflow, "jobs", "workflow")
        .as_mapping()
        .expect("workflow jobs must be a mapping");
    for (job_key, job) in jobs {
        let job_id = job_key.as_str().unwrap_or("<non-string job id>");
        for step in steps_of(job, job_id) {
            let Some(run_text) = step.get("run").and_then(Value::as_str) else {
                continue;
            };
            assert!(
                !run_text.contains(PINNED_XCODE_SCRIPT_MARKER),
                "job {job_id} carries an inline copy of the shared Xcode resolution"
            );
        }
    }
}
