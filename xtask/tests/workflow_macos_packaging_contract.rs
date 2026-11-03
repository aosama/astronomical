//! The macOS packaging lane contract, as typed Rust: parses
//! `.github/workflows/macos-packaging.yml` and pins what the lane promises.
//! It builds a real Stable bundle through `cargo xtask app package` and
//! smoke-tests that bundle; it never signs with an identity, notarizes, or
//! publishes; it gates only pull requests that touch packaging; and it
//! resolves the same Xcode as `ci.yml`, whose pin the native-build identity
//! depends on.

mod contract_support;

use contract_support::{
    find_step_by_name, macos_packaging_workflow_contract, required_field, required_job,
    required_text, run_text_of, steps_of, triggers_of, workflow_contract,
};
use serde_yaml::Value;

const PACKAGING_JOB_ID: &str = "package-stable";
const PACKAGE_STEP_NAME: &str = "Package the Stable bundle";
const SMOKE_TEST_STEP_NAME: &str = "Smoke-test the packaged Stable bundle";
const NATIVE_BOOTSTRAP_STEP_NAME: &str = "Provision verified native dependencies";
const STABLE_BUNDLE_PATH: &str = "target/astronomical-macos-stable.noindex/Astronomical.app";
const PINNED_XCODE_ACTION: &str = "./.github/actions/assert-pinned-xcode";
const PINNED_XCODE_VARIABLE: &str = "ASTRONOMICAL_PINNED_XCODE_VERSION";

/// Fragments that would mean the lane signs with a real identity, reaches
/// Apple's notary service, or publishes a release; CI holds no credentials
/// for any of them and must never try.
const FORBIDDEN_RELEASE_FRAGMENTS: [&str; 6] = [
    "--signing-identity",
    "notarytool",
    "xtask release",
    "gh release",
    "secrets.",
    "app-store",
];

fn packaging_job_run_texts(workflow: &Value) -> Vec<String> {
    let mut run_texts = Vec::new();
    for step in steps_of(required_job(workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID) {
        if let Some(run_text) = step.get("run").and_then(Value::as_str) {
            run_texts.push(run_text.to_owned());
        }
    }
    run_texts
}

#[test]
fn should_build_the_stable_bundle_through_app_package() {
    let workflow = macos_packaging_workflow_contract();
    let job_steps = steps_of(required_job(&workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID);
    let package_step = find_step_by_name(&job_steps, PACKAGE_STEP_NAME);

    assert_eq!(
        run_text_of(package_step, PACKAGE_STEP_NAME).trim(),
        "cargo xtask app package --channel stable"
    );
}

#[test]
fn should_smoke_test_exactly_the_bundle_the_package_step_produced() {
    let workflow = macos_packaging_workflow_contract();
    let job_steps = steps_of(required_job(&workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID);
    let smoke_test_text = run_text_of(
        find_step_by_name(&job_steps, SMOKE_TEST_STEP_NAME),
        SMOKE_TEST_STEP_NAME,
    );

    assert!(smoke_test_text.contains("cargo xtask app smoke-test --app-bundle"));
    assert!(
        smoke_test_text.contains(STABLE_BUNDLE_PATH),
        "the smoke test does not launch the packaged Stable bundle"
    );
    let mut package_index = None;
    let mut smoke_test_index = None;
    for (step_index, step) in job_steps.iter().enumerate() {
        match step.get("name").and_then(Value::as_str) {
            Some(PACKAGE_STEP_NAME) => package_index = Some(step_index),
            Some(SMOKE_TEST_STEP_NAME) => smoke_test_index = Some(step_index),
            _ => {}
        }
    }
    assert!(
        package_index.is_some() && package_index < smoke_test_index,
        "the smoke test must run after the bundle is packaged"
    );
}

#[test]
fn should_never_sign_with_an_identity_notarize_or_publish() {
    let workflow = macos_packaging_workflow_contract();
    for run_text in packaging_job_run_texts(&workflow) {
        for forbidden_fragment in FORBIDDEN_RELEASE_FRAGMENTS {
            assert!(
                !run_text.contains(forbidden_fragment),
                "the packaging lane runs a release-only operation ({forbidden_fragment}): {run_text}"
            );
        }
    }
    assert_eq!(
        required_text(
            required_field(&workflow, "permissions", "workflow"),
            "contents",
            "workflow permissions"
        ),
        "read",
        "the packaging lane must stay read-only"
    );
}

#[test]
fn should_gate_only_pull_requests_that_touch_packaging() {
    let workflow = macos_packaging_workflow_contract();
    let triggers = triggers_of(&workflow);
    assert!(
        triggers.get("schedule").is_some(),
        "the scheduled run is missing"
    );
    assert!(
        triggers.get("workflow_dispatch").is_some(),
        "the on-demand run is missing"
    );
    assert!(
        triggers.get("push").is_none(),
        "a cold release build on every push to main would queue behind itself"
    );
    let pull_request_paths = required_field(
        required_field(triggers, "pull_request", "triggers"),
        "paths",
        "pull_request trigger",
    )
    .as_sequence()
    .expect("the pull_request paths filter must be a sequence");
    let mut watches_packaging_pipeline = false;
    for path_filter in pull_request_paths {
        let path_filter_text = path_filter.as_str().unwrap_or_default();
        assert!(
            path_filter_text != "**" && !path_filter_text.is_empty(),
            "the pull_request filter must not match every change"
        );
        watches_packaging_pipeline |= path_filter_text.starts_with("xtask/src/app_bundle");
    }
    assert!(
        watches_packaging_pipeline,
        "pull requests that change the packaging pipeline must run this lane"
    );
}

#[test]
fn should_pin_the_same_xcode_as_the_ci_workflow() {
    let ci_pin = required_text(
        required_field(&workflow_contract(), "env", "ci workflow"),
        PINNED_XCODE_VARIABLE,
        "ci workflow env",
    )
    .to_owned();
    let workflow = macos_packaging_workflow_contract();
    let packaging_pin = required_text(
        required_field(&workflow, "env", "packaging workflow"),
        PINNED_XCODE_VARIABLE,
        "packaging workflow env",
    );
    assert_eq!(
        packaging_pin, ci_pin,
        "the packaging lane resolves a different Xcode than ci.yml"
    );
    let job_steps = steps_of(required_job(&workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID);
    let xcode_step = find_step_by_name(&job_steps, "Assert pinned Xcode toolchain");
    assert_eq!(
        required_text(xcode_step, "uses", "pinned-xcode"),
        PINNED_XCODE_ACTION
    );
    assert_eq!(
        required_text(
            required_field(xcode_step, "with", "pinned-xcode"),
            "xcode-version",
            "pinned-xcode"
        ),
        "${{ env.ASTRONOMICAL_PINNED_XCODE_VERSION }}"
    );
}

/// The package pipeline verifies the pinned native archives without
/// downloading them, so a fresh runner fails unless bootstrap ran first.
#[test]
fn should_provision_native_archives_before_packaging() {
    let workflow = macos_packaging_workflow_contract();
    let job_steps = steps_of(required_job(&workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID);
    let bootstrap_text = run_text_of(
        find_step_by_name(&job_steps, NATIVE_BOOTSTRAP_STEP_NAME),
        NATIVE_BOOTSTRAP_STEP_NAME,
    );
    assert_eq!(
        bootstrap_text.trim(),
        "cargo xtask bootstrap-native-dependencies"
    );
    let mut bootstrap_index = None;
    let mut package_index = None;
    for (step_index, step) in job_steps.iter().enumerate() {
        match step.get("name").and_then(Value::as_str) {
            Some(NATIVE_BOOTSTRAP_STEP_NAME) => bootstrap_index = Some(step_index),
            Some(PACKAGE_STEP_NAME) => package_index = Some(step_index),
            _ => {}
        }
    }
    assert!(
        bootstrap_index.is_some() && bootstrap_index < package_index,
        "native archives must be provisioned before the bundle is packaged"
    );
}
