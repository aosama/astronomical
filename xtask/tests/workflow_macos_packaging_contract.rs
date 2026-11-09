//! The macOS packaging lane contract, as typed Rust: parses
//! `.github/workflows/macos-packaging.yml` and pins what the lane promises.
//! It builds a real Stable bundle through `cargo xtask app package` and
//! smoke-tests that bundle; it never signs with an identity, notarizes, or
//! publishes; it gates only pull requests that touch packaging; and it
//! resolves the same Xcode as `ci.yml`, whose pin the native-build identity
//! depends on.

mod contract_support;

use contract_support::{
    find_step_by_id, find_step_by_name, macos_packaging_workflow_contract, required_field,
    required_job, required_text, run_text_of, steps_of, triggers_of, workflow_contract,
};
use serde_yaml::Value;

const PACKAGING_JOB_ID: &str = "package-stable";
const PACKAGE_STEP_NAME: &str = "Package the Stable bundle";
const SMOKE_TEST_STEP_NAME: &str = "Smoke-test the packaged Stable bundle";
const NATIVE_BOOTSTRAP_STEP_NAME: &str = "Provision verified native dependencies";
const STABLE_BUNDLE_PATH: &str = "target/astronomical-macos-stable.noindex/Astronomical.app";
const PINNED_XCODE_ACTION: &str = "./.github/actions/assert-pinned-xcode";
const PINNED_XCODE_VARIABLE: &str = "ASTRONOMICAL_PINNED_XCODE_VERSION";
const NATIVE_BUILD_RESTORE_STEP_NAME: &str = "Restore native-build-cache";
const NATIVE_BUILD_KEY_STEP_NAME: &str = "Compute the native build cache key";
const SCCACHE_ROUTING_STEP_NAME: &str = "Route Rust compilation through sccache";
const RELEASE_SCCACHE_RESTORE_STEP_NAME: &str = "Restore release-sccache-cache";
const CI_NATIVE_BUILD_JOB_ID: &str = "rust-core";

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

fn step_index_of(job_steps: &[&Value], step_name: &str) -> usize {
    job_steps
        .iter()
        .position(|step| step.get("name").and_then(Value::as_str) == Some(step_name))
        .unwrap_or_else(|| panic!("the packaging job has no step named {step_name}"))
}

fn cache_input<'a>(step: &'a Value, step_name: &str, input_name: &str) -> &'a str {
    required_text(
        required_field(step, "with", step_name),
        input_name,
        step_name,
    )
}

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

/// The packaging lane reuses the native MLX build ci.yml saves on main, so
/// its restore key must be ci.yml's key character for character: any drift
/// silently turns every packaging run back into a cold MLX build.
#[test]
fn should_restore_the_native_build_ci_saves_under_the_identical_key() {
    let ci_workflow = workflow_contract();
    let ci_steps = steps_of(
        required_job(&ci_workflow, CI_NATIVE_BUILD_JOB_ID),
        CI_NATIVE_BUILD_JOB_ID,
    );
    let ci_restore_step = find_step_by_id(&ci_steps, "native-build-cache");
    let workflow = macos_packaging_workflow_contract();
    let job_steps = steps_of(required_job(&workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID);
    let packaging_restore_step = find_step_by_name(&job_steps, NATIVE_BUILD_RESTORE_STEP_NAME);

    for input_name in ["key", "path"] {
        assert_eq!(
            cache_input(
                packaging_restore_step,
                NATIVE_BUILD_RESTORE_STEP_NAME,
                input_name
            ),
            cache_input(ci_restore_step, "native-build-cache", input_name),
            "the packaging lane's native-build {input_name} drifted from ci.yml"
        );
    }
}

#[test]
fn should_compute_the_native_key_and_restore_the_entry_before_packaging() {
    let workflow = macos_packaging_workflow_contract();
    let job_steps = steps_of(required_job(&workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID);
    let key_text = run_text_of(
        find_step_by_name(&job_steps, NATIVE_BUILD_KEY_STEP_NAME),
        NATIVE_BUILD_KEY_STEP_NAME,
    );
    assert!(
        key_text.contains("--profile core") && key_text.contains("--cache-environment"),
        "the key must be the core profile's, the profile app package prewarms: {key_text}"
    );
    let key_index = step_index_of(&job_steps, NATIVE_BUILD_KEY_STEP_NAME);
    let restore_index = step_index_of(&job_steps, NATIVE_BUILD_RESTORE_STEP_NAME);
    let package_index = step_index_of(&job_steps, PACKAGE_STEP_NAME);
    assert!(
        key_index < restore_index && restore_index < package_index,
        "the native build entry must be keyed, then restored, before the bundle is packaged"
    );
}

#[test]
fn should_route_rust_compilation_through_sccache_before_packaging() {
    let workflow = macos_packaging_workflow_contract();
    let job_steps = steps_of(required_job(&workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID);
    let routing_text = run_text_of(
        find_step_by_name(&job_steps, SCCACHE_ROUTING_STEP_NAME),
        SCCACHE_ROUTING_STEP_NAME,
    );
    for required_setting in [
        "RUSTC_WRAPPER=sccache",
        "CARGO_INCREMENTAL=0",
        "release-sccache",
    ] {
        assert!(
            routing_text.contains(required_setting),
            "the sccache routing step does not set {required_setting}"
        );
    }
    assert!(
        step_index_of(&job_steps, SCCACHE_ROUTING_STEP_NAME)
            < step_index_of(&job_steps, PACKAGE_STEP_NAME),
        "Rust must be routed through sccache before the bundle is packaged"
    );
}

/// Release objects are optimised and carry LTO bitcode, so they share nothing
/// with ci.yml's debug sccache; mixing the two would only evict each other.
#[test]
fn should_keep_release_objects_out_of_the_debug_sccache_family() {
    let workflow = macos_packaging_workflow_contract();
    let job_steps = steps_of(required_job(&workflow, PACKAGING_JOB_ID), PACKAGING_JOB_ID);
    let release_key = cache_input(
        find_step_by_name(&job_steps, RELEASE_SCCACHE_RESTORE_STEP_NAME),
        RELEASE_SCCACHE_RESTORE_STEP_NAME,
        "key",
    );
    assert!(release_key.starts_with("astronomical-v2-release-sccache-"));
    assert!(!release_key.starts_with("astronomical-v2-sccache-"));
}
