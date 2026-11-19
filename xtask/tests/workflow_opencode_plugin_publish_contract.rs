//! The OpenCode plugin's npm publish workflow, as typed Rust: parses
//! `.github/workflows/publish-opencode-plugin.yml` and pins what it promises.
//! It runs only for plugin tags, never on branches or pull requests; it checks
//! the tag against the package and its changelog, and that the tagged commit
//! is on main, before anything else; it
//! tests the plugin before publishing; it publishes through npm trusted
//! publishing with provenance and no stored token; and it never creates a
//! GitHub Release, which would take over the app's "latest release" links.

mod contract_support;

use contract_support::{
    find_step_by_name, opencode_plugin_publish_workflow_contract, required_field, required_job,
    required_text, run_text_of, steps_of, triggers_of,
};
use serde_yaml::Value;

const PUBLISH_JOB_ID: &str = "publish";
const RELEASE_CHECK_STEP_NAME: &str = "Check the tag against the package and its changelog";
const VERIFY_STEP_NAME: &str = "Install, check formatting and types, and test the plugin";
const PUBLISH_STEP_NAME: &str = "Publish to npm with provenance";

/// Fragments that would mean the workflow makes a GitHub Release or publishes
/// with a stored credential instead of trusted publishing.
const FORBIDDEN_FRAGMENTS: [&str; 4] = ["gh release", "secrets.", "NODE_AUTH_TOKEN", "NPM_TOKEN"];

fn publish_steps(workflow: &Value) -> Vec<&Value> {
    steps_of(required_job(workflow, PUBLISH_JOB_ID), PUBLISH_JOB_ID)
}

fn step_position(job_steps: &[&Value], step_name: &str) -> usize {
    let mut found_position = None;
    for (step_index, job_step) in job_steps.iter().enumerate() {
        if job_step.get("name").and_then(Value::as_str) == Some(step_name) {
            found_position = Some(step_index);
        }
    }
    found_position.unwrap_or_else(|| panic!("the publish job lost the step {step_name:?}"))
}

#[test]
fn should_run_only_for_plugin_tags() {
    let workflow = opencode_plugin_publish_workflow_contract();
    let triggers = triggers_of(&workflow)
        .as_mapping()
        .expect("the triggers must be a mapping");
    assert_eq!(triggers.len(), 1, "only a tag push may publish");
    let push_trigger = required_field(triggers_of(&workflow), "push", "on");
    assert!(push_trigger.get("branches").is_none());
    let tag_patterns = required_field(push_trigger, "tags", "on.push")
        .as_sequence()
        .expect("the tag patterns must be a list");
    assert_eq!(tag_patterns, &vec![Value::from("opencode-plugin-v*")]);
}

#[test]
fn should_grant_the_oidc_token_only_to_the_publish_job() {
    let workflow = opencode_plugin_publish_workflow_contract();
    let workflow_permissions = required_field(&workflow, "permissions", "workflow");
    assert!(workflow_permissions.get("id-token").is_none());
    let job_permissions = required_field(
        required_job(&workflow, PUBLISH_JOB_ID),
        "permissions",
        PUBLISH_JOB_ID,
    );
    assert_eq!(
        required_text(job_permissions, "id-token", PUBLISH_JOB_ID),
        "write"
    );
    assert_eq!(
        required_text(job_permissions, "contents", PUBLISH_JOB_ID),
        "read"
    );
}

#[test]
fn should_check_the_tag_and_test_the_plugin_before_publishing() {
    let workflow = opencode_plugin_publish_workflow_contract();
    let job_steps = publish_steps(&workflow);
    let check_step = find_step_by_name(&job_steps, RELEASE_CHECK_STEP_NAME);
    assert!(
        run_text_of(check_step, "release check")
            .contains("xtask opencode-plugin-release-check --require-on-main \"$GITHUB_REF_NAME\"")
    );
    let checkout_step = find_step_by_name(&job_steps, "Check out source");
    let checkout_settings = required_field(checkout_step, "with", "checkout");
    assert_eq!(
        required_field(checkout_settings, "fetch-depth", "checkout").as_u64(),
        Some(0),
        "the main-branch check needs the full history"
    );
    assert!(
        run_text_of(find_step_by_name(&job_steps, VERIFY_STEP_NAME), "verify")
            .contains("npm run verify")
    );
    let publish_position = step_position(&job_steps, PUBLISH_STEP_NAME);
    assert!(step_position(&job_steps, RELEASE_CHECK_STEP_NAME) < publish_position);
    assert!(step_position(&job_steps, VERIFY_STEP_NAME) < publish_position);
}

#[test]
fn should_publish_with_provenance_through_trusted_publishing() {
    let workflow = opencode_plugin_publish_workflow_contract();
    let job_steps = publish_steps(&workflow);
    let publish_text = run_text_of(find_step_by_name(&job_steps, PUBLISH_STEP_NAME), "publish");
    assert!(publish_text.contains("npm publish --provenance --access public"));
    let workflow_text = serde_yaml::to_string(&workflow).expect("the workflow re-serializes");
    for forbidden_fragment in FORBIDDEN_FRAGMENTS {
        assert!(
            !workflow_text.contains(forbidden_fragment),
            "the publish workflow must not contain {forbidden_fragment:?}"
        );
    }
}
