//! Shared accessors for the workflow structural contract tests: typed
//! navigation over the parsed `ci.yml` and composite-action YAML so each
//! contract test reads as the assertion it pins. Each contract binary uses a
//! subset of this module, so unused members are expected per binary.
#![allow(dead_code)]

use serde_yaml::Value;
use std::path::{Path, PathBuf};

pub(crate) const EXPECTED_MACOS_AUTHORITY: &str = "${{ always() && (needs.detect-changes.result != 'success' || needs.detect-changes.outputs.macos_verification_required == 'true') }}";
pub(crate) const EXPECTED_MACOS_CONCURRENCY_GROUP: &str =
    "macos-integration-${{ github.event_name }}-${{ github.ref }}";
pub(crate) const EXPECTED_SWIFT_CONCURRENCY_GROUP: &str =
    "macos-swift-node-${{ github.event_name }}-${{ github.ref }}";
pub(crate) const EXPECTED_OBSERVATORY_COMMAND: &str = "node --test --test-reporter=spec apps/daemon/console/console.test.js apps/daemon/console/library.test.js apps/daemon/console/library-fetch.test.js apps/daemon/console/connect.test.js";

pub(crate) fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("the workspace root should resolve above the xtask package")
}

fn parse_yaml_file(file_path: &Path) -> Value {
    let yaml_text = std::fs::read_to_string(file_path)
        .unwrap_or_else(|error| panic!("{} must be readable: {error}", file_path.display()));
    serde_yaml::from_str(&yaml_text)
        .unwrap_or_else(|error| panic!("{} must parse as YAML: {error}", file_path.display()))
}

pub(crate) fn workflow_contract() -> Value {
    parse_yaml_file(&repository_root().join(".github/workflows/ci.yml"))
}

pub(crate) fn composite_action_contract() -> Value {
    parse_yaml_file(&repository_root().join(".github/actions/astronomical-cache/action.yml"))
}

/// `on:` parses as a boolean key under YAML 1.1 schemas and as the string
/// "on" under YAML 1.2; accept either so the contract survives a parser swap.
pub(crate) fn triggers_of(workflow: &Value) -> &Value {
    let workflow_mapping = workflow
        .as_mapping()
        .expect("the workflow must be a mapping");
    workflow_mapping
        .iter()
        .find(|(key, _)| key.as_str() == Some("on") || **key == Value::Bool(true))
        .map(|(_, value)| value)
        .expect("the workflow triggers are missing")
}

pub(crate) fn required_field<'a>(value: &'a Value, field_name: &str, context: &str) -> &'a Value {
    value
        .get(field_name)
        .unwrap_or_else(|| panic!("{context} is missing {field_name}"))
}

pub(crate) fn required_text<'a>(value: &'a Value, field_name: &str, context: &str) -> &'a str {
    required_field(value, field_name, context)
        .as_str()
        .unwrap_or_else(|| panic!("{context} field {field_name} must be a string"))
}

pub(crate) fn required_bool(value: &Value, field_name: &str, context: &str) -> bool {
    required_field(value, field_name, context)
        .as_bool()
        .unwrap_or_else(|| panic!("{context} field {field_name} must be a boolean"))
}

pub(crate) fn required_u64(value: &Value, field_name: &str, context: &str) -> u64 {
    required_field(value, field_name, context)
        .as_u64()
        .unwrap_or_else(|| panic!("{context} field {field_name} must be an unsigned integer"))
}

pub(crate) fn required_job<'a>(workflow: &'a Value, job_id: &str) -> &'a Value {
    required_field(
        required_field(workflow, "jobs", "the workflow"),
        job_id,
        "the workflow jobs",
    )
}

pub(crate) fn steps_of<'a>(job: &'a Value, job_id: &str) -> Vec<&'a Value> {
    let step_sequence = required_field(job, "steps", &format!("job {job_id}"))
        .as_sequence()
        .unwrap_or_else(|| panic!("job {job_id} steps must be a sequence"));
    step_sequence.iter().collect()
}

pub(crate) fn find_step_by_id<'a>(job_steps: &[&'a Value], step_id: &str) -> &'a Value {
    job_steps
        .iter()
        .find(|step| step.get("id").and_then(Value::as_str) == Some(step_id))
        .unwrap_or_else(|| panic!("step id={step_id} is missing"))
}

pub(crate) fn find_step_by_name<'a>(job_steps: &[&'a Value], step_name: &str) -> &'a Value {
    job_steps
        .iter()
        .find(|step| step.get("name").and_then(Value::as_str) == Some(step_name))
        .unwrap_or_else(|| panic!("step name={step_name} is missing"))
}

pub(crate) fn run_text_of<'a>(step: &'a Value, step_id: &str) -> &'a str {
    required_text(step, "run", &format!("step id={step_id}"))
}

/// `actions/cache/restore@<40-hex>` shape without a regex dependency.
pub(crate) fn is_pinned_action_reference(uses_text: &str, action_name: &str) -> bool {
    let Some(after_action) = uses_text.strip_prefix(action_name) else {
        return false;
    };
    let Some(pinned_digest) = after_action.strip_prefix('@') else {
        return false;
    };
    pinned_digest.len() == 40
        && pinned_digest
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

pub(crate) fn path_lines(path_text: &str) -> Vec<&str> {
    path_text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect()
}
