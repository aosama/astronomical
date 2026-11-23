//! The code-shape ratchet backstop contract: the Linux config-contracts lane
//! must run the complexity ratchet (`cargo xtask crap-baseline`) and the
//! duplication ratchet (`cargo xtask check-duplication`) against the base
//! commit, with a SHA-verified cargo-crap. The git hooks run both first, but
//! a checkout whose hooks path names a missing folder runs no hook at all and
//! says nothing; six grown functions reached main that way on 2026-11-22.
//! Without this contract the backstop could be dropped or commented out and
//! every lane would stay green.

mod contract_support;

use contract_support::{
    find_step_by_name, is_pinned_action_reference, required_field, required_job, required_text,
    required_u64, run_text_of, steps_of, workflow_contract,
};
use serde_yaml::Value;

const CONFIG_CONTRACTS_JOB: &str = "config-contracts";
const PROVISION_STEP_NAME: &str = "Provision pinned cargo-crap";
const RATCHET_STEP_NAME: &str = "Run code-shape ratchets";
const INSTALL_ACTION: &str = "taiki-e/install-action";
const CARGO_CRAP_TOOL_PREFIX: &str = "cargo-crap@";
const COMPLEXITY_RATCHET_COMMAND: &str = "cargo xtask crap-baseline --against HEAD^1";
const DUPLICATION_RATCHET_COMMAND: &str = "cargo xtask check-duplication --against HEAD^1";

/// Whether `run_text` runs `command` on a line that is not commented out.
fn runs_command(run_text: &str, command: &str) -> bool {
    for run_line in run_text.lines() {
        let trimmed_line = run_line.trim_start();
        if !trimmed_line.starts_with('#') && trimmed_line.starts_with(command) {
            return true;
        }
    }
    false
}

/// `cargo-crap@0.9.0` pins one release; a bare name or a partial version
/// would follow whatever the action's manifest calls latest.
fn is_exact_cargo_crap_pin(tool_text: &str) -> bool {
    let Some(pinned_version) = tool_text.strip_prefix(CARGO_CRAP_TOOL_PREFIX) else {
        return false;
    };
    let version_components: Vec<&str> = pinned_version.split('.').collect();
    if version_components.len() != 3 {
        return false;
    }
    for version_component in version_components {
        if version_component.is_empty()
            || !version_component.bytes().all(|byte| byte.is_ascii_digit())
        {
            return false;
        }
    }
    true
}

fn position_of_step(job_steps: &[&Value], step_name: &str) -> usize {
    for (step_position, job_step) in job_steps.iter().enumerate() {
        if job_step.get("name").and_then(Value::as_str) == Some(step_name) {
            return step_position;
        }
    }
    panic!("{CONFIG_CONTRACTS_JOB} has no step named {step_name}");
}

#[test]
fn should_run_both_ratchets_against_the_base_commit_on_the_config_lane() {
    let workflow = workflow_contract();
    let job_steps = steps_of(
        required_job(&workflow, CONFIG_CONTRACTS_JOB),
        CONFIG_CONTRACTS_JOB,
    );
    let ratchet_step = find_step_by_name(&job_steps, RATCHET_STEP_NAME);
    let ratchet_text = run_text_of(ratchet_step, RATCHET_STEP_NAME);
    assert!(runs_command(ratchet_text, COMPLEXITY_RATCHET_COMMAND));
    assert!(runs_command(ratchet_text, DUPLICATION_RATCHET_COMMAND));
    assert!(
        position_of_step(&job_steps, PROVISION_STEP_NAME)
            < position_of_step(&job_steps, RATCHET_STEP_NAME),
        "cargo-crap must be provisioned before the ratchets run"
    );
}

#[test]
fn should_check_out_the_base_commit_for_the_ratchets() {
    let workflow = workflow_contract();
    let job_steps = steps_of(
        required_job(&workflow, CONFIG_CONTRACTS_JOB),
        CONFIG_CONTRACTS_JOB,
    );
    let checkout_step = find_step_by_name(&job_steps, "Check out source");
    let checkout_inputs = required_field(checkout_step, "with", "config-contracts checkout");
    assert!(
        required_u64(checkout_inputs, "fetch-depth", "config-contracts checkout") >= 2,
        "a one-commit checkout has no HEAD^1 for the ratchets to judge against"
    );
}

#[test]
fn should_provision_cargo_crap_through_a_sha_pinned_verified_install_action() {
    let workflow = workflow_contract();
    let job_steps = steps_of(
        required_job(&workflow, CONFIG_CONTRACTS_JOB),
        CONFIG_CONTRACTS_JOB,
    );
    let provision_step = find_step_by_name(&job_steps, PROVISION_STEP_NAME);
    let uses_text = required_text(provision_step, "uses", PROVISION_STEP_NAME);
    assert!(
        is_pinned_action_reference(uses_text, INSTALL_ACTION),
        "got {uses_text}"
    );
    let provision_inputs = required_field(provision_step, "with", PROVISION_STEP_NAME);
    assert!(is_exact_cargo_crap_pin(required_text(
        provision_inputs,
        "tool",
        PROVISION_STEP_NAME
    )));
    assert_eq!(
        required_text(provision_inputs, "fallback", PROVISION_STEP_NAME),
        "none"
    );
    assert!(provision_inputs.get("checksum").is_none());
}

#[test]
fn should_not_count_a_commented_out_or_missing_ratchet_as_running() {
    let ratchet_text =
        "# cargo xtask crap-baseline --against HEAD^1\ncargo xtask notices --check\n";
    assert!(!runs_command(ratchet_text, COMPLEXITY_RATCHET_COMMAND));
    assert!(!runs_command("", DUPLICATION_RATCHET_COMMAND));
    assert!(runs_command(
        "  cargo xtask check-duplication --against HEAD^1 --top 10\n",
        DUPLICATION_RATCHET_COMMAND
    ));
}

#[test]
fn should_accept_only_an_exact_cargo_crap_release_pin() {
    assert!(is_exact_cargo_crap_pin("cargo-crap@0.9.0"));
    assert!(!is_exact_cargo_crap_pin("cargo-crap"));
    assert!(!is_exact_cargo_crap_pin("cargo-crap@0.9"));
    assert!(!is_exact_cargo_crap_pin("cargo-crap@0.9.x"));
    assert!(!is_exact_cargo_crap_pin("cargo-nextest@0.9.0"));
}
