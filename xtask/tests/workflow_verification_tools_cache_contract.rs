//! The verification-tools cache owner across every ci.yml job that installs
//! the pinned cargo-about and sccache (issue #1986): each job restores and
//! saves the same path under the same key, the key rotates with the tool
//! pins, no prefix fallback can restore another pin's binaries, and a save
//! only follows a successful install. The disjointness of the macOS owners
//! stays in `workflow_cache_owner_contract`.

mod contract_support;

use contract_support::{
    find_step_by_id, required_field, required_job, required_text, steps_of, workflow_contract,
};
use serde_yaml::Value;

const COMPOSITE_CACHE_ACTION: &str = "./.github/actions/astronomical-cache";
const TOOL_PIN_KEY_INPUT: &str = "hashFiles('third-party/tool-pins/Cargo.toml')";
const INSTALL_SUCCESS_GATE: &str = "steps.verification-tools-install.outcome == 'success'";

/// One job's verification-tools cache wiring, read from its restore and
/// save steps.
struct VerificationToolsCache<'a> {
    restore_path: &'a str,
    restore_key: &'a str,
    save_path: &'a str,
    save_key: &'a str,
    save_condition: &'a str,
    has_restore_keys: bool,
}

fn cache_step_configuration<'a>(cache_step: &'a Value, context: &str) -> &'a Value {
    assert_eq!(
        required_text(cache_step, "uses", context),
        COMPOSITE_CACHE_ACTION,
        "{context} must go through the shared composite action"
    );
    required_field(cache_step, "with", context)
}

fn verification_tools_cache_of<'a>(
    job_steps: &[&'a Value],
    job_id: &str,
) -> VerificationToolsCache<'a> {
    let restore_context = format!("{job_id} verification-tools-cache restore");
    let save_context = format!("{job_id} verification-tools-cache save");
    let save_step = find_step_by_id(job_steps, "verification-tools-cache-save");
    let restore_configuration = cache_step_configuration(
        find_step_by_id(job_steps, "verification-tools-cache"),
        &restore_context,
    );
    let save_configuration = cache_step_configuration(save_step, &save_context);
    VerificationToolsCache {
        restore_path: required_text(restore_configuration, "path", &restore_context),
        restore_key: required_text(restore_configuration, "key", &restore_context),
        save_path: required_text(save_configuration, "path", &save_context),
        save_key: required_text(save_configuration, "key", &save_context),
        save_condition: required_text(save_step, "if", &save_context),
        has_restore_keys: restore_configuration.get("restore-keys").is_some(),
    }
}

/// The ci.yml jobs that run `install-verification-tools`.
const INSTALLING_JOB_IDS: [&str; 2] = ["rust-core", "notices-freshness"];

fn installing_job_caches(workflow: &Value) -> Vec<(&'static str, VerificationToolsCache<'_>)> {
    let mut job_caches = Vec::new();
    for job_id in INSTALLING_JOB_IDS {
        let job_steps = steps_of(required_job(workflow, job_id), job_id);
        job_caches.push((job_id, verification_tools_cache_of(&job_steps, job_id)));
    }
    job_caches
}

#[test]
fn should_save_the_same_path_and_key_each_job_restores() {
    let workflow = workflow_contract();
    for (job_id, tools_cache) in installing_job_caches(&workflow) {
        assert_eq!(
            tools_cache.restore_path, tools_cache.save_path,
            "{job_id} saves a different folder than it restores"
        );
        assert_eq!(
            tools_cache.restore_key, tools_cache.save_key,
            "{job_id} saves under a different key than it restores"
        );
    }
}

#[test]
fn should_key_every_job_identically_and_rotate_with_the_tool_pins() {
    let workflow = workflow_contract();
    let job_caches = installing_job_caches(&workflow);
    let (_, first_cache) = &job_caches[0];
    for (job_id, tools_cache) in &job_caches {
        assert_eq!(
            tools_cache.restore_key, first_cache.restore_key,
            "{job_id} keys verification tools differently; runner.os alone separates the hosts"
        );
    }
    assert!(
        first_cache.restore_key.contains(TOOL_PIN_KEY_INPUT),
        "the key must rotate with the tool pins, or a pin bump would restore old binaries and never save"
    );
}

#[test]
fn should_not_fall_back_to_another_pins_verification_tools() {
    let workflow = workflow_contract();
    for (job_id, tools_cache) in installing_job_caches(&workflow) {
        assert!(
            !tools_cache.has_restore_keys,
            "{job_id}: a prefix fallback could only restore another pin's binaries"
        );
    }
}

#[test]
fn should_save_only_after_a_successful_install() {
    let workflow = workflow_contract();
    for (job_id, tools_cache) in installing_job_caches(&workflow) {
        assert!(
            tools_cache.save_condition.contains(INSTALL_SUCCESS_GATE),
            "{job_id}: a failed install must never save a half-filled prefix under the primary key"
        );
    }
}
