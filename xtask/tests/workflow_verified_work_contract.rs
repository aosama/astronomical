//! The records that let a default-branch push skip the macOS lanes, as typed Rust:
//! parses `.github/workflows/ci.yml` and pins that a record can only mean those lanes
//! really passed, and that writing one can never fail the required check. What reads
//! the records is pinned by the `integration_classify_scope_journeys` test.

mod contract_support;

use contract_support::{
    find_step_by_name, required_field, required_job, required_text, required_u64, steps_of,
    workflow_contract,
};

/// A push skips the macOS lanes on these records, so they must mean those lanes
/// really passed, and the records must never be able to fail the required check.
#[test]
fn should_record_verified_work_only_after_the_verdict_passed_on_a_real_macos_run() {
    let workflow = workflow_contract();
    let record_job = required_job(&workflow, "record-verified-work");
    let record_condition = required_text(record_job, "if", "job record-verified-work");
    assert!(
        record_condition.contains("needs.verdict.result == 'success'"),
        "only a passed verdict may leave a record"
    );
    assert!(
        record_condition.contains("macos_verification_required == 'true'"),
        "a run that skipped the macOS lanes proves nothing about them"
    );
    for loosening_function in ["always()", "failure()", "cancelled()"] {
        assert!(
            !record_condition.contains(loosening_function),
            "{loosening_function} would record a run that did not pass"
        );
    }
    let verdict_job = required_job(&workflow, "verdict");
    let verdict_steps = steps_of(verdict_job, "verdict");
    assert_eq!(
        verdict_steps.len(),
        1,
        "the verdict stays a pure decision node, so nothing beside it can turn it red"
    );
    let verdict_needs = required_field(verdict_job, "needs", "job verdict");
    assert!(
        !verdict_needs
            .as_sequence()
            .expect("verdict needs must be a sequence")
            .iter()
            .any(|lane| lane.as_str() == Some("record-verified-work")),
        "losing a record costs the next merge a full run, and must never block a merge"
    );
    let record_steps = steps_of(record_job, "record-verified-work");
    let tree_step = find_step_by_name(&record_steps, "Record that this tree passed");
    assert!(
        required_text(tree_step, "if", "tree record").contains("'pull_request'"),
        "only a pull request's result is a record a later merge can reuse"
    );
    let tree_inputs = required_field(tree_step, "with", "tree record");
    assert!(
        required_text(tree_inputs, "name", "tree record inputs").starts_with("verified-tree-"),
        "the name must match what xtask/src/verified_tree.rs looks for"
    );
    let run_step = find_step_by_name(
        &record_steps,
        "Record that the default branch ran the macOS lanes",
    );
    assert!(
        required_text(run_step, "if", "run record").contains("'push'"),
        "only a default-branch push publishes caches"
    );
    let run_inputs = required_field(run_step, "with", "run record");
    assert_eq!(
        required_text(run_inputs, "name", "run record inputs"),
        "default-branch-macos-run",
        "the name must match what xtask/src/verified_tree.rs looks for"
    );
    assert!(
        required_u64(run_inputs, "retention-days", "run record inputs") < 7,
        "it must expire before GitHub evicts a cache left unused for seven days"
    );
}

#[test]
fn should_let_the_classifier_read_records_and_nothing_more() {
    let workflow = workflow_contract();
    let detection_job = required_job(&workflow, "detect-changes");
    let permissions = required_field(detection_job, "permissions", "job detect-changes");
    assert_eq!(
        required_text(permissions, "actions", "detect-changes permissions"),
        "read"
    );
    assert_eq!(
        required_text(permissions, "contents", "detect-changes permissions"),
        "read"
    );
}
