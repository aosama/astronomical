//! The run timeline, as typed Rust: parses `.github/workflows/ci.yml` and pins
//! the job that draws every job and step of a run from GitHub's own step
//! records (issue #2635).

mod contract_support;

use contract_support::{
    find_step_by_name, is_pinned_action_reference, required_field, required_job, required_text,
    steps_of, workflow_contract,
};
use serde_yaml::Value;

const TIMELINE_ACTION: &str = "Kesin11/actions-timeline";

/// The run's timing comes from GitHub's own step records, drawn by the
/// published actions-timeline action in a job of its own (issue #2635). It
/// must run after the verdict, also when a lane failed, since a failed run is
/// the one whose time matters most, and it must never gate a merge.
#[test]
fn should_draw_the_run_timeline_after_every_job_without_gating_the_merge() {
    let workflow = workflow_contract();
    let timeline_job = required_job(&workflow, "timeline");
    assert_eq!(
        required_text(timeline_job, "if", "job timeline"),
        "${{ always() }}",
        "the timeline must be drawn for failed and cancelled lanes too"
    );
    let timeline_needs = required_field(timeline_job, "needs", "job timeline")
        .as_sequence()
        .expect("timeline needs must be a sequence");
    assert!(
        timeline_needs.contains(&Value::from("verdict")),
        "the timeline must wait for the verdict, which waits for every lane"
    );
    let permissions = required_field(timeline_job, "permissions", "job timeline");
    assert_eq!(
        required_text(permissions, "actions", "timeline permissions"),
        "read",
        "the timeline reads the run's jobs and steps through the actions API"
    );
    let timeline_steps = steps_of(timeline_job, "timeline");
    let draw_step = find_step_by_name(&timeline_steps, "Draw the run timeline");
    assert!(is_pinned_action_reference(
        required_text(draw_step, "uses", "timeline step"),
        TIMELINE_ACTION
    ));
    let verdict_needs = required_field(required_job(&workflow, "verdict"), "needs", "job verdict")
        .as_sequence()
        .expect("verdict needs must be a sequence");
    assert!(
        !verdict_needs.contains(&Value::from("timeline")),
        "the timeline is a report, not a merge gate"
    );
}
