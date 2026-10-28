//! The orchestration crate's integration-lane entry: the workflow lane
//! contract requires every package in the integration selection to expose
//! this target. The modules' exhaustive coverage lives in their
//! `#[cfg(test)]` submodules; this canary keeps the lane observable.

use astronomical_native_build_orchestration::native_parallelism::resolve_native_parallel_job_count;

#[test]
fn should_run_the_native_build_orchestration_integration_smoke_contract() {
    // One representative resolution through the public API, enough to prove
    // the crate's test surface is alive inside the integration lane.
    let resolved_job_count = resolve_native_parallel_job_count(Some("2"), Some("3"), Some("3"), 3);
    assert_eq!(
        resolved_job_count, "2",
        "the native-specific override should win so the lane canary observes real resolution"
    );
}
