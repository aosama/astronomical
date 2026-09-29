//! Parallelism resolution contracts: the native build must always run with an
//! explicit, reported job count and never silently fall back to a serial build.

use std::num::NonZeroU32;

#[path = "../../build_parallelism.rs"]
mod build_parallelism;

use build_parallelism::{NativeBuildJobSource, resolve_native_build_jobs};

fn machine_parallelism_of(machine_job_count: Option<u32>) -> Option<NonZeroU32> {
    machine_job_count.and_then(NonZeroU32::new)
}

#[test]
fn should_prefer_the_explicit_override_over_every_other_source() {
    let resolved_jobs = resolve_native_build_jobs(
        |variable_name| match variable_name {
            "ASTRONOMICAL_NATIVE_BUILD_JOBS" => Some("3".to_owned()),
            "CARGO_BUILD_JOBS" => Some("7".to_owned()),
            "NUM_JOBS" => Some("9".to_owned()),
            _ => None,
        },
        || machine_parallelism_of(Some(11)),
    )
    .expect("an explicit override should resolve");

    assert_eq!(resolved_jobs.job_count.get(), 3);
    assert_eq!(resolved_jobs.source, NativeBuildJobSource::ExplicitOverride);
}

#[test]
fn should_use_the_cargo_provided_job_count_before_lower_priority_sources() {
    let resolved_jobs = resolve_native_build_jobs(
        |variable_name| match variable_name {
            "CARGO_BUILD_JOBS" => Some("5".to_owned()),
            "NUM_JOBS" => Some("9".to_owned()),
            _ => None,
        },
        || machine_parallelism_of(Some(11)),
    )
    .expect("a cargo-provided job count should resolve");

    assert_eq!(resolved_jobs.job_count.get(), 5);
    assert_eq!(resolved_jobs.source, NativeBuildJobSource::CargoBuildJobs);
}

#[test]
fn should_use_the_make_job_count_before_machine_parallelism() {
    let resolved_jobs = resolve_native_build_jobs(
        |variable_name| match variable_name {
            "NUM_JOBS" => Some("4".to_owned()),
            _ => None,
        },
        || machine_parallelism_of(Some(11)),
    )
    .expect("a make job count should resolve");

    assert_eq!(resolved_jobs.job_count.get(), 4);
    assert_eq!(resolved_jobs.source, NativeBuildJobSource::MakeJobs);
}

#[test]
fn should_use_machine_parallelism_as_the_last_resort() {
    let resolved_jobs =
        resolve_native_build_jobs(|_variable_name| None, || machine_parallelism_of(Some(8)))
            .expect("machine parallelism should resolve");

    assert_eq!(resolved_jobs.job_count.get(), 8);
    assert_eq!(
        resolved_jobs.source,
        NativeBuildJobSource::MachineParallelism
    );
}

#[test]
fn should_reject_a_malformed_explicit_override_instead_of_falling_through() {
    let resolution_error = resolve_native_build_jobs(
        |variable_name| match variable_name {
            "ASTRONOMICAL_NATIVE_BUILD_JOBS" => Some("four".to_owned()),
            "CARGO_BUILD_JOBS" => Some("7".to_owned()),
            _ => None,
        },
        || machine_parallelism_of(Some(11)),
    )
    .expect_err("a malformed explicit override must be a loud error");

    assert!(
        resolution_error.contains("ASTRONOMICAL_NATIVE_BUILD_JOBS"),
        "the error should name the misconfigured variable: {resolution_error}"
    );
}

#[test]
fn should_reject_a_zero_job_count() {
    let resolution_error = resolve_native_build_jobs(
        |variable_name| match variable_name {
            "ASTRONOMICAL_NATIVE_BUILD_JOBS" => Some("0".to_owned()),
            _ => None,
        },
        || machine_parallelism_of(Some(11)),
    )
    .expect_err("a zero job count must be a loud error");

    assert!(
        resolution_error.contains("at least 1"),
        "the error should explain the minimum: {resolution_error}"
    );
}

#[test]
fn should_reject_a_malformed_cargo_job_count_instead_of_falling_through() {
    let resolution_error = resolve_native_build_jobs(
        |variable_name| match variable_name {
            "CARGO_BUILD_JOBS" => Some("not-a-number".to_owned()),
            "NUM_JOBS" => Some("4".to_owned()),
            _ => None,
        },
        || machine_parallelism_of(Some(11)),
    )
    .expect_err("a malformed cargo job count must be a loud error");

    assert!(
        resolution_error.contains("CARGO_BUILD_JOBS"),
        "the error should name the misconfigured variable: {resolution_error}"
    );
}

#[test]
fn should_refuse_to_resolve_parallelism_when_the_machine_reports_none() {
    let resolution_error =
        resolve_native_build_jobs(|_variable_name| None, || machine_parallelism_of(None))
            .expect_err("unresolvable machine parallelism must be a loud error");

    assert!(
        resolution_error.contains("ASTRONOMICAL_NATIVE_BUILD_JOBS"),
        "the error should point at the explicit override escape hatch: {resolution_error}"
    );
}

#[test]
fn should_report_a_stable_log_value_for_every_job_source() {
    assert_eq!(
        NativeBuildJobSource::ExplicitOverride.log_value(),
        "explicit-override"
    );
    assert_eq!(
        NativeBuildJobSource::CargoBuildJobs.log_value(),
        "cargo-build-jobs"
    );
    assert_eq!(NativeBuildJobSource::MakeJobs.log_value(), "make-jobs");
    assert_eq!(
        NativeBuildJobSource::MachineParallelism.log_value(),
        "machine-parallelism"
    );
}

#[test]
fn should_accept_surrounding_whitespace_in_job_counts() {
    let resolved_jobs = resolve_native_build_jobs(
        |variable_name| match variable_name {
            "ASTRONOMICAL_NATIVE_BUILD_JOBS" => Some(" 6 ".to_owned()),
            _ => None,
        },
        || machine_parallelism_of(Some(11)),
    )
    .expect("a whitespace-padded job count should resolve");

    assert_eq!(resolved_jobs.job_count.get(), 6);
}
