use super::{MAXIMUM_WAITING_REPORTS, WaitingReports};
use crate::problem_reports::{MacDescription, ProblemKind, ProblemReport};

fn report_of_kind(problem: ProblemKind) -> ProblemReport {
    ProblemReport {
        app_version: "0.2.126".to_owned(),
        mac: MacDescription {
            chip_name: "Apple M5 Pro".to_owned(),
            memory_gb: 48,
            macos_version: "27.2".to_owned(),
        },
        model: None,
        problem,
        memory: None,
    }
}

#[test]
fn should_wait_until_the_report_is_taken_once() {
    let waiting_reports = WaitingReports::default();
    let report_number = waiting_reports.offer(report_of_kind(ProblemKind::ModelLoadFailed));
    assert_eq!(waiting_reports.waiting().len(), 1);
    assert_eq!(
        waiting_reports.take(report_number),
        Some(report_of_kind(ProblemKind::ModelLoadFailed))
    );
    assert_eq!(waiting_reports.take(report_number), None);
    assert!(waiting_reports.waiting().is_empty());
}

#[test]
fn should_keep_one_report_when_the_same_failure_repeats() {
    let waiting_reports = WaitingReports::default();
    let first_number = waiting_reports.offer(report_of_kind(ProblemKind::DownloadFailed));
    let repeat_number = waiting_reports.offer(report_of_kind(ProblemKind::DownloadFailed));
    assert_eq!(first_number, repeat_number);
    assert_eq!(waiting_reports.waiting().len(), 1);
}

#[test]
fn should_keep_only_the_newest_reports_when_too_many_wait() {
    let waiting_reports = WaitingReports::default();
    let problem_kinds = [
        ProblemKind::ModelLoadFailed,
        ProblemKind::DownloadFailed,
        ProblemKind::ReplyOutOfMemory,
        ProblemKind::EngineDidNotStart,
    ];
    for problem_kind in problem_kinds {
        waiting_reports.offer(report_of_kind(problem_kind));
    }
    let waiting = waiting_reports.waiting();
    assert_eq!(waiting.len(), MAXIMUM_WAITING_REPORTS);
    assert_eq!(waiting[0].report.problem, ProblemKind::DownloadFailed);
    assert_eq!(waiting[2].report.problem, ProblemKind::EngineDidNotStart);
}

#[test]
fn should_refuse_to_take_a_number_that_never_waited() {
    let waiting_reports = WaitingReports::default();
    assert_eq!(waiting_reports.take(42), None);
}
